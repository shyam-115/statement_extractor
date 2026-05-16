"""
OCR Engine — PDF/image ingestion + coordinate extraction.

Pipeline:
  PDF / image path
      ↓
  render_pages()  → list of numpy images
      ↓
  _run_ocr()      → raw PaddleOCR output
      ↓
  _build_tokens() → List[OCRToken] (normalized coordinates)

Design decisions:
- PDF pages are rasterised via PyMuPDF (fitz) at the configured DPI for
  maximum text sharpness.  pdf2image is used as a fallback.
- Rotation correction is delegated to PaddleOCR's angle classifier; we
  additionally apply a lightweight deskew step using OpenCV moments.
- All coordinate values emitted are **normalised** (0..1) relative to the
  rendered page dimensions, making all downstream layout logic
  resolution-independent.
"""
from __future__ import annotations

import logging
import os

# Set before Paddle is imported (mitigates Paddle 3.x + oneDNN CPU errors).
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ.setdefault("FLAGS_json_format_model", "0")

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import OCRConfig
from ..schemas import OCRToken
from ..parsing.numeric_parser import NumericParser

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Wraps PaddleOCR and converts its output into a normalised token stream.

    Parameters
    ----------
    config : OCRConfig
        Tuning parameters for the underlying OCR model.
    """

    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._paddle = self._init_paddle()
        self._numeric_parser = NumericParser()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_paddle(self):
        """Lazy-import PaddleOCR so the module can be imported even when the
        library is not yet installed (e.g. during unit tests with mocks)."""
        try:
            import inspect
            from paddleocr import PaddleOCR  # type: ignore

            # Paddle 3.x + oneDNN can fail on some CPUs; disable when requested.
            if self.config.disable_mkldnn:
                os.environ.setdefault("FLAGS_use_mkldnn", "0")

            sig = inspect.signature(PaddleOCR.__init__)
            params = set(sig.parameters)

            if "use_angle_cls" in params:
                # PaddleOCR 2.x
                return PaddleOCR(
                    use_angle_cls=self.config.use_angle_cls,
                    lang=self.config.lang,
                    det_db_thresh=self.config.det_db_thresh,
                    det_db_box_thresh=self.config.det_db_box_thresh,
                    rec_batch_num=self.config.rec_batch_num,
                    show_log=False,
                )

            # PaddleOCR 3.x (PaddleX pipeline) — PP-OCRv4 avoids oneDNN PIR bugs on CPU
            kwargs = {
                "lang": self.config.lang,
                "use_textline_orientation": self.config.use_angle_cls,
                "text_det_thresh": self.config.det_db_thresh,
                "text_det_box_thresh": self.config.det_db_box_thresh,
                "text_rec_score_thresh": self.config.min_confidence,
            }
            if "ocr_version" in params:
                kwargs["ocr_version"] = self.config.ocr_version
            if "use_doc_orientation_classify" in params:
                kwargs["use_doc_orientation_classify"] = False
            if "use_doc_unwarping" in params:
                kwargs["use_doc_unwarping"] = False
            if "text_recognition_batch_size" in params:
                kwargs["text_recognition_batch_size"] = self.config.rec_batch_num
            return PaddleOCR(**kwargs)
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is required.  "
                "Install via: pip install paddlepaddle paddleocr"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(self, file_path: str) -> Tuple[List[List[OCRToken]], List[np.ndarray]]:
        """
        Process a PDF or image file and return OCR tokens per page.

        Returns
        -------
        tokens_per_page : list of lists of OCRToken
        images          : list of rendered page images (numpy BGR arrays)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            images = self._pdf_to_images(str(path))
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
            images = [self._load_image(str(path))]
        else:
            # Attempt generic image load
            images = [self._load_image(str(path))]

        tokens_per_page: List[List[OCRToken]] = []
        for page_num, img in enumerate(images):
            img = self._deskew(img)
            tokens = self._run_ocr(img, page_num)
            tokens_per_page.append(tokens)
            logger.debug("Page %d → %d tokens", page_num, len(tokens))

        return tokens_per_page, images

    # ------------------------------------------------------------------
    # PDF → images
    # ------------------------------------------------------------------

    def _pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """Rasterise PDF pages using PyMuPDF (primary) or pdf2image (fallback)."""
        try:
            return self._pdf_via_pymupdf(pdf_path)
        except ImportError:
            logger.warning("PyMuPDF not available, falling back to pdf2image")
        try:
            return self._pdf_via_pdf2image(pdf_path)
        except ImportError:
            raise ImportError(
                "Either PyMuPDF (pip install pymupdf) or "
                "pdf2image (pip install pdf2image) is required to process PDFs."
            )

    def _pdf_via_pymupdf(self, pdf_path: str) -> List[np.ndarray]:
        import fitz  # type: ignore  # PyMuPDF

        doc = fitz.open(pdf_path)
        images: List[np.ndarray] = []
        zoom = self.config.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        doc.close()
        return images

    def _pdf_via_pdf2image(self, pdf_path: str) -> List[np.ndarray]:
        from pdf2image import convert_from_path  # type: ignore
        from PIL import Image as PILImage

        pil_images = convert_from_path(pdf_path, dpi=self.config.dpi)
        result: List[np.ndarray] = []
        for pil_img in pil_images:
            arr = np.array(pil_img.convert("RGB"))
            result.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return result

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            # Try PIL as fallback (handles unusual formats)
            try:
                from PIL import Image as PILImage
                pil = PILImage.open(path).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                raise ValueError(f"Cannot read image file: {path}") from exc
        return img

    # ------------------------------------------------------------------
    # Deskew
    # ------------------------------------------------------------------

    @staticmethod
    def _deskew(img: np.ndarray) -> np.ndarray:
        """
        Lightweight deskew using projection profile / image moments.
        Only corrects small skew angles (±15°) to avoid mangling
        already-straight documents.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 100:
            return img  # not enough text pixels — skip

        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect returns angles in [-90, 0); adjust to [-45, 45)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.3 or abs(angle) > 15:
            return img  # negligible or extreme — skip

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        logger.debug("Deskew applied: %.2f°", angle)
        return rotated

    # ------------------------------------------------------------------
    # OCR execution
    # ------------------------------------------------------------------

    def _run_ocr(self, img: np.ndarray, page_num: int) -> List[OCRToken]:
        """Run PaddleOCR on a single image and convert output to OCRTokens."""
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return []

        predict_fn = getattr(self._paddle, "predict", None) or self._paddle.ocr
        try:
            result = predict_fn(img)
        except TypeError:
            result = predict_fn(img, cls=self.config.use_angle_cls)

        tokens: List[OCRToken] = []
        for text, confidence, box_points in self._iter_ocr_lines(result):
            if confidence < self.config.min_confidence:
                continue
            text = text.strip()
            if not text:
                continue

            xs = [float(p[0]) for p in box_points]
            ys = [float(p[1]) for p in box_points]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            tokens.append(
                OCRToken(
                    text=text,
                    confidence=float(confidence),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    center_x=cx,
                    center_y=cy,
                    normalized_x=cx / w,
                    normalized_y=cy / h,
                    page_num=page_num,
                    is_numeric=self._numeric_parser.looks_like_number(text),
                    is_date=self._numeric_parser.is_date(text),
                )
            )

        return tokens

    @staticmethod
    def _iter_ocr_lines(result):
        """
        Yield (text, confidence, box_points) from PaddleOCR 2.x or 3.x output.
        """
        if not result:
            return

        # PaddleOCR 3.x: list[OCRResult] with rec_polys / rec_texts / rec_scores
        first = result[0]
        if hasattr(first, "get") and (
            "rec_texts" in first or "rec_polys" in first
        ):
            for page in result:
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                polys = page.get("rec_polys") or page.get("dt_polys") or []
                for text, score, poly in zip(texts, scores, polys):
                    if isinstance(text, (list, tuple)):
                        text = text[0]
                    box = OCREngine._poly_to_box(poly)
                    if box is not None:
                        yield str(text), float(score), box
            return

        # PaddleOCR 2.x: [[[box, (text, conf)], ...]]
        lines = first if isinstance(first, list) else result
        for line in lines:
            if line is None or len(line) < 2:
                continue
            box_points, rec = line[0], line[1]
            if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                text, confidence = rec[0], rec[1]
            else:
                text, confidence = str(rec), 1.0
            yield text, float(confidence), box_points

    @staticmethod
    def _poly_to_box(poly) -> Optional[list]:
        """Convert a detection polygon to a 4-point axis-aligned box."""
        if poly is None:
            return None
        arr = np.asarray(poly, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            return None
        xs, ys = arr[:, 0], arr[:, 1]
        return [
            [float(xs.min()), float(ys.min())],
            [float(xs.max()), float(ys.min())],
            [float(xs.max()), float(ys.max())],
            [float(xs.min()), float(ys.max())],
        ]
