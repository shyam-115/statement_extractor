"""
PDF Ingestor — structural parsing for digital PDFs.

Uses PyMuPDF dict-mode text extraction and drawing detection for
table boundaries.  Falls back to OCR when text density is too low.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np

from ..schemas import OCRToken
from ..parsing.numeric_parser import NumericParser

logger = logging.getLogger(__name__)

MIN_CHARS_PER_PAGE = 50


@dataclass
class TextSpan:
    """A single text span from dict-mode extraction."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font: str = ""
    size: float = 0.0
    flags: int = 0


@dataclass
class TextLine:
    """Line composed of spans in reading order."""
    spans: List[TextSpan] = field(default_factory=list)
    y_center: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.spans if s.text.strip())


@dataclass
class TextBlock:
    """Block of lines (paragraph / table cell region)."""
    lines: List[TextLine] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)


class PDFIngestor:
    """
    Structural PDF text extraction with reading-order tree.

    For digital PDFs: get_text("dict") + get_drawings() for table lines.
  OCR fallback when page has fewer than MIN_CHARS_PER_PAGE characters.
    """

    def __init__(self) -> None:
        self._numeric_parser = NumericParser()

    def extract_tokens(
        self,
        pdf_path: str,
        dpi: int = 200,
    ) -> Tuple[List[List[OCRToken]], List[np.ndarray], bool]:
        """
        Extract tokens per page using structural parsing.

        Returns
        -------
        tokens_per_page, page_images, used_native
        """
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        tokens_per_page: List[List[OCRToken]] = []
        images: List[np.ndarray] = []
        used_native = True

        for page_num, page in enumerate(doc):
            plain = page.get_text().strip()
            char_count = len(plain)

            if char_count < MIN_CHARS_PER_PAGE:
                used_native = False
                logger.debug(
                    "Page %d has %d chars (< %d) — needs OCR fallback",
                    page_num, char_count, MIN_CHARS_PER_PAGE,
                )
                tokens_per_page.append([])
            else:
                tokens = self._extract_page_structural(page, page_num, zoom)
                tokens_per_page.append(tokens)

            # Render page image for debug / line detection
            import cv2
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        doc.close()
        return tokens_per_page, images, used_native

    def get_table_lines(
        self,
        page: fitz.Page,
        zoom: float = 1.0,
    ) -> Tuple[List[float], List[float]]:
        """
        Detect horizontal and vertical lines from PDF drawings.

        Returns normalised y (horizontal) and x (vertical) coordinates.
        """
        h_lines: List[float] = []
        v_lines: List[float] = []
        rect = page.rect
        pw, ph = rect.width, rect.height

        try:
            drawings = page.get_drawings()
        except Exception:
            return h_lines, v_lines

        for d in drawings:
            for item in d.get("items", []):
                if not item:
                    continue
                kind = item[0]
                if kind == "l" and len(item) >= 3:
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 2:  # horizontal
                        y_norm = ((p1.y + p2.y) / 2) / ph
                        h_lines.append(y_norm)
                    elif abs(p1.x - p2.x) < 2:  # vertical
                        x_norm = ((p1.x + p2.x) / 2) / pw
                        v_lines.append(x_norm)
                elif kind == "re" and len(item) >= 2:
                    r = item[1]
                    if r.width > r.height * 3:  # horizontal bar
                        h_lines.append((r.y0 + r.y1) / 2 / ph)
                    elif r.height > r.width * 3:
                        v_lines.append((r.x0 + r.x1) / 2 / pw)

        return sorted(set(h_lines)), sorted(set(v_lines))

    def _extract_page_structural(
        self,
        page: fitz.Page,
        page_num: int,
        zoom: float,
    ) -> List[OCRToken]:
        """Build OCRTokens from dict-mode spans in reading order."""
        rect = page.rect
        w, h = rect.width * zoom, rect.height * zoom
        blocks = self._build_reading_order(page)
        tokens: List[OCRToken] = []

        for block in blocks:
            for line in block.lines:
                for span in line.spans:
                    text = span.text.strip()
                    if not text:
                        continue
                    x0 = span.x0 * zoom
                    y0 = span.y0 * zoom
                    x1 = span.x1 * zoom
                    y1 = span.y1 * zoom
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    tokens.append(
                        OCRToken(
                            text=text,
                            confidence=1.0,
                            x1=x0, y1=y0, x2=x1, y2=y1,
                            center_x=cx,
                            center_y=cy,
                            normalized_x=cx / w if w else 0,
                            normalized_y=cy / h if h else 0,
                            page_num=page_num,
                            is_numeric=self._numeric_parser.looks_like_number(text),
                            is_date=self._numeric_parser.is_date(text),
                        )
                    )
        return tokens

    def _build_reading_order(self, page: fitz.Page) -> List[TextBlock]:
        """
        spans → lines → blocks in top-to-bottom, left-to-right order.
        """
        d = page.get_text("dict")
        blocks_out: List[TextBlock] = []

        for block in d.get("blocks", []):
            if block.get("type") != 0:  # text only
                continue
            lines: List[TextLine] = []
            bbox = block.get("bbox", (0, 0, 0, 0))

            for line in block.get("lines", []):
                spans: List[TextSpan] = []
                for sp in line.get("spans", []):
                    text = sp.get("text", "")
                    if not text.strip():
                        continue
                    b = sp.get("bbox", (0, 0, 0, 0))
                    spans.append(
                        TextSpan(
                            text=text,
                            x0=b[0], y0=b[1], x1=b[2], y1=b[3],
                            font=sp.get("font", ""),
                            size=sp.get("size", 0),
                            flags=sp.get("flags", 0),
                        )
                    )
                if spans:
                    yc = sum((s.y0 + s.y1) / 2 for s in spans) / len(spans)
                    lines.append(TextLine(spans=spans, y_center=yc))

            if lines:
                lines.sort(key=lambda ln: ln.y_center)
                blocks_out.append(TextBlock(lines=lines, bbox=tuple(bbox)))

        blocks_out.sort(key=lambda b: b.bbox[1] if b.bbox else 0)
        return blocks_out

    @staticmethod
    def page_char_count(page: fitz.Page) -> int:
        return len(page.get_text().strip())

    @staticmethod
    def needs_ocr(pdf_path: str, min_chars: int = MIN_CHARS_PER_PAGE) -> bool:
        """True if any of the first 3 pages has insufficient extractable text."""
        try:
            doc = fitz.open(pdf_path)
            for i in range(min(len(doc), 3)):
                if PDFIngestor.page_char_count(doc[i]) < min_chars:
                    doc.close()
                    return True
            doc.close()
            return False
        except Exception:
            return True
