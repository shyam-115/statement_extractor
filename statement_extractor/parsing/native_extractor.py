"""
Native Extractor — High-fidelity structural text extraction for digital PDFs.

Delegates to PDFIngestor for dict-mode spans and drawing-based table hints.
Falls back to word-level extraction when structural parsing yields no tokens.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import fitz  # PyMuPDF
import numpy as np

from ..pdf.pdf_ingestor import PDFIngestor, MIN_CHARS_PER_PAGE
from ..schemas import OCRToken
from .numeric_parser import NumericParser

logger = logging.getLogger(__name__)


class NativeExtractor:
    """
    Extracts text directly from PDF drawing commands via structural parsing.
    """

    def __init__(self) -> None:
        self._numeric_parser = NumericParser()
        self._ingestor = PDFIngestor()

    def extract_tokens(self, pdf_path: str, dpi: int = 72) -> List[List[OCRToken]]:
        """
        Extract word-level tokens from a digital PDF.
        Uses structural dict-mode parsing; per-page fallback to get_text("words").
        """
        try:
            tokens_per_page, _, used_native = self._ingestor.extract_tokens(
                pdf_path, dpi=dpi
            )
        except Exception as exc:
            logger.warning("Structural extraction failed: %s — using words fallback", exc)
            return self._extract_words_fallback(pdf_path, dpi)

        if not used_native or not any(tokens_per_page):
            # Hybrid: fill empty pages with words fallback
            fallback = self._extract_words_fallback(pdf_path, dpi)
            for i, toks in enumerate(tokens_per_page):
                if not toks and i < len(fallback):
                    tokens_per_page[i] = fallback[i]
            if not any(tokens_per_page):
                return fallback

        return tokens_per_page

    def _extract_words_fallback(
        self, pdf_path: str, dpi: int
    ) -> List[List[OCRToken]]:
        """Legacy get_text('words') extraction."""
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.error("Failed to open PDF: %s", exc)
            return []

        tokens_per_page: List[List[OCRToken]] = []
        zoom = dpi / 72.0

        for page_num, page in enumerate(doc):
            rect = page.rect
            w, h = rect.width * zoom, rect.height * zoom
            words = page.get_text("words")
            tokens: List[OCRToken] = []

            for x0, y0, x1, y1, text, *_ in words:
                text = text.strip()
                if not text:
                    continue
                x0, y0, x1, y1 = x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom
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
            tokens_per_page.append(tokens)

        doc.close()
        return tokens_per_page

    @staticmethod
    def is_digital_pdf(pdf_path: str) -> bool:
        """True if PDF has sufficient extractable text (not a pure scan)."""
        try:
            doc = fitz.open(pdf_path)
            has_text = False
            for i in range(min(len(doc), 3)):
                if len(doc[i].get_text().strip()) >= MIN_CHARS_PER_PAGE:
                    has_text = True
                    break
            doc.close()
            return has_text
        except Exception:
            return False

    def get_page_images(self, pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
        """Render PDF pages to images for debug / line detection."""
        import cv2

        doc = fitz.open(pdf_path)
        images: List[np.ndarray] = []
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        doc.close()
        return images
