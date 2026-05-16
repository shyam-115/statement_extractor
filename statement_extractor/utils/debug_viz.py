"""
Debug Visualizer — OpenCV overlay renderer for inspection.

Draws on a copy of the page image:
  - OCR bounding boxes (green)
  - Row cluster boundaries (blue horizontal bands)
  - Column zone boundaries (red vertical bands)
  - Transaction regions (yellow)
  - Text annotations on each box

Usage
-----
    viz = DebugVisualizer(output_dir="debug_output")
    viz.render_page(image, tokens, rows, zones, page_num=0)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..schemas import ColumnZone, LogicalRow, OCRToken

logger = logging.getLogger(__name__)

# Colour palette (BGR)
_CLR_OCR_BOX    = (0, 220, 0)      # green
_CLR_ROW_BAND   = (255, 100, 0)    # blue
_CLR_COL_BAND   = (0, 0, 220)      # red
_CLR_TXN_REGION = (0, 220, 220)    # yellow
_CLR_HEADER     = (220, 0, 220)    # magenta
_CLR_CONT       = (180, 180, 0)    # teal
_ALPHA          = 0.25             # transparency for filled regions


class DebugVisualizer:
    """
    Renders OCR + layout debug overlays onto page images.

    Parameters
    ----------
    output_dir : Directory where rendered images are saved.
    """

    def __init__(self, output_dir: str = "debug_output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_page(
        self,
        image: np.ndarray,
        tokens: List[OCRToken],
        rows: List[LogicalRow],
        zones: List[ColumnZone],
        page_num: int = 0,
    ) -> np.ndarray:
        """
        Render all debug overlays on *image* and save to disk.

        Returns the annotated image (BGR numpy array).
        """
        canvas = image.copy()
        h, w = canvas.shape[:2]

        # 1. Column zone bands (vertical)
        canvas = self._draw_column_zones(canvas, zones, w, h)

        # 2. Row cluster bands (horizontal)
        canvas = self._draw_row_bands(canvas, rows, w, h)

        # 3. OCR bounding boxes
        canvas = self._draw_ocr_boxes(canvas, tokens, w, h)

        # 4. Text confidence overlay
        canvas = self._draw_token_labels(canvas, tokens, w, h)

        # Save
        out_path = self.output_dir / f"page_{page_num:03d}_debug.png"
        cv2.imwrite(str(out_path), canvas)
        logger.debug("Debug image saved: %s", out_path)

        return canvas

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_column_zones(
        self,
        img: np.ndarray,
        zones: List[ColumnZone],
        w: int,
        h: int,
    ) -> np.ndarray:
        overlay = img.copy()
        
        # Group zones by table to draw table boundaries
        table_bounds = set((z.top_boundary, z.bottom_boundary) for z in zones if z.bottom_boundary > 0)
        for top, bottom in table_bounds:
            t_y1, t_y2 = int(top), int(bottom)
            # Draw a solid border around the entire table block
            cv2.rectangle(img, (0, t_y1), (w, t_y2), (0, 150, 255), 2)

        for zone in zones:
            x1 = int(zone.left_boundary * w)
            x2 = int(zone.right_boundary * w)
            y1 = int(zone.top_boundary) if zone.top_boundary > 0 else 0
            y2 = int(zone.bottom_boundary) if zone.bottom_boundary > 0 else h
            
            # Draw column pillar strictly within the table boundaries
            cv2.rectangle(overlay, (x1, y1), (x2, y2), _CLR_COL_BAND, -1)
            label = zone.semantic_role or f"col{zone.column_id}"
            cv2.putText(
                img, label,
                (x1 + 2, y1 + 20 + zone.column_id * 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _CLR_COL_BAND, 1, cv2.LINE_AA,
            )
        return cv2.addWeighted(overlay, _ALPHA, img, 1 - _ALPHA, 0)

    def _draw_row_bands(
        self,
        img: np.ndarray,
        rows: List[LogicalRow],
        w: int,
        h: int,
    ) -> np.ndarray:
        overlay = img.copy()
        for row in rows:
            if not row.tokens:
                continue
            ys = [t.y1 for t in row.tokens]
            ye = [t.y2 for t in row.tokens]
            y_top = int(min(ys))
            y_bot = int(max(ye))
            colour = _CLR_HEADER if row.is_header else (
                _CLR_CONT if row.is_continuation else _CLR_ROW_BAND
            )
            cv2.rectangle(overlay, (0, y_top), (w, y_bot), colour, -1)
        return cv2.addWeighted(overlay, _ALPHA * 0.6, img, 1 - _ALPHA * 0.6, 0)

    @staticmethod
    def _draw_ocr_boxes(
        img: np.ndarray,
        tokens: List[OCRToken],
        w: int,
        h: int,
    ) -> np.ndarray:
        for token in tokens:
            x1, y1 = int(token.x1), int(token.y1)
            x2, y2 = int(token.x2), int(token.y2)
            colour = _CLR_TXN_REGION if token.is_numeric else _CLR_OCR_BOX
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1)
        return img

    @staticmethod
    def _draw_token_labels(
        img: np.ndarray,
        tokens: List[OCRToken],
        w: int,
        h: int,
    ) -> np.ndarray:
        for token in tokens:
            label = f"{token.text[:12]} ({token.confidence:.2f})"
            cv2.putText(
                img, label,
                (int(token.x1), max(int(token.y1) - 3, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                (50, 50, 200), 1, cv2.LINE_AA,
            )
        return img
