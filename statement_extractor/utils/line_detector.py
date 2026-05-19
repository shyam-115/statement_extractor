"""
Line Detector — Finds horizontal and vertical lines in images using OpenCV.
Used to assist row grouping and column detection by providing hard layout boundaries.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LineDetector:
    """
    Detects horizontal and vertical lines in an image.
    Returns normalised coordinates (0.0 - 1.0).
    """

    @staticmethod
    def detect_lines(img: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Returns (horizontal_y_coords, vertical_x_coords).
        Coordinates are normalised relative to image dimensions.
        """
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return [], []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert the image (lines should be white on black)
        gray = cv2.bitwise_not(gray)
        
        # Adaptive thresholding to handle uneven lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )
        
        # Extract horizontal lines
        # Create structure element roughly 1/30th of image width
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 30), 1))
        h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        
        # Extract vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h // 30)))
        v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        
        # Detect lines using Hough Transform
        hor_lines = cv2.HoughLinesP(
            h_mask, 1, np.pi / 180, 50, minLineLength=w // 10, maxLineGap=20
        )
        ver_lines = cv2.HoughLinesP(
            v_mask, 1, np.pi / 180, 50, minLineLength=h // 20, maxLineGap=20
        )
        
        y_coords: List[float] = []
        if hor_lines is not None:
            for line in hor_lines:
                x1, y1, x2, y2 = line[0]
                # Average y to get a single y-coordinate
                y = (y1 + y2) / 2.0
                y_coords.append(y / h)
                
        x_coords: List[float] = []
        if ver_lines is not None:
            for line in ver_lines:
                x1, y1, x2, y2 = line[0]
                x = (x1 + x2) / 2.0
                x_coords.append(x / w)
                
        # Deduplicate close lines
        def merge_close(coords: List[float], threshold: float = 0.005) -> List[float]:
            if not coords:
                return []
            coords.sort()
            merged = [coords[0]]
            for c in coords[1:]:
                if c - merged[-1] > threshold:
                    merged.append(c)
                else:
                    # Average them
                    merged[-1] = (merged[-1] + c) / 2.0
            return merged
            
        final_y = merge_close(y_coords)
        final_x = merge_close(x_coords)
        
        logger.debug("Detected %d horizontal lines and %d vertical lines", len(final_y), len(final_x))
        return final_y, final_x
