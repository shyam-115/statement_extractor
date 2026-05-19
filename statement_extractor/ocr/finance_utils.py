"""
OCR finance utilities — character confusion correction and numeric ensemble.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_NUMERIC_CONTEXT = re.compile(r"^\s*[\dOolISZ,\.]+\s*$")


def correct_numeric_confusion(text: str) -> str:
    """
    Replace O→0, l→1, S→5, Z→2 only in numeric-looking strings.
    """
    if not text or not _NUMERIC_CONTEXT.match(text.replace(" ", "")):
        return text
    mapping = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "s": "5", "Z": "2", "z": "2"}
    return "".join(mapping.get(c, c) for c in text)


def page_cache_key(image: np.ndarray, dpi: int, page_num: int) -> str:
    """Content hash for OCR cache lookup."""
    h = hashlib.sha256(image.tobytes())
    h.update(str(dpi).encode())
    h.update(str(page_num).encode())
    return h.hexdigest()


def load_cached_tokens(cache_dir: str, key: str) -> Optional[list]:
    path = Path(cache_dir) / f"{key}.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def save_cached_tokens(cache_dir: str, key: str, tokens: list) -> None:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"{key}.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(tokens, fh)


def numeric_ensemble_vote(candidates: List[str]) -> str:
    """
    Majority vote across OCR engine outputs for a numeric cell.
    Falls back to first non-empty candidate.
    """
    from collections import Counter
    cleaned = [correct_numeric_confusion(c.strip()) for c in candidates if c.strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]


def estimate_image_quality(img: np.ndarray) -> float:
    """
    Return 0–1 quality score (Laplacian variance heuristic).
    Low score → needs higher DPI.
    """
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    # Normalise: typical sharp scan ~500+, blurry ~50
    return min(1.0, var / 500.0)
