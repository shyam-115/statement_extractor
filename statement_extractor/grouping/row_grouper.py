"""
Row Grouper — y-axis DBSCAN clustering to form logical rows.

Algorithm
---------
1. Collect centre_y (normalised) of every OCR token on the page.
2. Run DBSCAN with eps = config.dbscan_eps_fraction (fraction of page height).
   min_samples=1 ensures every token is assigned to some cluster.
3. Sort clusters by ascending y (top → bottom reading order).
4. Within each cluster sort tokens by ascending x → natural reading order.
5. Narration continuation detection:
   - A row that contains NO numeric token AND whose y is within
     narration_y_gap_fraction of the previous row is marked as a
     continuation and later merged with its parent.
6. Header detection heuristic:
   - The first few rows that contain NO amounts and whose text matches
     common header vocabulary are flagged `is_header=True`.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from sklearn.cluster import DBSCAN

from ..config import RowGroupingConfig
from ..schemas import OCRToken, LogicalRow

logger = logging.getLogger(__name__)


class RowGrouper:
    """
    Groups a flat list of OCRTokens into LogicalRows via y-axis clustering.

    Parameters
    ----------
    config : RowGroupingConfig
    """

    def __init__(self, config: RowGroupingConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def group(self, tokens: List[OCRToken], page_num: int) -> List[LogicalRow]:
        """
        Cluster *tokens* into logical rows and return them in top-to-bottom
        reading order.

        Parameters
        ----------
        tokens   : OCRTokens from a single page
        page_num : page index (0-based)

        Returns
        -------
        List[LogicalRow]  sorted by y_center ascending
        """
        if not tokens:
            return []

        y_coords = np.array([[t.normalized_y] for t in tokens])
        eps = self.config.dbscan_eps_fraction
        labels = DBSCAN(eps=eps, min_samples=self.config.dbscan_min_samples).fit_predict(y_coords)

        # Group tokens by cluster label
        clusters: dict[int, List[OCRToken]] = {}
        for token, label in zip(tokens, labels):
            clusters.setdefault(label, []).append(token)

        # Build unsorted LogicalRows
        rows: List[LogicalRow] = []
        for row_id, (label, row_tokens) in enumerate(clusters.items()):
            # Sort tokens left → right
            sorted_tokens = sorted(row_tokens, key=lambda t: t.normalized_x)
            y_center = float(np.mean([t.normalized_y for t in sorted_tokens]))
            row = LogicalRow(
                row_id=row_id,
                tokens=sorted_tokens,
                page_num=page_num,
                y_center=y_center,
            )
            rows.append(row)

        # Sort rows top → bottom
        rows.sort(key=lambda r: r.y_center)
        # Re-assign sequential row IDs after sorting
        for idx, row in enumerate(rows):
            row.row_id = idx

        # Mark headers and narration continuations
        rows = self._detect_headers(rows)
        rows = self._detect_continuations(rows)

        return rows

    # ------------------------------------------------------------------
    # Merge continuation rows into their parent
    # ------------------------------------------------------------------

    def merge_continuations(self, rows: List[LogicalRow]) -> List[LogicalRow]:
        """
        Merge narration-continuation rows into the preceding data row.

        This is called *after* column assignment to glue multi-line
        narration text onto the parent transaction row.
        """
        merged: List[LogicalRow] = []
        for row in rows:
            if row.is_continuation and merged and not merged[-1].is_header:
                # Append tokens to the last row
                merged[-1].tokens.extend(row.tokens)
                # Rebuild full_text ordering
                merged[-1].tokens.sort(key=lambda t: (t.normalized_y, t.normalized_x))
            else:
                merged.append(row)
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_headers(self, rows: List[LogicalRow]) -> List[LogicalRow]:
        """
        Flag rows as headers if they appear in the top N rows and contain
        no parseable amounts.  We use a simple heuristic: if no token in the
        row has is_numeric=True and the row contains at least one token whose
        text length > 2, it is a candidate header.
        """
        max_header_rows = 6
        for row in rows[:max_header_rows]:
            has_numeric = any(t.is_numeric for t in row.tokens)
            has_alpha = any(len(t.text) > 2 and not t.text.replace(",", "").replace(".", "").isdigit()
                           for t in row.tokens)
            if not has_numeric and has_alpha:
                row.is_header = True
        return rows

    def _detect_continuations(self, rows: List[LogicalRow]) -> List[LogicalRow]:
        """
        Mark a row as a narration continuation if:
        - It has no numeric tokens
        - Its y_center is within narration_y_gap_fraction of the previous row
        - The previous row is not a continuation itself (to avoid runaway chains)
        """
        gap = self.config.narration_y_gap_fraction
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            curr = rows[i]
            if curr.is_header or prev.is_header:
                continue
            y_diff = curr.y_center - prev.y_center
            has_numeric = any(t.is_numeric or t.is_date for t in curr.tokens)
            if not has_numeric and y_diff <= gap:
                curr.is_continuation = True
        return rows
