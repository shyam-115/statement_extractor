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
        Merge narration-continuation rows into their parent data row.
        Supports merging UP (to previous row) or DOWN (to next row).
        """
        merged: List[LogicalRow] = []
        pending_down: List[LogicalRow] = []
        
        for row in rows:
            if getattr(row, 'is_continuation', False) and not row.is_header:
                merge_dir = getattr(row, 'merge_dir', 'up')
                if merge_dir == 'up':
                    if merged and not merged[-1].is_header:
                        merged[-1].tokens.extend(row.tokens)
                        merged[-1].tokens.sort(key=lambda t: (t.normalized_y, t.normalized_x))
                    else:
                        merged.append(row)
                else:
                    pending_down.append(row)
            else:
                if pending_down and not row.is_header:
                    for p_row in pending_down:
                        row.tokens.extend(p_row.tokens)
                    row.tokens.sort(key=lambda t: (t.normalized_y, t.normalized_x))
                    pending_down = []
                merged.append(row)
                
        if pending_down:
            merged.extend(pending_down)
            
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
        Mark orphaned rows and determine if they merge UP or DOWN to the nearest anchor.
        """
        from ..parsing.numeric_parser import NumericParser
        
        gap = self.config.narration_y_gap_fraction
        
        for r in rows:
            if r.is_header:
                r.is_anchor = False
                continue
            has_date = any(t.is_date for t in r.tokens)
            has_amount = any(NumericParser.is_amount(t.text) for t in r.tokens)
            r.is_anchor = has_date or has_amount
            r.is_continuation = not r.is_anchor
            r.merge_dir = 'up'
            
        i = 0
        while i < len(rows):
            if rows[i].is_continuation and not rows[i].is_header:
                j = i + 1
                while j < len(rows) and rows[j].is_continuation and not rows[j].is_header:
                    if rows[j].y_center - rows[j-1].y_center > gap:
                        break
                    j += 1
                    
                dist_up = float('inf')
                if i > 0 and rows[i-1].is_anchor:
                    dist_up = rows[i].y_center - rows[i-1].y_center
                    
                dist_down = float('inf')
                if j < len(rows) and rows[j].is_anchor:
                    dist_down = rows[j].y_center - rows[j-1].y_center
                    
                if dist_down < dist_up and dist_down <= gap * 2:
                    for k in range(i, j):
                        rows[k].merge_dir = 'down'
                else:
                    for k in range(i, j):
                        rows[k].merge_dir = 'up'
                i = j
            else:
                i += 1
                
        return rows
