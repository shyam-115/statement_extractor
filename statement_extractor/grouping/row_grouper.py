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

    def group(self, tokens: List[OCRToken], page_num: int, horizontal_lines: List[float] = None) -> List[LogicalRow]:
        """
        Cluster *tokens* into logical rows and return them in top-to-bottom
        reading order. Uses horizontal lines as hard boundaries if provided.

        Parameters
        ----------
        tokens           : OCRTokens from a single page
        page_num         : page index (0-based)
        horizontal_lines : Normalised y-coordinates of horizontal lines

        Returns
        -------
        List[LogicalRow]  sorted by y_center ascending
        """
        if not tokens:
            return []

        horizontal_lines = horizontal_lines or []
        horizontal_lines = sorted(horizontal_lines)
        
        import bisect

        # Group tokens into horizontal bins defined by the lines
        binned_tokens: dict[int, List[OCRToken]] = {}
        for token in tokens:
            bin_idx = bisect.bisect_right(horizontal_lines, token.normalized_y)
            binned_tokens.setdefault(bin_idx, []).append(token)

        rows: List[LogicalRow] = []
        row_id_offset = 0

        # Run DBSCAN independently within each bin
        eps = self._compute_eps(tokens)
        for bin_idx in sorted(binned_tokens.keys()):
            bin_toks = binned_tokens[bin_idx]
            y_coords = np.array([[t.normalized_y] for t in bin_toks])
            
            labels = DBSCAN(eps=eps, min_samples=self.config.dbscan_min_samples).fit_predict(y_coords)

            clusters: dict[int, List[OCRToken]] = {}
            for token, label in zip(bin_toks, labels):
                clusters.setdefault(label, []).append(token)

            bin_rows: List[LogicalRow] = []
            for _, row_tokens in clusters.items():
                sorted_tokens = sorted(row_tokens, key=lambda t: t.normalized_x)
                y_center = float(np.mean([t.normalized_y for t in sorted_tokens]))
                bin_rows.append(
                    LogicalRow(
                        row_id=0, # temporary
                        tokens=sorted_tokens,
                        page_num=page_num,
                        y_center=y_center,
                    )
                )
            
            # Sort bin rows and assign sequential IDs
            bin_rows.sort(key=lambda r: r.y_center)
            for r in bin_rows:
                r.row_id = row_id_offset
                row_id_offset += 1
                rows.append(r)

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

    def _compute_eps(self, tokens: List[OCRToken]) -> float:
        """
        Dynamic epsilon for y-axis DBSCAN.

        Uses median token height as a *fraction of the on-page vertical span*
        of the token set (not a hard-coded pixel page height).  This keeps
        eps ≈ half a text line for digital PDFs, avoiding single mega-rows
        when many table lines sit within the old 0.02–0.03 window.
        """
        base = self.config.dbscan_eps_fraction
        if not tokens:
            return base
        if not self.config.dynamic_eps_from_token_height:
            return base

        y_min = float(min(t.y1 for t in tokens))
        y_max = float(max(t.y2 for t in tokens))
        page_span = max(y_max - y_min, 1e-6)
        norm_heights = [(t.y2 - t.y1) / page_span for t in tokens]
        median_norm_h = float(np.median(norm_heights))

        # ~0.55 × line height: tokens on the same baseline cluster; adjacent
        # baselines (typically >0.7 × line height apart) form separate rows.
        eps = 0.55 * median_norm_h
        # Stay within a modest band around *base* so scanned docs can still
        # merge noisy baselines without reintroducing huge mega-clusters.
        eps = max(base * 0.5, min(eps, base * 2.0))

        # Skew multiplier (residual angle from token baseline variance)
        if len(tokens) >= 5:
            ys = np.array([t.normalized_y for t in tokens])
            xs = np.array([t.normalized_x for t in tokens])
            if np.std(xs) > 0.01:
                slope = np.polyfit(xs, ys, 1)[0]
                angle_deg = abs(float(np.degrees(np.arctan(slope))))
                if angle_deg > 0.5:
                    eps *= self.config.skew_eps_multiplier
                    eps = min(eps, base * 2.5)
        return float(eps)

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
        - It does NOT contain a date or a valid financial amount.
        - Its y_center is within narration_y_gap_fraction of the previous row.
        - The previous row is not a continuation itself.
        """
        from ..parsing.numeric_parser import NumericParser
        
        gap = self.config.narration_y_gap_fraction
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            curr = rows[i]
            if curr.is_header or prev.is_header:
                continue
            y_diff = curr.y_center - prev.y_center
            
            has_date = any(t.is_date for t in curr.tokens)
            has_amount = any(NumericParser.is_amount(t.text) for t in curr.tokens)
            
            if not has_date and not has_amount and y_diff <= gap:
                curr.is_continuation = True
        return rows
