"""
Column Detector — dynamic x-axis clustering to infer column bands.

Algorithm
---------
1. Collect the normalised_x centre of every *numeric* token across all rows
   (dates are also included because they sit in the date column).
2. Run DBSCAN on the 1-D x-positions.
3. For each cluster compute:
   - x_center   : median of all x values in the cluster
   - left/right  : min/max ± boundary_padding
   - support     : count of contributing tokens
4. Discard clusters with support < min_column_support.
5. Sort clusters left → right.
6. Return a list of ColumnZone objects ready for semantic assignment.

Design rationale
----------------
Using numeric-only tokens for clustering is intentional:
- Narration text sprawls horizontally and would blur column boundaries.
- Numeric values (amounts, dates) are tightly aligned within each column.
DBSCAN is preferred over k-means because the number of columns is unknown.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN

from ..config import ColumnDetectionConfig
from ..schemas import OCRToken, ColumnZone, LogicalRow

logger = logging.getLogger(__name__)

# Bare 4-digit calendar years (1900-2099) must not be used as column anchors.
# They are date fragments, not standalone amount or date column values.
_BARE_YEAR = re.compile(r"^(19|20)\d{2}$")


class ColumnDetector:
    """
    Infers vertical column zones from the x-distribution of numeric tokens.

    Parameters
    ----------
    config : ColumnDetectionConfig
    """

    def __init__(self, config: ColumnDetectionConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, rows: List[LogicalRow]) -> List[ColumnZone]:
        """
        Detect column zones from all non-header rows.

        Parameters
        ----------
        rows : LogicalRow list from the RowGrouper (all pages combined or per-page)

        Returns
        -------
        Sorted list of ColumnZone objects (left → right).
        """
        if not rows:
            return []

        if self.config.vertical_strip_detection:
            strips = self._split_vertical_strips(rows)
            if len(strips) > 1:
                all_zones: List[ColumnZone] = []
                for strip_rows in strips.values():
                    zones = self._detect_on_rows(strip_rows)
                    all_zones.extend(zones)
                if all_zones:
                    all_zones.sort(key=lambda z: z.x_center)
                    for idx, z in enumerate(all_zones):
                        z.column_id = idx
                    return all_zones

        return self._detect_on_rows(rows)

    def _detect_on_rows(self, rows: List[LogicalRow]) -> List[ColumnZone]:
        """Core column detection on a single vertical strip."""
        if self.config.numeric_fusion:
            rows = self._fuse_split_numeric_tokens(rows)

        # 1. Numeric Clustering (DBSCAN) - Find true data pillars.
        # Bare calendar years (e.g. "2026") are excluded: they are fragments
        # of date strings split by OCR, not real column anchors.
        numeric_x: List[float] = []
        for row in rows:
            if row.is_header:
                continue
            for token in row.tokens:
                if token.is_numeric or token.is_date:
                    # Skip bare year fragments — they drift from the true date
                    # column x and create phantom zones.
                    if _BARE_YEAR.match(token.text.strip()):
                        continue
                    numeric_x.append(token.normalized_x)

        if len(numeric_x) < self.config.dbscan_min_samples:
            logger.warning(
                "Not enough numeric tokens (%d) for reliable column detection",
                len(numeric_x),
            )
            return []

        # Short tables (many credit-card layouts) yield only 1 sample per row
        # per column — clusters of size 2 are common.  Relax support threshold.
        n_data_rows = max(
            1,
            len([r for r in rows if not r.is_header]),
        )
        min_support = self.config.min_column_support
        if n_data_rows <= 14:
            min_support = min(min_support, 2)

        X = np.array(numeric_x).reshape(-1, 1)
        labels = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
        ).fit_predict(X)

        zones: List[ColumnZone] = []
        for col_id, label in enumerate(set(labels)):
            if label == -1:
                continue  # DBSCAN noise
            mask = labels == label
            xs = X[mask, 0]
            support = int(np.sum(mask))
            if support < min_support:
                continue
            x_center = float(np.median(xs))
            left = float(np.min(xs)) - self.config.boundary_padding
            right = float(np.max(xs)) + self.config.boundary_padding
            zones.append(
                ColumnZone(
                    column_id=col_id,
                    x_center=x_center,
                    left_boundary=max(0.0, left),
                    right_boundary=min(1.0, right),
                    support=support,
                )
            )

        # Sort left → right and re-assign sequential IDs
        zones.sort(key=lambda z: z.x_center)
        for idx, zone in enumerate(zones):
            zone.column_id = idx

        logger.debug("Detected %d column zones", len(zones))
        for z in zones:
            logger.debug(
                "  Col %d: x=%.3f [%.3f, %.3f] support=%d",
                z.column_id, z.x_center, z.left_boundary, z.right_boundary, z.support,
            )

        return zones

    def _split_vertical_strips(
        self, rows: List[LogicalRow]
    ) -> Dict[int, List[LogicalRow]]:
        """
        Use vertical projection profile to find column gaps → page strips.
        """
        xs: List[float] = []
        for row in rows:
            for t in row.tokens:
                if t.is_numeric or t.is_date:
                    xs.append(t.normalized_x)
        if len(xs) < 10:
            return {0: rows}

        hist, edges = np.histogram(xs, bins=40, range=(0.0, 1.0))
        # Valleys in histogram indicate column gaps
        threshold = np.max(hist) * 0.15
        in_gap = hist < threshold
        gap_centers: List[float] = []
        for i, is_gap in enumerate(in_gap):
            if is_gap:
                gap_centers.append((edges[i] + edges[i + 1]) / 2)

        if not gap_centers:
            return {0: rows}

        # Use largest central gap as strip boundary
        mid_gaps = [g for g in gap_centers if 0.25 < g < 0.75]
        if not mid_gaps:
            return {0: rows}
        split_x = mid_gaps[len(mid_gaps) // 2]

        left, right = [], []
        for row in rows:
            cx = np.mean([t.normalized_x for t in row.tokens]) if row.tokens else 0.5
            (left if cx < split_x else right).append(row)
        if not left or not right:
            return {0: rows}
        return {0: left, 1: right}

    def _fuse_split_numeric_tokens(self, rows: List[LogicalRow]) -> List[LogicalRow]:
        """
        Merge adjacent numeric tokens that form a valid Indian number when concatenated.
        """
        import re
        from ..schemas import OCRToken

        indian_num = re.compile(r"^[\d,]+(?:\.\d{1,2})?$")
        fused_rows: List[LogicalRow] = []

        for row in rows:
            sorted_toks = sorted(row.tokens, key=lambda t: t.normalized_x)
            merged: List[OCRToken] = []
            i = 0
            while i < len(sorted_toks):
                t = sorted_toks[i]
                if (
                    t.is_numeric
                    and i + 1 < len(sorted_toks)
                    and sorted_toks[i + 1].is_numeric
                ):
                    nxt = sorted_toks[i + 1]
                    gap = nxt.normalized_x - t.normalized_x
                    if gap < 0.08:
                        combined = t.text + nxt.text
                        if indian_num.match(combined.replace(" ", "")):
                            merged.append(
                                OCRToken(
                                    text=combined,
                                    confidence=min(t.confidence, nxt.confidence),
                                    x1=t.x1, y1=min(t.y1, nxt.y1),
                                    x2=nxt.x2, y2=max(t.y2, nxt.y2),
                                    center_x=(t.center_x + nxt.center_x) / 2,
                                    center_y=(t.center_y + nxt.center_y) / 2,
                                    normalized_x=(t.normalized_x + nxt.normalized_x) / 2,
                                    normalized_y=(t.normalized_y + nxt.normalized_y) / 2,
                                    page_num=t.page_num,
                                    is_numeric=True,
                                    is_date=False,
                                )
                            )
                            i += 2
                            continue
                merged.append(t)
                i += 1
            if merged != row.tokens:
                row = LogicalRow(
                    row_id=row.row_id,
                    tokens=merged,
                    page_num=row.page_num,
                    y_center=row.y_center,
                    is_header=row.is_header,
                    is_table_header=row.is_table_header,
                    is_continuation=row.is_continuation,
                )
            fused_rows.append(row)
        return fused_rows

    # ------------------------------------------------------------------
    # Token assignment
    # ------------------------------------------------------------------

    @staticmethod
    def assign_token_to_column(
        token: OCRToken,
        zones: List[ColumnZone],
    ) -> int:
        """
        Return the column_id of the zone whose centre is nearest to
        *token*'s normalised x-position.  Returns -1 if no zone is close
        enough (threshold: half the gap to the nearest neighbour).
        """
        if not zones:
            return -1

        best_id = -1
        best_dist = float("inf")
        for zone in zones:
            dist = abs(token.normalized_x - zone.x_center)
            if dist < best_dist:
                best_dist = dist
                best_id = zone.column_id

        # Accept assignment only if token falls within zone boundaries
        if best_id >= 0:
            z = zones[best_id]
            if z.left_boundary <= token.normalized_x <= z.right_boundary:
                return best_id
            # Fallback: nearest centre within 2× eps
            if best_dist <= 0.05:
                return best_id

        return -1
