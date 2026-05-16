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
from typing import List

import numpy as np
from sklearn.cluster import DBSCAN

from ..config import ColumnDetectionConfig
from ..schemas import OCRToken, ColumnZone, LogicalRow

logger = logging.getLogger(__name__)


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

        # 1. Numeric Clustering (DBSCAN) - Find true data pillars
        numeric_x: List[float] = []
        for row in rows:
            if row.is_header:
                continue
            for token in row.tokens:
                if token.is_numeric or token.is_date:
                    numeric_x.append(token.normalized_x)

        if len(numeric_x) < self.config.dbscan_min_samples:
            logger.warning(
                "Not enough numeric tokens (%d) for reliable column detection",
                len(numeric_x),
            )
            return []

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
            if support < self.config.min_column_support:
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
