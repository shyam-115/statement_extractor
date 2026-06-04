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

        # 1. Header-Driven Column Generation (Primary)
        table_headers = [r for r in rows if r.is_table_header]
        if table_headers:
            # Use the tokens from the last strong header row (closest to data)
            last_header = table_headers[-1]
            sorted_tokens = sorted(last_header.tokens, key=lambda t: t.normalized_x)
            
            # Merge close tokens to form unified header blocks (e.g. "VALUE" + "DATE")
            merged_blocks = []
            if sorted_tokens:
                current_block = [sorted_tokens[0]]
                for token in sorted_tokens[1:]:
                    prev_token = current_block[-1]
                    # Calculate horizontal gap between tokens
                    gap = token.normalized_x1 - prev_token.normalized_x2
                    # If gap is very small (< 1.5% of page width), merge them
                    # This prevents merging distinct columns like "DATE" and "DESCRIPTION"
                    if gap < 0.015:
                        current_block.append(token)
                    else:
                        merged_blocks.append(current_block)
                        current_block = [token]
                merged_blocks.append(current_block)
            
            zones: List[ColumnZone] = []
            for col_id, block in enumerate(merged_blocks):
                x_center = sum(t.normalized_x for t in block) / len(block)
                text = " ".join(t.text for t in block)
                zones.append(ColumnZone(
                    column_id=col_id,
                    x_center=x_center,
                    left_boundary=0.0,
                    right_boundary=1.0,
                    support=10,
                    header_text=text,
                ))
            
            # Precisely calculate midpoints between header BLOCKS for strict boundaries
            for i in range(len(zones)):
                if i > 0:
                    zones[i].left_boundary = (merged_blocks[i-1][-1].normalized_x2 + merged_blocks[i][0].normalized_x1) / 2.0
                else:
                    zones[i].left_boundary = 0.0
                    
                if i < len(zones) - 1:
                    zones[i].right_boundary = (merged_blocks[i][-1].normalized_x2 + merged_blocks[i+1][0].normalized_x1) / 2.0
                else:
                    zones[i].right_boundary = 1.0
                    
            logger.debug("Created %d header-driven column zones", len(zones))
            return zones

        # 2. Fallback to Numeric Clustering (DBSCAN) - finding the mathematically true columns
        numeric_x: List[float] = []
        data_rows_count = 0
        for row in rows:
            if row.is_header or row.is_table_header:
                continue
            data_rows_count += 1
            for token in row.tokens:
                if token.is_numeric or token.is_date:
                    numeric_x.append(token.normalized_x)

        dynamic_min_samples = min(self.config.dbscan_min_samples, max(1, data_rows_count))
        dynamic_min_support = min(self.config.min_column_support, max(1, data_rows_count))

        if len(numeric_x) < dynamic_min_samples:
            logger.warning(
                "Not enough numeric tokens (%d) for reliable column detection",
                len(numeric_x),
            )
            return []

        X = np.array(numeric_x).reshape(-1, 1)
        labels = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=dynamic_min_samples,
        ).fit_predict(X)

        zones: List[ColumnZone] = []
        for col_id, label in enumerate(set(labels)):
            if label == -1:
                continue  # DBSCAN noise
            mask = labels == label
            xs = X[mask, 0]
            support = int(np.sum(mask))
            if support < dynamic_min_support:
                continue
            x_center = float(np.median(xs))
            zones.append(
                ColumnZone(
                    column_id=col_id,
                    x_center=x_center,
                    left_boundary=0.0,
                    right_boundary=1.0,
                    support=support,
                )
            )

        # 3. Sort left → right and assign sequential IDs
        zones.sort(key=lambda z: z.x_center)
        for idx, zone in enumerate(zones):
            zone.column_id = idx

        # 4. Establish Strict Contiguous Mathematical Boundary Walls
        for i in range(len(zones)):
            if i > 0:
                zones[i].left_boundary = (zones[i-1].x_center + zones[i].x_center) / 2.0
            else:
                zones[i].left_boundary = 0.0
                
            if i < len(zones) - 1:
                zones[i].right_boundary = (zones[i].x_center + zones[i+1].x_center) / 2.0
            else:
                zones[i].right_boundary = 1.0

        logger.debug("Detected %d generalized column zones", len(zones))
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
