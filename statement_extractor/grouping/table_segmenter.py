"""
Table Segmenter — splits a single page into distinct table blocks.
"""
from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

from ..config import ExtractorConfig
from ..schemas import LogicalRow

logger = logging.getLogger(__name__)


class TableSegmenter:
    """
    Groups a page's LogicalRows into separate tables based on vertical rhythm
    and explicit header keywords.
    """

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config
        hi = config.header_inference
        # Extract unique words > 2 chars from all header vocabularies
        self.header_words = set()
        for kw_list in [
            hi.date_keywords,
            hi.narration_keywords,
            hi.debit_keywords,
            hi.credit_keywords,
            hi.balance_keywords,
        ]:
            for kw in kw_list:
                for word in kw.lower().split():
                    if len(word) > 2:
                        self.header_words.add(word)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, page_rows: List[LogicalRow]) -> List[List[LogicalRow]]:
        """
        Split a list of single-page rows into one or more table blocks.
        """
        if not page_rows:
            return []

        tables: List[List[LogicalRow]] = []
        current_table: List[LogicalRow] = [page_rows[0]]

        # Calculate median y-gap between adjacent rows
        y_gaps = []
        for i in range(1, len(page_rows)):
            y_gaps.append(page_rows[i].y_center - page_rows[i - 1].y_center)

        median_gap = float(np.median(y_gaps)) if y_gaps else 0.02
        # A new table starts if the gap is > 3.0x median or at least 4% of page height
        gap_threshold = max(0.04, median_gap * 3.0)

        for i in range(1, len(page_rows)):
            row = page_rows[i]
            prev_row = page_rows[i - 1]

            gap = row.y_center - prev_row.y_center
            is_new_table = False

            if gap > gap_threshold:
                is_new_table = True
            elif self._is_strong_header(row):
                # Only split if the previous row wasn't also a header
                # (to keep multi-line headers together)
                if not prev_row.is_header:
                    is_new_table = True

            if is_new_table:
                tables.append(current_table)
                current_table = [row]
            else:
                current_table.append(row)

        if current_table:
            tables.append(current_table)

        if len(tables) > 1:
            logger.info("Segmented page into %d distinct tables", len(tables))

        return tables

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_strong_header(self, row: LogicalRow) -> bool:
        """
        Return True if the row matches >= 3 header keywords, OR
        matches >= 2 header keywords and contains no numbers.
        """
        text = row.full_text.lower()
        words = set(re.findall(r"\w+", text))
        matches = len(words.intersection(self.header_words))

        if matches >= 3:
            pass
        elif matches >= 2 and not any(t.is_numeric for t in row.tokens):
            pass
        else:
            return False

        row.is_header = True  # force flag update
        row.is_table_header = True
        return True
