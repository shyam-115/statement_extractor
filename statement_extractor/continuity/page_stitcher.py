"""
Cross-Page Stitcher — merge broken narrations and carry balance forward.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from ..schemas import Transaction

logger = logging.getLogger(__name__)

_INCOMPLETE_END = re.compile(r"[-–—]\s*$|\b\w{1,3}\s*$")


class CrossPageStitcher:
    """
    Stitches transactions across page boundaries.

    - Merges narrations ending with hyphen or incomplete word
    - Carries running balance when next page's first row lacks balance
    - Sets continuation metadata on stitched rows
    """

    def stitch(self, transactions: List[Transaction]) -> List[Transaction]:
        """Apply cross-page stitching in place and return the list."""
        if len(transactions) < 2:
            return transactions

        i = 0
        while i < len(transactions) - 1:
            curr = transactions[i]
            nxt = transactions[i + 1]

            if self._should_merge_narration(curr, nxt):
                merged_desc = f"{curr.description or ''} {nxt.description or ''}".strip()
                curr.description = merged_desc
                curr.continuation = True
                # Prefer financial fields from the row that has amounts
                if curr.debit is None and curr.credit is None:
                    curr.debit = nxt.debit
                    curr.credit = nxt.credit
                    curr.balance = nxt.balance or curr.balance
                    curr.txn_date = curr.txn_date or nxt.txn_date
                transactions.pop(i + 1)
                logger.debug("Stitched narration across pages %d→%d", curr.page_num, nxt.page_num)
                continue

            # Carry balance forward
            if (
                curr.balance is not None
                and nxt.balance is None
                and nxt.page_num > curr.page_num
            ):
                nxt.carried_balance = curr.balance

            i += 1

        return transactions

    @staticmethod
    def _should_merge_narration(curr: Transaction, nxt: Transaction) -> bool:
        """True if curr narration looks incomplete and nxt continues on next page."""
        if nxt.page_num <= curr.page_num:
            return False
        desc = (curr.description or "").strip()
        if not desc:
            return False
        if _INCOMPLETE_END.search(desc):
            return True
        # Hyphenated break at line end
        if desc.endswith("-") or desc.endswith("–"):
            return True
        # Next row has no date but has narration continuation
        if not nxt.txn_date and nxt.description and not (nxt.debit or nxt.credit):
            return True
        return False
