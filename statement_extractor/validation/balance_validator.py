"""
Balance Validator — fidelity-first pass-through.

Design philosophy
-----------------
Data is kept EXACTLY as extracted from the source document.
No arithmetic checks, no sum calculations, no value corrections,
no debit/credit swaps.  The validator's role is limited to:

  1. Marking all transactions as NEEDS_REVIEW (honest status when
     no external ground truth is available).
  2. Computing a document-level validated_ratio (always 0.0 in
     fidelity-first mode — caller should treat all rows as candidates
     for human review).

This guarantees zero hallucination and zero data mutation from the
validation stage.  Downstream consumers (human reviewers, reconciliation
systems) receive the raw extracted values and decide correctness themselves.
"""
from __future__ import annotations

import logging
from typing import List

from ..config import ValidationConfig
from ..schemas import Transaction, ValidationStatus

logger = logging.getLogger(__name__)


class BalanceValidator:
    """
    Fidelity-first pass-through validator.

    Does NOT modify any extracted transaction values.  Marks all
    transactions as NEEDS_REVIEW so downstream consumers know the
    data has not been independently verified.

    Parameters
    ----------
    config : ValidationConfig
    """

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, transactions: List[Transaction]) -> List[Transaction]:
        """
        Apply fidelity-first status marking.

        Parameters
        ----------
        transactions : extracted Transaction list

        Returns
        -------
        Same list with validation_status set to NEEDS_REVIEW on every row.
        No values are altered.
        """
        for txn in transactions:
            txn.validation_status = ValidationStatus.NEEDS_REVIEW

        logger.info(
            "Fidelity-first validation: %d transactions marked NEEDS_REVIEW "
            "(no arithmetic checks applied — data kept as-is from document)",
            len(transactions),
        )
        return transactions
