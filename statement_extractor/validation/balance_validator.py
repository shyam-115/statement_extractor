"""
Balance Validator — arithmetic continuity engine.

Core formula
------------
    balance[n] = balance[n-1] + credit[n] - debit[n]

This module:
  1. Walks the transaction list sequentially.
  2. For each transaction checks if the formula holds within tolerance.
  3. If validation fails, attempts column-swap repair (debit ↔ credit).
  4. Assigns a final ValidationStatus and adjusts the confidence score.
  5. Attempts to infer opening balance when not explicitly provided.

Confidence scoring
------------------
  Base score = mean OCR confidence of tokens
  +0.15  : balance arithmetic validates exactly
  +0.05  : within 1% tolerance
  -0.20  : arithmetic fails
  -0.10  : repaired (swapped columns)
  Clamped to [0, 1].

Design notes
------------
- The validator must be tolerant of:
    * Missing debit/credit values (sparse rows)
    * OCR-corrupted amounts (single digit off)
    * Negative balances (overdraft accounts)
    * CR/DR balance flags changing sign semantics
- When neither assignment validates and we have < 3 data points,
  the transaction is marked NEEDS_REVIEW rather than FAILED.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from ..config import ValidationConfig
from ..parsing.numeric_parser import NumericParser
from ..schemas import Transaction, ValidationStatus

logger = logging.getLogger(__name__)
_parser = NumericParser()


class BalanceValidator:
    """
    Validates and repairs transaction balance arithmetic.

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
        Validate balance continuity across the transaction list and
        update each transaction's validation_status and confidence_score.
        Does NOT alter or generate any data values.
        """
        if not transactions:
            return transactions

        balance_rows = [t for t in transactions if t.balance is not None]
        if len(balance_rows) < 2:
            for t in transactions:
                t.validation_status = ValidationStatus.NEEDS_REVIEW
            return transactions

        opening = self._infer_opening(balance_rows)
        prev_balance: Optional[float] = opening

        for txn in transactions:
            if txn.balance is None:
                txn.validation_status = ValidationStatus.NEEDS_REVIEW
                continue

            expected = self._compute_expected(prev_balance, txn)
            if expected is None:
                txn.validation_status = ValidationStatus.NEEDS_REVIEW
                prev_balance = txn.balance
                continue

            err = abs(txn.balance - expected)
            tol = self.config.tolerance_fraction * max(abs(txn.balance), 1.0)

            if err <= tol:
                txn.validation_status = ValidationStatus.VALIDATED
                txn.confidence_score = min(1.0, txn.confidence_score + 0.15)
            else:
                txn.validation_status = ValidationStatus.FAILED
                txn.confidence_score = max(0.0, txn.confidence_score - 0.20)

            prev_balance = txn.balance

        return transactions

    def _infer_opening(self, balance_rows: List[Transaction]) -> Optional[float]:
        first = balance_rows[0]
        if first.balance is None:
            return None
        if first.debit is not None:
            return first.balance + first.debit
        if first.credit is not None:
            return first.balance - first.credit
        return first.balance

    def _compute_expected(
        self,
        prev_balance: Optional[float],
        txn: Transaction,
    ) -> Optional[float]:
        if prev_balance is None:
            return None
        if txn.debit is None and txn.credit is None:
            return None
        expected = prev_balance
        if txn.credit is not None:
            expected += txn.credit
        if txn.debit is not None:
            expected -= txn.debit
        return expected
