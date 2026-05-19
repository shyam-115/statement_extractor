"""
Ledger Validator — non-mutating running-balance reconciliation.

Computes expected balance from debits/credits and compares against the
extracted balance column.  Adds validation_flags per row and a document
summary.  Does NOT modify any extracted numeric values.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..schemas import Transaction

logger = logging.getLogger(__name__)


@dataclass
class LedgerValidationSummary:
    """Document-level ledger validation summary."""
    ledger_mismatches: int = 0
    date_anomalies: List[str] = field(default_factory=list)
    duplicates_found: bool = False
    total_rows: int = 0
    rows_with_balance: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_mismatches": self.ledger_mismatches,
            "date_anomalies": self.date_anomalies,
            "duplicates_found": self.duplicates_found,
        }


class LedgerValidator:
    """
    Validates extracted transactions via running-balance arithmetic.

    Parameters
    ----------
    tolerance_absolute : float
        Minimum absolute tolerance (default 0.01).
    tolerance_fraction : float
        Fraction of balance used for relative tolerance (default 0.005 = 0.5%).
    """

    def __init__(
        self,
        tolerance_absolute: float = 0.01,
        tolerance_fraction: float = 0.005,
    ) -> None:
        self.tolerance_absolute = tolerance_absolute
        self.tolerance_fraction = tolerance_fraction

    def validate(
        self,
        transactions: List[Transaction],
        *,
        flag_duplicates: bool = True,
    ) -> Tuple[List[Transaction], LedgerValidationSummary]:
        """
        Validate transactions and attach validation_flags.

        Returns
        -------
        (transactions, summary) — transactions are mutated in-place only
        for validation_flags; numeric fields are unchanged.
        """
        summary = LedgerValidationSummary(total_rows=len(transactions))

        if not transactions:
            return transactions, summary

        # Duplicate detection (metadata only — never removes rows)
        if flag_duplicates:
            seen: Set[Tuple] = set()
            for txn in transactions:
                key = (
                    txn.txn_date or "",
                    txn.description or "",
                    txn.debit,
                    txn.credit,
                    txn.balance,
                )
                if key in seen and (txn.debit or txn.credit):
                    summary.duplicates_found = True
                    if "DUPLICATE" not in txn.validation_flags:
                        txn.validation_flags.append("DUPLICATE")
                seen.add(key)

        # Date anomaly detection
        prev_date: Optional[datetime] = None
        for i, txn in enumerate(transactions):
            parsed = self._parse_date(txn.txn_date)
            if parsed is None and txn.txn_date:
                summary.date_anomalies.append(
                    f"row_{i}: unparseable date '{txn.txn_date}'"
                )
                if "DATE_ANOMALY" not in txn.validation_flags:
                    txn.validation_flags.append("DATE_ANOMALY")
            elif parsed and prev_date and parsed < prev_date:
                summary.date_anomalies.append(
                    f"row_{i}: date {txn.txn_date} before previous {transactions[i-1].txn_date}"
                )
                if "DATE_OUT_OF_ORDER" not in txn.validation_flags:
                    txn.validation_flags.append("DATE_OUT_OF_ORDER")
            if parsed:
                prev_date = parsed

        # Running balance reconciliation
        running: Optional[float] = None
        anchor_idx: Optional[int] = None

        for i, txn in enumerate(transactions):
            if txn.balance is not None and running is None:
                running = txn.balance
                anchor_idx = i
                summary.rows_with_balance += 1
                continue

            if running is None:
                if txn.balance is not None:
                    running = txn.balance
                    anchor_idx = i
                    summary.rows_with_balance += 1
                continue

            delta = self._movement(txn)
            if delta is not None:
                running = running + delta

            if txn.balance is not None:
                summary.rows_with_balance += 1
                tol = self._tolerance(txn.balance)
                if abs(running - txn.balance) > tol:
                    summary.ledger_mismatches += 1
                    if "BALANCE_MISMATCH" not in txn.validation_flags:
                        txn.validation_flags.append("BALANCE_MISMATCH")
                running = txn.balance

        if anchor_idx is None and transactions:
            for txn in transactions:
                if txn.balance is not None:
                    if "NO_BALANCE_ANCHOR" not in txn.validation_flags:
                        txn.validation_flags.append("NO_BALANCE_ANCHOR")

        logger.info(
            "Ledger validation: %d mismatches, %d date anomalies, duplicates=%s",
            summary.ledger_mismatches,
            len(summary.date_anomalies),
            summary.duplicates_found,
        )
        return transactions, summary

    @staticmethod
    def validate_dicts(
        transactions: List[Dict[str, Any]],
        tolerance_absolute: float = 0.01,
        tolerance_fraction: float = 0.005,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate a list of transaction dicts (date, narration, debit, credit, balance).

        Useful for standalone validation without full Transaction models.
        """
        validator = LedgerValidator(tolerance_absolute, tolerance_fraction)
        txns = [
            Transaction(
                txn_date=t.get("date") or t.get("txn_date") or t.get("transaction_date"),
                description=t.get("narration") or t.get("description"),
                debit=t.get("debit"),
                credit=t.get("credit"),
                balance=t.get("balance"),
                page_num=t.get("page_num", 0),
            )
            for t in transactions
        ]
        validated, summary = validator.validate(txns)
        for orig, val in zip(transactions, validated):
            orig["validation_flags"] = val.validation_flags
        return transactions, summary.to_dict()

    def _tolerance(self, balance: float) -> float:
        rel = abs(balance) * self.tolerance_fraction
        return max(self.tolerance_absolute, rel)

    @staticmethod
    def _movement(txn: Transaction) -> Optional[float]:
        """Net change: credit adds, debit subtracts."""
        if txn.credit is not None and txn.debit is not None:
            return txn.credit - txn.debit
        if txn.credit is not None:
            return txn.credit
        if txn.debit is not None:
            return -txn.debit
        return None

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        for fmt in (
            "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
            "%Y-%m-%d", "%d %b %Y", "%d %B %Y",
        ):
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None
