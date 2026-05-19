"""Unit tests for LedgerValidator."""
from __future__ import annotations

import pytest

from statement_extractor.schemas import Transaction
from statement_extractor.validation.ledger_validator import LedgerValidator


class TestLedgerValidator:

    def setup_method(self):
        self.v = LedgerValidator(tolerance_absolute=0.01, tolerance_fraction=0.005)

    def _txn(self, **kwargs) -> Transaction:
        return Transaction(**kwargs)

    def test_balanced_sequence_no_flags(self):
        txns = [
            self._txn(balance=10000.0, page_num=0),
            self._txn(debit=1200.0, balance=8800.0, page_num=0),
            self._txn(credit=5000.0, balance=13800.0, page_num=0),
        ]
        result, summary = self.v.validate(txns)
        assert summary.ledger_mismatches == 0
        assert all("BALANCE_MISMATCH" not in t.validation_flags for t in result)

    def test_balance_mismatch_flagged(self):
        txns = [
            self._txn(balance=10000.0),
            self._txn(debit=1200.0, balance=9000.0),  # should be 8800
        ]
        result, summary = self.v.validate(txns)
        assert summary.ledger_mismatches >= 1
        assert any("BALANCE_MISMATCH" in t.validation_flags for t in result)

    def test_duplicate_detection(self):
        t = self._txn(
            txn_date="01/05/2024",
            description="UPI",
            debit=100.0,
            page_num=0,
        )
        txns = [t, self._txn(
            txn_date="01/05/2024",
            description="UPI",
            debit=100.0,
            page_num=0,
        )]
        _, summary = self.v.validate(txns)
        assert summary.duplicates_found

    def test_values_not_mutated(self):
        txns = [self._txn(debit=500.0, balance=1000.0)]
        orig_debit = txns[0].debit
        self.v.validate(txns)
        assert txns[0].debit == orig_debit

    def test_validate_dicts(self):
        rows = [
            {"date": "01/05/2024", "debit": None, "credit": None, "balance": 1000.0},
            {"date": "02/05/2024", "debit": 100.0, "credit": None, "balance": 900.0},
        ]
        out, summary = LedgerValidator.validate_dicts(rows)
        assert "validation_flags" in out[0]
        assert "ledger_mismatches" in summary
