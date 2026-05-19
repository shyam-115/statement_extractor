"""Unit tests for SemanticAmountResolver."""
from __future__ import annotations

from statement_extractor.financial.semantic_resolver import SemanticAmountResolver
from statement_extractor.schemas import Transaction


class TestSemanticAmountResolver:

    def setup_method(self):
        self.r = SemanticAmountResolver()

    def test_cashback_forces_credit(self):
        txn = Transaction(
            description="CASHBACK OFFER",
            debit=50.0,
            raw_text="50.00 DR",
        )
        result = self.r.resolve(txn, section_header="CASHBACK SUMMARY")
        assert result.credit == 50.0
        assert result.debit is None
        assert result.confidence >= 0.85

    def test_reversal_inverts_debit(self):
        txn = Transaction(description="NEFT REVERSAL", debit=1000.0)
        result = self.r.resolve(txn)
        assert result.credit == 1000.0
        assert result.rule_applied.startswith("reversal")

    def test_balance_trend(self):
        txn = Transaction(debit=500.0, balance=9500.0)
        result = self.r.resolve(txn, prev_balance=10000.0)
        assert result is not None
