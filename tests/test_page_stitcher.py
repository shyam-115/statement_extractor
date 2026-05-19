"""Unit tests for CrossPageStitcher."""
from __future__ import annotations

from statement_extractor.continuity.page_stitcher import CrossPageStitcher
from statement_extractor.schemas import Transaction


class TestCrossPageStitcher:

    def setup_method(self):
        self.s = CrossPageStitcher()

    def test_merge_hyphenated_narration(self):
        txns = [
            Transaction(
                description="NEFT TRANSFER TO ACME-",
                page_num=0,
                debit=100.0,
                balance=9900.0,
            ),
            Transaction(
                description="CORP LTD",
                page_num=1,
            ),
        ]
        result = self.s.stitch(txns)
        assert len(result) == 1
        assert "ACME" in result[0].description
        assert result[0].continuation is True

    def test_carry_balance_metadata(self):
        txns = [
            Transaction(balance=5000.0, page_num=0),
            Transaction(description="cont", page_num=1),
        ]
        result = self.s.stitch(txns)
        assert result[1].carried_balance == 5000.0
