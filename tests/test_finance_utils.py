"""Unit tests for OCR finance utilities."""
from __future__ import annotations

from statement_extractor.ocr.finance_utils import (
    correct_numeric_confusion,
    numeric_ensemble_vote,
)


class TestFinanceUtils:

    def test_correct_o_to_zero(self):
        assert correct_numeric_confusion("1,O00.00") == "1,000.00"

    def test_no_change_for_text(self):
        assert correct_numeric_confusion("NEFT PAYMENT") == "NEFT PAYMENT"

    def test_ensemble_majority(self):
        assert numeric_ensemble_vote(["1,000.00", "1,000.00", "1,O00.00"]) == "1,000.00"
