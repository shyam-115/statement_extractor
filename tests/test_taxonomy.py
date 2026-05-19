"""Unit tests for TransactionTaxonomy."""
from __future__ import annotations

import pytest

from statement_extractor.financial.taxonomy import TransactionTaxonomy


class TestTransactionTaxonomy:

    @pytest.mark.parametrize("narration,expected", [
        ("UPI/PhonePe/REF123", "UPI"),
        ("NEFT-SALARY CREDIT", "NEFT"),
        ("IMPS/123456", "IMPS"),
        ("ATM WDL", "ATM"),
        ("EMI DEDUCTION", "EMI"),
        ("CASHBACK REWARD", "CASHBACK"),
        ("INTEREST CHARGED", "INTEREST"),
        ("GROCERY STORE XYZ", ""),
    ])
    def test_classify(self, narration, expected):
        assert TransactionTaxonomy.classify(narration) == expected
