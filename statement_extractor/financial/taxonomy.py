"""
Transaction Taxonomy — classify narration into payment/channel types.
"""
from __future__ import annotations

import re
from typing import Optional

# Ordered by specificity (first match wins)
_TAXONOMY_PATTERNS = [
    ("UPI", re.compile(r"\b(?:upi|phonepe|gpay|google\s*pay|paytm|bhim)\b", re.I)),
    ("IMPS", re.compile(r"\bimps\b", re.I)),
    ("NEFT", re.compile(r"\bneft\b", re.I)),
    ("RTGS", re.compile(r"\brtgs\b", re.I)),
    ("ATM", re.compile(r"\b(?:atm|cash\s*wdl|cash\s*withdrawal)\b", re.I)),
    ("POS", re.compile(r"\b(?:pos|card\s*swipe|merchant)\b", re.I)),
    ("EMI", re.compile(r"\b(?:emi|installment|instalment)\b", re.I)),
    ("REVERSAL", re.compile(r"\breversal\b", re.I)),
    ("CASHBACK", re.compile(r"\b(?:cashback|cash\s*back)\b", re.I)),
    ("INTEREST", re.compile(r"\b(?:interest|int\.?\s*paid|int\.?\s*charged)\b", re.I)),
    ("CHARGE", re.compile(r"\b(?:charge|fee|penalty|service\s*charge)\b", re.I)),
    ("REFUND", re.compile(r"\brefund\b", re.I)),
    ("AUTOPAY", re.compile(r"\b(?:autopay|auto\s*pay|si\s*debit|standing\s*instruction)\b", re.I)),
]


class TransactionTaxonomy:
    """Classify transaction narration into a tx_type label."""

    @classmethod
    def classify(cls, narration: Optional[str], raw_text: str = "") -> str:
        """
        Return tx_type string (e.g. UPI, NEFT) or empty string if unknown.
        """
        text = f"{narration or ''} {raw_text or ''}".strip()
        if not text:
            return ""
        for label, pattern in _TAXONOMY_PATTERNS:
            if pattern.search(text):
                return label
        return ""

    @classmethod
    def enrich(cls, narration: Optional[str], raw_text: str = "") -> dict:
        """Return dict with tx_type field."""
        return {"tx_type": cls.classify(narration, raw_text)}
