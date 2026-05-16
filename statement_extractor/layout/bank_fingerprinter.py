"""
Bank Fingerprinter — lightweight keyword-based bank identity detection.

Strategy
--------
Scans the first N pages of OCR tokens for bank-specific name/watermark
keywords. Returns a BankProfile with:
  - bank_id          : short code, e.g. "ICICI", "HDFC"
  - bank_name        : full name
  - confidence       : 0–1 match score
  - date_format_hint : advisory date format
  - debit_left_of_credit : advisory column order
  - has_cr_dr_suffix : advisory amount suffix style

Design principles
-----------------
- Zero external dependencies — pure keyword matching.
- Purely advisory — the generic pipeline runs unchanged if bank is UNKNOWN.
- Conservative confidence scoring — only raises confidence when multiple
  signals agree to avoid false positives on generic financial text.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from ..config import BankFingerprintConfig
from ..schemas import BankProfile, OCRToken

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bank registry
# Each entry: (bank_id, bank_name, keyword_patterns, advisory_hints)
# ---------------------------------------------------------------------------
_BANK_REGISTRY = [
    {
        "bank_id": "ICICI",
        "bank_name": "ICICI Bank",
        "patterns": [
            re.compile(r"\bicici\b", re.IGNORECASE),
            re.compile(r"\bicicibank\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YYYY",
        "debit_left_of_credit": False,   # ICICI: Deposits (credit) left of Withdrawals
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "HDFC",
        "bank_name": "HDFC Bank",
        "patterns": [
            re.compile(r"\bhdfc\b", re.IGNORECASE),
            re.compile(r"\bhdfc\s*bank\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": True,
    },
    {
        "bank_id": "SBI",
        "bank_name": "State Bank of India",
        "patterns": [
            re.compile(r"\bsbi\b", re.IGNORECASE),
            re.compile(r"\bstate\s+bank\s+of\s+india\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD MMM YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "KOTAK",
        "bank_name": "Kotak Mahindra Bank",
        "patterns": [
            re.compile(r"\bkotak\b", re.IGNORECASE),
            re.compile(r"\bkotak\s*mahindra\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD-MM-YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": True,
    },
    {
        "bank_id": "AXIS",
        "bank_name": "Axis Bank",
        "patterns": [
            re.compile(r"\baxis\s*bank\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD-MM-YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "YES",
        "bank_name": "Yes Bank",
        "patterns": [
            re.compile(r"\byes\s*bank\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "IDFC",
        "bank_name": "IDFC First Bank",
        "patterns": [
            re.compile(r"\bidfc\b", re.IGNORECASE),
            re.compile(r"\bidfc\s*first\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "INDUSIND",
        "bank_name": "IndusInd Bank",
        "patterns": [
            re.compile(r"\bindusind\b", re.IGNORECASE),
            re.compile(r"\bindus\s*ind\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "PNB",
        "bank_name": "Punjab National Bank",
        "patterns": [
            re.compile(r"\bpnb\b", re.IGNORECASE),
            re.compile(r"\bpunjab\s+national\s+bank\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "CANARA",
        "bank_name": "Canara Bank",
        "patterns": [
            re.compile(r"\bcanara\s*bank\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD-MM-YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "BOI",
        "bank_name": "Bank of India",
        "patterns": [
            re.compile(r"\bbank\s+of\s+india\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "BOB",
        "bank_name": "Bank of Baroda",
        "patterns": [
            re.compile(r"\bbank\s+of\s+baroda\b", re.IGNORECASE),
            re.compile(r"\bbob\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "FEDERAL",
        "bank_name": "Federal Bank",
        "patterns": [
            re.compile(r"\bfederal\s*bank\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
    {
        "bank_id": "UCO",
        "bank_name": "UCO Bank",
        "patterns": [
            re.compile(r"\buco\s*bank\b", re.IGNORECASE),
        ],
        "date_format_hint": "DD/MM/YYYY",
        "debit_left_of_credit": True,
        "has_cr_dr_suffix": False,
    },
]


class BankFingerprinter:
    """
    Detects the issuing bank from OCR token streams.

    Parameters
    ----------
    config : BankFingerprintConfig
    """

    def __init__(self, config: BankFingerprintConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify(
        self,
        tokens_per_page: List[List[OCRToken]],
    ) -> BankProfile:
        """
        Scan the first ``config.scan_pages`` pages and return the best
        matching BankProfile, or an UNKNOWN profile if no match clears
        the confidence threshold.

        Parameters
        ----------
        tokens_per_page : list of per-page OCRToken lists

        Returns
        -------
        BankProfile
        """
        if not self.config.enabled:
            return BankProfile()

        pages_to_scan = tokens_per_page[: self.config.scan_pages]
        # Build a flat text corpus from scanned pages
        corpus = self._build_corpus(pages_to_scan)

        best_profile: Optional[BankProfile] = None
        best_score = 0.0

        for entry in _BANK_REGISTRY:
            score = self._match_score(corpus, entry["patterns"])
            if score > best_score:
                best_score = score
                best_profile = BankProfile(
                    bank_id=entry["bank_id"],
                    bank_name=entry["bank_name"],
                    confidence=score,
                    date_format_hint=entry.get("date_format_hint"),
                    debit_left_of_credit=entry.get("debit_left_of_credit"),
                    has_cr_dr_suffix=entry.get("has_cr_dr_suffix"),
                )

        if best_profile is not None and best_score >= self.config.min_match_score:
            logger.info(
                "Bank identified: %s (confidence=%.2f)",
                best_profile.bank_name, best_score,
            )
            return best_profile

        logger.info(
            "Bank not identified (best_score=%.2f < threshold=%.2f)",
            best_score, self.config.min_match_score,
        )
        return BankProfile()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_corpus(pages: List[List[OCRToken]]) -> str:
        """Concatenate all token texts into a single searchable string."""
        parts: List[str] = []
        for page in pages:
            for token in page:
                parts.append(token.text)
        return " ".join(parts)

    @staticmethod
    def _match_score(corpus: str, patterns: list) -> float:
        """
        Return a 0–1 score based on how many patterns match the corpus.

        Multiple matches increase confidence toward 1.0 using a
        diminishing-returns formula.
        """
        hits = sum(1 for p in patterns if p.search(corpus))
        if hits == 0:
            return 0.0
        # First hit = 0.75, second = 0.90, third+ = 1.0
        return min(1.0, 0.60 + 0.20 * hits)
