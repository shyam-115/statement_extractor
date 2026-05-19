"""
Numeric Parser — regex-based amount / date / reference detection.

Supports:
- Indian number format (1,00,000.00)
- Standard format (1,000,000.00)
- Negative values (-500.00)
- CR / DR suffixes  (500.00 CR)
- Pure integers
- Various date formats
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Matches amounts in Indian or standard comma-separated notation
_AMOUNT_CORE = r"""
    (?:
        (?:\d{1,3}(?:,\d{2})*(?:,\d{3})?(?:\.\d{1,4})?)   # Indian format
        |
        (?:\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?)               # Standard format
        |
        (?:\d+\.\d{1,4})                                    # Simple decimal
        |
        (?:\d{4,})                                          # Plain integer ≥4 digits
    )
"""

_AMOUNT_PATTERN = re.compile(
    r"^[+-]?\s*(?:INR|Rs\.?|₹|USD|\$|EUR|€)?\s*"
    + _AMOUNT_CORE
    + r"\s*(?:CR|DR|Cr|Dr|cr|dr)?$",
    re.VERBOSE,
)

# Strips everything that isn't a digit or decimal point
_CLEAN_AMOUNT = re.compile(r"[^\d.]")

# CR/DR suffix detection
_CR_SUFFIX = re.compile(r"\b(cr|credit)\b", re.IGNORECASE)
_DR_SUFFIX = re.compile(r"\b(dr|debit)\b", re.IGNORECASE)
_NEGATIVE   = re.compile(r"^-")

# Date patterns — ordered most → least specific
_DATE_PATTERNS = [
    re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}(?!\d)"),                    # DD/MM/YYYY
    re.compile(r"\b\d{4}[/-]\d{2}[/-]\d{2}(?!\d)"),                    # YYYY-MM-DD
    re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{2}(?!\d)"),                    # DD/MM/YY
    # Day + short month + 2- or 4-digit year (common on Indian CC statements)
    re.compile(
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|"
               r"Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]*\d{2,4}\b",
               re.IGNORECASE),                                       # 12 Jan 2024
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|"
               r"Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{1,2},?\s*\d{2,4}\b",
               re.IGNORECASE),                                       # Jan 12, 2024
    re.compile(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|"
               r"Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b",
               re.IGNORECASE),                                       # 02 March
    re.compile(r"\b\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|"
               r"Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b",
               re.IGNORECASE),                                       # 01March, 23March
    re.compile(r"\b\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|"
               r"Jul|Aug|Sep|Oct|Nov|Dec)\b",
               re.IGNORECASE),                                       # 4 Mar
    re.compile(r"\b\d{2}\d{2}\d{4}\b"),                             # DDMMYYYY (no sep)
]

# Reference / UTR patterns
_REFERENCE_PATTERNS = [
    re.compile(r"\b[A-Z]{3,6}\d{10,22}\b"),               # NEFT/IMPS/UPI ref
    re.compile(r"\bUTR\s*:?\s*[\w\d]{10,22}\b", re.IGNORECASE),
    re.compile(r"\b\d{16,22}\b"),                          # Long numeric ref
    # Generic refs must contain digits — avoids matching long English words
    # (e.g. merchant names ending in "INTERNATIONAL").
    re.compile(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{10,24}\b", re.IGNORECASE),
]


class NumericParser:
    """
    Utility class for recognising and parsing numeric values in raw OCR text.
    All methods are stateless; an instance is kept only for potential
    future caching.
    """

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def is_amount(text: str) -> bool:
        """Return True if *text* strictly matches a monetary amount format."""
        t = text.strip()
        if re.match(r"^-?0\d{3,}$", t) and "." not in t:
            return False
        if NumericParser.is_reference(t):
            return False
        if (
            t.replace(",", "").replace(".", "").isdigit()
            and len(t.replace(",", "").replace(".", "")) >= 11
            and "." not in t
        ):
            return False
        return bool(_AMOUNT_PATTERN.match(t)) and len(_CLEAN_AMOUNT.sub("", t)) >= 1

    @staticmethod
    def looks_like_number(text: str) -> bool:
        """
        Return True if *text* is predominantly numeric. 
        Highly resilient to OCR noise (e.g., '8,000,00', '500.00*').
        Used for spatial clustering rather than strict parsing.
        """
        t = text.strip()
        # Remove common currency symbols and signs
        t = re.sub(r"^[+-]?\s*(?:INR|Rs\.?|₹|USD|\$|EUR|€)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*(?:CR|DR|Cr|Dr|cr|dr)$", "", t, flags=re.IGNORECASE)
        
        digits = sum(1 for c in t if c.isdigit())
        if digits == 0:
            return False
            
        # Allowed chars: digits, commas, periods, spaces, OCR artifacts
        allowed = sum(1 for c in t if c.isdigit() or c in "., '")
        return allowed / len(t) >= 0.7

    @staticmethod
    def is_date(text: str) -> bool:
        """Return True if *text* contains a date-like pattern."""
        t = text.strip()
        # Long bare numbers are refs/account ids, not dates
        if t.isdigit() and len(t) > 6:
            return False
        for pattern in _DATE_PATTERNS:
            if pattern.search(t):
                return True
        return False

    @staticmethod
    def extract_amounts(text: str) -> List[float]:
        """Return all monetary values found in *text*, left-to-right."""
        values: List[float] = []
        for m in re.finditer(
            r"(?:[+-]?\s*)?(?:\d{1,3}(?:,\d{2})*(?:,\d{3})?|\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,4})?",
            text,
        ):
            raw = m.group(0)
            cleaned = _CLEAN_AMOUNT.sub("", raw)
            if not cleaned or cleaned == ".":
                continue
            try:
                val = float(cleaned)
            except ValueError:
                continue
            if val >= 0.01:
                values.append(val)
        return values

    @staticmethod
    def is_reference(text: str) -> bool:
        """Return True if *text* looks like a transaction reference."""
        for pattern in _REFERENCE_PATTERNS:
            if pattern.search(text.strip()):
                return True
        return False

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_amount(self, text: str) -> Optional[Tuple[float, str]]:
        """
        Parse a monetary string and return (value, sign).

        sign is one of: "debit", "credit", "unknown"

        Returns None if text cannot be parsed.
        """
        t = text.strip()
        if not self.is_amount(t):
            return None

        sign = "unknown"
        if _CR_SUFFIX.search(t):
            sign = "credit"
        elif _DR_SUFFIX.search(t):
            sign = "debit"
        if _NEGATIVE.match(t):
            sign = "debit"

        # Strip everything except digits and dot
        numeric_str = _CLEAN_AMOUNT.sub("", t)
        if not numeric_str or numeric_str == ".":
            return None
        try:
            value = float(numeric_str)
        except ValueError:
            return None

        return (value, sign)

    @staticmethod
    def extract_date(text: str) -> Optional[str]:
        """
        Extract and return the first date-like string from *text*.
        Returns None if no date found.
        """
        for pattern in _DATE_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(0).strip()
        return None

    @staticmethod
    def extract_reference(text: str) -> Optional[str]:
        """Extract a transaction reference/UTR from *text*."""
        for pattern in _REFERENCE_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(0).strip()
        return None

    # Compiled pattern for bare calendar years (1900-2099) with no decimal/comma formatting.
    # Such values are date fragments, not monetary amounts.
    _BARE_YEAR = re.compile(r"^(19|20)\d{2}$")

    def clean_amount_str(self, text: str) -> Optional[float]:
        """Parse and return the float value only, ignoring sign semantics, while healing OCR typos like 1,95.009.02"""
        t = text.strip()

        # Guard: do not hallucinate amounts from explicit dates (e.g. "02 Apr 2026" -> 22026.0)
        if self.is_date(t):
            return None

        # Guard: bare 4-digit calendar years (e.g. "2026", "1999") are date fragments,
        # not monetary amounts. Genuine amounts appear as "2,026.00" or "2026.00".
        if self._BARE_YEAR.match(t):
            return None

        # Guard: integers with leading zeros (e.g. "00380066") and no decimal 
        # are reference/ID numbers, not monetary amounts.
        if re.match(r"^-?0\d{3,}$", t) and "." not in t:
            return None

        # Guard: require the string to be predominantly numeric before aggressive stripping
        if not self.looks_like_number(t):
            return None

        
        # Remove trailing/leading text that isn't a digit, decimal, comma, or minus
        t = re.sub(r"[^\d.,-]", "", t)
        if not t:
            return None
            
        is_negative = t.startswith("-")
        t = t.lstrip("-")
        
        # Look for the true decimal separator (typically the LAST . or , followed by exactly 2 digits)
        m = re.search(r"[,.](\d{2})$", t)
        if m:
            cents = m.group(1)
            # Remove the cents and the delimiter
            t = t[:-3]
            # Remove all remaining . and , (treating them as thousands separators)
            t = re.sub(r"[,.]", "", t)
            t = f"{t}.{cents}"
        else:
            # Whole number or single decimal digit? Remove all . and , just to be safe
            t = re.sub(r"[,.]", "", t)
            
        if not t:
            return None
            
        try:
            val = float(t)
            return -val if is_negative else val
        except ValueError:
            return None
