"""
Transaction Reconstructor — maps OCR tokens onto column zones and
assembles raw Transaction objects.

Algorithm
---------
For each non-header, non-continuation LogicalRow:
  1. For every token in the row, find its nearest ColumnZone.
  2. Bucket tokens by role (date, narration, debit, credit, balance, reference).
  3. Resolve each bucket:
     - date      → NumericParser.extract_date()
     - narration → concatenate all non-numeric tokens in narration zone
     - debit/credit/balance → NumericParser.clean_amount_str()
     - reference → NumericParser.extract_reference()
  4. Build a raw Transaction with confidence = mean OCR confidence of tokens.

Multiline narration handling
-----------------------------
Continuation rows (is_continuation=True) are fused to their parent row
by the RowGrouper.merge_continuations() call before this module runs.
However, wide narration zones may also contain date/ref fragments — these
are extracted first so the narration field doesn't include them.

Single-amount-column handling
------------------------------
If only ONE numeric column exists (no explicit debit/credit split),
all amounts land in that column.  The balance validator will later
determine sign semantics from arithmetic continuity.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ..config import ExtractorConfig
from ..schemas import ColumnZone, LogicalRow, OCRToken, Transaction, ValidationStatus
from ..clustering.column_detector import ColumnDetector
from ..utils.row_filters import filter_data_rows, is_noise_row
from .numeric_parser import NumericParser

logger = logging.getLogger(__name__)


class TransactionReconstructor:
    """
    Builds Transaction objects from LogicalRows + ColumnZones.

    Parameters
    ----------
    config : ExtractorConfig
    """

    def __init__(self, config: ExtractorConfig) -> None:
        self.config = config
        self._parser = NumericParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        rows: List[LogicalRow],
        zones: List[ColumnZone],
    ) -> List[Transaction]:
        """
        Convert non-header rows into Transaction objects.

        Parameters
        ----------
        rows  : LogicalRow list (continuations already merged)
        zones : ColumnZone list with semantic_role set
        """
        transactions: List[Transaction] = []
        role_map: Dict[int, str] = {
            z.column_id: z.semantic_role
            for z in zones
            if z.semantic_role
        }

        for row in filter_data_rows(rows):
            txn = self._build_transaction(row, zones, role_map)
            if txn is not None:
                transactions.append(txn)

        return transactions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_transaction(
        self,
        row: LogicalRow,
        zones: List[ColumnZone],
        role_map: Dict[int, str],
    ) -> Optional[Transaction]:
        """
        Map every token in *row* to a column zone and build a Transaction.
        Returns None if the row carries no meaningful financial data.
        """
        # Bucket tokens by role
        buckets: Dict[str, List[OCRToken]] = {
            "date": [], "narration": [], "debit": [],
            "credit": [], "balance": [], "reference": [], "unknown": [],
        }

        for token in row.tokens:
            col_id = ColumnDetector.assign_token_to_column(token, zones)
            role = role_map.get(col_id, "unknown")

            # Override: numeric tokens in an "unknown" zone go to narration
            # only if they look like dates; amounts stay unknown for later resolution
            if role == "unknown":
                if token.is_date:
                    role = "date"
                elif not token.is_numeric:
                    role = "narration"

            buckets[role].append(token)

        # Require at least a date OR numeric amount to be a valid transaction row
        has_date = bool(buckets["date"]) or any(t.is_date for t in buckets["narration"])
        has_amount = any(
            bool(buckets[r]) for r in ("debit", "credit", "balance", "unknown")
            if any(t.is_numeric for t in buckets.get(r, []))
        )
        if not has_date and not has_amount:
            return None

        # --- Date ---
        txn_date = self._resolve_date(row, buckets["date"], buckets["narration"])

        # --- Reference ---
        reference_no = self._resolve_reference(buckets["reference"], buckets["narration"])

        # --- Amounts ---
        debit   = self._resolve_amount(buckets["debit"])
        credit  = self._resolve_amount(buckets["credit"])
        balance = self._resolve_amount(buckets["balance"])

        # Single-amount-column: unknown numeric tokens are stored in a
        # temporary field so the balance validator can resolve them later.
        unknown_amounts = [
            self._parser.clean_amount_str(t.text)
            for t in buckets["unknown"]
            if t.is_numeric
        ]
        unknown_amounts = [a for a in unknown_amounts if a is not None]

        # If debit/credit/balance are all missing but we have unknowns,
        # try to distribute them (rightmost → balance heuristic)
        if balance is None and debit is None and credit is None and unknown_amounts:
            unknown_tokens_sorted = sorted(
                [t for t in buckets["unknown"] if t.is_numeric],
                key=lambda t: t.normalized_x,
            )
            if len(unknown_amounts) == 1:
                balance = unknown_amounts[0]
            elif len(unknown_amounts) >= 2:
                # Rightmost → balance; leftmost movement → debit/credit TBD
                balance = self._parser.clean_amount_str(
                    unknown_tokens_sorted[-1].text
                )
                movement = self._parser.clean_amount_str(
                    unknown_tokens_sorted[0].text
                )
                if movement is not None:
                    debit = movement  # validator will swap to credit if needed

        # Balance column set but movement in another unassigned numeric zone
        elif balance is not None and debit is None and credit is None and unknown_amounts:
            debit = unknown_amounts[0]

        # --- Narration ---
        description = self._resolve_narration(
            buckets["narration"] + buckets["date"],
            buckets["unknown"],
            txn_date,
            reference_no
        )

        # ------------------------------------------------------------------
        # Generalized structural guard (bank-agnostic)
        # ------------------------------------------------------------------
        # By accounting definition, a debit/credit transaction MUST carry at
        # least one financial movement (debit OR credit).  A row that has only
        # a balance value and no movement is structurally metadata:
        #   - Opening balance marker   (first row of the statement)
        #   - Summary / totals footer  (e.g. "Statement Balance as on:")
        #   - Account info row         (e.g. "A/c No: 12345  Balance: X")
        #
        # EXCEPTION: if the row carries a *structurally valid* calendar date
        # (day 1-31, month 1-12) it is treated as an opening-balance entry and
        # kept.  An 8-digit credit card fragment that accidentally matches a
        # date regex (e.g. "68212345") will fail this check and be dropped.
        # ------------------------------------------------------------------
        if debit is None and credit is None:
            if not self._is_valid_calendar_date(txn_date):
                # No movement + no real date → definitively metadata, not a transaction
                return None
            # Has a real date but no movement → keep as opening-balance entry
            # (single-balance row; movement will be resolved later if needed)

        # Drop remaining true noise rows (already passed structural check)
        if is_noise_row(row):
            return None
        if not description and txn_date is None and balance is None:
            return None

        # --- Confidence ---
        all_tokens = row.tokens
        confidence = (
            float(sum(t.confidence for t in all_tokens) / len(all_tokens))
            if all_tokens else 0.0
        )
        # Clamp to 0-1 (PaddleOCR returns 0-1 but guard against edge cases)
        confidence = max(0.0, min(1.0, confidence))

        return Transaction(
            txn_date=txn_date,
            description=description,
            reference_no=reference_no,
            debit=debit,
            credit=credit,
            balance=balance,
            confidence_score=round(confidence, 4),
            validation_status=ValidationStatus.NEEDS_REVIEW,
            page_num=row.page_num,
            raw_text=row.full_text,
        )

    # ------------------------------------------------------------------
    # Field resolvers
    # ------------------------------------------------------------------

    def _resolve_date(
        self,
        row: LogicalRow,
        date_tokens: List[OCRToken],
        narration_tokens: List[OCRToken],
    ) -> Optional[str]:
        """Extract a date string from date column, narration, or full row text."""
        for token in date_tokens:
            d = self._parser.extract_date(token.text)
            if d and not re.fullmatch(r"\d{8,}", d.replace(" ", "")):
                return d
        for token in sorted(row.tokens, key=lambda t: t.normalized_x):
            d = self._parser.extract_date(token.text)
            if d and not re.fullmatch(r"\d{8,}", d.replace(" ", "")):
                return d
        for token in narration_tokens:
            d = self._parser.extract_date(token.text)
            if d:
                return d
        return self._parser.extract_date(row.full_text)

    def _resolve_reference(
        self,
        ref_tokens: List[OCRToken],
        narration_tokens: List[OCRToken],
    ) -> Optional[str]:
        """Extract a reference number, checking ref bucket then narration."""
        for token in ref_tokens:
            r = self._parser.extract_reference(token.text)
            if r:
                return r
        for token in narration_tokens:
            r = self._parser.extract_reference(token.text)
            if r:
                return r
        return None

    def _resolve_amount(self, tokens: List[OCRToken]) -> Optional[float]:
        """Parse the first valid amount from *tokens*, None if none found."""
        for token in tokens:
            val = self._parser.clean_amount_str(token.text)
            if val is not None:
                return val
        return None

    def _resolve_narration(
        self,
        narration_tokens: List[OCRToken],
        unknown_tokens: List[OCRToken],
        txn_date: Optional[str],
        reference_no: Optional[str],
    ) -> Optional[str]:
        """
        Build a clean narration string by joining non-date tokens 
        from the narration, date, and unknown buckets.
        """
        parts: List[str] = []
        tokens = sorted(narration_tokens + unknown_tokens, key=lambda t: t.normalized_x)
        for token in tokens:
            text = token.text.strip()
            if not text:
                continue
                
            # If OCR merged the date and narration (e.g. "01-03-2025BY-TRANSFER"),
            # strip the date out to recover the residual narration text.
            if txn_date and txn_date in text:
                text = text.replace(txn_date, "").strip()
                
            # Clean up residual artifacts from stripping
            text = re.sub(r"^[-/]+|[-/]+$", "", text).strip()
            
            if text and not self._parser.is_amount(text):
                parts.append(text)

        return " ".join(parts).strip() if parts else ""

    # ------------------------------------------------------------------
    # Structural date validator (bank-agnostic)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_calendar_date(date_str: Optional[str]) -> bool:
        """
        Return True only if *date_str* represents a structurally valid
        calendar date (day 1–31, month 1–12).

        Rejects:
        - None / empty
        - Bare digit strings (e.g. "68212345" — credit card fragment)
        - Values where day > 31 or month > 12
        - Pure numeric strings with no separator (reference numbers)

        Accepts:
        - "01-03-2025"  → day=01, month=03 → valid
        - "20/03/2025"  → valid
        - "01 Mar 2025" → valid (month name)
        - "33-03-2025"  → day=33 → INVALID (OCR error in doc is kept as-is
                          but this row has movement so it won't reach this check)
        """
        if not date_str:
            return False

        # If the date string is all digits (possibly with spaces) and ≥ 7 chars
        # it is likely a reference number / account number, not a real date.
        stripped = date_str.replace(" ", "").replace("-", "").replace("/", "")
        if stripped.isdigit() and len(stripped) >= 7:
            # Only accept DDMMYYYY / YYYYMMDD patterns with valid ranges
            s = stripped
            if len(s) == 8:
                # Try DD MM YYYY
                try:
                    day, month = int(s[0:2]), int(s[2:4])
                    if 1 <= day <= 31 and 1 <= month <= 12:
                        return True
                except ValueError:
                    pass
                # Try YYYY MM DD
                try:
                    month, day = int(s[4:6]), int(s[6:8])
                    if 1 <= day <= 31 and 1 <= month <= 12:
                        return True
                except ValueError:
                    pass
            return False  # bare long digit string — not a real date

        # Has separators or month names — apply range check on first two numeric parts
        parts = re.split(r"[-/\s,]+", date_str.strip())
        numeric_parts = []
        for p in parts:
            p = p.strip()
            if p.isdigit():
                numeric_parts.append(int(p))

        if len(numeric_parts) >= 2:
            # Heuristic: if either of the first two parts is clearly a year (>31),
            # the other is day/month. Check neither exceeds valid ranges.
            a, b = numeric_parts[0], numeric_parts[1]
            if a > 1900:
                # YYYY-MM-DD style: b is month
                return 1 <= b <= 12
            if b > 1900:
                # DD-MM-YYYY style: a is day
                return 1 <= a <= 31
            # DD-MM style (no year): validate day and month
            return 1 <= a <= 31 and 1 <= b <= 12

        # Contains a month name (e.g. "01 Mar 2025") — always structural valid
        _MONTH_NAMES = {
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        }
        date_lower = date_str.lower()
        if any(m in date_lower for m in _MONTH_NAMES):
            return True

        return False
