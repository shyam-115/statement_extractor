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
from typing import Dict, List, Optional, Tuple

from ..config import ExtractorConfig
from ..schemas import ColumnZone, LogicalRow, OCRToken, Transaction, ValidationStatus
from ..clustering.column_detector import ColumnDetector
from ..utils.row_filters import filter_data_rows, filter_header_rows_only, is_noise_row
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

        if self.config.pipeline.document_fidelity:
            candidate_rows = filter_header_rows_only(rows)
        else:
            candidate_rows = filter_data_rows(rows)

        for row in candidate_rows:
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
        # Bucket tokens by role.
        # 'index' tokens (serial number column) are bucketed separately
        # and completely excluded from financial resolution.
        buckets: Dict[str, List[OCRToken]] = {
            "index": [], "date": [], "value_date": [], "narration": [],
            "debit": [], "credit": [], "balance": [], "reference": [],
            "unknown": [], "noise": [],
        }

        for token in row.tokens:
            col_id = ColumnDetector.assign_token_to_column(token, zones)
            role = role_map.get(col_id, "unknown")

            # Tokens from serial-number / index columns are not financial data — skip entirely.
            if role == "index":
                buckets["index"].append(token)
                continue

            # Tokens from noise columns (cashback, rewards, etc.) — skip entirely.
            if role == "noise":
                buckets["noise"].append(token)
                continue

            # Override: numeric tokens in an "unknown" zone go to narration
            # only if they look like dates; amounts stay unknown for later resolution
            if role == "unknown":
                if token.is_date:
                    role = "date"
                elif not token.is_numeric:
                    role = "narration"

            buckets[role].append(token)

        fidelity = self.config.pipeline.document_fidelity

        # Require at least a date OR numeric amount to be a valid transaction row
        has_date = bool(buckets["date"]) or any(t.is_date for t in buckets["narration"])
        has_amount = any(
            bool(buckets[r]) for r in ("debit", "credit", "balance", "unknown")
            if any(t.is_numeric for t in buckets.get(r, []))
        )
        if not fidelity and not has_date and not has_amount:
            return None
        if fidelity and not row.tokens and not row.full_text.strip():
            return None

        # --- Date ---
        txn_date = self._resolve_date(
            row,
            buckets["date"] + buckets["value_date"],
            buckets["narration"],
        )

        # --- Reference ---
        reference_no = self._resolve_reference(buckets["reference"], buckets["narration"])

        # --- Amounts ---
        debit_val, debit_sign = self._resolve_amount_with_sign(buckets["debit"])
        credit_val, credit_sign = self._resolve_amount_with_sign(buckets["credit"])
        balance, _ = self._resolve_amount_with_sign(buckets["balance"])
        
        debit, credit = None, None

        # Apply explicit CR/DR suffix overrides over standard column assignments
        if debit_sign == "credit":
            credit = debit_val
        elif debit_val is not None:
            debit = debit_val

        if credit_sign == "debit":
            debit = credit_val
        elif credit_val is not None:
            credit = credit_val

        # --- Resolve unassigned numeric tokens via spatial heuristics ---
        unknown_numeric_tokens = sorted(
            [t for t in buckets["unknown"] if t.is_numeric],
            key=lambda t: t.normalized_x,
        )

        if unknown_numeric_tokens:
            debit_zone = next((z for z in zones if role_map.get(z.column_id) == "debit"), None)
            credit_zone = next((z for z in zones if role_map.get(z.column_id) == "credit"), None)

            # Prefer explicit Dr/Cr on unknown-bucket numerics (single-column CC layouts)
            # before treating a lone number as a running balance.
            signed_remaining: List[OCRToken] = []
            for t in unknown_numeric_tokens:
                if self._parser.is_reference(t.text):
                    continue
                parsed = self._parser.parse_amount(t.text)
                if parsed is None:
                    signed_remaining.append(t)
                    continue
                amount_val, sign = parsed
                if sign == "credit" and credit is None:
                    credit = amount_val
                elif sign == "debit" and debit is None:
                    debit = amount_val
                else:
                    signed_remaining.append(t)
            unknown_numeric_tokens = sorted(
                signed_remaining, key=lambda t: t.normalized_x
            )

            # If everything is missing, rightmost is balance, leftmost is movement
            if balance is None and debit is None and credit is None:
                if len(unknown_numeric_tokens) == 1:
                    lone = unknown_numeric_tokens[0]
                    lone_parsed = self._parser.parse_amount(lone.text)
                    if lone_parsed is None or lone_parsed[1] == "unknown":
                        balance = self._parser.clean_amount_str(lone.text)
                    unknown_numeric_tokens = []
                elif unknown_numeric_tokens:
                    balance = self._parser.clean_amount_str(unknown_numeric_tokens[-1].text)
                    unknown_numeric_tokens = unknown_numeric_tokens[:-1]
            
            # Distribute remaining unknown tokens to missing debit/credit slots
            for t in unknown_numeric_tokens:
                if self._parser.is_reference(t.text):
                    continue
                parsed = self._parser.parse_amount(t.text)
                if parsed is None:
                    continue
                amount_val, sign = parsed

                if debit is None and credit is None:
                    assigned = False
                    
                    if sign == "credit":
                        credit = amount_val
                        assigned = True
                    elif sign == "debit":
                        debit = amount_val
                        assigned = True

                    if not assigned and debit_zone and not credit_zone:
                        if t.normalized_x > debit_zone.right_boundary + 0.01:
                            credit = amount_val
                            assigned = True
                        else:
                            debit = amount_val
                            assigned = True
                    elif credit_zone and not debit_zone:
                        if t.normalized_x < credit_zone.left_boundary - 0.01:
                            debit = amount_val
                            assigned = True
                        else:
                            credit = amount_val
                            assigned = True
                            
                    if not assigned:
                        if debit_zone and credit_zone:
                            dist_debit = abs(t.normalized_x - debit_zone.x_center)
                            dist_credit = abs(t.normalized_x - credit_zone.x_center)
                            if dist_credit < dist_debit:
                                credit = amount_val
                            else:
                                debit = amount_val
                        else:
                            debit = amount_val  # absolute fallback

                elif debit is None and credit is not None:
                    if sign == "credit":
                        credit = amount_val  # override or append
                    else:
                        debit = amount_val
                elif credit is None and debit is not None:
                    if sign == "debit":
                        debit = amount_val
                    else:
                        credit = amount_val

        # --- Inline Dr/Cr reconciliation (esp. credit-card + multi-column PDFs) ---
        debit, credit = self._refine_debit_credit_from_inline_suffixes(
            row, debit, credit, balance
        )

        # --- Narration ---
        description = self._resolve_narration(
            buckets["narration"] + buckets["date"],
            buckets["unknown"],
            txn_date,
            reference_no
        )

        if not fidelity:
            # ------------------------------------------------------------------
            # Generalized structural guard (bank-agnostic)
            # ------------------------------------------------------------------
            if debit is None and credit is None:
                if not self._is_valid_calendar_date(txn_date):
                    return None

            if is_noise_row(row):
                return None
            if not description and txn_date is None and balance is None:
                return None
        elif not description:
            description = row.full_text.strip() or None

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
        val, _ = self._resolve_amount_with_sign(tokens)
        return val

    # Standalone CR/DR indicator tokens (case-insensitive)
    _CR_INDICATORS = frozenset({"cr", "cr.", "credit"})
    _DR_INDICATORS = frozenset({"dr", "dr.", "debit"})
    # Inline amount + Dr/Cr anywhere in the row (credit-card PDFs)
    _INLINE_DRCR_RE = re.compile(
        r"(?:INR|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)\s*(Dr|Cr)\b",
        re.IGNORECASE,
    )

    def _refine_debit_credit_from_inline_suffixes(
        self,
        row: LogicalRow,
        debit: Optional[float],
        credit: Optional[float],
        balance: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Align debit/credit with explicit Dr/Cr markers in *row.full_text*.

        Uses the largest amount that carries an inline suffix so credit-card
        rows with a cashback column still pick the correct main movement.

        Savings / passbook PDFs often append ``Cr`` only to the *running balance*
        (e.g. ``270.00  53762.47 Cr``).  That suffix must not overwrite a
        movement amount already taken from debit/credit columns, and must not
        be mistaken for a deposit when it matches the extracted balance.
        """
        pairs: List[Tuple[float, str]] = []
        for m in self._INLINE_DRCR_RE.finditer(row.full_text):
            parsed = self._parser.parse_amount(f"{m.group(1)} {m.group(2)}")
            if parsed is None:
                continue
            val, sign = parsed
            if sign == "unknown":
                continue
            pairs.append((val, sign))
        if not pairs:
            return debit, credit

        prim_val, prim_sign = max(pairs, key=lambda x: abs(x[0]))

        if debit is None and credit is None:
            # Ledger balance is often the only inline ``Cr`` on savings rows.
            # Do not fabricate a "credit" movement from it.
            if (
                balance is not None
                and prim_sign == "credit"
                and abs(prim_val - balance)
                <= max(abs(balance) * 0.015, 0.05)
            ):
                return None, None
            if prim_sign == "debit":
                return prim_val, None
            return None, prim_val

        eff_side = "debit" if debit is not None else "credit"
        eff_val = debit if debit is not None else credit
        if eff_val is None:
            return debit, credit

        tol = max(abs(prim_val) * 0.015, 0.05)
        if abs(eff_val - prim_val) <= tol:
            if (eff_side == "debit" and prim_sign != "debit") or (
                eff_side == "credit" and prim_sign != "credit"
            ):
                if prim_sign == "debit":
                    return prim_val, None
                return None, prim_val
            return debit, credit

        # ``prim_val`` is typically the suffixed running balance, not the
        # withdrawal/deposit cell — keep column / spatial assignments.
        return debit, credit

    def _resolve_amount_with_sign(self, tokens: List[OCRToken]) -> Tuple[Optional[float], str]:
        """
        Parse the first valid amount from *tokens*, returning (value, sign).

        Sign detection strategy (ordered by priority):
        1. Inline suffix — amount text itself contains CR/DR (e.g. "589.00 Cr")
        2. Adjacent token — a separate non-numeric token in the same bucket
           carries the CR/DR indicator (e.g. tokens ["589.00", "Cr"])
        3. Falls back to "unknown" if no indicator is found.
        """
        for token in tokens:
            if self._parser.is_reference(token.text):
                continue
            parsed = self._parser.parse_amount(token.text)
            if parsed is not None:
                val, sign = parsed
                if sign != "unknown":
                    return parsed
                # Amount found but no inline sign — scan all tokens in the
                # bucket for standalone CR/DR indicator tokens.
                for t in tokens:
                    t_lower = t.text.strip().lower()
                    if t_lower in self._CR_INDICATORS:
                        return val, "credit"
                    if t_lower in self._DR_INDICATORS:
                        return val, "debit"
                return val, "unknown"
        return None, "unknown"

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
