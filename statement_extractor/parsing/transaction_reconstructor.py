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
from ..utils.row_filters import filter_data_rows, is_noise_row
from .numeric_parser import NumericParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CR / DR narration patterns — most reliable signal in Indian bank statements.
# Banks embed these directly in the transaction description:
#   "NEFT CR-PAYEE", "UPI/DR - MERCHANT", "IMPS CR", "POS TXN", etc.
# ---------------------------------------------------------------------------
_NARRATION_CREDIT = re.compile(
    r"\b(?:UPI|NEFT|IMPS|RTGS|NACH|ENACH|ACH)[/\s\-]*CR\b"  # UPI/CR, NEFT CR-
    r"|\bCR-"                                                  # CR-PAYEE NAME
    r"|\bDEPOSIT(?:ED)?\b"                                    # DEPOSITED
    r"|\bRECEIVED\b",                                         # RECEIVED
    re.IGNORECASE,
)

_NARRATION_DEBIT = re.compile(
    r"\b(?:UPI|NEFT|IMPS|RTGS|NACH|ENACH|ACH)[/\s\-]*DR\b"  # UPI/DR, NEFT DR-
    r"|\bDR-"                                                  # DR-PAYEE NAME
    r"|\bPOS\s+(?:TXN|TRANSACTION|PURCHASE|DEBIT)\b"          # POS TXN
    r"|\bATM[\s\-]*(?:WDL|WTHDRL|WITHDRAWAL|CASH|W[DT])?\b"  # ATM WDL / ATM-WITHDRL
    r"|\bWITHDRA(?:W|WL|WN|WAL)?\b",                          # WITHDRAW / WITHDRAWAL / WITTHDRAN (OCR)
    re.IGNORECASE,
)


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

            # 'amount' sentinel marks a combined Deposits/Withdrawals column
            # (e.g. ICICI "Deposits Withdrawals" merged into one OCR zone).
            # Use the token's x-position relative to the zone midpoint to
            # determine sub-column: left half → credit (Deposits),
            #                       right half → debit (Withdrawals).
            # This correctly handles rows that have amounts in BOTH sub-columns.
            if role == "amount":
                if token.is_numeric:
                    amount_zone = next(
                        (z for z in zones if z.column_id == col_id), None
                    )
                    if (
                        amount_zone is not None
                        and (amount_zone.right_boundary - amount_zone.left_boundary) > 0.04
                    ):
                        mid = (amount_zone.left_boundary + amount_zone.right_boundary) / 2
                        role = "credit" if token.normalized_x <= mid else "debit"
                    else:
                        role = "unknown"
                else:
                    role = "narration"

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
        def get_all_amounts(tokens):
            return [
                (t, self._parser.clean_amount_str(t.text))
                for t in sorted(tokens, key=lambda x: (x.normalized_y, x.normalized_x))
                if t.is_numeric and self._parser.clean_amount_str(t.text) is not None
            ]

        debit_amts = get_all_amounts(buckets["debit"])
        credit_amts = get_all_amounts(buckets["credit"])
        balance_amts = get_all_amounts(buckets["balance"])
        unknown_amts = get_all_amounts(buckets["unknown"])
        
        # If a bucket has multiple amounts, keep the best one and reassign others to unknown
        if len(balance_amts) > 1:
            unknown_amts.extend(balance_amts[:-1])
            balance_amts = [balance_amts[-1]]
            
        if len(credit_amts) > 1:
            unknown_amts.extend(credit_amts[1:])
            credit_amts = [credit_amts[0]]
            
        if len(debit_amts) > 1:
            unknown_amts.extend(debit_amts[1:])
            debit_amts = [debit_amts[0]]

        # Sort unknown amounts by y then x again
        unknown_amts.sort(key=lambda x: (x[0].normalized_y, x[0].normalized_x))

        debit = debit_amts[0][1] if debit_amts else None
        credit = credit_amts[0][1] if credit_amts else None
        balance = balance_amts[0][1] if balance_amts else None
        
        unknown_amounts = [a[1] for a in unknown_amts]

        # If debit/credit/balance are all missing but we have unknowns,
        # try to distribute them (rightmost → balance heuristic)
        if balance is None and debit is None and credit is None and unknown_amounts:
            if len(unknown_amounts) == 1:
                balance = unknown_amounts[0]
            elif len(unknown_amounts) >= 2:
                balance = unknown_amounts[-1]
                # Tentatively assign leftmost to credit; validator swaps to debit if needed
                credit = unknown_amounts[0]
                
        # Balance column set but movement in another unassigned numeric zone.
        # Tentatively assign to credit so that opening-balance / deposit rows
        # (e.g. B/F, NEFT CR without explicit CR in description) are correct.
        # BalanceValidator will swap credit ↔ debit when arithmetic requires it.
        elif balance is not None and debit is None and credit is None and unknown_amounts:
            credit = unknown_amounts[0]
            
        # Movement set but balance missing and we have unknowns
        elif balance is None and (debit is not None or credit is not None) and unknown_amounts:
            balance = unknown_amounts[-1]

        # --- Narration ---
        description = self._resolve_narration(
            buckets["narration"] + buckets["date"], 
            buckets["unknown"], 
            txn_date, 
            reference_no
        )

        # --- CR/DR description correction (high-priority, pre-arithmetic) ---
        # Use embedded CR/DR markers from the raw row text to fix column
        # misassignments that occur when a token's x-coordinate falls in the
        # wrong zone (common on narrow/borderline amount columns).
        debit, credit = self._apply_description_correction(row.full_text, debit, credit)

        # Drop rows that are only metadata (no balance and no movement)
        if balance is None and debit is None and credit is None:
            return None
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
        tokens = sorted(narration_tokens + unknown_tokens, key=lambda t: (t.normalized_y, t.normalized_x))
        for token in tokens:
            text = token.text.strip()
            if not text:
                continue

            # Strip any date found in the token. This handles cases where OCR 
            # merged a date with narration/reference text (e.g. "03-03-2025 REF123").
            d = self._parser.extract_date(text)
            if d:
                text = text.replace(d, "").strip()
            
            # Also ensure txn_date is stripped if it somehow missed the extract_date regex
            if txn_date and txn_date in text:
                text = text.replace(txn_date, "").strip()

            if not text:
                continue

            # Clean up residual artifacts from stripping
            text = re.sub(r"^[-/]+|[-/]+$", "", text).strip()

            if text and not self._parser.is_amount(text):
                parts.append(text)

        return " ".join(parts).strip() if parts else ""

    def _apply_description_correction(
        self,
        raw_text: str,
        debit: Optional[float],
        credit: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Correct debit/credit misassignment using CR/DR markers embedded in
        the raw transaction text.

        Indian banks reliably prefix or suffix CR/DR to indicate direction:
          - "UPI/CR", "NEFT CR-PAYEE"  → credit transaction
          - "UPI/DR", "POS TXN"        → debit transaction

        Only acts when exactly one side carries a value; when both sides are
        set (or both are None) the validator's arithmetic will resolve it.
        """
        has_debit = debit is not None
        has_credit = credit is not None

        # Nothing to do when both sides are filled or both are empty
        if has_debit == has_credit:
            return debit, credit

        is_credit_txn = bool(_NARRATION_CREDIT.search(raw_text))
        is_debit_txn = bool(_NARRATION_DEBIT.search(raw_text))

        if is_credit_txn and not is_debit_txn:
            if has_debit:
                logger.debug(
                    "CR/DR hint: moving debit %.2f → credit | %s",
                    debit, raw_text[:80],
                )
                return None, debit
        elif is_debit_txn and not is_credit_txn:
            if has_credit:
                logger.debug(
                    "CR/DR hint: moving credit %.2f → debit | %s",
                    credit, raw_text[:80],
                )
                return credit, None

        return debit, credit
