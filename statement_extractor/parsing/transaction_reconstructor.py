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
            buckets["narration"], buckets["unknown"], txn_date, reference_no
        )

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
        Build a clean narration string by joining non-date, non-reference
        tokens from the narration and unknown buckets.
        """
        parts: List[str] = []
        for token in narration_tokens:
            text = token.text.strip()
            if not text:
                continue
            # Skip tokens that were already captured as date/reference
            if txn_date and text == txn_date:
                continue
            if reference_no and text == reference_no:
                continue
            # Skip tokens that are purely numeric (likely misrouted amounts)
            if token.is_numeric:
                continue
            parts.append(text)

        # Non-numeric unknown tokens also contribute to narration
        for token in unknown_tokens:
            if not token.is_numeric and not token.is_date:
                text = token.text.strip()
                if text:
                    parts.append(text)

        if not parts:
            return None
        return " ".join(parts)
