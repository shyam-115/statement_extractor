"""
Semantic Amount Resolver — context-aware CR/DR polarity resolution.

Resolves debit/credit ambiguity using section headers, column headers,
running balance trends, and keyword rules.  Does not mutate source tokens.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..schemas import ColumnZone, LogicalRow, Transaction

logger = logging.getLogger(__name__)

_CREDIT_SECTION = re.compile(
    r"\b(?:cashback|cash\s*back|refund|reward|reversal\s+credit)\b",
    re.IGNORECASE,
)
_REVERSAL = re.compile(r"\breversal\b", re.IGNORECASE)


@dataclass
class ResolvedAmount:
    """Result of semantic amount resolution."""
    debit: Optional[float] = None
    credit: Optional[float] = None
    confidence: float = 0.0
    rule_applied: str = ""


class SemanticAmountResolver:
    """
    Resolves debit/credit polarity using contextual rules.

    Parameters
    ----------
    section_header : optional text from table/section header row
    column_headers : mapping role -> header text
    prev_balance : running balance from previous transaction
    """

    def resolve(
        self,
        txn: Transaction,
        section_header: str = "",
        column_headers: Optional[dict] = None,
        prev_balance: Optional[float] = None,
        row: Optional[LogicalRow] = None,
    ) -> ResolvedAmount:
        """
        Return resolved debit/credit with confidence score.

        Starting point is the transaction's current debit/credit values.
        """
        column_headers = column_headers or {}
        debit = txn.debit
        credit = txn.credit
        narration = (txn.description or "") + " " + (txn.raw_text or "")
        context = f"{section_header} {narration}".upper()
        confidence = 0.5
        rule = "passthrough"

        # Rule 1: Explicit column headers
        if column_headers.get("debit") and debit is not None and credit is None:
            confidence = 0.85
            rule = "explicit_debit_column"
            return ResolvedAmount(debit=debit, credit=None, confidence=confidence, rule_applied=rule)
        if column_headers.get("credit") and credit is not None and debit is None:
            confidence = 0.85
            rule = "explicit_credit_column"
            return ResolvedAmount(debit=None, credit=credit, confidence=confidence, rule_applied=rule)

        # Rule 2: Cashback / refund sections → credit
        if _CREDIT_SECTION.search(context):
            amount = credit or debit
            if amount is not None:
                confidence = 0.9
                rule = "cashback_refund_section"
                return ResolvedAmount(
                    debit=None, credit=amount, confidence=confidence, rule_applied=rule
                )

        # Rule 3: Reversal → invert usual polarity
        if _REVERSAL.search(context):
            if debit is not None and credit is None:
                confidence = 0.8
                rule = "reversal_invert_to_credit"
                return ResolvedAmount(
                    debit=None, credit=debit, confidence=confidence, rule_applied=rule
                )
            if credit is not None and debit is None:
                confidence = 0.8
                rule = "reversal_invert_to_debit"
                return ResolvedAmount(
                    debit=credit, credit=None, confidence=confidence, rule_applied=rule
                )

        # Rule 4: Running balance trend
        if prev_balance is not None and txn.balance is not None:
            # Credit-card rows often carry a *rewards/cashback* column in
            # ``balance`` — it is not a running ledger balance.  When the
            # source row already states explicit Dr/Cr amounts, do not infer
            # polarity from (prev_balance, balance) deltas.
            if not re.search(r"\b(?:Dr|Cr)\b", txn.raw_text or "", re.IGNORECASE):
                resolved = self._resolve_from_balance_trend(
                    prev_balance, txn.balance, debit, credit
                )
                if resolved is not None:
                    d, c, conf = resolved
                    return ResolvedAmount(
                        debit=d, credit=c, confidence=conf, rule_applied="balance_trend"
                    )

        # Rule 5: Both set — trust as-is with lower confidence
        if debit is not None or credit is not None:
            confidence = 0.6
            rule = "existing_assignment"
            return ResolvedAmount(
                debit=debit, credit=credit, confidence=confidence, rule_applied=rule
            )

        return ResolvedAmount(debit=None, credit=None, confidence=0.0, rule_applied="none")

    def apply_to_transaction(
        self,
        txn: Transaction,
        **kwargs,
    ) -> Transaction:
        """
        Apply resolution to a transaction (mutates debit/credit only when
        confidence >= 0.75).  Original values preserved in raw_text context.
        """
        result = self.resolve(txn, **kwargs)
        if result.confidence >= 0.75:
            if result.debit != txn.debit or result.credit != txn.credit:
                txn.debit = result.debit
                txn.credit = result.credit
        return txn

    @staticmethod
    def _resolve_from_balance_trend(
        prev_balance: float,
        curr_balance: float,
        debit: Optional[float],
        credit: Optional[float],
    ) -> Optional[Tuple[Optional[float], Optional[float], float]]:
        """Pick polarity that makes arithmetic consistent with balance change."""
        delta = curr_balance - prev_balance
        if abs(delta) < 0.001:
            return None

        amount = debit or credit
        if amount is None:
            return None

        # Credit increases balance
        if delta > 0:
            if debit is not None and credit is None:
                # Would decrease balance with debit — flip to credit
                return None, amount, 0.82
            return None, amount if credit is None else credit, 0.78
        # Debit decreases balance
        if credit is not None and debit is None:
            return amount, None, 0.82
        return amount if debit is None else debit, None, 0.78

    @staticmethod
    def column_header_map(zones: List[ColumnZone]) -> dict:
        """Build role -> header text map from ColumnZones."""
        return {
            z.semantic_role: (z.header_text or z.semantic_role or "")
            for z in zones
            if z.semantic_role
        }
