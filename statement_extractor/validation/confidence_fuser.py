"""
Confidence Fuser — multi-factor confidence scoring for extracted transactions.

Fidelity-first design: no arithmetic validation is performed. The fused
score combines three signals that measure extraction quality without
modifying or second-guessing the source document values:

  1. OCR confidence        (0–1, weight 0.40) — mean token-level OCR score
  2. Date parse quality    (0–1, weight 0.25) — was a date successfully extracted?
  3. Amount parse quality  (0–1, weight 0.25) — are amount fields populated?
  4. Column assignment     (0–1, weight 0.10) — how tightly do tokens sit in zones?

Rationale
---------
OCR confidence is the primary signal. Date and amount presence indicate that
the row contains the expected financial fields. Column assignment tightness
distinguishes well-aligned rows from OCR fragments that landed in the wrong zone.
No arithmetic is computed — values are kept exactly as extracted.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..config import ConfidenceFusionConfig
from ..schemas import ColumnZone, ConfidenceFactors, OCRToken, Transaction

logger = logging.getLogger(__name__)


class ConfidenceFuser:
    """
    Computes fused confidence scores for Transaction objects.

    Parameters
    ----------
    config : ConfidenceFusionConfig
    """

    def __init__(self, config: ConfidenceFusionConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        txn: Transaction,
        row_tokens: List[OCRToken],
        zones: List[ColumnZone],
    ) -> Transaction:
        """
        Compute and attach a fused confidence score to *txn*.

        Parameters
        ----------
        txn        : Transaction object to score
        row_tokens : OCR tokens from the source LogicalRow
        zones      : column zones used during reconstruction

        Returns
        -------
        The same Transaction object with updated confidence_score and
        confidence_factors.
        """
        if not self.config.enabled:
            return txn

        cfg = self.config

        # ── Factor 1: OCR confidence ────────────────────────────────────
        ocr_score = self._ocr_score(row_tokens)

        # ── Factor 2: Date parse quality ───────────────────────────────
        date_score = 1.0 if txn.txn_date else 0.0

        # ── Factor 3: Amount parse quality ─────────────────────────────
        amount_score = self._amount_score(txn)

        # ── Factor 4: Column assignment tightness ──────────────────────
        col_score = self._column_assignment_score(row_tokens, zones)

        # ── Weighted fusion ─────────────────────────────────────────────
        fused = (
            cfg.weight_ocr_confidence   * ocr_score
            + cfg.weight_date_parse     * date_score
            + cfg.weight_amount_parse   * amount_score
            + cfg.weight_column_assignment * col_score
        )
        fused = max(0.0, min(1.0, fused))

        factors = ConfidenceFactors(
            ocr_confidence=round(ocr_score, 4),
            arithmetic_score=0.0,          # disabled — fidelity-first mode
            date_parse_score=round(date_score, 4),
            amount_parse_score=round(amount_score, 4),
            column_assignment_score=round(col_score, 4),
            fused_score=round(fused, 4),
        )

        txn.confidence_score = round(fused, 4)
        txn.confidence_factors = factors
        return txn

    def fuse_all(
        self,
        transactions: List[Transaction],
        row_tokens_map: Optional[dict] = None,
        zones: Optional[List[ColumnZone]] = None,
    ) -> List[Transaction]:
        """
        Fuse confidence for a list of transactions.

        When row_tokens_map/zones are unavailable, a simplified fusion
        using only OCR confidence + date/amount presence is applied.
        """
        for txn in transactions:
            tokens = (row_tokens_map or {}).get(id(txn), [])
            zs = zones or []
            self.fuse(txn, tokens, zs)
        return transactions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ocr_score(tokens: List[OCRToken]) -> float:
        """Mean OCR confidence across all tokens in the row."""
        if not tokens:
            return 0.5  # no tokens — neutral
        return float(sum(t.confidence for t in tokens) / len(tokens))

    @staticmethod
    def _amount_score(txn: Transaction) -> float:
        """
        Score how many expected amount fields are present.

        Full row (movement + balance) → 1.0
        Partial row (only balance or only movement) → 0.60
        No amounts → 0.0
        """
        has_movement = (txn.debit is not None) or (txn.credit is not None)
        has_balance = txn.balance is not None
        if has_movement and has_balance:
            return 1.0
        if has_balance or has_movement:
            return 0.60
        return 0.0

    @staticmethod
    def _column_assignment_score(
        tokens: List[OCRToken],
        zones: List[ColumnZone],
    ) -> float:
        """
        Score how tightly each token sits within its assigned zone centre.

        Returns 1.0 when all tokens are perfectly centred in their zones,
        decreasing toward 0.0 as tokens drift away from zone centres.
        Distance of 0.10 normalised units → score 0.0 for that token.
        """
        if not tokens or not zones:
            return 0.5  # unknown — neutral

        zone_centres = [z.x_center for z in zones]
        total_score = 0.0
        scored = 0

        for token in tokens:
            dists = [abs(token.normalized_x - c) for c in zone_centres]
            min_dist = min(dists)
            # Perfect alignment (dist=0) → 1.0; dist ≥ 0.10 → 0.0
            token_score = max(0.0, 1.0 - min_dist / 0.10)
            total_score += token_score
            scored += 1

        return total_score / scored if scored > 0 else 0.5
