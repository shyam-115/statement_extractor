"""
Header Inference — semantic role assignment for detected column zones.

Algorithm
---------
For each ColumnZone (already sorted left→right):
  1. Collect all tokens from header rows that overlap the zone's x-range.
  2. Build a candidate text string from those tokens.
  3. Fuzzy-match against every semantic vocabulary list using rapidfuzz.
  4. The role with the highest score above the threshold wins.

Fallback (when no header rows are detected):
  - Positional heuristics are applied:
    * Leftmost zone → date
    * Rightmost zone → balance
    * Second-rightmost zone → debit OR credit (resolved by balance validator)
    * Wide zone spanning > 35% of page width → narration

Design notes
------------
- rapidfuzz.fuzz.partial_ratio is used instead of ratio() to handle
  cases where OCR inserts extra characters or splits tokens.
- A zone can receive at most one role.  If two zones match the same role
  the one with the higher score wins.
- The narration zone is inferred by exclusion after all numeric roles
  are assigned — it is typically the widest non-numeric zone.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..config import HeaderInferenceConfig
from ..schemas import ColumnZone, LogicalRow

logger = logging.getLogger(__name__)

# Semantic role order (for tie-breaking)
ROLE_PRIORITY = ["date", "reference", "debit", "credit", "balance", "narration"]


class HeaderInference:
    """
    Assigns semantic roles to ColumnZones using fuzzy header matching.

    Parameters
    ----------
    config : HeaderInferenceConfig
    """

    def __init__(self, config: HeaderInferenceConfig) -> None:
        self.config = config
        self._vocab: Dict[str, List[str]] = {
            "date":      config.date_keywords,
            "narration": config.narration_keywords,
            "debit":     config.debit_keywords,
            "credit":    config.credit_keywords,
            "balance":   config.balance_keywords,
            "reference": config.reference_keywords,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(
        self,
        zones: List[ColumnZone],
        rows: List[LogicalRow],
    ) -> List[ColumnZone]:
        """
        Assign semantic_role to each zone and return the updated list.
        """
        if not zones:
            return zones

        # Check for both page-level headers and table-level headers
        header_rows = [r for r in rows if r.is_header or r.is_table_header]

        used_fallback = False
        if header_rows:
            zones = self._match_from_headers(zones, header_rows)
            # If we matched at least debit or credit from explicit headers, we trust the order
            has_explicit_roles = any(z.semantic_role in ("debit", "credit") for z in zones)
            if not has_explicit_roles:
                used_fallback = True
        else:
            logger.info("No header rows found — applying positional heuristics")
            used_fallback = True

        # Fill any unassigned zones with positional fallback
        zones = self._positional_fallback(zones)

        # Withdrawal/debit is typically left of deposit/credit on statements.
        # ONLY apply this risky swap if we used positional fallback.
        # ICICI and some others put Deposits (Credit) on the left!
        if used_fallback:
            zones = self._ensure_debit_left_of_credit(zones)

        # Log final role assignment
        for z in zones:
            logger.debug("Column %d (x=%.3f) → %s", z.column_id, z.x_center, z.semantic_role)

        return zones

    # ------------------------------------------------------------------
    # Header-based matching
    # ------------------------------------------------------------------

    def _match_from_headers(
        self,
        zones: List[ColumnZone],
        header_rows: List[LogicalRow],
    ) -> List[ColumnZone]:
        """
        For each zone, collect overlapping header tokens and fuzzy-match.
        """
        try:
            from rapidfuzz import fuzz  # type: ignore
        except ImportError:
            logger.warning("rapidfuzz not available — falling back to simple matching")
            return self._simple_match_from_headers(zones, header_rows)

        # Build per-zone candidate text from header tokens
        zone_texts: Dict[int, str] = {}
        for zone in zones:
            tokens_in_zone = []
            for hrow in header_rows:
                for token in hrow.tokens:
                    if zone.left_boundary <= token.normalized_x <= zone.right_boundary:
                        tokens_in_zone.append(token.text)
            original_candidate = " ".join(tokens_in_zone).strip()
            if original_candidate:
                zone.header_text = original_candidate
            zone_texts[zone.column_id] = original_candidate.lower()

        # Score each zone against each role vocabulary
        # Structure: {role: (best_zone_id, best_score)}
        role_winner: Dict[str, tuple] = {}

        for zone in zones:
            candidate = zone_texts.get(zone.column_id, "")
            if not candidate:
                continue
            for role, keywords in self._vocab.items():
                for kw in keywords:
                    score = fuzz.partial_ratio(candidate, kw.lower())
                    if score >= self.config.fuzzy_threshold:
                        current_best = role_winner.get(role)
                        if current_best is None or score > current_best[1]:
                            role_winner[role] = (zone.column_id, score)

        # Assign roles — each zone gets at most one role
        assigned_zones: set = set()
        for role in ROLE_PRIORITY:
            winner = role_winner.get(role)
            if winner and winner[0] not in assigned_zones:
                zone_id, _ = winner
                for z in zones:
                    if z.column_id == zone_id:
                        z.semantic_role = role
                        assigned_zones.add(zone_id)
                        break

        return zones

    def _simple_match_from_headers(
        self,
        zones: List[ColumnZone],
        header_rows: List[LogicalRow],
    ) -> List[ColumnZone]:
        """Fallback: exact substring matching (no rapidfuzz)."""
        zone_texts: Dict[int, str] = {}
        for zone in zones:
            tokens_in_zone = []
            for hrow in header_rows:
                for token in hrow.tokens:
                    if zone.left_boundary <= token.normalized_x <= zone.right_boundary:
                        tokens_in_zone.append(token.text)
            original_candidate = " ".join(tokens_in_zone).strip()
            if original_candidate:
                zone.header_text = original_candidate
            zone_texts[zone.column_id] = original_candidate.lower()

        assigned: set = set()
        for role in ROLE_PRIORITY:
            for zone in zones:
                if zone.column_id in assigned:
                    continue
                candidate = zone_texts.get(zone.column_id, "")
                for kw in self._vocab[role]:
                    if kw.lower() in candidate:
                        zone.semantic_role = role
                        assigned.add(zone.column_id)
                        break

        return zones

    # ------------------------------------------------------------------
    # Positional fallback
    # ------------------------------------------------------------------

    def _positional_fallback(self, zones: List[ColumnZone]) -> List[ColumnZone]:
        """
        Apply positional heuristics for any zone still missing a role.

        Heuristics (left → right):
        - First zone without a role → date (if leftmost)
        - Rightmost unassigned zone with high support → balance
        - Wide unassigned zone → narration
        - Remaining → debit / credit (pair assignment)
        """
        unassigned = [z for z in zones if z.semantic_role is None]
        if not unassigned:
            return zones

        assigned_roles = {z.semantic_role for z in zones if z.semantic_role}

        # Leftmost unassigned → date
        if "date" not in assigned_roles and unassigned:
            # Check if leftmost zone overall is unassigned
            leftmost = min(unassigned, key=lambda z: z.x_center)
            if leftmost.x_center < 0.25:  # must be in left quarter
                leftmost.semantic_role = "date"
                unassigned.remove(leftmost)
                assigned_roles.add("date")

        # Rightmost unassigned → balance
        if "balance" not in assigned_roles and unassigned:
            rightmost = max(unassigned, key=lambda z: z.x_center)
            rightmost.semantic_role = "balance"
            unassigned.remove(rightmost)
            assigned_roles.add("balance")

        # Widest remaining unassigned → narration
        if "narration" not in assigned_roles and unassigned:
            widest = max(
                unassigned,
                key=lambda z: z.right_boundary - z.left_boundary,
            )
            if (widest.right_boundary - widest.left_boundary) > 0.15:
                widest.semantic_role = "narration"
                unassigned.remove(widest)
                assigned_roles.add("narration")

        # Pair remaining for debit/credit (second-rightmost → debit, third → credit)
        if unassigned:
            remaining = sorted(unassigned, key=lambda z: z.x_center, reverse=True)
            roles_to_assign = []
            if "debit" not in assigned_roles:
                roles_to_assign.append("debit")
            if "credit" not in assigned_roles:
                roles_to_assign.append("credit")
            for zone, role in zip(remaining, roles_to_assign):
                zone.semantic_role = role

        return zones

    @staticmethod
    def _ensure_debit_left_of_credit(zones: List[ColumnZone]) -> List[ColumnZone]:
        """Swap debit/credit roles when columns are reversed (common OCR/header error)."""
        debit_z = next((z for z in zones if z.semantic_role == "debit"), None)
        credit_z = next((z for z in zones if z.semantic_role == "credit"), None)
        if (
            debit_z is not None
            and credit_z is not None
            and debit_z.x_center > credit_z.x_center
        ):
            debit_z.semantic_role = "credit"
            credit_z.semantic_role = "debit"
            logger.info(
                "Swapped debit/credit column roles (debit was right of credit)"
            )
        return zones
