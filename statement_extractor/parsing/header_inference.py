"""
Header Inference — probability-scored semantic role assignment for column zones.

Algorithm
---------
For each ColumnZone (already sorted left→right):
  1. Collect all tokens from header rows that overlap the zone's x-range.
  2. Build a candidate text string from those tokens.
  3. Score the candidate against EVERY semantic vocabulary list using rapidfuzz.
     Unlike the previous binary winner-takes-all approach, we compute a full
     probability distribution across roles.
  4. Resolve conflicts using:
     a. Score gap: winner must outscore second-best by at least MARGIN.
     b. Positional prior: date is leftmost, balance is rightmost.
     c. Superrole resolution: if both debit and credit score similarly,
        use column position relative to balance to disambiguate.
  5. Zones with no convincing match fall through to positional fallback.

Positional fallback (no header rows detected)
---------------------------------------------
  * Leftmost zone (x < 0.25) → date
  * Rightmost zone → balance
  * Widest unassigned zone (width > 15%) → narration
  * Remaining pair (right-to-left) → debit, credit

Design notes
------------
- rapidfuzz.fuzz.partial_ratio is used to handle OCR-inserted extra chars.
- A zone receives at most one role.
- The narration zone is inferred by exclusion after all numeric roles assigned.
- Bank profile hints (from BankFingerprinter) can override debit_left_of_credit.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from ..config import HeaderInferenceConfig
from ..schemas import BankProfile, ColumnZone, LogicalRow

logger = logging.getLogger(__name__)

# Role priority for tie-breaking when scores are equal
ROLE_PRIORITY = ["date", "reference", "debit", "credit", "balance", "narration"]

# Minimum score margin between first and second-best role to accept a match.
# If the gap is smaller the assignment is considered ambiguous.
_MIN_SCORE_MARGIN = 8   # rapidfuzz score points (0–100)


class HeaderInference:
    """
    Assigns semantic roles to ColumnZones using probability-scored fuzzy matching.

    Parameters
    ----------
    config      : HeaderInferenceConfig
    bank_profile: Optional BankProfile from BankFingerprinter (advisory hints)
    """

    def __init__(
        self,
        config: HeaderInferenceConfig,
        bank_profile: Optional[BankProfile] = None,
    ) -> None:
        self.config = config
        self.bank_profile = bank_profile or BankProfile()
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

    def update_bank_profile(self, bank_profile: BankProfile) -> None:
        """Update advisory bank hints without reconstructing the object."""
        self.bank_profile = bank_profile

    def infer(
        self,
        zones: List[ColumnZone],
        rows: List[LogicalRow],
    ) -> List[ColumnZone]:
        """
        Assign semantic_role to each zone and return the updated list.

        Parameters
        ----------
        zones : ColumnZone list to annotate
        rows  : All rows in the current table (headers used for matching)
        """
        if not zones:
            return zones

        header_rows = [r for r in rows if r.is_header or r.is_table_header]

        used_fallback = False
        if header_rows:
            zones = self._match_from_headers(zones, header_rows)
            has_explicit_roles = any(
                z.semantic_role in ("debit", "credit") for z in zones
            )
            if not has_explicit_roles:
                used_fallback = True
        else:
            logger.info("No header rows found — applying positional heuristics")
            used_fallback = True

        # Fill any still-unassigned zones with positional fallback
        zones = self._positional_fallback(zones)

        # Apply debit/credit order correction only when bank profile doesn't
        # give us an explicit advisory — and only when we used positional fallback.
        if used_fallback and self.bank_profile.debit_left_of_credit is None:
            zones = self._ensure_debit_left_of_credit(zones)
        elif (
            self.bank_profile.debit_left_of_credit is not None
            and used_fallback
        ):
            zones = self._apply_bank_column_order(zones)

        for z in zones:
            logger.debug(
                "Column %d (x=%.3f) → %s [header: %r]",
                z.column_id, z.x_center, z.semantic_role, z.header_text,
            )

        return zones

    # ------------------------------------------------------------------
    # Header-based probability-scored matching
    # ------------------------------------------------------------------

    def _match_from_headers(
        self,
        zones: List[ColumnZone],
        header_rows: List[LogicalRow],
    ) -> List[ColumnZone]:
        """
        For each zone, build a candidate text from overlapping header tokens
        and compute a score distribution across all semantic roles.
        """
        try:
            from rapidfuzz import fuzz  # type: ignore
        except ImportError:
            logger.warning("rapidfuzz not available — falling back to simple matching")
            return self._simple_match_from_headers(zones, header_rows)

        # ── Step 1: Build per-zone candidate text ──────────────────────
        # Map each header token to the NEAREST data pillar (zone).
        # This solves the alignment mismatch where headers are left-aligned
        # but the numeric data in the column is right-aligned.
        zone_texts: Dict[int, str] = {z.column_id: "" for z in zones}
        
        for hrow in header_rows:
            for token in hrow.tokens:
                if not zones:
                    continue
                # Find the nearest zone based on center distance
                best_zone = min(zones, key=lambda z: abs(token.normalized_x - z.x_center))
                
                # Assign to the nearest zone if it's reasonably close (e.g., within 15% of page width)
                if abs(token.normalized_x - best_zone.x_center) < 0.15:
                    current = zone_texts[best_zone.column_id]
                    zone_texts[best_zone.column_id] = (current + " " + token.text).strip()
                    
        for zone in zones:
            candidate = zone_texts.get(zone.column_id, "")
            if candidate:
                zone.header_text = candidate

        # ── Step 2: Score matrix {zone_id: {role: best_score}} ─────────
        score_matrix: Dict[int, Dict[str, float]] = {
            z.column_id: {role: 0.0 for role in self._vocab}
            for z in zones
        }

        for zone in zones:
            candidate = zone_texts.get(zone.column_id, "")
            if not candidate:
                continue
            candidate_lower = candidate.lower()
            for role, keywords in self._vocab.items():
                best_role_score = 0.0
                for kw in keywords:
                    s = float(fuzz.partial_ratio(candidate_lower, kw.lower()))
                    if s > best_role_score:
                        best_role_score = s
                score_matrix[zone.column_id][role] = best_role_score

        # ── Step 3: Resolve assignments with conflict detection ─────────
        zones = self._resolve_score_matrix(zones, score_matrix)

        return zones

    def _resolve_score_matrix(
        self,
        zones: List[ColumnZone],
        score_matrix: Dict[int, Dict[str, float]],
    ) -> List[ColumnZone]:
        """
        Convert a score matrix into unique role assignments.

        Algorithm:
        1. For each role (in priority order), find the zone with the
           highest score above threshold AND the required score margin.
        2. Each zone gets at most one role; each role is assigned at most once.
        3. Ambiguous assignments (gap < _MIN_SCORE_MARGIN) are left unassigned
           and resolved by positional fallback.
        """
        threshold = self.config.fuzzy_threshold
        assigned_zones: set = set()
        assigned_roles: set = set()

        for role in ROLE_PRIORITY:
            best_zone_id: Optional[int] = None
            best_score = 0.0

            for zone in zones:
                if zone.column_id in assigned_zones:
                    continue
                zone_scores = score_matrix.get(zone.column_id, {})
                role_score = zone_scores.get(role, 0.0)
                if role_score < threshold:
                    continue

                # Check margin: winner must be clearly ahead of all other roles
                other_scores = [
                    s for r, s in zone_scores.items()
                    if r != role and r not in assigned_roles
                ]
                second_best = max(other_scores) if other_scores else 0.0
                gap = role_score - second_best

                if role_score > best_score and gap >= _MIN_SCORE_MARGIN:
                    best_score = role_score
                    best_zone_id = zone.column_id

            if best_zone_id is not None:
                for z in zones:
                    if z.column_id == best_zone_id:
                        z.semantic_role = role
                        assigned_zones.add(best_zone_id)
                        assigned_roles.add(role)
                        logger.debug(
                            "Header match: zone %d → %s (score=%.1f)",
                            best_zone_id, role, best_score,
                        )
                        break

        return zones

    def _simple_match_from_headers(
        self,
        zones: List[ColumnZone],
        header_rows: List[LogicalRow],
    ) -> List[ColumnZone]:
        """Fallback exact substring matching when rapidfuzz is not installed."""
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
        Apply positional heuristics for zones still missing a role.

        Heuristics (left → right):
        - First zone (x < 0.25) without a role → date
        - Rightmost unassigned → balance
        - Widest unassigned zone (width > 15%) → narration
        - Remaining pair (right to left) → debit, credit
        """
        unassigned = [z for z in zones if z.semantic_role is None]
        if not unassigned:
            return zones

        assigned_roles = {z.semantic_role for z in zones if z.semantic_role}

        # Leftmost unassigned → date
        if "date" not in assigned_roles and unassigned:
            leftmost = min(unassigned, key=lambda z: z.x_center)
            if leftmost.x_center < 0.25:
                leftmost.semantic_role = "date"
                unassigned.remove(leftmost)
                assigned_roles.add("date")

        # Rightmost unassigned → balance
        if "balance" not in assigned_roles and unassigned:
            rightmost = max(unassigned, key=lambda z: z.x_center)
            rightmost.semantic_role = "balance"
            unassigned.remove(rightmost)
            assigned_roles.add("balance")

        # Widest remaining → narration
        if "narration" not in assigned_roles and unassigned:
            widest = max(
                unassigned,
                key=lambda z: z.right_boundary - z.left_boundary,
            )
            if (widest.right_boundary - widest.left_boundary) > 0.15:
                widest.semantic_role = "narration"
                unassigned.remove(widest)
                assigned_roles.add("narration")

        # Remaining pair → debit / credit (right to left)
        if unassigned:
            remaining = sorted(unassigned, key=lambda z: z.x_center, reverse=True)
            roles_to_assign: List[str] = []
            if "debit" not in assigned_roles:
                roles_to_assign.append("debit")
            if "credit" not in assigned_roles:
                roles_to_assign.append("credit")
            for zone, role in zip(remaining, roles_to_assign):
                zone.semantic_role = role

        return zones

    # ------------------------------------------------------------------
    # Column order correction
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_debit_left_of_credit(zones: List[ColumnZone]) -> List[ColumnZone]:
        """
        Swap debit/credit roles when columns are reversed.
        Applied only in positional fallback mode — not when header matching succeeded.
        """
        debit_z  = next((z for z in zones if z.semantic_role == "debit"),  None)
        credit_z = next((z for z in zones if z.semantic_role == "credit"), None)
        if (
            debit_z is not None
            and credit_z is not None
            and debit_z.x_center > credit_z.x_center
        ):
            debit_z.semantic_role  = "credit"
            credit_z.semantic_role = "debit"
            logger.info("Swapped debit/credit column roles (debit was right of credit)")
        return zones

    def _apply_bank_column_order(self, zones: List[ColumnZone]) -> List[ColumnZone]:
        """
        Apply bank-profile advisory for debit/credit column order.
        Only called when bank_profile.debit_left_of_credit is explicitly set.
        """
        debit_z  = next((z for z in zones if z.semantic_role == "debit"),  None)
        credit_z = next((z for z in zones if z.semantic_role == "credit"), None)
        if debit_z is None or credit_z is None:
            return zones

        want_debit_left = self.bank_profile.debit_left_of_credit
        debit_is_left = debit_z.x_center < credit_z.x_center

        if want_debit_left != debit_is_left:
            debit_z.semantic_role  = "credit"
            credit_z.semantic_role = "debit"
            logger.info(
                "Applied bank profile '%s' column order: swapped debit/credit",
                self.bank_profile.bank_id,
            )
        return zones
