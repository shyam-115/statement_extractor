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
import re
from typing import Dict, List, Optional

from ..config import HeaderInferenceConfig
from ..schemas import ColumnZone, LogicalRow
from .numeric_parser import NumericParser

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

        # Check ONLY for table-level headers, page-level headers can hijack numeric columns
        header_rows = [r for r in rows if r.is_table_header]

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
        zones = self._positional_fallback(zones, rows)

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
        matches = []

        for zone in zones:
            candidate = zone_texts.get(zone.column_id, "")
            logger.info(f"[DEBUG] Zone {zone.column_id} text: '{candidate}'")
            if not candidate:
                continue
            for role, keywords in self._vocab.items():
                best_score = 0
                for kw in keywords:
                    kw_lower = kw.lower()
                    if len(kw_lower) <= 3:
                        # For short abbreviations ('cr', 'dr', 'dt'), require exact word match
                        import re
                        words = set(re.findall(r'\w+', candidate))
                        score = 100.0 if kw_lower in words else 0.0
                    else:
                        score = fuzz.partial_ratio(candidate, kw_lower)
                    
                    if score > best_score:
                        best_score = score
                if best_score >= self.config.fuzzy_threshold:
                    matches.append((best_score, role, zone.column_id))

        logger.info(f"[DEBUG] Matches: {matches}")
        
        # ── Detect combined Deposits/Withdrawals columns ──────────────────
        # When ONE zone scores high for BOTH debit AND credit (e.g. ICICI's
        # "Deposits Withdrawals" header), assign neither — mark it 'amount'
        # so the per-row CR/DR description hints resolve direction instead.
        zone_score_map: Dict[int, Dict[str, float]] = {}
        for sc, rl, zid in matches:
            zone_score_map.setdefault(zid, {})
            if sc > zone_score_map[zid].get(rl, 0):
                zone_score_map[zid][rl] = sc

        combined_zones: set = set()
        for zid, role_scores in zone_score_map.items():
            if (
                role_scores.get('debit', 0) >= self.config.fuzzy_threshold
                and role_scores.get('credit', 0) >= self.config.fuzzy_threshold
            ):
                # Guard against false positives caused by boundary-padding bleed:
                # adjacent DEBITS and CREDITS columns may both capture each other's
                # header text, scoring high for both roles.  Only mark as combined
                # when the zone text uses genuine deposit-type AND withdrawal-type
                # vocabulary (e.g. ICICI's "Deposits Withdrawals").
                # The words 'debit'/'credit' alone are NOT sufficient.
                zone_text = zone_texts.get(zid, '')
                _DEPOSIT_WORDS    = r'\b(?:deposit|deposits)\b'
                _WITHDRAWAL_WORDS = r'\b(?:withdrawal|withdrawals)\b'
                has_genuine_deposit    = bool(re.search(_DEPOSIT_WORDS,    zone_text, re.IGNORECASE))
                has_genuine_withdrawal = bool(re.search(_WITHDRAWAL_WORDS, zone_text, re.IGNORECASE))
                if has_genuine_deposit and has_genuine_withdrawal:
                    combined_zones.add(zid)
                    logger.info(
                        "Zone %d is a combined Deposits/Withdrawals column (‘%s’) — "
                        "marking as 'amount'; per-row CR/DR hints will resolve direction",
                        zid, zone_text[:40],
                    )

        # Remove debit/credit candidates for combined zones from the match pool
        matches = [
            (s, r, zid) for s, r, zid in matches
            if not (zid in combined_zones and r in ('debit', 'credit'))
        ]

        # Assign roles greedily by highest score
        matches.sort(
            key=lambda x: (x[0], -ROLE_PRIORITY.index(x[1]) if x[1] in ROLE_PRIORITY else -99),
            reverse=True
        )

        assigned_zones: set = set()
        assigned_roles: set = set()

        for score, role, zone_id in matches:
            if zone_id not in assigned_zones and role not in assigned_roles:
                for z in zones:
                    if z.column_id == zone_id:
                        z.semantic_role = role
                        assigned_zones.add(zone_id)
                        assigned_roles.add(role)
                        break

        # Stamp combined zones with sentinel 'amount' so positional fallback
        # does not overwrite them with debit/credit.
        for z in zones:
            if z.column_id in combined_zones and z.semantic_role is None:
                z.semantic_role = "amount"

        for z in zones:
            logger.info(f"[DEBUG] Assigned Zone {z.column_id}: {z.semantic_role}")

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

    def _positional_fallback(self, zones: List[ColumnZone], rows: List[LogicalRow]) -> List[ColumnZone]:
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
        # INSTEAD OF blindly picking leftmost, let's scan the first few data rows
        # to find which column actually contains dates!
        if "date" not in assigned_roles and unassigned:
            date_zone = None
            parser = NumericParser()
            
            # Count date-like tokens in each zone
            zone_date_counts = {z.column_id: 0 for z in unassigned}
            # Look at non-header rows
            data_rows = [r for r in rows if not r.is_header]
            for row in data_rows[:10]:
                for t in row.tokens:
                    if parser.extract_date(t.text):
                        # Find which zone this token belongs to
                        for z in unassigned:
                            if z.left_boundary <= t.normalized_x <= z.right_boundary:
                                zone_date_counts[z.column_id] += 1
                                break
            
            # Pick the zone with the most dates (must have at least 1)
            best_zone_id = max(zone_date_counts, key=zone_date_counts.get)
            if zone_date_counts[best_zone_id] > 0:
                date_zone = next(z for z in unassigned if z.column_id == best_zone_id)
            else:
                # Fallback to leftmost if no dates found
                leftmost = min(unassigned, key=lambda z: z.x_center)
                if leftmost.x_center < 0.25:  # must be in left quarter
                    date_zone = leftmost
                    
            if date_zone:
                date_zone.semantic_role = "date"
                unassigned.remove(date_zone)
                assigned_roles.add("date")

        # Rightmost unassigned → balance
        if "balance" not in assigned_roles and unassigned:
            rightmost = max(unassigned, key=lambda z: z.x_center)
            rightmost.semantic_role = "balance"
            unassigned.remove(rightmost)
            assigned_roles.add("balance")

        # Removed widest unassigned -> narration logic because ColumnDetector
        # only outputs numeric columns when falling back to DBSCAN. Assigning
        # narration to a numeric column breaks amount extraction.

        # Pair remaining for debit/credit (standard Indian layout: Debit left, Credit right)
        # INSTEAD OF blindly assigning left-to-right, let's scan for columns that actually contain amounts!
        if unassigned:
            zone_amount_counts = {z.column_id: 0 for z in unassigned}
            # Look at non-header rows
            data_rows = [r for r in rows if not r.is_header]
            
            parser = NumericParser()
            for row in data_rows[:15]:
                for t in row.tokens:
                    if parser.clean_amount_str(t.text) is not None:
                        # Find which zone this token belongs to
                        for z in unassigned:
                            if z.left_boundary <= t.normalized_x <= z.right_boundary:
                                zone_amount_counts[z.column_id] += 1
                                break
                                
            # Filter zones that actually have amounts
            amount_zones = [z for z in unassigned if zone_amount_counts[z.column_id] > 0]
            amount_zones.sort(key=lambda z: z.x_center)
            
            if len(amount_zones) == 1:
                if "debit" not in assigned_roles and "credit" not in assigned_roles:
                    amount_zones[0].semantic_role = "amount"
                    unassigned.remove(amount_zones[0])
            elif len(amount_zones) >= 2:
                roles_to_assign = []
                if "debit" not in assigned_roles:
                    roles_to_assign.append("debit")
                if "credit" not in assigned_roles:
                    roles_to_assign.append("credit")
                for zone, role in zip(amount_zones[-2:], roles_to_assign): # take the last two if more than 2
                    zone.semantic_role = role
                    unassigned.remove(zone)
                    assigned_roles.add(role)
                    
        return zones


