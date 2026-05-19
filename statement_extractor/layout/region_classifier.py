"""
Layout Region Classifier — scores table segments by structural signals.

Algorithm
---------
Each table segment (a List[LogicalRow]) is scored across five dimensions:

1. Date density     — fraction of rows containing a parseable date
2. Numeric density  — fraction of rows with ≥1 numeric token
3. Row count        — raw count (very short = noise/footer)
4. Keyword presence — header keywords present (date/debit/credit/balance)
5. Footer signals   — boilerplate patterns that indicate footer/summary regions

The weighted score is compared against ``transaction_score_threshold``.
Regions above the threshold are classified as TRANSACTION_TABLE.

Design principles
-----------------
- Pure Python — no ML model required.  All signals are deterministic.
- Fast — O(n·tokens) per page; does not duplicate OCR work.
- Conservative — when uncertain, falls back to the existing heuristic filter
  in extractor.py (date+amount+narration check) rather than discarding rows.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from ..config import LayoutConfig
from ..schemas import LogicalRow, RegionType
from ..parsing.numeric_parser import NumericParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Footer / noise boilerplate patterns
# ---------------------------------------------------------------------------
_FOOTER_PATTERNS = re.compile(
    r"""
    \b(?:
        page\s*\d+\s*of\s*\d+               |
        generated\s+on                       |
        statement\s+period                   |
        statement\s+date                     |
        total\s+(?:debit|credit|dr|cr)       |
        closing\s+balance                    |
        opening\s+balance                    |
        net\s+(?:debit|credit)               |
        this\s+is\s+a\s+computer             |
        authorised\s+signatory               |
        branch\s+code                        |
        ifsc\s+code                          |
        micr\s+code                          |
        account\s+number                     |
        account\s+type                       |
        account\s+holder                     |
        nominee                              |
        customer\s+id                        |
        contact\s+us                         |
        toll\s+free                          |
        visit\s+us                           |
        https?://                            |
        www\.                                |
        \.com\b                              |
        for\s+the\s+period                   |
        summary\s+of\s+transactions?         |
        number\s+of\s+transactions?          |
        terms\s+and\s+conditions             |
        important\s+instructions             |
        bank\s+never\s+asks                  |
        never\s+share\s+your                 |
        do\s+not\s+share                     |
        subject\s+to                         |
        insurance\s+policy
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Strong METADATA anti-patterns — presence of ANY of these in header rows
# immediately disqualifies the region as a TRANSACTION_TABLE.
# These are specific to FD/investment tables, account summary tables, etc.
# ---------------------------------------------------------------------------
_METADATA_HEADER_PATTERNS = re.compile(
    r"""
    \b(?:
        # Fixed deposit / investment table indicators
        deposit\s*no\.?                       |
        r\.?o\.?i\.?                          |  # Rate of Interest
        p\.?i\.?b\.?                          |  # Principal Interest Balance
        mat\.?\s*(?:amt|date|amount)          |  # Maturity Amount/Date
        fixed\s+deposit                       |
        fd\s+(?:number|no|amount)             |
        maturity\s+(?:date|amount|value)      |
        open\s+date                           |
        interest\s+rate                       |
        tenure                                |

        # Account summary table indicators
        account\s+type                        |
        a\/c\s+balance                        |
        (?:fixxed|fixed)\s+deposit.*linked    |
        total\s+balance\s*\(                  |  # TOTAL BALANCE (I+II)
        nomination                            |

        # Loan / credit card table indicators
        due\s+date                            |
        outstanding\s+amount                  |
        minimum\s+(?:due|payment)             |
        credit\s+limit                        |
        available\s+limit                     |

        # Investment / mutual fund indicators
        nav\s*(?:date|value)?                 |
        units\s+(?:allotted|held|redeemed)    |
        folio\s*(?:no|number)?               |
        scheme\s+name
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Patterns that strongly suggest a transaction row (date-like at row start)
_DATE_AT_START = re.compile(
    r"^\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|"
    r"^\s*\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    re.IGNORECASE,
)

# Header vocabulary words (shared with HeaderInference; duplicated for isolation)
_HEADER_VOCAB = frozenset([
    "date", "narration", "description", "particulars", "debit", "credit",
    "balance", "withdrawal", "deposit", "reference", "details", "remarks",
    "txn", "transaction", "amount", "dr", "cr",
])

# Table-intent keyword patterns (density scoring)
_EMI_KEYWORDS = re.compile(
    r"\b(?:emi|amortization|amortisation|installment|instalment|"
    r"principal|outstanding|tenure)\b",
    re.IGNORECASE,
)
_REWARD_KEYWORDS = re.compile(
    r"\b(?:reward\s*points?|cashback|cash\s*back|loyalty|miles)\b",
    re.IGNORECASE,
)
_CHARGES_KEYWORDS = re.compile(
    r"\b(?:charges?|fees?|penalty|service\s*charge|late\s*fee)\b",
    re.IGNORECASE,
)
_INTEREST_KEYWORDS = re.compile(
    r"\b(?:interest\s+charged|interest\s+rate|int\.?\s*calculation|"
    r"apr|annual\s+percentage)\b",
    re.IGNORECASE,
)
_CC_KEYWORDS = re.compile(
    r"\b(?:card\s*no|credit\s*card|minimum\s+due|payment\s+due|"
    r"available\s+credit|credit\s+limit)\b",
    re.IGNORECASE,
)
_EMI_COLUMN_PATTERN = re.compile(
    r"\b(?:principal|interest|outstanding|emi\s*amount|instalment)\b",
    re.IGNORECASE,
)
_EMI_DETAILS_HEADER = re.compile(
    r"\b(?:active\s+)?emi\s+details\b",
    re.IGNORECASE,
)


class LayoutRegionClassifier:
    """
    Classifies a table segment into a RegionType using multi-signal scoring.

    Parameters
    ----------
    config : LayoutConfig
    """

    def __init__(self, config: LayoutConfig) -> None:
        self.config = config
        self._parser = NumericParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        rows: List[LogicalRow],
        col_roles_found: set,
    ) -> Tuple[RegionType, float]:
        """
        Classify *rows* (a single table segment) into a RegionType.

        Parameters
        ----------
        rows           : LogicalRow list for one table segment
        col_roles_found: set of semantic roles found by HeaderInference
                         (e.g. {"date", "debit", "balance"})

        Returns
        -------
        (RegionType, score)  where score is 0–1
        """
        if not rows:
            return RegionType.NOISE, 0.0

        cfg = self.config

        data_rows = [r for r in rows if not r.is_header and not r.is_table_header]
        header_rows = [r for r in rows if r.is_header or r.is_table_header]
        n = len(data_rows)

        if n < cfg.min_transaction_rows:
            logger.debug(
                "Region has only %d data rows — classifying as NOISE", n
            )
            return RegionType.NOISE, 0.0

        # ── Anti-pattern gate: strong metadata signals in headers ───────
        # If ANY metadata header pattern is found, immediately return METADATA.
        # This prevents FD tables, account summaries, and loan tables from
        # being mis-classified as TRANSACTION_TABLEs regardless of scores.
        header_text = " ".join(r.full_text for r in header_rows)
        if header_text and _METADATA_HEADER_PATTERNS.search(header_text):
            logger.debug(
                "Region has metadata header pattern — classifying as METADATA_TABLE"
            )
            return RegionType.METADATA_TABLE, 0.0

        # ── Multi-date-column signal: FD tables typically have 2+ date columns ──
        # Count distinct date column positions (normalised x). If 2+ zones with
        # the date role exist (e.g. OPEN DATE and MAT.DATE), downgrade to METADATA.
        date_zone_count = sum(
            1 for r in header_rows
            for t in r.tokens
            if self._parser.is_date(t.text)
        )
        # Heuristic: real transaction tables have 1 date column.
        # FD/investment tables typically show 2 (open date + maturity date).
        if date_zone_count >= 2:
            logger.debug(
                "Region has %d date tokens in headers — likely FD/investment table",
                date_zone_count,
            )
            return RegionType.METADATA_TABLE, 0.10

        # ── Signal 1: Footer pattern density ───────────────────────────
        footer_hits = sum(
            1 for r in data_rows if _FOOTER_PATTERNS.search(r.full_text)
        )
        footer_fraction = footer_hits / n
        if footer_fraction >= cfg.footer_row_fraction_threshold:
            logger.debug(
                "Region footer fraction %.2f ≥ threshold %.2f → FOOTER",
                footer_fraction, cfg.footer_row_fraction_threshold,
            )
            return RegionType.FOOTER, footer_fraction

        # ── Signal 2: Date density ──────────────────────────────────────
        date_rows = sum(
            1 for r in data_rows
            if any(t.is_date for t in r.tokens)
            or _DATE_AT_START.search(r.full_text)
        )
        date_fraction = date_rows / n

        # ── Signal 3: Numeric density ───────────────────────────────────
        numeric_rows = sum(
            1 for r in data_rows
            if any(t.is_numeric for t in r.tokens)
        )
        numeric_fraction = numeric_rows / n

        # ── Signal 4: Column role coverage ─────────────────────────────
        # How well do the detected column roles cover a transaction schema?
        role_score = self._role_coverage_score(col_roles_found)

        # ── Signal 5: Header keyword presence ──────────────────────────
        header_rows = [r for r in rows if r.is_header or r.is_table_header]
        header_keyword_score = self._header_keyword_score(header_rows)

        # ── Weighted composite score ────────────────────────────────────
        # Weights are tuned so that role coverage + date density dominate.
        score = (
            0.30 * role_score
            + 0.25 * date_fraction
            + 0.20 * numeric_fraction
            + 0.15 * header_keyword_score
            + 0.10 * min(1.0, n / 5.0)  # reward larger tables
        )
        # Penalty: if footer fraction is non-trivial, reduce score
        score *= (1.0 - footer_fraction * 0.5)

        logger.debug(
            "Region classifier: n=%d date=%.2f num=%.2f role=%.2f "
            "header_kw=%.2f footer=%.2f → score=%.3f",
            n, date_fraction, numeric_fraction, role_score,
            header_keyword_score, footer_fraction, score,
        )

        # ── Table-intent classification (fine-grained) ─────────────────
        intent_type, intent_score = self._classify_table_intent(
            rows, header_rows, data_rows, col_roles_found, n,
            date_fraction, numeric_fraction, score,
        )
        if intent_type is not None:
            return intent_type, max(score, intent_score)

        if score >= cfg.transaction_score_threshold:
            has_balance = "balance" in col_roles_found
            if _CC_KEYWORDS.search(header_text) or (
                not has_balance and "credit" in col_roles_found
            ):
                return RegionType.CREDIT_CARD_TRANSACTIONS, score
            return RegionType.BANK_TRANSACTIONS, score

        # Below threshold — decide between metadata and noise
        if numeric_fraction >= cfg.min_numeric_row_fraction:
            return RegionType.METADATA_TABLE, score

        return RegionType.NOISE, score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _role_coverage_score(roles: set) -> float:
        """
        Score how well the detected roles match a transaction schema.

        Minimum viable: date + (debit or credit or balance) = 0.5
        Full schema:    date + narration + debit + credit + balance = 1.0
        """
        if not roles:
            return 0.0

        has_date     = "date" in roles
        has_narration = "narration" in roles
        has_debit    = "debit" in roles
        has_credit   = "credit" in roles
        has_balance  = "balance" in roles

        score = 0.0
        if has_date:
            score += 0.30
        if has_narration:
            score += 0.20
        if has_debit or has_credit:
            score += 0.30
        if has_balance:
            score += 0.20
        return score

    def _classify_table_intent(
        self,
        rows: List[LogicalRow],
        header_rows: List[LogicalRow],
        data_rows: List[LogicalRow],
        col_roles_found: set,
        n: int,
        date_fraction: float,
        numeric_fraction: float,
        base_score: float,
    ) -> Tuple[Optional[RegionType], float]:
        """
        Detect specialised table types (EMI, rewards, charges, etc.).
        Returns (None, 0) if no strong intent signal.
        """
        all_text = " ".join(r.full_text for r in rows)
        header_text = " ".join(r.full_text for r in header_rows)
        emi_hits = len(_EMI_KEYWORDS.findall(all_text))

        # EMI schedule grid (distinct from purchase transaction grids)
        if _EMI_DETAILS_HEADER.search(header_text):
            return RegionType.EMI_SCHEDULE, min(1.0, 0.55 + emi_hits * 0.08)

        # EMI schedule: keyword density + principal/interest columns
        emi_column_header = (
            _EMI_COLUMN_PATTERN.search(header_text)
            and emi_hits >= 1
        )
        if emi_hits >= 3 or emi_column_header:
            # Merchant lines on credit-card statements often contain words such as
            # "principal", "interest", or "amortization" (EMI breakdowns).  Those
            # are still transaction grids if date + amount columns are present
            # with healthy row-level date/numeric density.
            strong_txn_grid = (
                date_fraction >= self.config.min_date_row_fraction
                and numeric_fraction >= self.config.min_numeric_row_fraction
                and "date" in col_roles_found
                and ("debit" in col_roles_found or "credit" in col_roles_found)
            )
            if strong_txn_grid:
                return None, 0.0

            return RegionType.EMI_SCHEDULE, min(1.0, 0.5 + emi_hits * 0.1)

        # Reward summary
        reward_hits = len(_REWARD_KEYWORDS.findall(all_text))
        if reward_hits >= 2 and date_fraction < 0.2:
            return RegionType.REWARD_SUMMARY, min(1.0, 0.4 + reward_hits * 0.15)

        # Interest calculation
        if _INTEREST_KEYWORDS.search(header_text) and date_fraction < 0.3:
            return RegionType.INTEREST_CALCULATION, 0.75

        # Charges table
        charge_hits = len(_CHARGES_KEYWORDS.findall(header_text))
        if charge_hits >= 2 and n < 15 and date_fraction < 0.3:
            return RegionType.CHARGES_TABLE, min(1.0, 0.5 + charge_hits * 0.1)

        return None, 0.0

    @staticmethod
    def _header_keyword_score(header_rows: List[LogicalRow]) -> float:
        """
        Return fraction of transaction-relevant keywords found in header rows.
        """
        if not header_rows:
            return 0.0

        all_words: set = set()
        for row in header_rows:
            for token in row.tokens:
                for word in re.findall(r"\w+", token.text.lower()):
                    all_words.add(word)

        hits = len(all_words & _HEADER_VOCAB)
        # A full transaction header typically has ~5 relevant words
        return min(1.0, hits / 5.0)
