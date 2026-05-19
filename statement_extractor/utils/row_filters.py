"""
Row filters — drop header, footer, and noise rows before transaction assembly.

Two-stage filtering
-------------------
1. ``is_noise_row()``     — single-row heuristic: boilerplate, summary lines,
                            account metadata, bank URLs, lone keywords.
2. ``is_footer_row()``    — structural footer detection: rows near the bottom
                            of a page that match summary/totals patterns.
3. ``filter_data_rows()`` — applies both filters and strips header flags.

Design notes
------------
Patterns are ordered from most specific (exact matches) to most general
(regex) to minimise false positives.  All matches are case-insensitive.
No transaction row should ever be silently dropped — when in doubt, keep.
"""
from __future__ import annotations

import re
from typing import List

from ..schemas import LogicalRow

# ---------------------------------------------------------------------------
# Core noise patterns — rows that are definitively NOT transactions
# ---------------------------------------------------------------------------
_NOISE_EXACT: frozenset = frozenset({
    "TRANSACTIONS", "DATE", "DESCRIPTION", "NARRATION", "PARTICULARS",
    "WITHDRAWAL", "DEPOSIT", "BALANCE", "DEBIT", "CREDIT",
    "DR", "CR", "REF NO", "REFERENCE", "CHEQUE NO", "CHQ NO",
    "AMOUNT", "DETAILS", "REMARKS",
})

_NOISE_ROW_RE = re.compile(
    r"""
    ^\s*(?:
        # Table header echoes
        ref\.?\s*no\.?                              |
        chq\.?\s*(?:no\.?|number)                   |
        instrument\s*no                             |

        # Summary / totals lines
        summary\s+of\s+transactions?                |
        total\s+(?:debit|credit|dr|cr|amount|transactions?) |
        net\s+(?:debit|credit|total)                |
        number\s+of\s+transactions?                 |
        no\.\s*of\s+transactions?                   |

        # Account metadata lines
        account\s+(?:number|no\.?|type|holder|name|status|branch) |
        customer\s+(?:id|name|type)                 |
        primary\s+account                           |
        nominee                                     |
        joint\s+holder                              |
        ifsc\s+code                                 |
        micr\s+code                                 |
        branch\s+code                               |
        sort\s+code                                 |

        # Statement metadata
        statement\s+(?:date|period|of\s+account)    |
        for\s+the\s+period                          |
        on\s+\w+.*\bto\b.*\d{1,2}\s                |
        generated\s+on                              |

        # Opening/closing balance markers (summary, not a transaction)
        opening\s+balance                           |
        closing\s+balance                           |
        available\s+balance                         |
        current\s+balance                           |

        # Page markers
        page\s+\d+\s+of\s+\d+                      |
        continued\s+(?:on\s+next\s+page|overleaf)   |

        # Footer boilerplate
        this\s+is\s+a\s+computer                    |
        authorised\s+signatory                      |
        important\s+(?:points?|information|note)    |
        terms?\s+(?:and\s+)?conditions?             |
        contact\s+us                                |
        toll\s+free                                 |
        customer\s+(?:care|service)                 |

        # URL / web artifacts
        https?://                                   |
        www\.                                       |
        \.com\b                                     |
        \.in\b                                      |

        # Known bank artifact patterns
        kotak-mahindra                              |
        \.avif\s*\(                                 |
        \.pdf\s*\(
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Footer structural pattern — totals / summary rows at page bottom
# ---------------------------------------------------------------------------
_FOOTER_SUMMARY_RE = re.compile(
    r"""
    \b(?:
        total\s*(?:debit|credit|dr|cr|amount|balance|transactions?)?  |
        sub[\s\-]?total                                               |
        grand\s+total                                                 |
        net\s+(?:debit|credit|balance|total)                          |
        closing\s+balance                                             |
        opening\s+balance                                             |
        brought\s+forward                                             |
        carried\s+forward                                             |
        balance\s+b/f                                                 |
        balance\s+c/f                                                 |
        balance\s+brought\s+forward                                   |
        balance\s+carried\s+forward
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bare account / reference number — 8+ consecutive digits, no date pattern
_BARE_LONG_NUMBER_RE = re.compile(r"^\d{8,}$")

# EMI / interest schedule rows (rate + future date) without Dr/Cr amounts
_EMI_RATE_ROW_RE = re.compile(
    r"\b\d{1,2}\s*%\s*(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|\d{4})",
    re.IGNORECASE,
)
# Compact APR / tenure band (percent touching digits) with no Dr/Cr
_EMI_DENSE_GRID_RE = re.compile(r"\b\d{1,2}\s*%\s*\d", re.IGNORECASE)


def is_noise_row(row: LogicalRow) -> bool:
    """
    Return True if the row is definitively NOT a ledger transaction.

    Kept conservative: only returns True when very high confidence
    the row is boilerplate, not when ambiguous.
    """
    text = row.full_text.strip()
    if not text:
        return True

    # Exact match against known header/label words
    if text.upper() in _NOISE_EXACT:
        return True

    # Regex boilerplate match
    if _NOISE_ROW_RE.search(text):
        return True

    # Combined header echo: "DESCRIPTION DATE" / "DATE DESCRIPTION" with no amounts
    upper = text.upper()
    has_header_echo = (
        "DESCRIPTION" in upper or "NARRATION" in upper or "PARTICULARS" in upper
    )
    has_date_kw = ("DATE" in upper or "VALUE DATE" in upper)
    if has_header_echo and has_date_kw and not any(t.is_numeric for t in row.tokens):
        return True

    # Bare long numeric string — likely an account/ref number, not a date
    clean = text.replace(" ", "")
    if _BARE_LONG_NUMBER_RE.match(clean):
        return True

    # EMI / interest schedule grid (e.g. "16% 20 MAY 26") without ledger Dr/Cr
    if _EMI_RATE_ROW_RE.search(text) and not re.search(
        r"\b(?:Dr|Cr)\b", text, re.IGNORECASE
    ):
        return True

    if (
        _EMI_DENSE_GRID_RE.search(text)
        and not re.search(r"\b(?:Dr|Cr)\b", text, re.IGNORECASE)
        and sum(1 for ch in text if ch.isdigit()) >= 12
    ):
        return True

    return False


def is_footer_row(row: LogicalRow) -> bool:
    """
    Return True if the row appears to be a totals/summary footer line.

    More aggressive than is_noise_row — targets summary arithmetic rows
    that have a numeric value but represent aggregates, not transactions.
    """
    text = row.full_text.strip()
    if not text:
        return True
    return bool(_FOOTER_SUMMARY_RE.search(text))


def filter_header_rows_only(rows: List[LogicalRow]) -> List[LogicalRow]:
    """
    Return all non-header rows without noise/footer heuristics.

    Used in document-fidelity mode so every ledger line in the source
    is emitted — nothing is dropped as boilerplate or duplicate.
    """
    return [r for r in rows if not r.is_header and not r.is_table_header]


def filter_data_rows(rows: List[LogicalRow]) -> List[LogicalRow]:
    """
    Return only rows that are candidates for transaction reconstruction.

    Removes:
    - rows flagged is_header or is_table_header
    - rows matching is_noise_row()
    - rows matching is_footer_row()
    """
    result: List[LogicalRow] = []
    for r in rows:
        if r.is_header or r.is_table_header:
            continue
        if is_noise_row(r):
            continue
        if is_footer_row(r):
            continue
        result.append(r)
    return result
