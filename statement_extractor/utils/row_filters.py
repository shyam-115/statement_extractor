"""Heuristics to drop header/footer/noise rows before transaction assembly."""
from __future__ import annotations

import re
from typing import List

from ..schemas import LogicalRow

_SKIP_ROW_RE = re.compile(
    r"""
    ^\s*(?:
        ref\.?\s*no\.?              |
        summary\s+of                |
        transactions?\s*$           |
        important\s+points?         |
        ^date\s*$                   |
        ^description\s*$            |
        ^withdrawal\s*$             |
        ^deposit\s*$                |
        ^balance\s*$                |
        on\s+\w+.*to\s+\d{1,2}\s    |
        primary\s+account           |
        https?://                   |
        kotak-mahindra              |
        \.avif\s*\(
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_noise_row(row: LogicalRow) -> bool:
    """Return True if the row is unlikely to be a ledger transaction."""
    text = row.full_text.strip()
    if not text:
        return True
    if _SKIP_ROW_RE.search(text):
        return True
    upper = text.upper()
    if upper in {"TRANSACTIONS", "DATE", "DESCRIPTION", "WITHDRAWAL", "DEPOSIT", "BALANCE"}:
        return True
    if "DESCRIPTION" in upper and "DATE" in upper and not any(
        t.is_numeric for t in row.tokens
    ):
        return True
    # Ref / account numbers mis-parsed as dates
    if re.fullmatch(r"\d{8,}", text.replace(" ", "")):
        return True
    return False


def filter_data_rows(rows: List[LogicalRow]) -> List[LogicalRow]:
    """Remove header and noise rows while keeping transaction candidates."""
    return [r for r in rows if not r.is_header and not is_noise_row(r)]
