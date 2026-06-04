"""
Exporters — JSON, CSV, and Markdown output writers.

All functions accept an ExtractionResult and a file path.
They sanitise None values and ensure the output directory exists.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas import ExtractionResult, Transaction

logger = logging.getLogger(__name__)

_CSV_FIELDS = [
    "txn_date", "description", "reference_no",
    "debit", "credit", "balance",
    "confidence_score", "validation_status", "page_num",
]


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_to_json(result: ExtractionResult, output_path: str) -> None:
    """Serialise *result* to a pretty-printed JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "source_file":           result.source_file,
        "total_pages":           result.total_pages,
        "total_transactions":    len(result.transactions),
        "total_extracted_tables": len(result.extracted_tables),
        "column_mapping":        result.column_mapping,
        "extraction_warnings":   result.extraction_warnings,
        "extracted_tables":      [t.dict() for t in result.extracted_tables],
        "transactions":          [_txn_to_dict(t) for t in result.transactions],
    }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("JSON saved → %s (%d transactions)", path, len(result.transactions))


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_to_csv(result: ExtractionResult, output_path: str) -> None:
    """Write *result* transactions to a UTF-8 CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for txn in result.transactions:
            writer.writerow(_txn_to_dict(txn))

    logger.info("CSV saved → %s (%d transactions)", path, len(result.transactions))


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def save_to_markdown(result: ExtractionResult, output_path: str) -> None:
    """
    Write *result* transactions to a GitHub-Flavored Markdown file.

    The output contains:
    - A metadata header (source file, page count, transaction count)
    - A formatted transaction table with Indian number formatting
    - A summary section (total debits, credits, net change, closing balance)

    Indian number format: last 3 digits grouped, then groups of 2 (e.g. 1,20,000.00).
    Empty debit/credit cells are shown as '—'.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    source = Path(result.source_file).name

    # ── Header ────────────────────────────────────────────────────────
    lines.append("# Statement Extraction Report")
    lines.append("")
    lines.append(f"**Source file:** `{source}`  ")
    lines.append(f"**Total pages:** {result.total_pages}  ")
    lines.append(f"**Transactions extracted:** {len(result.transactions)}  ")
    lines.append("")

    if result.extraction_warnings:
        lines.append("> [!WARNING]")
        for w in result.extraction_warnings:
            lines.append(f"> {w}")
        lines.append("")

    # ── Transaction table ──────────────────────────────────────────────
    lines.append("## Transactions")
    lines.append("")
    lines.append(
        "| Date | Description | Debit (₹) | Credit (₹) | Balance (₹) | Status |"
    )
    lines.append(
        "| :--- | :---------- | --------: | ---------: | ----------: | :----- |"
    )

    total_debit: float = 0.0
    total_credit: float = 0.0

    for txn in result.transactions:
        date_str    = txn.txn_date or "—"
        desc        = (txn.description or "").strip()
        debit_str   = _fmt_inr(txn.debit)
        credit_str  = _fmt_inr(txn.credit)
        balance_str = _fmt_inr(txn.balance)
        status      = txn.validation_status.value if txn.validation_status else "—"

        if txn.debit is not None:
            total_debit += txn.debit
        if txn.credit is not None:
            total_credit += txn.credit

        lines.append(
            f"| {date_str} | {desc} | {debit_str} | {credit_str} | {balance_str} | {status} |"
        )

    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    lines.append("| Particulars     | Amount (₹)         |")
    lines.append("| :-------------- | -----------------: |")
    lines.append(f"| Total Debits    | {_fmt_inr(total_debit)}  |")
    lines.append(f"| Total Credits   | {_fmt_inr(total_credit)} |")
    net  = total_credit - total_debit
    sign = "+" if net >= 0 else ""
    lines.append(f"| Net Change      | {sign}{_fmt_inr(abs(net))} |")
    closing = result.transactions[-1].balance if result.transactions else None
    if closing is not None:
        lines.append(f"| Closing Balance | {_fmt_inr(closing)} |")
    lines.append("")

    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("Markdown saved → %s (%d transactions)", path, len(result.transactions))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_inr(value: Optional[float]) -> str:
    """
    Format *value* with Indian number grouping (e.g. 12,450.00 or 1,20,000.00).
    Returns '—' for None.
    """
    if value is None:
        return "—"
    negative = value < 0
    abs_val  = abs(value)

    # Split into integer and decimal parts
    integer_digits = str(int(abs_val))
    decimal_part   = f"{abs_val:.2f}".split(".")[1]

    # Indian grouping: last 3, then groups of 2 left-wards
    if len(integer_digits) <= 3:
        indian = integer_digits
    else:
        last3  = integer_digits[-3:]
        rest   = integer_digits[:-3]
        groups: List[str] = []
        while rest:
            groups.append(rest[-2:] if len(rest) >= 2 else rest)
            rest = rest[:-2]
        groups.reverse()
        indian = ",".join(groups) + "," + last3

    formatted = f"{indian}.{decimal_part}"
    return f"-{formatted}" if negative else formatted


def _txn_to_dict(txn: Transaction) -> Dict[str, Any]:
    """Convert a Transaction to a plain dict with safe defaults."""
    return {
        "txn_date":          txn.txn_date or "",
        "description":       txn.description or "",
        "reference_no":      txn.reference_no or "",
        "debit":             txn.debit if txn.debit is not None else "",
        "credit":            txn.credit if txn.credit is not None else "",
        "balance":           txn.balance if txn.balance is not None else "",
        "confidence_score":  round(txn.confidence_score, 4),
        "validation_status": txn.validation_status.value,
        "page_num":          txn.page_num,
        "raw_text":          txn.raw_text,
    }
