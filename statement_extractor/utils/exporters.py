"""
Exporters — JSON and CSV output writers.

Both functions accept an ExtractionResult and a file path.
They sanitise None values and ensure the output directory exists.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict

from ..schemas import ExtractionResult, Transaction

logger = logging.getLogger(__name__)

_CSV_FIELDS = [
    "txn_date", "description", "reference_no",
    "debit", "credit", "balance",
    "confidence_score", "validation_status", "page_num",
]


def save_to_json(result: ExtractionResult, output_path: str) -> None:
    """
    Serialise *result* to a pretty-printed JSON file.

    Parameters
    ----------
    result      : ExtractionResult from StatementExtractor
    output_path : Destination file path (created if missing)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "source_file": result.source_file,
        "total_pages": result.total_pages,
        "total_transactions": len(result.transactions),
        "column_mapping": result.column_mapping,
        "extraction_warnings": result.extraction_warnings,
        "transactions": [_txn_to_dict(t) for t in result.transactions],
    }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("JSON saved → %s (%d transactions)", path, len(result.transactions))


def save_to_csv(result: ExtractionResult, output_path: str) -> None:
    """
    Write *result* transactions to a UTF-8 CSV file.

    Parameters
    ----------
    result      : ExtractionResult from StatementExtractor
    output_path : Destination file path (created if missing)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for txn in result.transactions:
            row = _txn_to_dict(txn)
            writer.writerow(row)

    logger.info("CSV saved → %s (%d transactions)", path, len(result.transactions))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _txn_to_dict(txn: Transaction) -> Dict[str, Any]:
    """Convert a Transaction to a plain dict with safe defaults."""
    return {
        "txn_date":         txn.txn_date or "",
        "description":      txn.description or "",
        "reference_no":     txn.reference_no or "",
        "debit":            txn.debit if txn.debit is not None else "",
        "credit":           txn.credit if txn.credit is not None else "",
        "balance":          txn.balance if txn.balance is not None else "",
        "confidence_score": round(txn.confidence_score, 4),
        "validation_status": txn.validation_status.value,
        "page_num":         txn.page_num,
        "raw_text":         txn.raw_text,
    }
