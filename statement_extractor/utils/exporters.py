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
    "transaction_date", "description", "reference_no",
    "debit", "credit", "balance",
    "tx_type", "validation_flags",
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
        "schema_version": 1,
        "source_file": result.source_file,
        "total_pages": result.total_pages,
        "total_transactions": len(result.transactions),
        "total_extracted_tables": len(result.extracted_tables),
        "column_mapping": result.column_mapping,
        "extraction_warnings": result.extraction_warnings,
        "extracted_tables": [t.dict() for t in result.extracted_tables],
        "transactions": [_txn_to_dict(t) for t in result.transactions],
    }
    if result.validation_summary is not None:
        data["validation_summary"] = result.validation_summary.model_dump()

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
    """
    Canonical transaction record for JSON/CSV consumers.

    Uses one field name per concept (no duplicate aliases).  Debit/credit
    are positive magnitudes; use ``tx_type`` / document context for Dr/Cr.
    """
    return {
        "transaction_date": txn.txn_date,
        "description":      txn.description or "",
        "reference_no":     txn.reference_no or "",
        "debit":            txn.debit if txn.debit is not None else None,
        "credit":           txn.credit if txn.credit is not None else None,
        "balance":          txn.balance if txn.balance is not None else None,
        "tx_type":          txn.tx_type or "",
        "validation_flags": list(txn.validation_flags),
        "continuation":     txn.continuation,
        "confidence_score": round(txn.confidence_score, 4),
        "validation_status": txn.validation_status.value,
        "page_num":         txn.page_num,
        "raw_text":         txn.raw_text,
    }
