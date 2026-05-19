#!/usr/bin/env python3
"""
Statement Extractor CLI — command-line interface for the extraction engine.

Usage examples
--------------
# Basic extraction (JSON output):
    python -m statement_extractor.cli extract statement.pdf

# Save as CSV:
    python -m statement_extractor.cli extract statement.pdf --csv output.csv

# Save as JSON + debug overlay images:
    python -m statement_extractor.cli extract statement.pdf \\
        --json output.json --debug

# Process all pages with custom DPI:
    python -m statement_extractor.cli extract statement.pdf \\
        --dpi 300 --json output.json

# Quiet mode (suppress progress logs):
    python -m statement_extractor.cli extract statement.pdf --quiet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ExtractorConfig, OCRConfig
from .extractor import StatementExtractor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="statement-extractor",
        description="Generalized financial statement extraction engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── extract ────────────────────────────────────────────────────────
    extract = sub.add_parser("extract", help="Extract transactions from a statement file")
    extract.add_argument(
        "file",
        metavar="FILE",
        help="Path to PDF, PNG, JPG, TIFF or other image file",
    )
    extract.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Save output as JSON to PATH",
    )
    extract.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="Save output as CSV to PATH",
    )
    extract.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Render debug overlay images (saved to output/debug_output/)",
    )
    extract.add_argument(
        "--debug-dir",
        metavar="DIR",
        default="output/debug_output",
        help="Directory to save debug images (default: output/debug_output)",
    )
    extract.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF rendering (default: 200)",
    )
    extract.add_argument(
        "--lang",
        default="en",
        help="PaddleOCR language code (default: en)",
    )
    extract.add_argument(
        "--max-pages",
        type=int,
        default=0,
        metavar="N",
        help="Process only the first N pages (0 = all)",
    )
    extract.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress info logs",
    )
    extract.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Use GPU for PaddleOCR (requires paddlepaddle-gpu)",
    )

    return parser


def _run_extract(args: argparse.Namespace) -> int:
    # Logging level
    level = logging.WARNING if args.quiet else logging.INFO
    logging.getLogger("statement_extractor").setLevel(level)

    # Build config
    ocr_cfg = OCRConfig(
        lang=args.lang,
        dpi=args.dpi,
        use_gpu=args.gpu,
    )
    config = ExtractorConfig(
        ocr=ocr_cfg,
        debug=args.debug,
        debug_output_dir=args.debug_dir,
        max_pages=args.max_pages,
    )

    # Run extraction
    extractor = StatementExtractor(config)
    result = extractor.extract(args.file)

    # Print warnings
    if result.extraction_warnings and not args.quiet:
        for w in result.extraction_warnings:
            print(f"[WARN] {w}", file=sys.stderr)

    # Output
    if args.json:
        extractor.save_json(result, args.json)
        if not args.quiet:
            print(f"JSON saved → {args.json}")

    if args.csv:
        extractor.save_csv(result, args.csv)
        if not args.quiet:
            print(f"CSV saved  → {args.csv}")

    # Always print summary JSON to stdout
    summary = {
        "source_file":     result.source_file,
        "total_pages":     result.total_pages,
        "total_txn":       len(result.transactions),
        "column_mapping":  result.column_mapping,
        "warnings":        result.extraction_warnings,
        "transactions":    [
            {
                "transaction_date": t.txn_date,
                "description": (t.description or "")[:60],
                "debit":      t.debit,
                "credit":     t.credit,
                "balance":    t.balance,
                "status":     t.validation_status.value,
                "confidence": round(t.confidence_score, 3),
            }
            for t in result.transactions
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        sys.exit(_run_extract(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
