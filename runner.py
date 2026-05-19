#!/usr/bin/env python3
"""
Batch Statement Processing Runner

Processes all financial statement documents (PDFs and images) in the input directory,
extracts transaction tables, saves outputs to the output directory, and generates
isolated debug visualization page overlays.

Optimized to load PaddleOCR weights only once and reuse the extractor instance.
"""
from __future__ import annotations

import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure statement_extractor can be imported from the current workspace
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from statement_extractor import StatementExtractor
    from statement_extractor.config import ExtractorConfig, OCRConfig
    from statement_extractor.utils.debug_viz import DebugVisualizer
    from statement_extractor.schemas import ValidationStatus
except ImportError as exc:
    print(f"\n[ERROR] Required packages or statement_extractor module could not be imported: {exc}", file=sys.stderr)
    print("Please ensure you are running this script inside the active virtual environment.", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_runner")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enterprise-grade batch processor for bank statement extraction."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Directory containing statement files to process (default: input)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory where extraction results will be saved (default: output)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for rasterizing PDF pages (default: 200)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU for PaddleOCR if available (default: False)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress detailed extraction progress logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve()

    # Enforce safe path limits (workspace boundaries) to align with security guidelines
    workspace_root = Path(__file__).resolve().parent
    try:
        input_path.relative_to(workspace_root)
        output_path.relative_to(workspace_root)
    except ValueError:
        logger.error("Security boundary violation: Input and output paths must reside inside the workspace.")
        sys.exit(1)

    if not input_path.exists():
        logger.error("Input directory does not exist: %s", input_path)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    debug_base_dir = output_path / "debug_output"
    debug_base_dir.mkdir(parents=True, exist_ok=True)

    # Scan for supported document files
    supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".avif", ".webp"}
    input_files = sorted(
        [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in supported_extensions]
    )

    if not input_files:
        logger.warning("No supported statement files found in: %s", input_path)
        sys.exit(0)

    print("=" * 80)
    print(f"  BATCH RUNNER: Processing {len(input_files)} statement document(s)")
    print("=" * 80)

    # Initialize Extractor Config with reuse strategy
    ocr_config = OCRConfig(
        dpi=args.dpi,
        use_gpu=args.use_gpu,
    )
    config = ExtractorConfig(
        ocr=ocr_config,
        debug=True,  # Always enable debug rendering as requested
        debug_output_dir=str(debug_base_dir),
    )

    # Suppress verbose logger if quiet
    if args.quiet:
        logging.getLogger("statement_extractor").setLevel(logging.WARNING)

    logger.info("Initializing StatementExtractor pipeline (this can take a few seconds)...")
    try:
        extractor = StatementExtractor(config)
    except Exception as e:
        logger.error("Failed to initialize StatementExtractor: %s", e, exc_info=True)
        sys.exit(1)

    results_summary: List[Dict[str, Any]] = []

    for i, file_path in enumerate(input_files, start=1):
        file_name = file_path.name
        file_stem = file_path.stem
        print(f"\n[{i}/{len(input_files)}] Processing: {file_name}")
        print("-" * 60)

        # Configure file-specific isolated debug directory to prevent name collisions
        file_debug_dir = debug_base_dir / file_stem
        file_debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Inject the file-specific visualizer
        extractor.config.debug_output_dir = str(file_debug_dir)
        extractor._visualizer = DebugVisualizer(str(file_debug_dir))

        start_time = time.perf_counter()
        status = "SUCCESS"
        error_msg = ""
        txn_count = 0
        validation_ratio = 0.0
        bank_id = "UNKNOWN"

        try:
            # Execute extraction
            result = extractor.extract(str(file_path))
            
            # Record metrics
            txn_count = len(result.transactions)
            validation_ratio = result.validated_ratio
            bank_id = result.bank_profile.bank_id if result.bank_profile else "UNKNOWN"

            # Save structured output JSON
            json_out_path = output_path / f"{file_stem}_output.json"
            extractor.save_json(result, str(json_out_path))

            # Save structured output CSV (production value-add)
            csv_out_path = output_path / f"{file_stem}_output.csv"
            extractor.save_csv(result, str(csv_out_path))

            logger.info("Saved JSON → %s", json_out_path.name)
            logger.info("Saved CSV  → %s", csv_out_path.name)
            logger.info("Debug visualization images generated in: %s", file_debug_dir.relative_to(workspace_root))

        except Exception as exc:
            status = "FAILED"
            error_msg = str(exc)
            logger.error("Failed to process file '%s': %s", file_name, exc, exc_info=True)

        elapsed = time.perf_counter() - start_time
        results_summary.append({
            "name": file_name,
            "status": status,
            "duration": elapsed,
            "txns": txn_count,
            "bank": bank_id,
            "valid_ratio": validation_ratio,
            "error": error_msg,
        })

    # Print Premium Execution Summary Table
    print("\n" + "=" * 105)
    print(f"{'FILE NAME':<25} | {'STATUS':<8} | {'BANK':<12} | {'TXNS':<6} | {'VALID %':<8} | {'TIME (s)':<8} | {'REMARKS/ERROR':<25}")
    print("-" * 105)
    for r in results_summary:
        remarks = r["error"] if r["status"] == "FAILED" else f"Parsed successfully"
        print(f"{r['name']:<25} | {r['status']:<8} | {r['bank']:<12} | {r['txns']:<6} | {r['valid_ratio']*100:>6.1f}% | {r['duration']:>7.2f} | {remarks:<25}")
    print("=" * 105)

    print("\nBatch extraction process complete.\n")


if __name__ == "__main__":
    main()
