"""
StatementExtractor — main orchestration entrypoint.

Full pipeline
-------------
  Input (PDF / image)
      ↓
  OCREngine.process_file()
      → tokens_per_page, images
      ↓
  RowGrouper.group()           (per page)
      → LogicalRow list
      ↓
  RowGrouper.merge_continuations()
      → merged LogicalRow list
      ↓
  ColumnDetector.detect()      (all pages combined)
      → ColumnZone list
      ↓
  HeaderInference.infer()
      → ColumnZone list with semantic_role
      ↓
  TransactionReconstructor.reconstruct()
      → raw Transaction list
      ↓
  BalanceValidator.validate()
      → validated Transaction list
      ↓
  ExtractionResult
      → save_to_json() / save_to_csv()

Thread safety
-------------
  Each StatementExtractor instance holds a single PaddleOCR model.
  Do NOT share one instance across threads; create one per worker.

Usage
-----
    from statement_extractor import StatementExtractor

    extractor = StatementExtractor()
    result = extractor.extract("statement.pdf")

    extractor.save_json(result, "output.json")
    extractor.save_csv(result, "output.csv")
    print(result)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from .config import ExtractorConfig
from .ocr.engine import OCREngine
from .grouping.row_grouper import RowGrouper
from .grouping.table_segmenter import TableSegmenter
from .clustering.column_detector import ColumnDetector
from .parsing.header_inference import HeaderInference
from .parsing.transaction_reconstructor import TransactionReconstructor
from .validation.balance_validator import BalanceValidator
from .schemas import ExtractionResult, LogicalRow, OCRToken, ColumnZone, ExtractedTable
from .utils.exporters import save_to_json, save_to_csv
from .utils.debug_viz import DebugVisualizer

logger = logging.getLogger(__name__)


class StatementExtractor:
    """
    Generalized financial statement extraction engine.

    Parameters
    ----------
    config : ExtractorConfig, optional
        Override any default configuration values.  If None, all defaults
        are used.

    Examples
    --------
    >>> extractor = StatementExtractor()
    >>> result = extractor.extract("bank_statement.pdf")
    >>> print(result.transactions[0].txn_date)
    """

    def __init__(self, config: Optional[ExtractorConfig] = None) -> None:
        self.config = config or ExtractorConfig()
        self._setup_logging()

        logger.info("Initialising StatementExtractor …")
        self._ocr         = OCREngine(self.config.ocr)
        self._row_grouper = RowGrouper(self.config.row_grouping)
        self._table_segmenter = TableSegmenter(self.config)
        self._col_detector = ColumnDetector(self.config.column_detection)
        self._header_inf  = HeaderInference(self.config.header_inference)
        self._reconstructor = TransactionReconstructor(self.config)
        self._validator   = BalanceValidator(self.config.validation)

        if self.config.debug:
            self._visualizer = DebugVisualizer(self.config.debug_output_dir)
        else:
            self._visualizer = None

        logger.info("StatementExtractor ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, file_path: str) -> ExtractionResult:
        """
        Run the full extraction pipeline on *file_path*.

        Parameters
        ----------
        file_path : Path to a PDF, PNG, JPG, TIFF, or other image file.

        Returns
        -------
        ExtractionResult containing validated transactions and metadata.
        """
        logger.info("Extracting: %s", file_path)
        warnings: List[str] = []

        # ── Stage 1: OCR ────────────────────────────────────────────────
        try:
            tokens_per_page, images = self._ocr.process_file(file_path)
        except Exception as exc:
            logger.error("OCR failed: %s", exc, exc_info=True)
            return ExtractionResult(
                source_file=file_path,
                extraction_warnings=[f"OCR error: {exc}"],
            )

        total_pages = len(tokens_per_page)
        if self.config.max_pages > 0:
            tokens_per_page = tokens_per_page[: self.config.max_pages]
            images          = images[: self.config.max_pages]

        # ── Stage 2: Row grouping (per page) ────────────────────────────
        all_rows: List[LogicalRow] = []
        all_tokens: List[OCRToken] = []

        for page_num, tokens in enumerate(tokens_per_page):
            all_tokens.extend(tokens)
            rows = self._row_grouper.group(tokens, page_num)
            rows = self._row_grouper.merge_continuations(rows)
            all_rows.extend(rows)

            if self._visualizer and page_num < len(images):
                try:
                    self._visualizer.render_page(
                        images[page_num], tokens, rows, [], page_num
                    )
                except Exception as viz_exc:
                    logger.debug("Debug render failed on page %d: %s", page_num, viz_exc)

        if not all_rows:
            warnings.append("No rows extracted — document may be blank or unreadable")
            return ExtractionResult(
                source_file=file_path,
                total_pages=total_pages,
                extraction_warnings=warnings,
            )

        # ── Stage 3-5: Per-table extraction ────────────────────────────────
        all_transactions: List[Transaction] = []
        all_extracted_tables: List[ExtractedTable] = []
        page_zones: Dict[int, List[ColumnZone]] = {}
        combined_mapping: Dict[str, str] = {}

        for page_num in range(total_pages):
            page_rows = [r for r in all_rows if r.page_num == page_num]
            if not page_rows:
                page_zones[page_num] = []
                continue

            page_zones[page_num] = []
            tables = self._table_segmenter.segment(page_rows)

            for table_idx, table_rows in enumerate(tables):
                # 3. Column detection per table
                zones = self._col_detector.detect(table_rows)
                if not zones:
                    warnings.append(f"Column detection produced no zones on page {page_num + 1}, table {table_idx + 1}")
                    continue

                # Dynamically calculate the format-agnostic absolute Y-boundaries for the visualizer
                valid_rows = [r for r in table_rows if r.tokens]
                if valid_rows:
                    min_y = min(min(t.y1 for t in r.tokens) for r in valid_rows)
                    max_y = max(max(t.y2 for t in r.tokens) for r in valid_rows)
                else:
                    min_y, max_y = 0.0, 0.0
                    
                for z in zones:
                    z.top_boundary = min_y
                    z.bottom_boundary = max_y

                # 4. Header inference (pass ONLY table_rows to prevent header leakage across tables)
                zones = self._header_inf.infer(zones, table_rows)
                
                col_mapping = {z.column_id: z.semantic_role for z in zones if z.semantic_role}
                
                # --- SAFE HEADER INHERITANCE ---
                # If no headers were matched for this table, try to inherit from the last known transaction table
                # This safely handles continuation pages without stealing headers from unrelated tables below it.
                if not col_mapping and getattr(self, "_last_transaction_zones", None):
                    for z in zones:
                        for prev_z in self._last_transaction_zones:
                            if abs(z.x_center - prev_z.x_center) < 0.05:
                                z.semantic_role = prev_z.semantic_role
                                z.header_text = prev_z.header_text
                                break
                    col_mapping = {z.column_id: z.semantic_role for z in zones if z.semantic_role}

                page_zones[page_num].extend(zones)
                
                # --- BUILD GENERIC TABLE ---
                table_headers = []
                for z in zones:
                    h_name = z.header_text or z.semantic_role or f"Column_{z.column_id}"
                    table_headers.append(h_name)
                    
                data_rows = [r for r in table_rows if not r.is_header]
                table_data_rows = []
                for r in data_rows:
                    row_dict = {h: [] for h in table_headers}
                    for token in r.tokens:
                        col_id = self._col_detector.assign_token_to_column(token, zones)
                        if col_id >= 0:
                            for z, h_name in zip(zones, table_headers):
                                if z.column_id == col_id:
                                    row_dict[h_name].append(token.text)
                                    break
                    row_dict = {k: " ".join(v) for k, v in row_dict.items()}
                    table_data_rows.append(row_dict)
                    
                table_id_str = f"p{page_num + 1}_t{table_idx + 1}"
                all_extracted_tables.append(ExtractedTable(
                    table_id=table_id_str,
                    headers=table_headers,
                    rows=table_data_rows
                ))

                # --- TRANSACTION FILTERING ---
                roles_found = set(col_mapping.values())
                has_date = "date" in roles_found
                has_amount = bool({"debit", "credit", "balance"}.intersection(roles_found))
                # Stricter requirement: A true transaction table must also have a narration/description column
                # This prevents Fixed Deposit tables from accidentally passing
                has_narration = "narration" in roles_found
                
                if has_date and has_amount and has_narration:
                    logger.info("%s identified as TRANSACTION table (roles: %s)", table_id_str, col_mapping)
                    for k, v in col_mapping.items():
                        combined_mapping[f"{table_id_str}_col{k}"] = v
                        
                    self._last_transaction_zones = zones
                    table_txns = self._reconstructor.reconstruct(data_rows, zones)
                    all_transactions.extend(table_txns)
                else:
                    logger.debug("%s skipped (not a transaction table, saving as generic)", table_id_str)

        if not all_transactions and not all_extracted_tables:
            warnings.append("No data reconstructed")
            return ExtractionResult(
                source_file=file_path,
                total_pages=total_pages,
                column_mapping=combined_mapping,
                extraction_warnings=warnings,
            )

        logger.info("Reconstructed %d raw transactions and %d generic tables across %d page(s)", 
                    len(all_transactions), len(all_extracted_tables), total_pages)

        # ── Stage 6: Balance validation ───────────────────────────────────
        transactions = self._validator.validate(all_transactions)

        # ── Stage 7: Debug re-render with zones ───────────────────────────
        if self._visualizer:
            for page_num, tokens in enumerate(tokens_per_page):
                if page_num < len(images):
                    page_rows = [r for r in all_rows if r.page_num == page_num]
                    zones = page_zones.get(page_num, [])
                    try:
                        self._visualizer.render_page(
                            images[page_num], tokens, page_rows, zones, page_num
                        )
                    except Exception as viz_exc:
                        logger.debug("Final debug render failed: %s", viz_exc)

        result = ExtractionResult(
            transactions=transactions,
            extracted_tables=all_extracted_tables,
            total_pages=total_pages,
            source_file=str(Path(file_path).resolve()),
            extraction_warnings=warnings,
            column_mapping=combined_mapping,
        )

        logger.info(
            "Extraction complete: %d transactions from %d page(s)",
            len(transactions), total_pages,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience save methods
    # ------------------------------------------------------------------

    def save_json(self, result: ExtractionResult, output_path: str) -> None:
        """Save *result* to a JSON file at *output_path*."""
        save_to_json(result, output_path)

    def save_csv(self, result: ExtractionResult, output_path: str) -> None:
        """Save *result* transactions to a CSV file at *output_path*."""
        save_to_csv(result, output_path)

    # ------------------------------------------------------------------
    # Repr / str
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"StatementExtractor("
            f"lang={self.config.ocr.lang!r}, "
            f"debug={self.config.debug})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _setup_logging() -> None:
        """Configure root logger if no handlers are attached yet."""
        root = logging.getLogger("statement_extractor")
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            root.addHandler(handler)
            root.setLevel(logging.INFO)
