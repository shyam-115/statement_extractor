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
from .clustering.column_detector import ColumnDetector
from .parsing.header_inference import HeaderInference
from .parsing.transaction_reconstructor import TransactionReconstructor
from .validation.balance_validator import BalanceValidator
from .schemas import ExtractionResult, LogicalRow, OCRToken, ColumnZone
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

        # ── Stage 3: Column detection ────────────────────────────────────
        zones: List[ColumnZone] = self._col_detector.detect(all_rows)
        if not zones:
            warnings.append("Column detection produced no zones — output may be degraded")

        # ── Stage 4: Header semantic inference ───────────────────────────
        zones = self._header_inf.infer(zones, all_rows)

        col_mapping = {
            z.column_id: z.semantic_role for z in zones if z.semantic_role
        }
        logger.info("Column mapping: %s", col_mapping)

        # ── Stage 5: Transaction reconstruction ──────────────────────────
        data_rows = [r for r in all_rows if not r.is_header]
        transactions = self._reconstructor.reconstruct(data_rows, zones)
        if not transactions:
            warnings.append("No transactions reconstructed")
            return ExtractionResult(
                source_file=file_path,
                total_pages=total_pages,
                column_mapping=col_mapping,
                extraction_warnings=warnings,
            )

        logger.info("Reconstructed %d raw transactions", len(transactions))

        # ── Stage 6: Balance validation ───────────────────────────────────
        transactions = self._validator.validate(transactions)

        # ── Stage 7: Debug re-render with zones ───────────────────────────
        if self._visualizer:
            for page_num, tokens in enumerate(tokens_per_page):
                if page_num < len(images):
                    page_rows = [r for r in all_rows if r.page_num == page_num]
                    try:
                        self._visualizer.render_page(
                            images[page_num], tokens, page_rows, zones, page_num
                        )
                    except Exception as viz_exc:
                        logger.debug("Final debug render failed: %s", viz_exc)

        result = ExtractionResult(
            transactions=transactions,
            total_pages=total_pages,
            source_file=str(Path(file_path).resolve()),
            extraction_warnings=warnings,
            column_mapping={str(k): str(v) for k, v in col_mapping.items()},
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
