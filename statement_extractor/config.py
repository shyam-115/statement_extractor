"""
Global configuration and tuning parameters for the extraction engine.
All thresholds are adaptive — document-level overrides are applied
dynamically at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class OCRConfig:
    """PaddleOCR engine settings."""
    use_angle_cls: bool = False         # textline orientation (PP-OCRv4: keep off on CPU)
    lang: str = "en"                    # primary OCR language
    ocr_version: str = "PP-OCRv4"       # PP-OCRv5 can fail on some CPU builds
    use_gpu: bool = False               # switch to True if CUDA available
    det_db_thresh: float = 0.3          # detection binary threshold
    det_db_box_thresh: float = 0.5      # box confidence threshold
    rec_batch_num: int = 6
    min_confidence: float = 0.30        # tokens below this are discarded
    dpi: int = 200                      # PDF→image render DPI
    disable_mkldnn: bool = True         # avoids some Paddle 3.x CPU oneDNN errors


@dataclass
class RowGroupingConfig:
    """Parameters that control y-axis row clustering."""
    dbscan_eps_fraction: float = 0.012  # fraction of page height for DBSCAN eps
    dbscan_min_samples: int = 1
    min_row_tokens: int = 1
    # narration continuation: if a line contains NO numeric token and sits
    # within this fraction of page height below the previous row it is merged
    narration_y_gap_fraction: float = 0.025


@dataclass
class ColumnDetectionConfig:
    """Parameters for dynamic x-axis column clustering."""
    dbscan_eps: float = 0.025           # normalized-x eps for DBSCAN
    dbscan_min_samples: int = 2         # minimum hits to form a column band
    min_column_support: int = 3         # columns with fewer hits are dropped
    boundary_padding: float = 0.015     # extra padding added to column boundaries


@dataclass
class HeaderInferenceConfig:
    """Fuzzy semantic header matching settings."""
    fuzzy_threshold: int = 75           # rapidfuzz ratio threshold (0-100)
    max_header_rows: int = 5            # how many top rows to scan for headers
    # Semantic vocabulary (lower-cased, order matters for priority)
    date_keywords: List[str] = field(default_factory=lambda: [
        "date", "txn date", "transaction date", "value date",
        "posting date", "tran date", "dt",
    ])
    narration_keywords: List[str] = field(default_factory=lambda: [
        "narration", "description", "particulars", "details",
        "remarks", "transaction details", "particuler", "detail",
    ])
    debit_keywords: List[str] = field(default_factory=lambda: [
        "debit", "withdrawal", "withdrawals", "dr", "paid out",
        "money out", "amount dr", "debit amount", "withdraw",
        "debits", "debit(dr)",
    ])
    credit_keywords: List[str] = field(default_factory=lambda: [
        "credit", "deposit", "deposits", "cr", "received",
        "money in", "amount cr", "credit amount", "credits",
        "credit(cr)",
    ])
    balance_keywords: List[str] = field(default_factory=lambda: [
        "balance", "closing balance", "available balance",
        "running balance", "bal", "balance (cr/dr)", "net balance",
        "outstanding balance",
    ])
    reference_keywords: List[str] = field(default_factory=lambda: [
        "reference", "ref no", "ref", "chq no", "cheque no",
        "utr", "transaction id", "txn id", "instrument no",
        "chq/ref", "reference no", "ref number",
    ])


@dataclass
class ValidationConfig:
    """Balance continuity validator settings."""
    tolerance_fraction: float = 0.01    # 1% tolerance on arithmetic check
    min_validated_ratio: float = 0.60   # at least 60% rows must validate
    max_repair_attempts: int = 2        # how many column-swap permutations to try


@dataclass
class ExtractorConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    row_grouping: RowGroupingConfig = field(default_factory=RowGroupingConfig)
    column_detection: ColumnDetectionConfig = field(default_factory=ColumnDetectionConfig)
    header_inference: HeaderInferenceConfig = field(default_factory=HeaderInferenceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    debug: bool = False
    debug_output_dir: str = "debug_output"
    max_pages: int = 0                  # 0 = all pages
