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
    numeric_min_confidence: float = 0.20  # lower threshold for amount cells
    dpi: int = 200                      # Base DPI
    adaptive_dpi: bool = True           # Adjust DPI based on document type
    digital_dpi: int = 200              # DPI for digital PDFs
    scanned_dpi: int = 300              # DPI for scans/images
    low_quality_dpi: int = 250          # medium quality scans
    disable_mkldnn: bool = True         # avoids some Paddle 3.x CPU oneDNN errors
    cache_dir: str = "./ocr_cache"      # content-hashed page OCR cache
    ocr_cache_enabled: bool = True
    finance_ocr_correction: bool = True  # O→0, l→1 in numeric contexts
    numeric_ensemble: bool = False       # Tesseract+EasyOCR vote (optional deps)


@dataclass
class RowGroupingConfig:
    """Parameters that control y-axis row clustering."""
    dbscan_eps_fraction: float = 0.008  # fraction of page height for DBSCAN eps
    dynamic_eps_from_token_height: bool = True  # eps = 0.5 × median token height
    dbscan_min_samples: int = 1
    min_row_tokens: int = 1
    skew_eps_multiplier: float = 1.5    # scale eps when residual skew > 0.5°
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
    vertical_strip_detection: bool = True  # split page into strips before clustering
    numeric_fusion: bool = True         # merge split Indian number tokens


@dataclass
class HeaderInferenceConfig:
    """Fuzzy semantic header matching settings."""
    fuzzy_threshold: int = 75           # rapidfuzz ratio threshold (0-100)
    max_header_rows: int = 5            # how many top rows to scan for headers
    # Semantic vocabulary (lower-cased, order matters for priority)
    date_keywords: List[str] = field(default_factory=lambda: [
        "date", "txn date", "transaction date",
        "posting date", "tran date", "dt",
    ])
    value_date_keywords: List[str] = field(default_factory=lambda: [
        "value date", "value dt", "val date", "val dt", "value dat",
    ])
    narration_keywords: List[str] = field(default_factory=lambda: [
        "narration", "description", "particulars", "details",
        "remarks", "transaction details", "particuler", "detail",
        "merchant category", "emi eligibility", "fx transactions",
    ])
    debit_keywords: List[str] = field(default_factory=lambda: [
        "debit", "withdrawal", "withdrawals", "dr", "paid out",
        "money out", "amount dr", "debit amount", "withdraw",
        "debits", "debit(dr)",
        "amount (rs", "amount(rs", "amount (in inr)", "amount in inr",
        "amount(rs.)", "charges",
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
        "chq/ref", "chq./ref", "chq ref", "reference no", "ref number",
        "chq./ref.no", "chq. no",
    ])
    # Serial-number / row-index column headers — these are NOT financial columns.
    # Zones matched by these keywords are assigned the "index" role and are
    # completely excluded from debit / credit / balance resolution.
    index_keywords: List[str] = field(default_factory=lambda: [
        "#", "no", "no.", "sno", "s.no", "sr.no", "sl.no", "sl no",
        "sr no", "serial no", "serial", "item no", "s no",
    ])
    noise_keywords: List[str] = field(default_factory=lambda: [
        "cashback", "reward", "rewards", "points", "cash back",
        "reward points", "promo", "cb", "reward pts",
    ])


@dataclass
class ValidationConfig:
    """
    Validation settings.

    When ledger_validation is enabled, LedgerValidator runs non-mutating
    balance checks.  BalanceValidator still applies fidelity-first status.
    """
    min_validated_ratio: float = 0.60
    ledger_validation: bool = True
    tolerance_absolute: float = 0.01
    tolerance_fraction: float = 0.005


@dataclass
class LayoutConfig:
    """
    Settings for the layout-aware region classifier.

    The classifier scores each table segment as one of:
    TRANSACTION_TABLE | METADATA_TABLE | FOOTER | HEADER_BLOCK | NOISE
    and routes only TRANSACTION_TABLE regions to the reconstruction pipeline.
    """
    enabled: bool = True
    min_transaction_rows: int = 2       # below this, cannot be a transaction table
    # Minimum fraction of rows that must have a parseable date to classify
    # a region as a transaction table.
    min_date_row_fraction: float = 0.25
    # Minimum fraction of rows with at least one numeric token
    min_numeric_row_fraction: float = 0.40
    # Maximum fraction of rows allowed to match footer patterns before
    # classifying a region as FOOTER
    footer_row_fraction_threshold: float = 0.60
    # Number of rows at the bottom of a page to treat as candidate footer zone
    footer_candidate_rows: int = 5
    # Score threshold above which a table is accepted as TRANSACTION_TABLE
    transaction_score_threshold: float = 0.45


@dataclass
class BankFingerprintConfig:
    """
    Settings for lightweight bank detection from document text.

    Bank identification is purely optional and advisory — when a bank is
    identified its profile may supply column-order hints or date-format hints
    to improve accuracy. The generic pipeline always runs unchanged if the
    bank is UNKNOWN.
    """
    enabled: bool = True
    scan_pages: int = 2         # only scan first N pages for bank identity
    # Minimum keyword score (0–1) to accept a bank identification
    min_match_score: float = 0.40


@dataclass
class ConfidenceFusionConfig:
    """
    Weights for the multi-factor confidence fuser.

    Arithmetic validation is disabled (fidelity-first mode).
    Weights are split across OCR confidence, date parse, amount presence,
    and column assignment tightness.
    """
    enabled: bool = True
    weight_ocr_confidence: float = 0.40
    weight_date_parse: float = 0.25
    weight_amount_parse: float = 0.25
    weight_column_assignment: float = 0.10


@dataclass
class PipelineConfig:
    """Post-extraction pipeline feature flags."""
    # When True, keep every table row and extracted value as in the source
    # document — no row drops, merges, or amount rewrites.
    document_fidelity: bool = True
    semantic_resolver: bool = False
    transaction_taxonomy: bool = True
    cross_page_stitching: bool = False
    early_exit_on_full_table: bool = False  # skip pages after full txn table found
    min_transactions_for_early_exit: int = 5


@dataclass
class ExtractorConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    row_grouping: RowGroupingConfig = field(default_factory=RowGroupingConfig)
    column_detection: ColumnDetectionConfig = field(default_factory=ColumnDetectionConfig)
    header_inference: HeaderInferenceConfig = field(default_factory=HeaderInferenceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    bank_fingerprint: BankFingerprintConfig = field(default_factory=BankFingerprintConfig)
    confidence_fusion: ConfidenceFusionConfig = field(default_factory=ConfidenceFusionConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    debug: bool = False
    debug_output_dir: str = "output/debug_output"
    max_pages: int = 0                  # 0 = all pages
