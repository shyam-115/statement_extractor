"""
Pydantic schemas — all internal and output data models live here.
This is the canonical definition file; schemas/__init__.py re-exports from here.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Internal pipeline models
# ---------------------------------------------------------------------------

class OCRToken(BaseModel):
    """A single text token extracted by the OCR engine."""
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    normalized_x: float        # center_x / page_width
    normalized_y: float        # center_y / page_height
    page_num: int = 0
    is_numeric: bool = False
    is_date: bool = False

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class LogicalRow(BaseModel):
    """A horizontally grouped set of OCR tokens forming one statement row."""
    row_id: int
    tokens: List[OCRToken]
    page_num: int
    y_center: float            # representative y of the row (normalised)
    is_header: bool = False
    is_continuation: bool = False   # True if this row is a narration continuation

    @property
    def full_text(self) -> str:
        return " ".join(
            t.text for t in sorted(self.tokens, key=lambda t: t.normalized_x)
        )


class ColumnZone(BaseModel):
    """A detected vertical column band inferred from numeric x-positions."""
    column_id: int
    x_center: float
    left_boundary: float
    right_boundary: float
    support: int = 0           # number of numeric tokens that voted for this column
    semantic_role: Optional[str] = None  # date|narration|debit|credit|balance|reference


class ValidationStatus(str, Enum):
    VALIDATED    = "validated"
    REPAIRED     = "repaired"
    NEEDS_REVIEW = "needs_review"
    FAILED       = "failed"


# ---------------------------------------------------------------------------
# Output transaction model
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """Final extracted and validated transaction record."""
    txn_date:          Optional[str]   = None
    description:       Optional[str]   = None
    reference_no:      Optional[str]   = None
    debit:             Optional[float] = None
    credit:            Optional[float] = None
    balance:           Optional[float] = None
    confidence_score:  float           = Field(default=0.0, ge=0.0, le=1.0)
    validation_status: ValidationStatus = ValidationStatus.NEEDS_REVIEW
    page_num:          int             = 0
    raw_text:          str             = ""


class ExtractionResult(BaseModel):
    """Top-level output of the extraction engine."""
    transactions:        List[Transaction] = Field(default_factory=list)
    total_pages:         int               = 0
    source_file:         str               = ""
    extraction_warnings: List[str]         = Field(default_factory=list)
    column_mapping:      Dict[str, str]    = Field(default_factory=dict)

    def save_to_json(self, output_path: str) -> None:
        """Write this result to a JSON file."""
        from ..utils.exporters import save_to_json
        save_to_json(self, output_path)

    def save_to_csv(self, output_path: str) -> None:
        """Write transactions to a CSV file."""
        from ..utils.exporters import save_to_csv
        save_to_csv(self, output_path)
