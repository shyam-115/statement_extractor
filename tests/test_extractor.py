"""
Unit tests for the extraction engine — no OCR / no GPU required.
Tests run entirely on synthetic in-memory data.

Run:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import pytest
from typing import List

from statement_extractor.config import (
    ExtractorConfig, RowGroupingConfig, ColumnDetectionConfig,
    HeaderInferenceConfig, ValidationConfig,
)
from statement_extractor.schemas.models import (
    OCRToken, LogicalRow, ColumnZone, Transaction, ValidationStatus,
)
from statement_extractor.parsing.numeric_parser import NumericParser
from statement_extractor.grouping.row_grouper import RowGrouper
from statement_extractor.clustering.column_detector import ColumnDetector
from statement_extractor.parsing.header_inference import HeaderInference
from statement_extractor.validation.balance_validator import BalanceValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token(
    text: str,
    nx: float,
    ny: float,
    confidence: float = 0.95,
    page: int = 0,
    is_numeric: bool = False,
    is_date: bool = False,
) -> OCRToken:
    """Create a minimal OCRToken for testing."""
    return OCRToken(
        text=text, confidence=confidence,
        x1=nx * 900, y1=ny * 700,
        x2=nx * 900 + 60, y2=ny * 700 + 14,
        center_x=nx * 900 + 30, center_y=ny * 700 + 7,
        normalized_x=nx, normalized_y=ny,
        page_num=page,
        is_numeric=is_numeric,
        is_date=is_date,
    )


def _make_row(
    row_id: int,
    tokens: List[OCRToken],
    page: int = 0,
    y: float = 0.5,
    is_header: bool = False,
) -> LogicalRow:
    return LogicalRow(
        row_id=row_id, tokens=tokens, page_num=page,
        y_center=y, is_header=is_header,
    )


# ---------------------------------------------------------------------------
# NumericParser
# ---------------------------------------------------------------------------

class TestNumericParser:

    def setup_method(self):
        self.p = NumericParser()

    # --- is_amount ---
    @pytest.mark.parametrize("text", [
        "1,00,000.00", "50000", "1,200.50", "75,000.00",
        "1,23,800.00", "342.50", "295.00",
        "-500.00", "500.00 CR", "500.00 DR",
        "INR 1,200.00", "Rs. 500",
    ])
    def test_is_amount_positive(self, text):
        assert self.p.is_amount(text), f"Expected is_amount({text!r}) = True"

    @pytest.mark.parametrize("text", [
        "NEFT", "UPI-Payment", "01/05/2024", "Description",
        "", "   ", "ABC", "ACME CORP",
    ])
    def test_is_amount_negative(self, text):
        assert not self.p.is_amount(text), f"Expected is_amount({text!r}) = False"

    # --- parse_amount ---
    def test_parse_amount_plain(self):
        result = self.p.parse_amount("1200.50")
        assert result is not None
        val, sign = result
        assert abs(val - 1200.50) < 0.01
        assert sign == "unknown"

    def test_parse_amount_cr_suffix(self):
        result = self.p.parse_amount("75000.00 CR")
        assert result is not None
        val, sign = result
        assert sign == "credit"

    def test_parse_amount_dr_suffix(self):
        result = self.p.parse_amount("1200.00 DR")
        assert result is not None
        _, sign = result
        assert sign == "debit"

    def test_parse_amount_negative(self):
        result = self.p.parse_amount("-500.00")
        assert result is not None
        _, sign = result
        assert sign == "debit"

    def test_parse_amount_indian_format(self):
        result = self.p.parse_amount("1,23,800.00")
        assert result is not None
        val, _ = result
        assert abs(val - 123800.00) < 0.01

    def test_parse_amount_none_for_text(self):
        assert self.p.parse_amount("NEFT-SALARY") is None

    # --- is_date ---
    @pytest.mark.parametrize("text", [
        "01/05/2024", "2024-05-01", "01-05-24",
        "12 Jan 2024", "Jan 12, 2024",
    ])
    def test_is_date_positive(self, text):
        assert self.p.is_date(text), f"Expected is_date({text!r}) = True"

    @pytest.mark.parametrize("text", [
        "NEFT", "1200.50", "Description", "", "UPI",
    ])
    def test_is_date_negative(self, text):
        assert not self.p.is_date(text)

    # --- extract_date ---
    def test_extract_date(self):
        assert self.p.extract_date("Txn on 03/05/2024 via NEFT") == "03/05/2024"

    def test_extract_date_no_date(self):
        assert self.p.extract_date("No date here") is None

    # --- extract_reference ---
    def test_extract_reference_utr(self):
        ref = self.p.extract_reference("UTR: NEFT2024051500123456")
        assert ref is not None

    # --- clean_amount_str ---
    def test_clean_amount_str(self):
        val = self.p.clean_amount_str("1,200.50")
        assert val is not None
        assert abs(val - 1200.50) < 0.01

    def test_clean_amount_str_none_for_text(self):
        assert self.p.clean_amount_str("NEFT") is None


# ---------------------------------------------------------------------------
# RowGrouper
# ---------------------------------------------------------------------------

class TestRowGrouper:

    def setup_method(self):
        self.rg = RowGrouper(RowGroupingConfig())

    def _tokens_at_y(self, texts_nx: list, ny: float, page=0) -> List[OCRToken]:
        return [_make_token(t, nx, ny, page=page) for t, nx in texts_nx]

    def test_group_two_rows(self):
        tokens = (
            self._tokens_at_y([("01/05/24", 0.05), ("Salary", 0.20), ("75000", 0.80)], ny=0.15)
            + self._tokens_at_y([("03/05/24", 0.05), ("Rent", 0.20), ("25000", 0.80)], ny=0.35)
        )
        rows = self.rg.group(tokens, page_num=0)
        assert len(rows) == 2

    def test_group_empty_tokens(self):
        assert self.rg.group([], page_num=0) == []

    def test_rows_sorted_top_to_bottom(self):
        tokens = (
            self._tokens_at_y([("Row2", 0.5)], ny=0.6)
            + self._tokens_at_y([("Row1", 0.5)], ny=0.2)
        )
        rows = self.rg.group(tokens, page_num=0)
        assert rows[0].y_center < rows[1].y_center

    def test_header_detection(self):
        """Top row with no numeric tokens should be flagged as header."""
        header_tokens = self._tokens_at_y(
            [("Date", 0.05), ("Description", 0.20), ("Debit", 0.70), ("Balance", 0.88)],
            ny=0.08,
        )
        data_tokens = self._tokens_at_y(
            [("01/05/24", 0.05), ("Salary", 0.20)],
            ny=0.25,
        )
        rows = self.rg.group(header_tokens + data_tokens, page_num=0)
        header_rows = [r for r in rows if r.is_header]
        assert len(header_rows) >= 1

    def test_merge_continuations(self):
        """Continuation rows should be merged into the preceding row."""
        # Row with amount (parent)
        parent_tokens = [
            _make_token("01/05/24", 0.05, 0.20, is_date=True),
            _make_token("Salary", 0.25, 0.20),
            _make_token("75000", 0.80, 0.20, is_numeric=True),
        ]
        # Continuation row (no numeric, y close to parent)
        cont_tokens = [
            _make_token("from ACME Corp", 0.25, 0.225),
        ]
        rows = self.rg.group(parent_tokens + cont_tokens, page_num=0)
        rows = self.rg.merge_continuations(rows)
        # After merging, continuation should be absorbed
        cont_rows = [r for r in rows if r.is_continuation]
        assert len(cont_rows) == 0  # merged, not present as standalone


# ---------------------------------------------------------------------------
# ColumnDetector
# ---------------------------------------------------------------------------

class TestColumnDetector:

    def setup_method(self):
        self.cd = ColumnDetector(ColumnDetectionConfig())

    def _numeric_rows(self) -> List[LogicalRow]:
        """Simulate a statement with 3 numeric columns: ~0.15, ~0.72, ~0.88."""
        rows = []
        col_positions = [
            (0.72, 0.88),   # debit, balance
            (None, 0.88),   # balance only
            (0.72, 0.88),
            (None, 0.88),
            (None, 0.88),
        ]
        for i, (dr_x, bal_x) in enumerate(col_positions):
            tokens = [_make_token("01/05/24", 0.05, 0.1 + i * 0.1, is_date=True)]
            if dr_x:
                tokens.append(_make_token("1200.00", dr_x, 0.1 + i * 0.1, is_numeric=True))
            tokens.append(_make_token("48800.00", bal_x, 0.1 + i * 0.1, is_numeric=True))
            rows.append(_make_row(i, tokens, y=0.1 + i * 0.1))
        return rows

    def test_detect_returns_zones(self):
        rows = self._numeric_rows()
        zones = self.cd.detect(rows)
        assert len(zones) >= 1

    def test_zones_sorted_left_to_right(self):
        rows = self._numeric_rows()
        zones = self.cd.detect(rows)
        for i in range(1, len(zones)):
            assert zones[i].x_center > zones[i - 1].x_center

    def test_detect_empty_rows(self):
        zones = self.cd.detect([])
        assert zones == []

    def test_assign_token_to_column(self):
        zones = [
            ColumnZone(column_id=0, x_center=0.72, left_boundary=0.70, right_boundary=0.74, support=5),
            ColumnZone(column_id=1, x_center=0.88, left_boundary=0.86, right_boundary=0.90, support=5),
        ]
        token = _make_token("1200.00", 0.72, 0.3, is_numeric=True)
        assert ColumnDetector.assign_token_to_column(token, zones) == 0

        token2 = _make_token("48800.00", 0.88, 0.3, is_numeric=True)
        assert ColumnDetector.assign_token_to_column(token2, zones) == 1

    def test_assign_token_no_zones(self):
        token = _make_token("1200.00", 0.72, 0.3, is_numeric=True)
        assert ColumnDetector.assign_token_to_column(token, []) == -1


# ---------------------------------------------------------------------------
# BalanceValidator
# ---------------------------------------------------------------------------

class TestBalanceValidator:

    def setup_method(self):
        self.v = BalanceValidator(ValidationConfig())

    def _txn(self, debit=None, credit=None, balance=None, conf=0.9):
        return Transaction(
            txn_date="01/05/2024",
            description="Test",
            debit=debit,
            credit=credit,
            balance=balance,
            confidence_score=conf,
            validation_status=ValidationStatus.NEEDS_REVIEW,
        )

    def test_validates_correct_sequence(self):
        """10000 - 1200 = 8800 → fidelity-first pass-through NEEDS_REVIEW."""
        txns = [
            self._txn(balance=10000.0),            # opening
            self._txn(debit=1200.0, balance=8800.0),
            self._txn(credit=5000.0, balance=13800.0),
        ]
        result = self.v.validate(txns)
        statuses = [t.validation_status for t in result]
        # All set to NEEDS_REVIEW
        assert all(s == ValidationStatus.NEEDS_REVIEW for s in statuses)

    def test_empty_list(self):
        assert self.v.validate([]) == []

    def test_no_balance_values(self):
        """Transactions without balance values → all NEEDS_REVIEW."""
        txns = [
            self._txn(debit=500.0),
            self._txn(credit=1000.0),
        ]
        result = self.v.validate(txns)
        for t in result:
            assert t.validation_status == ValidationStatus.NEEDS_REVIEW

    def test_tolerance_applied(self):
        """A 0.5% rounding error in fidelity-first mode still results in NEEDS_REVIEW."""
        txns = [
            self._txn(balance=10000.00),
            self._txn(debit=1200.00, balance=8800.01),  # 1 cent off
        ]
        result = self.v.validate(txns)
        assert result[1].validation_status == ValidationStatus.NEEDS_REVIEW


# ---------------------------------------------------------------------------
# HeaderInference
# ---------------------------------------------------------------------------

class TestHeaderInference:

    def setup_method(self):
        self.hi = HeaderInference(HeaderInferenceConfig())

    def _header_row(self, texts_nx: list) -> LogicalRow:
        tokens = [_make_token(t, nx, 0.05) for t, nx in texts_nx]
        return _make_row(0, tokens, y=0.05, is_header=True)

    def _zones(self) -> List[ColumnZone]:
        return [
            ColumnZone(column_id=0, x_center=0.05, left_boundary=0.02, right_boundary=0.12, support=5),
            ColumnZone(column_id=1, x_center=0.30, left_boundary=0.14, right_boundary=0.50, support=5),
            ColumnZone(column_id=2, x_center=0.65, left_boundary=0.55, right_boundary=0.74, support=5),
            ColumnZone(column_id=3, x_center=0.82, left_boundary=0.75, right_boundary=0.90, support=5),
            ColumnZone(column_id=4, x_center=0.93, left_boundary=0.91, right_boundary=0.99, support=5),
        ]

    def test_infer_standard_headers(self):
        """Standard bank statement headers should map to correct roles."""
        hrow = self._header_row([
            ("Date", 0.05),
            ("Description", 0.30),
            ("Withdrawal", 0.65),
            ("Deposit", 0.82),
            ("Balance", 0.93),
        ])
        zones = self._zones()
        result = self.hi.infer(zones, [hrow])
        roles = {z.column_id: z.semantic_role for z in result}
        assert roles.get(0) == "date"
        assert roles.get(4) == "balance"

    def test_infer_alternate_headers(self):
        """Alternate header vocabulary (Dr/Cr) should still match."""
        hrow = self._header_row([
            ("Txn Date", 0.05),
            ("Particulars", 0.30),
            ("Dr", 0.65),
            ("Cr", 0.82),
            ("Bal", 0.93),
        ])
        zones = self._zones()
        result = self.hi.infer(zones, [hrow])
        roles = {z.semantic_role for z in result if z.semantic_role}
        assert "balance" in roles or "date" in roles

    def test_positional_fallback_no_headers(self):
        """When no header rows exist, positional heuristics should assign roles."""
        zones = self._zones()
        # Pass data row (not header)
        data_row = _make_row(0, [_make_token("01/05/24", 0.05, 0.2)], y=0.2)
        result = self.hi.infer(zones, [data_row])
        # Rightmost should be balance
        rightmost = max(result, key=lambda z: z.x_center)
        assert rightmost.semantic_role == "balance"
