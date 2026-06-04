#!/usr/bin/env python3
"""
verify.py — runs without pytest, no OCR model loaded.
Tests all pure-Python modules (no PaddleOCR, no GPU).
"""
import sys
import traceback

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
results = []

def check(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"       {e}")
        results.append((name, False, traceback.format_exc()))

# ── 1. Schema imports ────────────────────────────────────────────────────────
print("\n[1] Schema imports")

def t_schema_imports():
    from statement_extractor.schemas.models import (
        OCRToken, LogicalRow, ColumnZone,
        Transaction, ValidationStatus, ExtractionResult,
    )

check("schemas.models imports", t_schema_imports)

def t_schema_pkg_imports():
    from statement_extractor.schemas import (
        OCRToken, LogicalRow, ColumnZone,
        Transaction, ValidationStatus, ExtractionResult,
    )

check("schemas package re-exports", t_schema_pkg_imports)

# ── 2. Config ────────────────────────────────────────────────────────────────
print("\n[2] Config")

def t_config():
    from statement_extractor.config import (
        ExtractorConfig, OCRConfig, RowGroupingConfig,
        ColumnDetectionConfig, HeaderInferenceConfig, ValidationConfig,
    )
    cfg = ExtractorConfig()
    assert cfg.ocr.dpi == 200
    assert cfg.row_grouping.dbscan_eps_fraction > 0
    assert cfg.column_detection.dbscan_eps > 0

check("ExtractorConfig defaults", t_config)

# ── 3. NumericParser ─────────────────────────────────────────────────────────
print("\n[3] NumericParser")
from statement_extractor.parsing.numeric_parser import NumericParser
p = NumericParser()

def t_amount_indian():
    assert p.is_amount("1,23,800.00"), "Indian format should match"
    assert p.is_amount("75,000.00"), "Standard format should match"
    assert p.is_amount("342.50"), "Decimal should match"
check("is_amount (Indian/standard formats)", t_amount_indian)

def t_amount_suffixes():
    r = p.parse_amount("500.00 CR")
    assert r and r[1] == "credit"
    r2 = p.parse_amount("1200.00 DR")
    assert r2 and r2[1] == "debit"
check("parse_amount CR/DR suffixes", t_amount_suffixes)

def t_amount_negative():
    r = p.parse_amount("-500.00")
    assert r and r[1] == "debit"
check("parse_amount negative value", t_amount_negative)

def t_date_formats():
    assert p.is_date("01/05/2024")
    assert p.is_date("2024-05-01")
    assert p.is_date("12 Jan 2024")
    assert not p.is_date("NEFT")
    assert not p.is_date("1200.50")
check("is_date multi-format", t_date_formats)

def t_extract_date():
    d = p.extract_date("Transaction on 03/05/2024 via NEFT")
    assert d == "03/05/2024", f"Got {d!r}"
check("extract_date from sentence", t_extract_date)

def t_clean_amount():
    val = p.clean_amount_str("1,200.50")
    assert val is not None and abs(val - 1200.50) < 0.01
check("clean_amount_str", t_clean_amount)

# ── 4. RowGrouper ────────────────────────────────────────────────────────────
print("\n[4] RowGrouper")
from statement_extractor.grouping.row_grouper import RowGrouper
from statement_extractor.schemas.models import OCRToken
from statement_extractor.config import RowGroupingConfig

def _tok(text, nx, ny, is_num=False, is_date=False):
    return OCRToken(
        text=text, confidence=0.95,
        x1=nx*900, y1=ny*700, x2=nx*900+60, y2=ny*700+14,
        center_x=nx*900+30, center_y=ny*700+7,
        normalized_x=nx, normalized_y=ny,
        is_numeric=is_num, is_date=is_date,
    )

rg = RowGrouper(RowGroupingConfig())

def t_row_grouping():
    tokens = (
        [_tok("01/05/24", 0.05, 0.15, is_date=True), _tok("Salary", 0.25, 0.15), _tok("75000", 0.85, 0.15, is_num=True)]
        + [_tok("03/05/24", 0.05, 0.35, is_date=True), _tok("Rent",   0.25, 0.35), _tok("25000", 0.85, 0.35, is_num=True)]
    )
    rows = rg.group(tokens, page_num=0)
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    assert rows[0].y_center < rows[1].y_center, "Rows not sorted top→bottom"
check("group() produces 2 sorted rows", t_row_grouping)

def t_header_detection():
    tokens = [_tok("Date", 0.05, 0.05), _tok("Description", 0.25, 0.05),
              _tok("Debit", 0.72, 0.05), _tok("Balance", 0.90, 0.05),
              _tok("01/05/24", 0.05, 0.20, is_date=True), _tok("Salary", 0.25, 0.20)]
    rows = rg.group(tokens, 0)
    headers = [r for r in rows if r.is_header]
    assert len(headers) >= 1
check("header detection", t_header_detection)

# ── 5. ColumnDetector ────────────────────────────────────────────────────────
print("\n[5] ColumnDetector")
from statement_extractor.clustering.column_detector import ColumnDetector
from statement_extractor.schemas.models import LogicalRow, ColumnZone
from statement_extractor.config import ColumnDetectionConfig

cd = ColumnDetector(ColumnDetectionConfig())

def _row(rid, tokens, y):
    return LogicalRow(row_id=rid, tokens=tokens, page_num=0, y_center=y)

def t_column_detect():
    rows = [
        _row(0, [_tok("01/05", 0.05, 0.1, is_date=True), _tok("1200", 0.72, 0.1, is_num=True), _tok("48800", 0.88, 0.1, is_num=True)], 0.1),
        _row(1, [_tok("03/05", 0.05, 0.2, is_date=True), _tok("5000",  0.72, 0.2, is_num=True), _tok("43800", 0.88, 0.2, is_num=True)], 0.2),
        _row(2, [_tok("05/05", 0.05, 0.3, is_date=True), _tok("2000",  0.72, 0.3, is_num=True), _tok("41800", 0.88, 0.3, is_num=True)], 0.3),
        _row(3, [_tok("07/05", 0.05, 0.4, is_date=True), _tok("500",   0.72, 0.4, is_num=True), _tok("41300", 0.88, 0.4, is_num=True)], 0.4),
    ]
    zones = cd.detect(rows)
    assert len(zones) >= 2, f"Expected ≥2 zones, got {len(zones)}"
    for i in range(1, len(zones)):
        assert zones[i].x_center > zones[i-1].x_center
check("detect() finds sorted column zones", t_column_detect)

def t_assign_token():
    zones = [
        ColumnZone(column_id=0, x_center=0.72, left_boundary=0.70, right_boundary=0.74, support=4),
        ColumnZone(column_id=1, x_center=0.88, left_boundary=0.86, right_boundary=0.90, support=4),
    ]
    t_near_0 = _tok("1200", 0.72, 0.1, is_num=True)
    t_near_1 = _tok("48800", 0.88, 0.1, is_num=True)
    assert ColumnDetector.assign_token_to_column(t_near_0, zones) == 0
    assert ColumnDetector.assign_token_to_column(t_near_1, zones) == 1
    assert ColumnDetector.assign_token_to_column(_tok("X", 0.5, 0.1), []) == -1
check("assign_token_to_column()", t_assign_token)

# ── 6. BalanceValidator ──────────────────────────────────────────────────────
print("\n[6] BalanceValidator")
from statement_extractor.validation.balance_validator import BalanceValidator
from statement_extractor.schemas.models import ValidationStatus
from statement_extractor.config import ValidationConfig

v = BalanceValidator(ValidationConfig())

def _txn(debit=None, credit=None, balance=None):
    from statement_extractor.schemas.models import Transaction
    return Transaction(
        txn_date="01/05/2024", description="Test",
        debit=debit, credit=credit, balance=balance,
        confidence_score=0.9,
        validation_status=ValidationStatus.NEEDS_REVIEW,
    )

def t_validate_correct():
    txns = [_txn(balance=50000.0), _txn(debit=1200.0, balance=48800.0), _txn(credit=75000.0, balance=123800.0)]
    result = v.validate(txns)
    statuses = {t.validation_status for t in result}
    assert ValidationStatus.VALIDATED in statuses
check("validate correct arithmetic sequence", t_validate_correct)

def t_validate_empty():
    assert v.validate([]) == []
check("validate empty list", t_validate_empty)

def t_validate_no_balance():
    txns = [_txn(debit=500.0), _txn(credit=1000.0)]
    result = v.validate(txns)
    for t in result:
        assert t.validation_status == ValidationStatus.NEEDS_REVIEW
check("validate: no balance → NEEDS_REVIEW", t_validate_no_balance)

def t_validate_tolerance():
    txns = [_txn(balance=10000.00), _txn(debit=1200.00, balance=8800.01)]  # 1 cent off
    result = v.validate(txns)
    assert result[1].validation_status == ValidationStatus.VALIDATED
check("validate within tolerance (1 cent off)", t_validate_tolerance)



# ── 7. Exporters ─────────────────────────────────────────────────────────────
print("\n[7] Exporters")
import tempfile, json, os
from statement_extractor.utils.exporters import save_to_json, save_to_csv
from statement_extractor.schemas.models import ExtractionResult, Transaction

def t_json_export():
    result = ExtractionResult(
        source_file="test.pdf",
        total_pages=1,
        transactions=[_txn(debit=1200.0, balance=48800.0)],
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        save_to_json(result, path)
        with open(path) as f:
            data = json.load(f)
        assert data["total_transactions"] == 1
        assert data["transactions"][0]["debit"] == 1200.0
    finally:
        os.unlink(path)
check("save_to_json()", t_json_export)

def t_csv_export():
    result = ExtractionResult(
        source_file="test.pdf",
        total_pages=1,
        transactions=[_txn(credit=75000.0, balance=123800.0)],
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        save_to_csv(result, path)
        with open(path, encoding="utf-8-sig") as f:
            content = f.read()
        assert "credit" in content.lower()
        assert "75000" in content
    finally:
        os.unlink(path)
check("save_to_csv()", t_csv_export)

def t_result_save_methods():
    result = ExtractionResult(
        source_file="test.pdf",
        total_pages=1,
        transactions=[_txn(debit=100.0, balance=9900.0)],
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        jpath = f.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        cpath = f.name
    try:
        result.save_to_json(jpath)
        result.save_to_csv(cpath)
        with open(jpath) as f:
            assert json.load(f)["total_transactions"] == 1
        with open(cpath, encoding="utf-8-sig") as f:
            assert "debit" in f.read().lower()
    finally:
        os.unlink(jpath)
        os.unlink(cpath)

check("ExtractionResult.save_to_json/csv", t_result_save_methods)

# ── Summary ──────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\n{'='*55}")
print(f"  Results: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} failed)")
    for name, ok, tb in results:
        if not ok:
            print(f"\n  FAILED: {name}")
            print(tb)
else:
    print("  — ALL PASSED ✓")
print(f"{'='*55}\n")

sys.exit(0 if failed == 0 else 1)
