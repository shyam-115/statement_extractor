"""Tests for transaction reconstruction — BOB-style balance ``Cr`` suffix handling."""
from __future__ import annotations

from statement_extractor.config import ExtractorConfig
from statement_extractor.parsing.transaction_reconstructor import TransactionReconstructor
from statement_extractor.schemas.models import LogicalRow, OCRToken


def _tok(text: str, nx: float, *, is_numeric: bool = False, is_date: bool = False) -> OCRToken:
    return OCRToken(
        text=text,
        confidence=0.95,
        x1=nx * 900,
        y1=350,
        x2=nx * 900 + 80,
        y2=364,
        center_x=nx * 900 + 40,
        center_y=357,
        normalized_x=nx,
        normalized_y=0.5,
        page_num=0,
        is_numeric=is_numeric,
        is_date=is_date,
    )


def test_refine_does_not_replace_withdrawal_with_balance_cr_suffix() -> None:
    """BOB rows look like ``270.00 53762.47 Cr`` — balance must not become credit."""
    rec = TransactionReconstructor(ExtractorConfig())
    row = LogicalRow(
        row_id=1,
        tokens=[
            _tok("02-04-2026", 0.05, is_date=True),
            _tok("UPI/120931185007", 0.2),
            _tok("270.00", 0.75, is_numeric=True),
            _tok("53762.47", 0.88, is_numeric=True),
            _tok("Cr", 0.93),
        ],
        page_num=0,
        y_center=0.5,
    )
    d, c = rec._refine_debit_credit_from_inline_suffixes(
        row, debit=270.0, credit=None, balance=53762.47
    )
    assert d == 270.0
    assert c is None


def test_refine_does_not_invent_credit_when_only_balance_has_cr() -> None:
    """Opening row: date + balance with ``Cr``, no withdrawal — no fake credit."""
    rec = TransactionReconstructor(ExtractorConfig())
    row = LogicalRow(
        row_id=2,
        tokens=[
            _tok("01-04-2026", 0.05, is_date=True),
            _tok("54032.47", 0.88, is_numeric=True),
            _tok("Cr", 0.93),
        ],
        page_num=0,
        y_center=0.5,
    )
    d, c = rec._refine_debit_credit_from_inline_suffixes(
        row, debit=None, credit=None, balance=54032.47
    )
    assert d is None
    assert c is None


def test_refine_still_assigns_explicit_dr_when_no_balance_extracted() -> None:
    rec = TransactionReconstructor(ExtractorConfig())
    row = LogicalRow(
        row_id=3,
        tokens=[
            _tok("500.00", 0.5, is_numeric=True),
            _tok("Dr", 0.55),
        ],
        page_num=0,
        y_center=0.5,
    )
    d, c = rec._refine_debit_credit_from_inline_suffixes(
        row, debit=None, credit=None, balance=None
    )
    assert d == 500.0
    assert c is None
