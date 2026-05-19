"""Unit tests for enhanced LayoutRegionClassifier."""
from __future__ import annotations

from statement_extractor.config import LayoutConfig
from statement_extractor.layout.region_classifier import LayoutRegionClassifier
from statement_extractor.schemas import LogicalRow, RegionType, OCRToken


def _token(text: str, nx: float = 0.1) -> OCRToken:
    return OCRToken(
        text=text, confidence=1.0,
        x1=0, y1=0, x2=10, y2=10,
        center_x=nx * 100, center_y=50,
        normalized_x=nx, normalized_y=0.5,
    )


def _row(text: str, y: float = 0.2, header: bool = False) -> LogicalRow:
    return LogicalRow(
        row_id=0,
        tokens=[_token(text)],
        page_num=0,
        y_center=y,
        is_header=header,
    )


class TestRegionClassifierIntent:

    def setup_method(self):
        self.clf = LayoutRegionClassifier(LayoutConfig())

    def test_emi_schedule_detected(self):
        rows = [
            _row("EMI Schedule Principal Interest Outstanding", header=True),
            _row("1 5000 200 4800"),
            _row("2 5000 180 4600"),
            _row("3 5000 160 4400"),
        ]
        rtype, score = self.clf.classify(rows, set())
        assert rtype == RegionType.EMI_SCHEDULE

    def test_bank_transactions(self):
        rows = [
            _row("Date Particulars Debit Credit Balance", header=True),
        ]
        for i in range(5):
            rows.append(_row(f"01/05/2024 NEFT {1000+i} {50000+i}", y=0.2 + i * 0.05))
        rtype, _ = self.clf.classify(
            rows, {"date", "narration", "debit", "credit", "balance"}
        )
        assert rtype in (
            RegionType.BANK_TRANSACTIONS,
            RegionType.TRANSACTION_TABLE,
            RegionType.CREDIT_CARD_TRANSACTIONS,
        )
