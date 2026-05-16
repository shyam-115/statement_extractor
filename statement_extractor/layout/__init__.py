"""
Layout intelligence package.

Components
----------
region_classifier : LayoutRegionClassifier
    Scores each table segment (List[LogicalRow]) and returns a RegionType.
    Only TRANSACTION_TABLE regions enter the reconstruction pipeline.

bank_fingerprinter : BankFingerprinter
    Lightweight keyword-based bank identity detector.
    Returns a BankProfile with optional extraction hints.
"""
from .region_classifier import LayoutRegionClassifier
from .bank_fingerprinter import BankFingerprinter

__all__ = ["LayoutRegionClassifier", "BankFingerprinter"]
