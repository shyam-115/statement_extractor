"""
Statement Extractor — Generalized Financial Statement Extraction Engine.
"""

__version__ = "1.0.0"
__all__ = ["StatementExtractor"]


def __getattr__(name: str):
    if name == "StatementExtractor":
        from .extractor import StatementExtractor
        return StatementExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
