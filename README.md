# Statement Extractor

Generalized enterprise-grade financial statement extraction engine.

Extracts structured transactions from **any** bank statement — PDF, scanned image,
screenshot, or passbook photo — without bank-specific templates.

---

## Architecture

```
Input (PDF / Image)
      ↓
OCREngine          — PaddleOCR + deskew + coordinate normalisation
      ↓
RowGrouper         — DBSCAN y-axis clustering → LogicalRow list
      ↓
merge_continuations— multiline narration fusion
      ↓
ColumnDetector     — DBSCAN x-axis clustering → ColumnZone list
      ↓
HeaderInference    — fuzzy semantic role assignment (rapidfuzz)
      ↓
TransactionReconstructor — token → column → Transaction objects
      ↓
BalanceValidator   — arithmetic continuity + column-swap repair
      ↓
ExtractionResult   → JSON / CSV
```

---

## Project Layout

```
statement_extractor/
├── __init__.py
├── __main__.py          # python -m statement_extractor
├── extractor.py         # StatementExtractor — main entrypoint
├── config.py            # All tuning parameters (dataclasses)
├── cli.py               # argparse CLI
├── demo.py              # Synthetic demo (no real PDF needed)
├── ocr/
│   └── engine.py        # PaddleOCR wrapper + deskew
├── grouping/
│   └── row_grouper.py   # DBSCAN y-clustering
├── clustering/
│   └── column_detector.py # DBSCAN x-clustering
├── parsing/
│   ├── numeric_parser.py       # Amount / date / reference regex
│   ├── header_inference.py     # Fuzzy header → role mapping
│   └── transaction_reconstructor.py
├── validation/
│   └── balance_validator.py   # Arithmetic continuity engine
├── schemas/
│   ├── __init__.py
│   └── models.py         # Pydantic models
└── utils/
    ├── debug_viz.py      # OpenCV overlays
    └── exporters.py      # JSON / CSV writers
```

---

## Quick Start

### 1. Activate virtualenv

```bash
source venv/bin/activate
```

### 2. Verify modules (no OCR model needed)

```bash
python verify.py
```

### 3. Run the synthetic demo

```bash
python -m statement_extractor.demo
```

### 4. Extract a real statement

```python
from statement_extractor import StatementExtractor

extractor = StatementExtractor()
result = extractor.extract("my_statement.pdf")

extractor.save_json(result, "output.json")
extractor.save_csv(result, "output.csv")
```

### 5. CLI usage

```bash
# Basic extraction
python -m statement_extractor extract statement.pdf

# Save JSON + CSV + debug overlays
python -m statement_extractor extract statement.pdf \
    --json output.json \
    --csv  output.csv \
    --debug

# Custom DPI + GPU
python -m statement_extractor extract statement.pdf \
    --dpi 300 --gpu --json output.json
```

### 6. Run unit tests

```bash
# With pytest
venv/bin/pip install pytest
venv/bin/python -m pytest tests/ -v

# Without pytest (standalone)
python verify.py
```

---

## Configuration

All parameters live in `config.py`. Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `ocr.dpi` | 200 | PDF render DPI |
| `ocr.lang` | `"en"` | PaddleOCR language |
| `ocr.use_gpu` | `False` | CUDA acceleration |
| `ocr.min_confidence` | 0.30 | OCR confidence threshold |
| `row_grouping.dbscan_eps_fraction` | 0.012 | Row y-gap (fraction of page height) |
| `column_detection.dbscan_eps` | 0.025 | Column x-gap (normalised) |
| `header_inference.fuzzy_threshold` | 75 | rapidfuzz score threshold |
| `validation.tolerance_fraction` | 0.01 | Balance arithmetic tolerance (1%) |
| `debug` | `False` | Render OpenCV debug overlays |

```python
from statement_extractor import StatementExtractor
from statement_extractor.config import ExtractorConfig, OCRConfig

config = ExtractorConfig(
    ocr=OCRConfig(dpi=300, lang="en", use_gpu=True),
    debug=True,
    debug_output_dir="debug_output",
)
extractor = StatementExtractor(config)
```

---

## Output Schema

```json
{
  "source_file": "statement.pdf",
  "total_pages": 3,
  "total_transactions": 24,
  "column_mapping": {"0": "date", "1": "narration", "2": "debit", "3": "balance"},
  "extraction_warnings": [],
  "transactions": [
    {
      "txn_date": "03/05/2024",
      "description": "UPI-PHONEPE-Grocery Store",
      "reference_no": null,
      "debit": 1200.0,
      "credit": null,
      "balance": 48800.0,
      "confidence_score": 0.94,
      "validation_status": "validated",
      "page_num": 0,
      "raw_text": "03/05/2024 UPI-PHONEPE-Grocery Store 1,200.00 48,800.00"
    }
  ]
}
```

### Validation Statuses

| Status | Meaning |
|---|---|
| `validated` | Balance arithmetic checks out exactly |
| `repaired` | Debit/credit were swapped; auto-corrected |
| `needs_review` | Missing amounts — cannot validate |
| `failed` | Arithmetic mismatch — manual review needed |

---

## Supported Formats

| Format | Support |
|---|---|
| Digital PDFs | ✅ Native text layer via PyMuPDF |
| Scanned PDFs | ✅ Rasterised + PaddleOCR |
| PNG / JPG / TIFF | ✅ Direct image input |
| Rotated scans | ✅ Deskew + PaddleOCR angle classifier |
| Indian number format | ✅ 1,23,800.00 |
| CR/DR suffixes | ✅ Semantic sign inference |
| Multiline narrations | ✅ Continuation row merging |
| Missing table borders | ✅ Coordinate-based layout |
| Varying column orders | ✅ Dynamic DBSCAN clustering |
| Unknown banks | ✅ No templates required |

---

## Debug Visualization

Enable `debug=True` to get per-page annotated images in `debug_output/`:

- 🟥 Red bands — detected column zones
- 🟦 Blue bands — grouped row clusters  
- 🟩 Green boxes — OCR token bounding boxes
- 🟨 Yellow boxes — numeric tokens

---

## Requirements

```
paddlepaddle    >= 2.6.2
paddleocr       >= 2.9.1
opencv-python-headless
Pillow          >= 10.0.0
pymupdf         >= 1.24.0
pdf2image       >= 1.17.0
numpy           >= 1.26.0
pandas          >= 2.2.0
scikit-learn    >= 1.5.0
pydantic        >= 2.5.0
rapidfuzz       >= 3.8.0
```
