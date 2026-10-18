# Installation

CortexMark requires Python 3.11 or newer. Install the source checkout:

```bash
git clone https://github.com/PhiniteLab/cortexmark.git
cd cortexmark
git checkout v0.3.5a1
```

## Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
mkdir -p workspace/papers
```

## Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
New-Item -ItemType Directory -Force workspace/papers
```

## Optional engines

Run installation commands from the repository root in the active virtual environment:

| Command | Purpose |
|---|---|
| `python -m pip install .` | Base installation with MarkItDown |
| `python -m pip install ".[ocr]"` | Adds RapidOCR for scanned pages |
| `python -m pip install ".[docling]"` | Adds Docling for layout-aware extraction |
| `python -m pip install ".[docling,ocr]"` | Both optional paths |

Use `--engine markitdown` for the lighter CPU workflow. The default `auto` uses Docling when installed, otherwise MarkItDown. The base and RapidOCR paths do not require PyTorch. Docling has additional dependencies and may download model files on first use. Allow additional setup time and disk space for optional engines.

Conversion needs no API key, LLM, or cloud service. Without the OCR extra, scanned-page OCR is unavailable. OCR may merge spaces or misread text; compare extracted passages against the PDF. Development dependencies are optional for users converting papers.

## Verify and run

```bash
cortexmark run --help
cortexmark run workspace/papers/paper.pdf --engine markitdown
```

Put your own papers in ignored `workspace/papers/`; no bundled corpus is required. Output defaults to ignored `cortexmark-output/`. Continue to [Quick Start](quickstart.md).

This is an early preview. Read the [known issues](https://github.com/PhiniteLab/cortexmark/blob/main/KNOWN_ISSUES.md) and compare important converted content with the PDF.

## Development and documentation

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
python -m mkdocs build --strict
```

The development extra currently includes both optional conversion/OCR engines as well as test and lint tools. See [Contributing](../contributing.md) for checks.
