# CortexMark

CortexMark is a standalone Python CLI that turns your own PDF papers into local Markdown bundles for reading, annotation, and retrieval workflows.

**Early preview:** conversion and diagnostic reports have known limitations. Review the [known issues](KNOWN_ISSUES.md) before relying on outputs for thesis work. Report scores do not certify extraction accuracy or scientific correctness.

## Install from source

Requires Python 3.11–3.14; CI currently covers Python 3.11–3.13. Clone the repository:

```bash
git clone https://github.com/PhiniteLab/cortexmark.git
cd cortexmark
git checkout v0.3.5a2
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
mkdir -p workspace/papers
```

On Windows (PowerShell):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
New-Item -ItemType Directory -Force workspace/papers
```

Place your own PDFs under `workspace/papers/`. The base install provides the lighter CPU path using MarkItDown. Add only the engines you need, from the repository root in your active environment:

| Local install | Purpose |
|---|---|
| `python -m pip install .` | Base PDF text extraction |
| `python -m pip install ".[ocr]"` | Add RapidOCR for scanned pages |
| `python -m pip install ".[docling]"` | Add Docling for layout-aware extraction |
| `python -m pip install ".[docling,ocr]"` | Both optional paths |

The base and RapidOCR paths do not require PyTorch. Docling brings additional model/runtime dependencies and may download models on first use. No development extra is required to convert papers.

## Convert your papers

Start with the lighter CPU path:

```bash
cortexmark run workspace/papers/paper.pdf --engine markitdown
```

Process a whole directory recursively:

```bash
cortexmark run workspace/papers/ --engine markitdown
```

Request analysis, validation diagnostics, retrieval records, and a JSON batch summary:

```bash
cortexmark run workspace/papers/ --engine markitdown --stages convert clean chunk analyze validate --rag --json
```

Each PDF must have a unique filename stem within a directory run. Avoid the filename `raw.pdf`, which currently collides with an internal artifact name; see [R08](KNOWN_ISSUES.md). The default output root is `cortexmark-output/`; choose another with `--out workspace/results/`.

```text
cortexmark run <pdf|dir> [--out DIR] [--engine auto|docling|markitdown|dual] [--stages STAGE...] [--ocr auto|on|off] [--rag] [--json]
```

- `auto` is the CLI default: Docling when installed, otherwise MarkItDown. Explicit `markitdown` keeps the lighter path even if Docling is installed.
- `dual` combines the engines and falls back to MarkItDown when Docling is unavailable.
- `--ocr auto` uses RapidOCR on scanned pages when available; `on` requests OCR recovery and `off` disables it.
- `--stages` defaults to `convert clean chunk`. Add `analyze validate` for optional reports; omitting earlier stages requires existing outputs.
- `--json` changes terminal output; completed documents still receive `summary.json`.

## Read the output

For `paper.pdf`, files are written under `cortexmark-output/paper/`:

| Path | Contents |
|---|---|
| `raw.md` | Conversion output before Markdown cleanup |
| `paper.md` | Cleaned Markdown |
| `chunks/` | Heading-based Markdown chunks |
| `summary.json` | Engine, OCR use, counts, and report aggregates |
| `ir.json` | Page-aware source blocks and normalized bounding boxes |
| `consensus.json` | Token agreement between PDFium text and engine output |
| `grounding.json` | Source-text support checks for metadata/topic fields |
| `rag.jsonl` | Retrieval records, requested with `--rag` |
| `semantic_chunks/`, `crossref_report.json`, `algorithm_report.json`, `notation_report.json` | Requested with `analyze` |
| `qa.json`, `formula_validation.json`, `scientific_qa.json`, `citation_context.json` | Requested with `validate` |

IR, consensus, grounding, and individual analysis/validation steps can fail independently, leaving reports absent while the main conversion succeeds. A failed document may leave partial files. Directory runs continue after document failures and can return exit code zero on partial success: inspect warnings and the `--json` failure list.

Confidence, grounding, formula diagnostics, and QA badges are heuristics. They do not establish semantic fidelity, mathematical correctness, citation accuracy, or complete extraction. Check important passages, equations, tables, code, and references against the PDF. Known issues also affect retrieval IDs and exports; inspect records before building downstream indexes.

OCR can merge spaces, reorder words or lines, and omit recognized text when box arrays are incomplete. See [R30](KNOWN_ISSUES.md). Installing a compatible OCR engine enables processing; it does not guarantee transcription or layout accuracy.

## Local data and privacy

Conversion needs no API key, LLM, or cloud service. Optional engines may download model assets during setup or first use. Keep papers and notes in ignored `workspace/`; `sessions/` and `cortexmark-output/` are also local ignored areas. Logs, summaries, and exported bundles can contain document text and local paths: review them before sharing.

## Advanced tools

The configuration-driven runner remains available:

```bash
python -m cortexmark.run_pipeline --config configs/pipeline.yaml --input workspace/papers/
```

It supports conversion, cleanup, chunks, template rendering, analysis, validation, and named sessions. Individual `cortexmark convert`, `clean`, `chunk`, `render`, `analyze`, and `validate` commands remain available. Rendering populates templates in its source tree; use a local working copy.

Standalone utilities include figure catalogs, OCR/formula diagnostics, citation graphs, document classification, version diffs, multi-format export, static-site generation, and manual plugin dispatch. See [API Modules](docs/api/modules.md), [Configuration](docs/guide/configuration.md), and [Legacy Pipeline Stages](docs/guide/pipeline-stages.md).

See [Installation](docs/getting-started/installation.md), [Quick Start](docs/getting-started/quickstart.md), and [Contributing](CONTRIBUTING.md) for more detail.

## License

MIT License. Author: Mehmet İşcan. See [LICENSE](LICENSE) for copyright and terms.
