# Requirements, Inputs, and Outputs

## Requirements

- Python 3.11 or newer
- this repository installed locally with `python -m pip install .`
- one or more local PDF files

Optional extras:

- `python -m pip install ".[docling]"` adds Docling for layout-aware extraction.
- `python -m pip install ".[ocr]"` adds torch-free RapidOCR for scanned pages.

The base MarkItDown path and the RapidOCR path do not require PyTorch. CortexMark does not require an API key, LLM, or cloud service.

## Inputs

`cortexmark run` accepts either:

- one `.pdf` file, or
- a directory, searched recursively for `.pdf` files.

```bash
cortexmark run workspace/papers/paper.pdf
cortexmark run workspace/papers/ --out workspace/results/
```

Keep personal papers and notes under the ignored `workspace/` directory. PDF stems must be unique within a directory run because each stem names its bundle directory. If one PDF cannot be processed, CortexMark reports the failure and continues with the remaining documents. Partial batches can return exit code zero; inspect warnings and the `--json` failure list. A failed document may leave partial files.

## Per-document output bundle

The default output root is `cortexmark-output/`. `--out DIR` changes only that root.

`raw.md`, `<stem>.md`, and `chunks/` come from the default stages (`convert clean chunk`), and completed documents get `summary.json`. IR, consensus, and grounding reports are attempted independently of `--stages`; failures are logged and can leave those artifacts absent. `--rag` adds `rag.jsonl`; adding `analyze`/`validate` to `--stages` requests the remaining artifacts. Each analysis/validation step can also fail independently. See [Configuration](configuration.md#-stages) for the `--stages` flag. The tree below lists possible artifacts, rather than a guarantee that every file exists.

```text
<output-root>/
└── <stem>/
    ├── raw.md
    ├── <stem>.md
    ├── chunks/
    │   └── chunk_*.md
    ├── ir.json
    ├── consensus.json
    ├── grounding.json
    ├── summary.json
    ├── rag.jsonl                    # only with --rag
    ├── semantic_chunks/              # only with --stages analyze
    ├── crossref_report.json          # only with --stages analyze
    ├── algorithm_report.json         # only with --stages analyze
    ├── notation_report.json          # only with --stages analyze
    ├── qa.json                       # only with --stages validate
    ├── formula_validation.json       # only with --stages validate
    ├── scientific_qa.json            # only with --stages validate
    └── citation_context.json         # only with --stages validate
```

| Artifact | Contents |
|---|---|
| `raw.md` | Direct engine output; locally recovered OCR text carries an OCR provenance marker |
| `<stem>.md` | Cleaned Markdown |
| `chunks/` | Heading-based Markdown chunks |
| `ir.json` | Pages and page-aware text blocks |
| `consensus.json` | Cross-source token agreement and heuristic confidence |
| `grounding.json` | Source-text support checks for generated fields |
| `summary.json` | Run metadata and aggregate counts |
| `rag.jsonl` | RAG-ready chunk records, emitted only by `--rag` |
| `semantic_chunks/` | Semantically split chunks (`--stages analyze`) |
| `crossref_report.json` | Internal cross-reference resolution (`--stages analyze`) |
| `algorithm_report.json` | Extracted pseudocode/algorithm blocks (`--stages analyze`) |
| `notation_report.json` | Mathematical notation glossary (`--stages analyze`) |
| `qa.json` | Aggregate quality-assurance badge and issue list (`--stages validate`) |
| `formula_validation.json` | Formula syntax and structure diagnostics (`--stages validate`) |
| `scientific_qa.json` | Scientific-writing QA issues (`--stages validate`) |
| `citation_context.json` | Citation context and co-citation/self-citation analysis (`--stages validate`) |

## Page-aware IR

`ir.json` is a torch-free intermediate representation built with pypdfium2 for born-digital pages and RapidOCR for scanned pages. Each block includes:

- `page_no`
- `bbox`, normalized to top-left `[0,1]` coordinates
- `engine` (`pypdfium2` or `rapidocr`)
- `page_type` (`born-digital`, `scanned`, or `empty`)
- extracted `text`, confidence, and evidence level

## Consensus and grounding

`consensus.json` separately reads the PDF and compares PDFium page tokens with the selected engine's Markdown, reporting per-page coverage, low-agreement pages, and document confidence. It does not consume `ir.json` and may run the engine again.

`grounding.json` treats cleaned extraction text as the source channel and checks metadata/topic values against it. It reports grounded and ungrounded values separately.

Confidence and grounding are text-matching heuristics, not semantic verification. QA badges and formula diagnostics do not establish mathematical correctness or citation accuracy. Inspect warnings and compare important text, equations, tables, and references against the PDF before thesis use. Summary defaults may represent an unavailable report rather than a measured zero.

## Legacy outputs

The advanced `python -m cortexmark.run_pipeline` workflow uses the configured `outputs/raw_md/`, `outputs/cleaned_md/`, `outputs/chunks/`, `outputs/semantic_chunks/`, and `outputs/quality/` trees. Those paths belong to the legacy configured pipeline, not to `cortexmark run` bundles.
