# Quick Start

[Install from source](installation.md), create `workspace/papers/`, and add your own PDF. The explicit MarkItDown engine uses the lighter CPU path:

```bash
cortexmark run workspace/papers/paper.pdf --engine markitdown
```

Process every PDF under a directory recursively:

```bash
cortexmark run workspace/papers/ --engine markitdown
```

Request optional analysis, validation diagnostics, retrieval records, and a JSON batch summary:

```bash
cortexmark run workspace/papers/ --engine markitdown --stages convert clean chunk analyze validate --rag --json
```

The default output root is `cortexmark-output/`; `--out workspace/results/` changes it. Each PDF needs a unique filename stem in a directory run. Avoid `raw.pdf` because of the known raw/cleaned artifact collision. This early preview has unresolved [known issues](https://github.com/PhiniteLab/cortexmark/blob/main/KNOWN_ISSUES.md).

## Controls

```text
cortexmark run <pdf|dir> [--out DIR] [--engine auto|docling|markitdown|dual] [--stages STAGE...] [--ocr auto|on|off] [--rag] [--json]
```

`auto` selects Docling when installed, otherwise MarkItDown. `dual` combines engines when available. `--ocr auto` uses optional RapidOCR for scanned pages; `on` requests OCR recovery and `off` disables it. Install `.[ocr]` from the repository root before processing scanned papers. See [Configuration](../guide/configuration.md) for all options and stage prerequisites.

## Inspect results

A successful default run writes `raw.md`, cleaned `paper.md`, `chunks/`, and `summary.json` under `cortexmark-output/paper/`. `ir.json`, `consensus.json`, and `grounding.json` are attempted independently. Analysis and validation add optional reports; `--rag` adds `rag.jsonl`. See [Requirements, Inputs, and Outputs](../guide/inputs-and-outputs.md#per-document-output-bundle) for details.

Individual report failures can leave artifacts absent. Document failures can leave partial directories. Batch processing continues after failures and can return exit code zero for partial success: inspect warnings and the `--json` failure list. `--json` controls stdout; completed bundles retain their summaries.

Compare important passages, equations, tables, code, and references against the PDF. Confidence, grounding, formula diagnostics, and QA badges are heuristics, without semantic, mathematical, or citation verification. Known export issues can affect retrieval record IDs and formatting.

## Keep data local

`workspace/`, `sessions/`, and `cortexmark-output/` are ignored local areas. Keep personal papers and notes there. Conversion needs no API key, LLM, or cloud service, although optional engines may download models. Outputs and logs can contain document text and local paths; review them before sharing.

## Advanced workflow

The separate configuration-driven runner retains selected stages, named sessions, and template rendering:

```bash
python -m cortexmark.run_pipeline --config configs/pipeline.yaml --input workspace/papers/
```

See [Legacy Pipeline Stages](../guide/pipeline-stages.md), [Configuration](../guide/configuration.md), and [API Modules](../api/modules.md) for retained stage commands and standalone utilities.
