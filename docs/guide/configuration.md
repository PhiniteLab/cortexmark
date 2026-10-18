# Configuration

## One-command workflow

`cortexmark run` is zero-config. It does not read `configs/pipeline.yaml`; its behavior is controlled by command-line options:

```text
cortexmark run <pdf|dir> [--out DIR] [--engine auto|docling|markitdown|dual] [--stages STAGE...] [--ocr auto|on|off] [--rag] [--json]
```

| Option | Default | Effect |
|---|---|---|
| `<pdf|dir>` | required | One PDF or a directory searched recursively for PDFs |
| `--out DIR` | `cortexmark-output` | Root directory for per-document bundles |
| `--engine` | `auto` | `auto` chooses Docling when installed, otherwise MarkItDown; `dual` runs both engines and merges the result (degrades to `markitdown` if Docling is unavailable) |
| `--stages` | `convert clean chunk` | One or more of `convert`, `clean`, `chunk`, `analyze`, `validate`; see below |
| `--ocr` | `auto` | `auto` OCRs scanned pages when RapidOCR is available; `on` requests OCR; `off` disables it |
| `--rag` | off | Adds `rag.jsonl` to each bundle |
| `--json` | off | Prints a machine-readable run summary to stdout |

Install optional engines from the repository root with:

```bash
python -m pip install ".[docling]"
python -m pip install ".[ocr]"
```

### `--stages`

`--stages` controls which of `convert`, `clean`, `chunk`, `analyze`, and `validate` run for this invocation. The default is `convert clean chunk`. Omitting `convert` or `clean` requires the corresponding prior-run output (`raw.md`, `<stem>.md`) to already exist in the document bundle under `--out`; otherwise the run fails. Omitting `chunk` reuses existing chunk files if present; `--rag` requires chunks.

Adding `analyze` and/or `validate` writes extra, opt-in artifacts to the same bundle directory:

```bash
cortexmark run paper.pdf --stages convert clean chunk analyze validate
```

| Stage | Extra artifacts |
|---|---|
| `analyze` | `semantic_chunks/`, `crossref_report.json`, `algorithm_report.json`, `notation_report.json` |
| `validate` | `qa.json`, `formula_validation.json`, `scientific_qa.json`, `citation_context.json` |

Each `analyze`/`validate` sub-step is isolated: a failure in one (for example, `semantic_chunk` raising on structureless text) is logged as a warning and does not drop the rest of the bundle. See [Requirements, Inputs, and Outputs](inputs-and-outputs.md#per-document-output-bundle) for what each artifact contains.

## Legacy advanced pipeline

`configs/pipeline.yaml` is read only by the older multi-stage workflow and its module commands. Run it explicitly with:

```bash
python -m cortexmark.run_pipeline --config configs/pipeline.yaml --input workspace/papers/
```

The checked-in file configures source and output paths, the conversion engine, Docling settings, Markdown cleanup, chunk split levels, template rendering, logging, and the idempotency manifest. Relative paths resolve from the configuration file's directory: `../data/raw` in `configs/pipeline.yaml` means repository-root `data/raw`. A compact example is:

```yaml
source_id: default

paths:
  data_raw: ../data/raw
  output_raw_md: ../outputs/raw_md
  output_cleaned_md: ../outputs/cleaned_md
  output_chunks: ../outputs/chunks
  output_quality: ../outputs/quality
  output_semantic_chunks: ../outputs/semantic_chunks

convert:
  engine: dual
  docling:
    device: auto
    num_threads: 1
    do_ocr: false
    do_table_structure: true
    table_structure_mode: accurate

clean:
  min_repeated_header_count: 3
  max_repeated_header_length: 80

chunk:
  split_levels: [1, 2]

idempotency:
  enabled: true
  manifest_file: ../outputs/.manifest.json
```

Legacy command-line arguments override matching configuration values. The legacy runner also accepts `--session-name`, `--stages`, `--engine`, and `--no-manifest`. Named sessions use the ignored `sessions/` directory. Keep private input trees and notes in ignored local directories such as `workspace/`; configuration files may contain sensitive local paths, so review them before sharing.

For its stage order and output trees, see [Legacy Pipeline Stages](pipeline-stages.md).

## Clean generated files

```bash
make clean-local-artifacts CONFIG=configs/pipeline.yaml
make clean-outputs CONFIG=configs/pipeline.yaml
```

The first target removes known caches and build artifacts, preserving configured inputs and results. The second removes configured output directories and the manifest after checking all targets. Output cleanup is limited to the project tree; external paths, overlapping output targets, and protected source/input trees are refused before deletion. Directory symlinks are never traversed. Cache artifacts that overlap protected data are skipped visibly. Neither target removes the Python environment.
