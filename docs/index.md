# CortexMark

CortexMark is a standalone Python CLI for converting your own PDF papers into local Markdown bundles. This is an **early preview**, with unresolved [known issues](https://github.com/PhiniteLab/cortexmark/blob/main/KNOWN_ISSUES.md); review converted content against the source PDF.

[Install from source](getting-started/installation.md), place a paper under `workspace/papers/`, then run:

```bash
cortexmark run workspace/papers/paper.pdf --engine markitdown
```

The base install provides a lighter CPU path. Optional RapidOCR handles scanned pages; optional Docling adds layout-aware extraction and may download models. The base and RapidOCR paths do not require PyTorch.

Results default to `cortexmark-output/<stem>/`: raw and cleaned Markdown, heading-based chunks, and a summary. Additional page IR, token consensus, grounding, analysis, and validation reports are diagnostic and may be absent after warnings. Retrieval records are optional. Scores do not certify semantic fidelity, mathematics, or citations.

Keep your own papers and notes in ignored `workspace/`. Conversion needs no API key, LLM, or cloud service; review outputs and logs before sharing.

| Topic | Page |
|---|---|
| Install and choose engines | [Installation](getting-started/installation.md) |
| Convert and inspect papers | [Quick Start](getting-started/quickstart.md) |
| Inputs and artifact details | [Requirements, Inputs, and Outputs](guide/inputs-and-outputs.md) |
| CLI options and legacy configuration | [Configuration](guide/configuration.md) |
| Runtime design | [Architecture Overview](architecture/overview.md) |
| Optional stage runner and templates | [Legacy Pipeline Stages](guide/pipeline-stages.md) |
| Python modules and standalone utilities | [API Modules](api/modules.md) |

The legacy runner, template rendering, and standalone utilities remain available for advanced use.
