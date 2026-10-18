# Repository orientation

CortexMark is a local PDF-to-Markdown toolkit. Start with the root [README](../README.md) for installation and conversion of your own papers.

| Path | Purpose |
|---|---|
| `cortexmark/` | CLI, conversion and OCR, Markdown processing, reports, standalone utilities |
| `configs/pipeline.yaml` | Configuration for the optional legacy runner |
| `tests/` | Synthetic and temporary-input regression tests |
| `docs/` | MkDocs user guide and Python API reference |
| `.github/` | Repository automation and contribution templates |
| `workspace/`, `sessions/`, `cortexmark-output/` | Ignored local inputs, notes, and results; created as needed |

[ARCHITECTURE.md](ARCHITECTURE.md) explains the runtime paths and report limitations. [CODE-MAP.md](CODE-MAP.md) provides a source inventory. Update this orientation and the architecture when relevant structure or behavior changes; no private maintenance tool is required to read or use the repository.
