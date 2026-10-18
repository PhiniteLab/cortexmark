# Changelog

## 0.3.5a2 — 2026-10-18

- Select supported OCR distributions by Python version and include the modern ONNX Runtime backend explicitly.
- Bound the Python requirement to match MarkItDown compatibility; CI covers Python 3.11, 3.12, and 3.13.
- Coordinate MarkItDown 0.1.8 and Ruff 0.17.0 across dependencies, the lock file, hooks, and CI.
- Update GitHub Actions while preserving manually requested Pages deployment and hidden site artifacts.
- Direct optional-engine installation errors to the source checkout.
- Document the existing OCR spacing, reading-order, and incomplete-geometry limitations as R30.

## 0.3.5a1 — 2026-10-18

- Prepare a self-contained student repository for converting users' own papers.
- Remove historical audit records and project-specific corpus instructions from the distribution.
- Document installation from local source, optional extraction engines, local data directories, and output limitations.
- Retain the CortexMark CLI, configurable legacy pipeline, template rendering, and standalone utilities.
- Start a standalone public source repository with a fresh Git history and source-based installation.
- Document the unresolved audit findings and publish this release as an early preview.
