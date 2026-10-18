# Contributing

Clone this repository and work from its root directory. Use Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The development extra includes both optional extraction engines for integration work. Ordinary use needs only the runtime extras described in the README.

## Make a scoped change

1. Inspect `git status` before editing. Preserve unrelated worktree changes, including changes made by another contributor or tool; do not reset or overwrite them.
2. Create a branch for a coherent fix. Identify the affected behavior and keep edits limited to that scope.
3. Use synthetic or temporary inputs for regressions. Keep your papers, notes, and generated results in ignored `workspace/`, `sessions/`, or output directories.
4. Run the narrow checks relevant to the change, then the repository checks before proposing a merge. Report existing failures separately from new failures.
5. Describe the problem, resulting behavior, checks performed, and remaining limitations in the pull request.

Commit subjects follow `vL.M.N - type(scope): concise change and reason`. Keep private papers and generated results out of commits.

## Checks and conventions

```bash
python -m pytest tests/ -k "test_name" -v  # Focused regression check
make lint
make test
make typecheck
pre-commit run --all-files
```

Python code uses Ruff (120-character lines) and Pyright. Inspect `pyproject.toml` and `.pre-commit-config.yaml` for the active rules. Add meaningful tests for behavior changes; use `tmp_path` for filesystem cases. Avoid committing personal PDFs or generated research output.

## Clean generated files

`make clean-local-artifacts` removes known caches and build artifacts while preserving configured inputs and results. `make clean-outputs CONFIG=configs/pipeline.yaml` removes the selected configuration's generated outputs and manifest. It checks every target before deleting anything and refuses external paths or protected source/input trees. Cache artifacts that overlap configured data are skipped with a message. These targets retain the local Python environment.

The shared conversion/cleanup/chunking tests include `tests/test_pipeline_structure.py`. CLI orchestration is covered in `tests/test_run_cli.py`; page extraction and report behavior also have dedicated test modules under `tests/`.

## Project structure

- `cortexmark/` contains the CLI, conversion engines, Markdown processing, reports, and standalone utilities.
- `configs/pipeline.yaml` configures the legacy runner. The primary `cortexmark run` command builds its own configuration.
- `docs/` contains the MkDocs documentation; install `.[docs]` to build it locally.
- `cortexmark.plugin` offers experimental manual hook discovery. The pipeline does not automatically execute plugins.

When adding a module, follow adjacent modules' interfaces and update only the relevant callers, tests, and documentation. The architecture guides describe the two execution paths; the source checkout also includes `REPO-INFO/ARCHITECTURE.md`.

## Report problems

Use this repository's issue tracker for ordinary bugs. Include Python/platform versions, the command, and safe error output. A small synthetic reproducer is preferable to sharing a private paper. Remove sensitive paths and document contents from logs. For vulnerabilities, follow `SECURITY.md` in the repository root.
