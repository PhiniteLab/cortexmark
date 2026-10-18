"""Documentation navigation, package metadata, and runtime surface consistency."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import yaml

from cortexmark.cli import build_run_parser

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"


def test_mkdocs_navigation_targets_exist() -> None:
    def check_entries(entries: list) -> None:
        for entry in entries:
            for target in entry.values():
                if isinstance(target, list):
                    check_entries(target)
                else:
                    assert (DOCS_DIR / target).is_file(), f"Missing navigation target: {target}"

    config = yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8"))
    assert config["nav"]
    check_entries(config["nav"])


def test_mkdocs_repository_matches_package_metadata_when_configured() -> None:
    config = yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8"))
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config.get("repo_url") == metadata["project"].get("urls", {}).get("Repository")
    if config.get("repo_name") or config.get("edit_uri"):
        assert config.get("repo_url")


def test_configuration_md_documents_stages_and_dual_engine() -> None:
    """cli.py's real --stages/--engine choices must be reflected in the primary-command docs."""
    parser = build_run_parser()
    engine_action = next(a for a in parser._actions if a.dest == "engine")
    stages_action = next(a for a in parser._actions if a.dest == "stages")
    assert engine_action.choices is not None
    assert stages_action.choices is not None

    text = (DOCS_DIR / "guide" / "configuration.md").read_text(encoding="utf-8")
    for choice in engine_action.choices:
        assert choice in text, f"--engine choice {choice!r} missing from configuration.md"
    assert "--stages" in text
    for stage in stages_action.choices:
        assert stage in text, f"--stages choice {stage!r} missing from configuration.md"
    # The extra artifacts analyze/validate produce must be named, not just the flag.
    for artifact in ("semantic_chunks", "crossref_report.json", "qa.json", "formula_validation.json"):
        assert artifact in text


def test_quickstart_documents_controls_and_index_links_to_configuration() -> None:
    """The quickstart exposes controls; the landing page links to their reference."""
    quickstart = (DOCS_DIR / "getting-started" / "quickstart.md").read_text(encoding="utf-8")
    index = (DOCS_DIR / "index.md").read_text(encoding="utf-8")
    assert "--stages" in quickstart
    assert "dual" in quickstart
    assert "(guide/configuration.md)" in index


def test_inputs_and_outputs_lists_analyze_and_validate_artifacts() -> None:
    text = (DOCS_DIR / "guide" / "inputs-and-outputs.md").read_text(encoding="utf-8")
    for artifact in (
        "semantic_chunks",
        "crossref_report.json",
        "algorithm_report.json",
        "notation_report.json",
        "qa.json",
        "formula_validation.json",
        "scientific_qa.json",
        "citation_context.json",
    ):
        assert artifact in text, f"{artifact} missing from inputs-and-outputs.md bundle contract"


def test_plugins_doc_states_not_invoked_by_pipeline() -> None:
    text = (DOCS_DIR / "architecture" / "plugins.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "standalone" in lowered or "experimental" in lowered
    assert "no effect on a real conversion run" in lowered or "never" in lowered


def test_modules_doc_covers_bundle_producing_modules() -> None:
    """page_ir/consensus/grounding/ocr/pdfutil/paths/cli generate the primary bundle and must
    have autodoc coverage; they were previously omitted entirely."""
    text = (DOCS_DIR / "api" / "modules.md").read_text(encoding="utf-8")
    for module in (
        "page_ir",
        "consensus",
        "grounding",
        "ocr",
        "scientific_ir",
        "citation_ir",
        "pdfutil",
        "paths",
        "cli",
    ):
        assert f"::: cortexmark.{module}" in text, f"missing autodoc stanza for cortexmark.{module}"


def test_modules_doc_separates_orphaned_utilities() -> None:
    text = (DOCS_DIR / "api" / "modules.md").read_text(encoding="utf-8")
    core_heading = text.index("## Core pipeline modules")
    standalone_heading = text.index("## Standalone utilities")
    assert core_heading < standalone_heading

    core_section = text[core_heading:standalone_heading]
    standalone_section = text[standalone_heading:]
    for orphan in ("diff", "ghpages", "citations", "doc_type", "formula_score", "parallel", "plugin"):
        assert f"::: cortexmark.{orphan}" in standalone_section
        assert f"::: cortexmark.{orphan}\n" not in core_section


@pytest.mark.parametrize(
    "module",
    ["page_ir", "consensus", "grounding", "ocr", "scientific_ir", "citation_ir", "pdfutil", "paths", "cli"],
)
def test_newly_documented_modules_are_actually_importable(module: str) -> None:
    """A docs stanza for a module that cannot import would break `mkdocs build --strict`."""
    import importlib

    importlib.import_module(f"cortexmark.{module}")
