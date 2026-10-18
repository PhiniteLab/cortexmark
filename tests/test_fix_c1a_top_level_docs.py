"""Portable package metadata and top-level documentation consistency checks."""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import yaml

from cortexmark.cli import build_run_parser

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _metadata() -> dict:
    return tomllib.loads(_read("pyproject.toml"))


def test_package_metadata_references_existing_readme_and_entrypoint() -> None:
    project = _metadata()["project"]
    assert (REPO_ROOT / project["readme"]).is_file()
    for entrypoint in project["scripts"].values():
        module, function = entrypoint.split(":", 1)
        assert callable(getattr(importlib.import_module(module), function))
    for url in project.get("urls", {}).values():
        parsed = urlparse(url)
        assert parsed.scheme == "https" and parsed.netloc


def test_commitizen_version_files_exist() -> None:
    for entry in _metadata()["tool"]["commitizen"]["version_files"]:
        target_path = entry.split(":", 1)[0]
        assert (REPO_ROOT / target_path).is_file(), f"Missing version file: {entry}"


def test_documented_install_extras_exist() -> None:
    extras = _metadata()["project"]["optional-dependencies"]
    for filename in ("README.md", "CONTRIBUTING.md"):
        text = _read(filename)
        documented = re.findall(r"pip install(?: -e)? [\"']?\.\[([^\]]+)\]", text)
        assert documented, f"No local install example in {filename}"
        for group in documented:
            assert set(group.split(",")) <= extras.keys()


def test_readme_exposes_supported_conversion_controls() -> None:
    readme = _read("README.md")
    parser = build_run_parser()
    for name in ("engine", "ocr", "stages"):
        action = next(action for action in parser._actions if action.dest == name)
        assert f"--{name}" in readme
        assert action.choices is not None
        for choice in action.choices:
            assert choice in readme


def test_top_level_relative_document_links_exist() -> None:
    for filename in ("README.md", "CONTRIBUTING.md", "SECURITY.md"):
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", _read(filename)):
            parsed = urlparse(target)
            if parsed.scheme or not parsed.path:
                continue
            assert (REPO_ROOT / parsed.path).exists(), f"Broken link in {filename}: {target}"


def test_contributing_has_no_hardcoded_test_count() -> None:
    for line in _read("CONTRIBUTING.md").splitlines():
        if "make test" in line:
            assert not re.search(r"\b\d+\s+tests?\b", line), f"Hardcoded test count: {line}"


def test_dependabot_ecosystem_directories_exist() -> None:
    config = yaml.safe_load(_read(".github/dependabot.yml"))
    for update in config["updates"]:
        target = REPO_ROOT / update["directory"].lstrip("/")
        assert target.is_dir(), f"Missing dependabot directory: {target}"
