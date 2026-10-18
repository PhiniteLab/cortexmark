"""OCR extras must select installable engines and their runtime by Python version."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OCR_PACKAGES = {"rapidocr-onnxruntime", "rapidocr", "onnxruntime"}


def _selected_requirements(extra: str | None, python_version: str) -> dict[str, Requirement]:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = list(project["dependencies"])
    if extra:
        requirements.extend(project["optional-dependencies"][extra])
    environment = {
        **default_environment(),
        "python_version": python_version,
        "python_full_version": f"{python_version}.0",
        "extra": extra or "",
    }
    selected = {}
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            selected[requirement.name] = requirement
    return selected


@pytest.mark.parametrize("extra", ["ocr", "dev"])
@pytest.mark.parametrize("python_version", ["3.11", "3.12"])
def test_legacy_python_selects_supported_legacy_ocr(extra: str, python_version: str) -> None:
    selected = _selected_requirements(extra, python_version)
    assert "rapidocr-onnxruntime" in selected
    assert selected["rapidocr-onnxruntime"].specifier.contains("1.4.4")
    assert "rapidocr" not in selected


@pytest.mark.parametrize("extra", ["ocr", "dev"])
def test_python313_selects_modern_ocr_and_explicit_onnx_runtime(extra: str) -> None:
    selected = _selected_requirements(extra, "3.13")
    assert "rapidocr-onnxruntime" not in selected
    assert {"rapidocr", "onnxruntime"} <= selected.keys()
    assert selected["rapidocr"].specifier.contains("3.3.0")
    assert selected["rapidocr"].specifier.contains("3.10.0")
    assert not selected["rapidocr"].specifier.contains("4.0.0")
    assert selected["onnxruntime"].specifier.contains("1.20.1")


@pytest.mark.parametrize("extra", [None, "docs"])
@pytest.mark.parametrize("python_version", ["3.11", "3.13"])
def test_unrelated_install_modes_do_not_require_ocr(extra: str | None, python_version: str) -> None:
    selected = _selected_requirements(extra, python_version)
    assert not OCR_PACKAGES.intersection(selected)


@pytest.mark.parametrize(
    ("python_version", "supported"),
    [("3.10", False), ("3.11", True), ("3.12", True), ("3.13", True), ("3.14", True), ("3.15", False)],
)
def test_declared_python_range_respects_runtime_dependency_support(python_version: str, supported: bool) -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert SpecifierSet(project["requires-python"]).contains(python_version) is supported
