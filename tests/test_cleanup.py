"""Cleanup commands preserve student data and reject unsafe plans before deletion."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from cortexmark import paths
from cortexmark.cleanup import cleanup_local_artifacts, cleanup_outputs
from cortexmark.common import load_config, reset_config_cache, resolve_configured_path, resolve_manifest_path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_KEYS = ("output_raw_md", "output_cleaned_md", "output_chunks", "output_quality", "output_semantic_chunks")


def _sentinel(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("keep this content", encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> tuple[Path, Path, dict]:
    monkeypatch.delenv("CONFIG", raising=False)
    for name, keys in vars(paths).items():
        if name.endswith("_ENV_KEYS"):
            for key in keys:
                monkeypatch.delenv(key, raising=False)
    root = tmp_path / "student project"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("CORTEXMARK_PROJECT_ROOT", str(root))
    _sentinel(root / "pyproject.toml")
    cfg = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
    config = root / "configs/pipeline.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    reset_config_cache()
    return root, config, cfg


def _save(config: Path, cfg: dict) -> None:
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    reset_config_cache()


def test_shipped_config_resolves_inputs_outputs_and_manifest_at_project_root(project) -> None:
    root, config, _cfg = project
    cfg = load_config(config)
    assert resolve_configured_path(cfg, "data_raw", "data/raw") == root / "data/raw"
    for key in OUTPUT_KEYS:
        assert resolve_configured_path(cfg, key, "unused").parent == root / "outputs"
    assert resolve_manifest_path(cfg) == root / "outputs/.manifest.json"


def test_make_cleanup_honors_selected_config_with_spaces_and_quotes(project) -> None:
    root, config, cfg = project
    custom_config = config.with_name("student's custom config.yaml")
    for key in OUTPUT_KEYS:
        cfg["paths"][key] = f"../selected/{key}"
        _sentinel(root / "selected" / key / "result.md")
    cfg["idempotency"]["manifest_file"] = "../selected/manifest.json"
    _save(custom_config, cfg)
    _sentinel(root / "selected/manifest.json")
    untouched = [_sentinel(root / "outputs/raw_md/default.md"), _sentinel(root / "data/raw/paper.txt")]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [
            "make",
            "--file",
            str(REPO_ROOT / "Makefile"),
            "clean-outputs",
            f"CONFIG={custom_config}",
            f"PYTHON={sys.executable}",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list((root / "selected").iterdir()) == []
    assert all(path.is_file() for path in untouched)


@pytest.mark.parametrize(
    "unsafe",
    [
        ".",
        "data/raw",
        "REPO-INFO",
        "configs",
        "workspace",
        "sessions",
        ".venv",
        "cortexmark",
        "outputs/raw_md",
        "../outside",
    ],
)
def test_unsafe_last_output_target_leaves_all_earlier_outputs_untouched(project, unsafe: str) -> None:
    root, config, cfg = project
    first_output = _sentinel(root / "outputs/raw_md/result.md")
    protected = _sentinel(root / unsafe / "sentinel.md")
    cfg["paths"]["output_semantic_chunks"] = f"../{unsafe}"
    _save(config, cfg)
    with pytest.raises(ValueError):
        cleanup_outputs(config)
    assert first_output.is_file()
    assert protected.is_file()


def test_manifest_cannot_alias_selected_config(project) -> None:
    root, config, cfg = project
    first_output = _sentinel(root / "outputs/raw_md/result.md")
    cfg["idempotency"]["manifest_file"] = "pipeline.yaml"
    _save(config, cfg)
    before = config.read_bytes()
    with pytest.raises(ValueError):
        cleanup_outputs(config)
    assert config.read_bytes() == before
    assert first_output.is_file()


def test_output_directory_type_error_is_checked_before_deletion(project) -> None:
    root, config, _cfg = project
    first_output = _sentinel(root / "outputs/raw_md/result.md")
    last_output = _sentinel(root / "outputs/semantic_chunks")
    with pytest.raises(ValueError):
        cleanup_outputs(config)
    assert first_output.is_file() and last_output.is_file()


def test_terminal_output_symlink_is_unlinked_without_deleting_its_target(project) -> None:
    root, config, cfg = project
    target = _sentinel(root / "saved-results/result.md")
    link = root / "linked-output"
    link.symlink_to(target.parent, target_is_directory=True)
    cfg["paths"]["output_raw_md"] = "../linked-output"
    _save(config, cfg)
    cleanup_outputs(config)
    assert not link.is_symlink()
    assert target.is_file()


def test_external_symlink_last_target_is_rejected_before_any_deletion(project) -> None:
    root, config, cfg = project
    first_output = _sentinel(root / "outputs/raw_md/result.md")
    external = _sentinel(root.parent / "outside/result.md")
    link = root / "external-results"
    link.symlink_to(external.parent, target_is_directory=True)
    cfg["paths"]["output_semantic_chunks"] = "../external-results"
    _save(config, cfg)
    with pytest.raises(ValueError):
        cleanup_outputs(config)
    assert first_output.is_file() and external.is_file() and link.is_symlink()


def test_cache_cleanup_preserves_configured_artifacts_and_student_state(project) -> None:
    root, config, cfg = project
    cfg["paths"]["data_raw"] = "../.ruff_cache"
    cfg["paths"]["output_raw_md"] = "../dist"
    cfg["paths"]["output_quality"] = "../site"
    cfg["idempotency"]["manifest_file"] = "../coverage.xml"
    _save(config, cfg)
    protected = [
        _sentinel(root / name)
        for name in (
            ".ruff_cache/input.txt",
            "dist/result.md",
            "site/report.md",
            "coverage.xml",
            ".venv/pyvenv.cfg",
            "sessions/student/notes.md",
            "workspace/paper.txt",
            "configs/outputs/.manifest.json",
        )
    ]
    caches = [_sentinel(root / ".pytest_cache/cache"), _sentinel(root / "cortexmark/__pycache__/cached.pyc")]
    cleanup_local_artifacts(config)
    assert all(path.is_file() for path in protected)
    assert all(not path.exists() for path in caches)


def test_repeated_cleanup_with_absent_targets_succeeds(project) -> None:
    _root, config, _cfg = project
    cleanup_outputs(config)
    cleanup_outputs(config)
    cleanup_local_artifacts(config)


@pytest.mark.parametrize("selection", ["environment", "dotenv"])
@pytest.mark.parametrize("target", ["outputs", "local-artifacts"])
def test_cleanup_preserves_configured_session_root(project, monkeypatch, selection: str, target: str) -> None:
    root, config, cfg = project
    session_input = _sentinel(root / "build/student-sessions/student/data/raw/paper.txt")
    first_output = _sentinel(root / "outputs/raw_md/result.md")
    cache = _sentinel(root / ".pytest_cache/cache")
    if selection == "environment":
        monkeypatch.setenv("CORTEXMARK_SESSIONS_DIR", "build/student-sessions")
    else:
        (root / ".env").write_text("SESSIONS_DIR=build/student-sessions\n", encoding="utf-8")
    if target == "outputs":
        cfg["paths"]["output_semantic_chunks"] = "../build"
        _save(config, cfg)
        with pytest.raises(ValueError):
            cleanup_outputs(config)
    else:
        cleanup_local_artifacts(config)
        assert not cache.exists()
    assert session_input.is_file()
    assert first_output.is_file()


def test_cache_cleanup_preserves_session_root_reached_through_alias(project, monkeypatch) -> None:
    root, config, _cfg = project
    session_input = _sentinel(root / "build/student-sessions/student/data/raw/paper.txt")
    alias = root / "session-alias"
    alias.symlink_to(root / "build/student-sessions", target_is_directory=True)
    monkeypatch.setenv("CORTEXMARK_SESSIONS_DIR", str(alias))
    cleanup_local_artifacts(config)
    assert session_input.is_file()
    assert alias.is_symlink()


def _make_cleanup(root: Path, config: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = ["make", "--file", str(REPO_ROOT / "Makefile"), "clean-outputs", f"PYTHON={sys.executable}"]
    if config is not None:
        command.append(f"CONFIG={config}")
    return subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("config_key", paths.CONFIG_ENV_KEYS)
@pytest.mark.parametrize("selection", ["environment", "dotenv"])
def test_make_cleanup_uses_runtime_config_discovery(project, monkeypatch, config_key: str, selection: str) -> None:
    root, _config, cfg = project
    custom_config = root / "student-configs/student config.yaml"
    custom_config.parent.mkdir()
    for key in OUTPUT_KEYS:
        cfg["paths"][key] = f"../selected/{key}"
    cfg["idempotency"]["manifest_file"] = "../selected/manifest.json"
    _save(custom_config, cfg)
    selected = _sentinel(root / "selected/output_raw_md/result.md")
    default = _sentinel(root / "outputs/raw_md/default.md")
    if selection == "environment":
        monkeypatch.setenv(config_key, str(custom_config.relative_to(root)))
    else:
        (root / ".env").write_text(f'{config_key}="{custom_config.relative_to(root)}"\n', encoding="utf-8")
    result = _make_cleanup(root)
    assert result.returncode == 0, result.stderr
    assert not selected.exists()
    assert default.is_file()


@pytest.mark.parametrize("selection", ["command-line", "environment"])
def test_make_explicit_config_takes_precedence_over_runtime_discovery(project, monkeypatch, selection: str) -> None:
    root, config, cfg = project
    discovered_config = config.with_name("discovered.yaml")
    for key in OUTPUT_KEYS:
        cfg["paths"][key] = f"../discovered/{key}"
    cfg["idempotency"]["manifest_file"] = "../discovered/manifest.json"
    _save(discovered_config, cfg)
    discovered = _sentinel(root / "discovered/output_raw_md/result.md")
    explicit = _sentinel(root / "outputs/raw_md/explicit.md")
    monkeypatch.setenv("CORTEXMARK_CONFIG", str(discovered_config))
    if selection == "environment":
        monkeypatch.setenv("CONFIG", str(config))
    result = _make_cleanup(root, config if selection == "command-line" else None)
    assert result.returncode == 0, result.stderr
    assert not explicit.exists()
    assert discovered.is_file()


def test_discovered_config_cannot_spoof_its_location_metadata(project, monkeypatch) -> None:
    root, _config, cfg = project
    custom_config = root / "student-configs/student.yaml"
    custom_config.parent.mkdir()
    cfg.update(__config_file__=str(root / "fake.yaml"), __config_dir__=str(root), __project_root__=str(root.parent))
    cfg["idempotency"]["manifest_file"] = "student.yaml"
    _save(custom_config, cfg)
    monkeypatch.setenv("CORTEXMARK_CONFIG", str(custom_config))
    output = _sentinel(root / "outputs/raw_md/result.md")
    before = custom_config.read_bytes()
    result = _make_cleanup(root)
    assert result.returncode != 0
    assert custom_config.read_bytes() == before
    assert output.is_file()


@pytest.mark.parametrize(
    "name",
    [
        "sessions/student/notes.md",
        "workspace/papers/metadata.json",
        "data/raw/student/notes.txt",
        "data/processed/student/table.csv",
        "outputs/result.json",
        "cortexmark-output/paper/raw.md",
        "examples/cortexmark-output/paper/raw.md",
        "examples/cortexmark-output/paper/ir.json",
        "examples/cortexmark-output/paper/rag.jsonl",
        "LITERATURE-EXAMPLE/notes.md",
        "paper.pdf",
        "paper.PDF",
        ".env.local",
    ],
)
def test_student_data_is_ignored_for_every_file_type(name: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", name], cwd=REPO_ROOT, capture_output=True, check=False
    )
    assert result.returncode == 0, name


@pytest.mark.parametrize("name", ["cortexmark/cleanup.py", "tests/test_cleanup.py", "configs/pipeline.yaml", "uv.lock"])
def test_portable_source_and_lock_are_not_ignored(name: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", name], cwd=REPO_ROOT, capture_output=True, check=False
    )
    assert result.returncode == 1, name
