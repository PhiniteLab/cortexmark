"""Conservative cleanup for the repository's two Make cleanup targets."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cortexmark.common import (
    _resolve_config_path,
    get_path_settings,
    load_config,
    resolve_configured_path,
    resolve_manifest_path,
)
from cortexmark.paths import (
    CHUNKS_ENV_KEYS,
    CLEANED_MD_ENV_KEYS,
    MANIFEST_ENV_KEYS,
    OUTPUT_DIR_ENV_KEYS,
    RAW_MD_ENV_KEYS,
    REPORTS_ENV_KEYS,
    SEMANTIC_ENV_KEYS,
    expand_runtime_tokens,
    first_present,
    merged_runtime_env,
)

_OUTPUTS = (
    ("output_raw_md", "raw_md", RAW_MD_ENV_KEYS),
    ("output_cleaned_md", "cleaned_md", CLEANED_MD_ENV_KEYS),
    ("output_chunks", "chunks", CHUNKS_ENV_KEYS),
    ("output_quality", "quality", REPORTS_ENV_KEYS),
    ("output_semantic_chunks", "semantic_chunks", SEMANTIC_ENV_KEYS),
)
_CACHE_NAMES = (
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "cortexmark.egg-info",
    "build",
    "dist",
    "site",
    "htmlcov",
    ".coverage",
    "coverage.xml",
)
_SOURCE_NAMES = (
    "cortexmark",
    "tests",
    "docs",
    "REPO-INFO",
    "configs",
    "workspace",
    "sessions",
    "LITERATURE-EXAMPLE",
    "tasks",
    "tasks-archive",
    ".github",
    "README.md",
    "LICENSE",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "mkdocs.yml",
    "pyrightconfig.json",
    ".editorconfig",
    ".gitattributes",
    ".gitmodules",
    ".gitignore",
    ".pre-commit-config.yaml",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CHANGELOG.md",
)


def _overlaps(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _context(config_path: Path | None) -> tuple[Path, dict[str, Any]]:
    selected_path, root = _resolve_config_path(config_path)
    cfg = dict(load_config(selected_path))
    # Internal location metadata must describe the selected file, not YAML values.
    cfg.update(
        __project_root__=str(root),
        __config_file__=str(selected_path),
        __config_dir__=str(selected_path.parent),
    )
    return root, cfg


def _output_paths(root: Path, cfg: dict[str, Any]) -> tuple[list[Path], Path]:
    """Retain lexical paths so cleanup can unlink a symlink without following it.

    Match the runtime's environment/config/default precedence, then cross-check
    against its resolver so future path-policy changes fail closed here.
    """
    env = merged_runtime_env(root)
    base = Path(cfg["__config_dir__"])
    paths = cfg.get("paths", {})
    paths = paths if isinstance(paths, Mapping) else {}

    def lexical(raw: str, anchor: Path) -> Path:
        path = Path(expand_runtime_tokens(raw, env))
        return path if path.is_absolute() else anchor / path

    output_root = lexical(first_present(env, OUTPUT_DIR_ENV_KEYS) or "outputs", root)
    directories = []
    for key, name, env_keys in _OUTPUTS:
        direct = first_present(env, env_keys)
        if direct:
            path = lexical(direct, root)
        elif first_present(env, OUTPUT_DIR_ENV_KEYS):
            path = output_root / name
        elif paths.get(key):
            path = lexical(str(paths[key]), base)
        else:
            path = output_root / name
        if path.resolve() != resolve_configured_path(cfg, key, f"outputs/{name}"):
            raise ValueError(f"Cleanup path policy differs from runtime for {key}")
        directories.append(path)

    idem = cfg.get("idempotency", {})
    idem = idem if isinstance(idem, Mapping) else {}
    manifest_env = first_present(env, MANIFEST_ENV_KEYS)
    if manifest_env:
        manifest = lexical(manifest_env, root)
    elif idem.get("manifest_file"):
        manifest = lexical(str(idem["manifest_file"]), base)
    else:
        manifest = output_root / ".manifest.json"
    if manifest.resolve() != resolve_manifest_path(cfg):
        raise ValueError("Cleanup manifest policy differs from runtime")
    return directories, manifest


def _protected_paths(root: Path, cfg: dict[str, Any]) -> list[Path]:
    settings = get_path_settings(cfg)
    protected = [(root / name).resolve() for name in (".git", ".venv", "venv", "env", "ENV", ".env", ".envrc")] + [
        settings.raw_data_dir,
        settings.sessions_dir,
        Path(cfg["__config_file__"]),
    ]
    protected.extend(path.resolve() for pattern in (".env.*", ".venv*", "venv*") for path in root.glob(pattern))
    protected.extend(path.resolve() for path in root.iterdir() if (path / "pyvenv.cfg").is_file())
    return protected


def _validate_path(path: Path, root: Path, protected: list[Path]) -> None:
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Refusing cleanup outside the project or at its root: {path}")
    if any(_overlaps(resolved, item) for item in protected):
        raise ValueError(f"Refusing cleanup of a protected path: {path}")
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f"Refusing cleanup through a directory symlink: {path}")


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        # shutil.rmtree does not follow directory symlinks inside the tree.
        shutil.rmtree(path)
    print(f"Removed {path}")


def cleanup_outputs(config_path: Path | None = None) -> None:
    """Remove only configured output directories and manifest after full preflight.

    External outputs, input/source overlap and symlink ancestors are refused.
    A terminal symlink to an otherwise safe target is unlinked, never traversed.
    """
    root, cfg = _context(config_path)
    directories, manifest = _output_paths(root, cfg)
    protected = _protected_paths(root, cfg) + [(root / name).resolve() for name in _SOURCE_NAMES]
    targets = [*directories, manifest]
    for index, path in enumerate(targets):
        _validate_path(path, root, protected)
        if any(_overlaps(path.resolve(), previous.resolve()) for previous in targets[:index]):
            raise ValueError(f"Refusing overlapping cleanup targets: {path}")
        if path.exists() and not path.is_symlink() and path.is_dir() != (path != manifest):
            raise ValueError(f"Unexpected cleanup target type: {path}")
    for path in targets:
        if path.exists() or path.is_symlink():
            _remove(path)


def cleanup_local_artifacts(config_path: Path | None = None) -> None:
    """Remove known caches/build artifacts while preserving configured data/results."""
    root, cfg = _context(config_path)
    directories, manifest = _output_paths(root, cfg)
    protected = _protected_paths(root, cfg) + [p.resolve() for p in [*directories, manifest]]
    protected.extend((root / name).resolve() for name in ("data", "sessions", "workspace", "cortexmark-output"))
    candidates = [root / name for name in _CACHE_NAMES]
    for source in (root / "cortexmark", root / "tests"):
        if source.is_symlink() or any(_overlaps(source.resolve(), p) for p in protected):
            continue
        for current, dirs, _files in os.walk(source, followlinks=False):
            if "__pycache__" in dirs:
                candidates.append(Path(current) / "__pycache__")
                dirs.remove("__pycache__")
    targets = []
    for path in candidates:
        if not (path.exists() or path.is_symlink()):
            continue
        if any(_overlaps(path.resolve(), item) for item in protected):
            print(f"Skipped protected artifact {path}")
            continue
        _validate_path(path, root, protected)
        targets.append(path)
    for path in targets:
        _remove(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("outputs", "local-artifacts"))
    args = parser.parse_args()
    config = Path(os.environ["CONFIG"]) if "CONFIG" in os.environ else None
    try:
        if args.target == "outputs":
            cleanup_outputs(config)
        else:
            cleanup_local_artifacts(config)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cleanup refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
