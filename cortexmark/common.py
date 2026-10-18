"""Shared utilities for the CortexMark."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from cortexmark.paths import (
    CONFIG_ENV_KEYS,
    MANIFEST_ENV_KEYS,
    SOURCE_ID_ENV_KEYS,
    PathSettings,
    SessionPathSettings,
    build_path_settings,
    find_project_root,
    first_present,
    load_dotenv_values,
    merged_runtime_env,
    resolve_binary,
    resolve_manifest_file,
    resolve_portable_path,
    resolve_session_settings,
)

PROJECT_ROOT = find_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"
DEFAULT_SOURCE_ID = "default"
ENGINE_ENV_KEYS: tuple[str, ...] = ("PIPELINE_ENGINE", "CORTEXMARK_ENGINE")

# Keyed by resolved config-file path -> (mtime, cfg). A single unkeyed slot would
# serve a stale config across two in-process calls that resolve to different files
# (different cwd/env) or across an on-disk edit of the same file; see reset_config_cache.
_CONFIG_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def detect_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── Configuration and path settings ─────────────────────────────────────────


def runtime_environ(cfg: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return merged runtime env (process env overrides .env) for the active project root."""
    project_root = (
        Path(str(cfg.get("__project_root__"))).resolve()
        if cfg and cfg.get("__project_root__")
        else find_project_root(start=Path.cwd())
    )
    return merged_runtime_env(project_root)


def runtime_env_value(*keys: str, cfg: Mapping[str, Any] | None = None) -> str | None:
    """Return the first configured runtime env value across process env and .env."""
    return first_present(runtime_environ(cfg), keys)


def _resolve_config_path(path: Path | None = None) -> tuple[Path, Path]:
    project_root = find_project_root(start=Path.cwd())
    merged_env = merged_runtime_env(project_root)

    if path is not None:
        config_path = resolve_portable_path(path, project_root=project_root, base_dir=Path.cwd(), environ=merged_env)
        return config_path, project_root

    env_value = first_present(merged_env, CONFIG_ENV_KEYS)
    if env_value:
        config_path = resolve_portable_path(
            env_value, project_root=project_root, base_dir=project_root, environ=merged_env
        )
        return config_path, project_root

    return (project_root / "configs" / "pipeline.yaml").resolve(), project_root


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and cache the pipeline YAML configuration.

    The cache is keyed by the *resolved* config-file path and its mtime, so a
    call that resolves to a different file (a different cwd, ``PROJECT_ROOT``,
    or ``CORTEXMARK_CONFIG`` than an earlier call in the same process) never
    returns another config's cached dict, and an on-disk edit to the same
    config file is picked up on the next call instead of serving a stale copy.
    """
    config_path, project_root = _resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cache_key = str(config_path)
    mtime = config_path.stat().st_mtime
    cached = _CONFIG_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}")
    cfg.setdefault("__config_file__", str(config_path))
    cfg.setdefault("__config_dir__", str(config_path.parent))
    cfg.setdefault("__project_root__", str(project_root))
    _CONFIG_CACHE[cache_key] = (mtime, cfg)
    return cfg


def reset_config_cache() -> None:
    """Clear cached config (useful for tests)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = {}


def get_path_settings(cfg: Mapping[str, Any] | None = None) -> PathSettings:
    """Return the central path settings object for the current runtime."""
    project_root = Path(str(cfg.get("__project_root__"))).resolve() if cfg and cfg.get("__project_root__") else None
    return build_path_settings(cfg, project_root=project_root)


def resolve_path(raw: str | Path, *, base_dir: Path | str | None = None) -> Path:
    """Resolve a path relative to *base_dir* or the discovered project root."""
    settings = get_path_settings()
    return resolve_portable_path(
        raw, project_root=settings.project_root, base_dir=Path(base_dir) if base_dir is not None else None
    )


def resolve_relative_path(raw: str | Path, base_dir: Path | str | None = None) -> Path:
    """Resolve *raw* relative to *base_dir* when given, otherwise project root."""
    return resolve_path(raw, base_dir=Path(base_dir) if base_dir is not None else None)


def config_base_dir(cfg: Mapping[str, Any]) -> Path:
    """Return the directory of the loaded config when known."""
    raw = cfg.get("__config_dir__")
    if raw:
        return Path(str(raw)).resolve()
    return get_path_settings(cfg).config_dir


def get_source_id(cfg: Mapping[str, Any], *, default: str = DEFAULT_SOURCE_ID) -> str:
    """Return the generic source identifier used for default input/output scoping."""
    project_root = Path(str(cfg.get("__project_root__"))).resolve() if cfg.get("__project_root__") else None
    dotenv = load_dotenv_values(project_root or find_project_root())
    merged_env = {**dotenv, **os.environ}
    env_source = first_present(merged_env, SOURCE_ID_ENV_KEYS)
    if env_source:
        return env_source
    source_id = cfg.get("source_id")
    if source_id:
        return str(source_id)
    return default


def resolve_configured_path(cfg: Mapping[str, Any], key: str, fallback: str) -> Path:
    """Resolve a named config path using env/.env/config/default precedence."""
    settings = get_path_settings(cfg)
    mapping = {
        "data_raw": settings.raw_data_dir,
        "output_raw_md": settings.raw_md_dir,
        "output_cleaned_md": settings.cleaned_md_dir,
        "output_chunks": settings.chunks_dir,
        "output_quality": settings.quality_dir,
        "output_semantic_chunks": settings.semantic_chunks_dir,
    }
    if key in mapping:
        return mapping[key]

    paths = cfg.get("paths", {})
    raw = paths.get(key) if isinstance(paths, Mapping) else None
    if raw:
        return resolve_relative_path(str(raw), base_dir=config_base_dir(cfg))
    return resolve_relative_path(fallback, base_dir=settings.project_root)


def resolve_output_subdir(cfg: Mapping[str, Any], *parts: str) -> Path:
    """Resolve a path under the portable outputs root."""
    settings = get_path_settings(cfg)
    return (settings.outputs_dir.joinpath(*parts)).resolve()


def get_session_path_settings(cfg: Mapping[str, Any], session_name: str) -> SessionPathSettings:
    """Return the resolved directory layout for ``session_name``."""
    project_root = Path(str(cfg.get("__project_root__"))).resolve() if cfg and cfg.get("__project_root__") else None
    return resolve_session_settings(session_name, cfg, project_root=project_root)


def resolve_manifest_path(cfg: Mapping[str, Any], *, session_name: str | None = None) -> Path:
    """Return the manifest path with optional session scoping."""
    if session_name:
        return get_session_path_settings(cfg, session_name).manifest_path
    manifest_path = resolve_manifest_file(cfg, project_root=get_path_settings(cfg).project_root)
    return manifest_path.resolve()


def resolve_quality_dir(cfg: Mapping[str, Any], *, session_name: str | None = None) -> Path:
    """Return output quality directory, optionally scoped by session."""
    if session_name:
        return get_session_path_settings(cfg, session_name).quality_dir
    quality_dir = get_path_settings(cfg).quality_dir
    return quality_dir.resolve()


def resolve_quality_report_path(
    cfg: Mapping[str, Any],
    filename: str,
    *,
    session_name: str | None = None,
) -> Path:
    """Return a quality report output path with optional session scoping."""
    return (resolve_quality_dir(cfg, session_name=session_name) / filename).resolve()


def resolve_plugin_dir(cfg: Mapping[str, Any] | None = None) -> Path:
    """Return the plugin discovery directory."""
    return get_path_settings(cfg).plugin_dir.resolve()


# ── Logging ──────────────────────────────────────────────────────────────────


def setup_logging(name: str, cfg: dict[str, Any] | None = None) -> logging.Logger:
    """Create a configured logger for a pipeline script."""
    log_cfg = (cfg or {}).get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    datefmt = log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)
    return logger


# ── File-system helpers ──────────────────────────────────────────────────────


def mirror_directory_tree(input_root: Path, output_root: Path) -> Path:
    """Replicate the directory tree of *input_root* under *output_root*.

    Returns the created target root (``output_root / input_root.name``).
    """
    target_root = output_root / input_root.name
    target_root.mkdir(parents=True, exist_ok=True)
    for directory in sorted(p for p in input_root.rglob("*") if p.is_dir()):
        (target_root / directory.relative_to(input_root)).mkdir(parents=True, exist_ok=True)
    return target_root


def file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def config_fingerprint(config: object) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Manifest (idempotency) ──────────────────────────────────────────────────


class Manifest:
    """Track source hashes, stage fingerprints, and outputs for idempotent stages."""

    def __init__(self, path: Path, *, stage: str | None = None) -> None:
        self.path = path
        self._stage = stage
        self._data: dict[str, object] = {}
        if path.exists():
            try:
                with path.open(encoding="utf-8") as fh:
                    loaded = json.load(fh)
            except (OSError, UnicodeError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                self._data = loaded

    def scoped(self, stage: str) -> Manifest:
        """Return a view of this manifest namespaced to *stage*.

        The view shares the same backing store and file path as the parent
        manifest -- ``save()`` on either persists both -- but keys every
        record as ``"{stage}:{source}"`` instead of the bare source path.
        Use this whenever more than one stage consumes the same source file
        (e.g. chunk and semantic-chunk both read the cleaned .md) so their
        idempotency records do not overwrite one another.
        """
        view = Manifest.__new__(Manifest)
        view.path = self.path
        view._stage = stage
        view._data = self._data
        return view

    @staticmethod
    def _output_paths(outputs: Iterable[Path]) -> list[str]:
        return [str(path.resolve()) for path in outputs]

    def _key(self, source: Path) -> str:
        return f"{self._stage}:{source}" if self._stage else str(source)

    def needs_update(
        self,
        source: Path,
        *,
        fingerprint: str = "",
        outputs: Iterable[Path] = (),
    ) -> bool:
        """Return True when the source, stage fingerprint, or outputs changed."""
        key = self._key(source)
        record = self._data.get(key)
        if not isinstance(record, dict):
            return True

        current_hash = file_hash(source)
        if record.get("hash") != current_hash or record.get("fingerprint") != fingerprint:
            return True

        recorded_outputs = record.get("outputs")
        if not isinstance(recorded_outputs, list) or not all(isinstance(path, str) for path in recorded_outputs):
            return True

        expected_outputs = self._output_paths(outputs)
        if expected_outputs and recorded_outputs != expected_outputs:
            return True
        return any(not Path(path).exists() for path in recorded_outputs)

    def record(
        self,
        source: Path,
        *,
        fingerprint: str = "",
        outputs: Iterable[Path] = (),
    ) -> None:
        """Store the current source hash, stage fingerprint, and output paths."""
        self._data[self._key(source)] = {
            "hash": file_hash(source),
            "fingerprint": fingerprint,
            "outputs": self._output_paths(outputs),
        }

    def save(self) -> None:
        """Persist the manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, sort_keys=True)
            fh.write("\n")


__all__ = [
    "CONFIG_ENV_KEYS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_SOURCE_ID",
    "ENGINE_ENV_KEYS",
    "MANIFEST_ENV_KEYS",
    "PROJECT_ROOT",
    "Manifest",
    "PathSettings",
    "config_base_dir",
    "config_fingerprint",
    "detect_device",
    "file_hash",
    "find_project_root",
    "get_path_settings",
    "get_session_path_settings",
    "get_source_id",
    "load_config",
    "load_dotenv_values",
    "merged_runtime_env",
    "mirror_directory_tree",
    "reset_config_cache",
    "resolve_binary",
    "resolve_configured_path",
    "resolve_manifest_path",
    "resolve_output_subdir",
    "resolve_path",
    "resolve_plugin_dir",
    "resolve_portable_path",
    "resolve_quality_dir",
    "resolve_quality_report_path",
    "resolve_relative_path",
    "runtime_env_value",
    "runtime_environ",
    "setup_logging",
]
