"""Central path and binary resolution helpers for portable CortexMark runtime behavior."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_PROJECT_ROOT", "PROJECT_ROOT")
CONFIG_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_CONFIG", "PIPELINE_CONFIG", "CONFIG_PATH")
DATA_DIR_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_DATA_DIR", "DATA_DIR")
RAW_DATA_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_RAW_DATA_DIR", "RAW_DATA_DIR")
PROCESSED_DATA_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_PROCESSED_DATA_DIR", "PROCESSED_DATA_DIR")
ARTIFACTS_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_ARTIFACTS_DIR", "ARTIFACTS_DIR")
OUTPUT_DIR_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_OUTPUT_DIR", "OUTPUT_DIR")
RAW_MD_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_OUTPUT_RAW_MD", "OUTPUT_RAW_MD")
CLEANED_MD_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_OUTPUT_CLEANED_MD", "OUTPUT_CLEANED_MD")
CHUNKS_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_OUTPUT_CHUNKS", "OUTPUT_CHUNKS")
REPORTS_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_REPORT_DIR", "REPORT_DIR")
SEMANTIC_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_OUTPUT_SEMANTIC_CHUNKS", "OUTPUT_SEMANTIC_CHUNKS")
SESSIONS_DIR_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_SESSIONS_DIR", "SESSIONS_DIR")
LOG_DIR_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_LOG_DIR", "LOG_DIR")
CHECKPOINT_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_CHECKPOINT_DIR", "CHECKPOINT_DIR")
CACHE_DIR_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_CACHE_DIR", "CACHE_DIR")
TEMP_DIR_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_TEMP_DIR", "TEMP_DIR")
MODEL_DIR_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_MODEL_DIR", "MODEL_DIR")
EXTERNAL_BIN_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_EXTERNAL_BIN_DIR", "EXTERNAL_BIN_DIR")
PLUGIN_DIR_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_PLUGIN_DIR", "PLUGIN_DIR")
TEST_RESOURCES_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_TEST_RESOURCES_DIR", "TEST_RESOURCES_DIR")
MANIFEST_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_MANIFEST_FILE", "MANIFEST_FILE")
SOURCE_ID_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_SOURCE_ID", "SOURCE_ID")
ENV_FILE_ENV_KEYS: tuple[str, ...] = ("CORTEXMARK_ENV_FILE", "ENV_FILE")

_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git", "configs/pipeline.yaml")
_PATH_TOKEN_RE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")


@dataclass(frozen=True)
class PathSettings:
    """Resolved path settings for portable runtime behavior."""

    project_root: Path
    src_root: Path
    config_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    artifacts_dir: Path
    outputs_dir: Path
    raw_md_dir: Path
    cleaned_md_dir: Path
    chunks_dir: Path
    reports_dir: Path
    quality_dir: Path
    semantic_chunks_dir: Path
    sessions_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    cache_dir: Path
    temp_dir: Path
    model_dir: Path
    external_bin_dir: Path | None
    plugin_dir: Path
    tests_resources_dir: Path

    def ensure_directories(self, *paths: Path) -> None:
        """Create the supplied directories when they are missing."""
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SessionPathSettings:
    """Resolved per-session directory layout rooted under ``sessions/<session_name>``."""

    session_name: str
    session_dir_name: str
    session_root: Path
    data_dir: Path
    raw_data_dir: Path
    outputs_dir: Path
    raw_md_dir: Path
    cleaned_md_dir: Path
    chunks_dir: Path
    quality_dir: Path
    semantic_chunks_dir: Path
    manifest_path: Path

    def ensure_directories(self) -> None:
        """Create the per-session directories when they do not exist."""
        for directory in (
            self.session_root,
            self.data_dir,
            self.raw_data_dir,
            self.outputs_dir,
            self.raw_md_dir,
            self.cleaned_md_dir,
            self.chunks_dir,
            self.quality_dir,
            self.semantic_chunks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def first_present(mapping: Mapping[str, str], keys: Iterable[str]) -> str | None:
    """Return the first non-empty value for *keys* from *mapping*."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _candidate_roots(start: Path) -> list[Path]:
    return [start, *start.parents]


def _looks_like_root(candidate: Path) -> bool:
    return any((candidate / marker).exists() for marker in _ROOT_MARKERS)


def find_project_root(start: Path | None = None, *, environ: Mapping[str, str] | None = None) -> Path:
    """Discover the project root from environment override or filesystem markers."""
    env_map = environ if environ is not None else os.environ
    override = first_present(env_map, PROJECT_ROOT_ENV_KEYS)
    if override:
        return Path(override).expanduser().resolve()

    search_starts: list[Path] = []
    if start is not None:
        search_starts.append(start.resolve())
    search_starts.extend([Path.cwd().resolve(), Path(__file__).resolve().parent])

    seen: set[Path] = set()
    for search_start in search_starts:
        for candidate in _candidate_roots(search_start):
            if candidate in seen:
                continue
            seen.add(candidate)
            if _looks_like_root(candidate):
                return candidate.resolve()

    return Path(__file__).resolve().parents[1]


def load_dotenv_values(project_root: Path, *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Parse a lightweight .env file without mutating the process environment."""
    env_map = environ if environ is not None else os.environ
    env_file_raw = first_present(env_map, ENV_FILE_ENV_KEYS)
    dotenv_path = Path(env_file_raw).expanduser() if env_file_raw else project_root / ".env"
    if not dotenv_path.is_absolute():
        dotenv_path = (project_root / dotenv_path).resolve()
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        stripped = value.strip()
        if stripped and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
            stripped = stripped[1:-1]
        values[key] = stripped
    return values


def merged_runtime_env(project_root: Path, *, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a merged env mapping where process env overrides .env values."""
    process_env = dict(environ if environ is not None else os.environ)
    dotenv = load_dotenv_values(project_root, environ=process_env)
    merged = {**dotenv, **process_env}
    merged.setdefault("PROJECT_ROOT", str(project_root))
    merged.setdefault("CORTEXMARK_PROJECT_ROOT", str(project_root))
    return merged


def expand_runtime_tokens(raw: str, variables: Mapping[str, str]) -> str:
    """Expand ${VAR}, $VAR, and ~ using the merged runtime environment."""

    def repl(match: re.Match[str]) -> str:
        key = match.group("braced") or match.group("plain")
        return variables.get(key, match.group(0))

    expanded = _PATH_TOKEN_RE.sub(repl, raw)
    return os.path.expanduser(expanded)


def sanitize_session_name(name: str, *, default: str = "session") -> str:
    """Return a cross-platform-safe directory name derived from a session name."""
    collapsed = " ".join(name.strip().split())
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", collapsed).rstrip(" .")
    return sanitized or default


def resolve_portable_path(
    raw: str | Path,
    *,
    project_root: Path,
    base_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a path using cwd/config/workspace aware rules."""
    path_like = (
        raw
        if isinstance(raw, Path)
        else Path(expand_runtime_tokens(str(raw), merged_runtime_env(project_root, environ=environ)))
    )
    if path_like.is_absolute():
        return path_like.resolve()
    base = base_dir.resolve() if base_dir is not None else project_root.resolve()
    return (base / path_like).resolve()


def _select_configured_path(
    *,
    direct_env_keys: Iterable[str],
    root_env_keys: Iterable[str] = (),
    config_value: str | None = None,
    config_base_dir: Path,
    project_root: Path,
    default_relative: str,
    parent_default: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    merged_env = merged_runtime_env(project_root, environ=environ)
    direct_raw = first_present(merged_env, direct_env_keys)
    if direct_raw:
        return resolve_portable_path(direct_raw, project_root=project_root, base_dir=project_root, environ=merged_env)

    parent_override_raw = first_present(merged_env, root_env_keys)
    if parent_override_raw:
        parent_override = resolve_portable_path(
            parent_override_raw,
            project_root=project_root,
            base_dir=project_root,
            environ=merged_env,
        )
        return (parent_override / default_relative).resolve()

    if config_value:
        return resolve_portable_path(
            config_value, project_root=project_root, base_dir=config_base_dir, environ=merged_env
        )

    default_parent = parent_default if parent_default is not None else project_root
    return (default_parent / default_relative).resolve()


def build_path_settings(
    cfg: Mapping[str, object] | None = None,
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> PathSettings:
    """Build the repo-wide central path settings object."""
    effective_root = project_root or find_project_root(environ=environ)
    config_dir = (
        Path(str(cfg.get("__config_dir__"))).resolve()
        if cfg and cfg.get("__config_dir__")
        else effective_root.resolve()
    )
    paths_cfg = cfg.get("paths", {}) if cfg else {}
    if not isinstance(paths_cfg, Mapping):
        paths_cfg = {}

    data_dir = _select_configured_path(
        direct_env_keys=DATA_DIR_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="data",
        environ=environ,
    )
    raw_data_dir = _select_configured_path(
        direct_env_keys=RAW_DATA_ENV_KEYS,
        root_env_keys=DATA_DIR_ENV_KEYS,
        config_value=str(paths_cfg.get("data_raw")) if paths_cfg.get("data_raw") else None,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="raw",
        parent_default=data_dir,
        environ=environ,
    )
    processed_data_dir = _select_configured_path(
        direct_env_keys=PROCESSED_DATA_ENV_KEYS,
        root_env_keys=DATA_DIR_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="processed",
        parent_default=data_dir,
        environ=environ,
    )
    outputs_dir = _select_configured_path(
        direct_env_keys=OUTPUT_DIR_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="outputs",
        environ=environ,
    )
    raw_md_dir = _select_configured_path(
        direct_env_keys=RAW_MD_ENV_KEYS,
        root_env_keys=OUTPUT_DIR_ENV_KEYS,
        config_value=str(paths_cfg.get("output_raw_md")) if paths_cfg.get("output_raw_md") else None,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="raw_md",
        parent_default=outputs_dir,
        environ=environ,
    )
    cleaned_md_dir = _select_configured_path(
        direct_env_keys=CLEANED_MD_ENV_KEYS,
        root_env_keys=OUTPUT_DIR_ENV_KEYS,
        config_value=str(paths_cfg.get("output_cleaned_md")) if paths_cfg.get("output_cleaned_md") else None,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="cleaned_md",
        parent_default=outputs_dir,
        environ=environ,
    )
    chunks_dir = _select_configured_path(
        direct_env_keys=CHUNKS_ENV_KEYS,
        root_env_keys=OUTPUT_DIR_ENV_KEYS,
        config_value=str(paths_cfg.get("output_chunks")) if paths_cfg.get("output_chunks") else None,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="chunks",
        parent_default=outputs_dir,
        environ=environ,
    )
    reports_dir = _select_configured_path(
        direct_env_keys=REPORTS_ENV_KEYS,
        root_env_keys=OUTPUT_DIR_ENV_KEYS,
        config_value=str(paths_cfg.get("output_quality")) if paths_cfg.get("output_quality") else None,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="quality",
        parent_default=outputs_dir,
        environ=environ,
    )
    semantic_chunks_dir = _select_configured_path(
        direct_env_keys=SEMANTIC_ENV_KEYS,
        root_env_keys=OUTPUT_DIR_ENV_KEYS,
        config_value=str(paths_cfg.get("output_semantic_chunks")) if paths_cfg.get("output_semantic_chunks") else None,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="semantic_chunks",
        parent_default=outputs_dir,
        environ=environ,
    )
    sessions_dir = _select_configured_path(
        direct_env_keys=SESSIONS_DIR_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="sessions",
        environ=environ,
    )
    logs_dir = _select_configured_path(
        direct_env_keys=LOG_DIR_ENV_KEYS,
        root_env_keys=OUTPUT_DIR_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="logs",
        parent_default=outputs_dir,
        environ=environ,
    )
    artifacts_dir = _select_configured_path(
        direct_env_keys=ARTIFACTS_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="artifacts",
        environ=environ,
    )
    checkpoints_dir = _select_configured_path(
        direct_env_keys=CHECKPOINT_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="checkpoints",
        environ=environ,
    )
    cache_dir = _select_configured_path(
        direct_env_keys=CACHE_DIR_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative=str(Path(".cache") / "cortexmark"),
        environ=environ,
    )
    temp_dir = _select_configured_path(
        direct_env_keys=TEMP_DIR_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative=str(Path("tmp") / "cortexmark"),
        environ=environ,
    )
    model_dir = _select_configured_path(
        direct_env_keys=MODEL_DIR_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="models",
        environ=environ,
    )
    plugin_dir = _select_configured_path(
        direct_env_keys=PLUGIN_DIR_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative="plugins",
        environ=environ,
    )
    tests_resources_dir = _select_configured_path(
        direct_env_keys=TEST_RESOURCES_ENV_KEYS,
        config_base_dir=config_dir,
        project_root=effective_root,
        default_relative=str(Path("tests") / "resources"),
        environ=environ,
    )
    external_bin_raw = first_present(merged_runtime_env(effective_root, environ=environ), EXTERNAL_BIN_ENV_KEYS)
    external_bin_dir = (
        resolve_portable_path(external_bin_raw, project_root=effective_root, base_dir=effective_root, environ=environ)
        if external_bin_raw
        else None
    )

    return PathSettings(
        project_root=effective_root.resolve(),
        src_root=(effective_root / "cortexmark").resolve(),
        config_dir=config_dir.resolve(),
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        artifacts_dir=artifacts_dir,
        outputs_dir=outputs_dir,
        raw_md_dir=raw_md_dir,
        cleaned_md_dir=cleaned_md_dir,
        chunks_dir=chunks_dir,
        reports_dir=reports_dir,
        quality_dir=reports_dir,
        semantic_chunks_dir=semantic_chunks_dir,
        sessions_dir=sessions_dir,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        cache_dir=cache_dir,
        temp_dir=temp_dir,
        model_dir=model_dir,
        external_bin_dir=external_bin_dir,
        plugin_dir=plugin_dir,
        tests_resources_dir=tests_resources_dir,
    )


def resolve_session_settings(
    session_name: str,
    cfg: Mapping[str, object] | None = None,
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> SessionPathSettings:
    """Resolve the portable directory layout for a named session."""
    settings = build_path_settings(cfg, project_root=project_root, environ=environ)
    session_dir_name = sanitize_session_name(session_name)
    session_root = (settings.sessions_dir / session_dir_name).resolve()
    outputs_dir = (session_root / "outputs").resolve()
    quality_dir = (outputs_dir / "quality").resolve()
    return SessionPathSettings(
        session_name=session_name,
        session_dir_name=session_dir_name,
        session_root=session_root,
        data_dir=(session_root / "data").resolve(),
        raw_data_dir=(session_root / "data" / "raw").resolve(),
        outputs_dir=outputs_dir,
        raw_md_dir=(outputs_dir / "raw_md").resolve(),
        cleaned_md_dir=(outputs_dir / "cleaned_md").resolve(),
        chunks_dir=(outputs_dir / "chunks").resolve(),
        quality_dir=quality_dir,
        semantic_chunks_dir=(outputs_dir / "semantic_chunks").resolve(),
        manifest_path=(outputs_dir / ".manifest.json").resolve(),
    )


def _is_executable_file(path: Path) -> bool:
    """Return True when *path* exists and is executable (or a Windows executable candidate)."""
    if not path.exists() or path.is_dir():
        return False
    if os.name == "nt":
        return True
    return os.access(path, os.X_OK)


def resolve_manifest_file(
    cfg: Mapping[str, object] | None = None,
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the manifest path using env > .env > config > defaults."""
    effective_root = project_root or find_project_root(environ=environ)
    config_dir = (
        Path(str(cfg.get("__config_dir__"))) if cfg and cfg.get("__config_dir__") else effective_root / "configs"
    )
    merged_env = merged_runtime_env(effective_root, environ=environ)
    manifest_env = first_present(merged_env, MANIFEST_ENV_KEYS)
    if manifest_env:
        return resolve_portable_path(
            manifest_env, project_root=effective_root, base_dir=effective_root, environ=merged_env
        )

    idem_cfg = cfg.get("idempotency", {}) if cfg else {}
    manifest_cfg = idem_cfg.get("manifest_file") if isinstance(idem_cfg, Mapping) else None
    if manifest_cfg:
        return resolve_portable_path(
            str(manifest_cfg), project_root=effective_root, base_dir=config_dir, environ=merged_env
        )

    settings = build_path_settings(cfg, project_root=effective_root, environ=merged_env)
    return (settings.outputs_dir / ".manifest.json").resolve()


def resolve_binary(
    name: str,
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    explicit: str | Path | None = None,
    env_keys: Iterable[str] = (),
    required: bool = False,
) -> str | None:
    """Resolve an external binary via explicit path, env vars, external_bin_dir, then PATH."""
    effective_root = project_root or find_project_root(environ=environ)
    merged_env = merged_runtime_env(effective_root, environ=environ)

    candidates: list[str | Path] = []
    if explicit:
        candidates.append(explicit)
    env_value = first_present(merged_env, tuple(env_keys)) if env_keys else None
    if env_value:
        candidates.append(env_value)

    settings = build_path_settings(project_root=effective_root, environ=merged_env)
    if settings.external_bin_dir is not None:
        candidates.append(settings.external_bin_dir / name)
        if os.name == "nt" and not name.lower().endswith(".exe"):
            candidates.append(settings.external_bin_dir / f"{name}.exe")

    for candidate in candidates:
        resolved = resolve_portable_path(
            candidate, project_root=effective_root, base_dir=effective_root, environ=merged_env
        )
        if _is_executable_file(resolved):
            return str(resolved)

    found = shutil.which(name, path=merged_env.get("PATH"))
    if found:
        return found

    if required:
        searched = ", ".join(env_keys) if env_keys else "PATH"
        raise FileNotFoundError(
            f"Required executable '{name}' was not found. Set one of [{searched}] or EXTERNAL_BIN_DIR."
        )
    return None
