"""Standalone experimental hooks for custom processing stages.

Provides:
  - PluginBase: base class for pipeline plugins
  - PluginRegistry: discover, register, and run plugins
  - Hook points: pre_convert, post_convert, pre_clean, post_clean,
    pre_chunk, post_chunk, post_pipeline
  - File-based plugin loading from a configurable directory

Plugins are Python modules placed in a ``plugins/`` directory that define
a subclass of ``PluginBase``. The pipeline runners do not invoke these hooks;
callers must load plugins and execute hooks explicitly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import load_config, resolve_plugin_dir, setup_logging

# ── Hook names ───────────────────────────────────────────────────────────────

VALID_HOOKS = frozenset(
    [
        "pre_convert",
        "post_convert",
        "pre_clean",
        "post_clean",
        "pre_chunk",
        "post_chunk",
        "post_pipeline",
    ]
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class PluginInfo:
    """Metadata about a registered plugin."""

    name: str
    description: str = ""
    hooks: list[str] = field(default_factory=list)
    source_file: str = ""
    enabled: bool = True


@dataclass
class HookResult:
    """Result of executing a hook."""

    plugin_name: str
    hook: str
    success: bool
    message: str = ""


# ── Plugin base class ────────────────────────────────────────────────────────


class PluginBase:
    """Base class for pipeline plugins.

    Override the desired hook methods in your subclass.
    """

    name: str = "unnamed"
    description: str = ""
    enabled: bool = True

    def pre_convert(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    def post_convert(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    def pre_clean(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    def post_clean(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    def pre_chunk(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    def post_chunk(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    def post_pipeline(self, context: dict[str, Any]) -> dict[str, Any]:
        return context

    def get_hooks(self) -> list[str]:
        """Return the list of hooks this plugin overrides."""
        hooks: list[str] = []
        for hook_name in VALID_HOOKS:
            method = getattr(self, hook_name, None)
            if method is not None and method.__func__ is not getattr(PluginBase, hook_name):
                hooks.append(hook_name)
        return hooks

    def info(self) -> PluginInfo:
        return PluginInfo(
            name=self.name,
            description=self.description,
            hooks=self.get_hooks(),
            enabled=self.enabled,
        )


# ── Registry ─────────────────────────────────────────────────────────────────


class PluginRegistry:
    """Discover, register, and execute pipeline plugins."""

    def __init__(self) -> None:
        self._plugins: list[PluginBase] = []
        self._log = logging.getLogger("plugins")

    @property
    def plugins(self) -> list[PluginBase]:
        return list(self._plugins)

    def register(self, plugin: PluginBase) -> None:
        """Register a plugin instance."""
        if not isinstance(plugin, PluginBase):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected PluginBase, got {type(plugin).__name__}")
        self._plugins.append(plugin)
        self._log.debug("registered plugin: %s", plugin.name)

    def discover(self, plugin_dir: Path) -> list[PluginInfo]:
        """Discover and load plugins from a directory.

        Each ``.py`` file in *plugin_dir* is loaded as a module; any
        ``PluginBase`` subclass found is instantiated and registered.
        """
        if not plugin_dir.is_dir():
            return []

        loaded: list[PluginInfo] = []
        for py_file in sorted(plugin_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"plugins.{py_file.stem}", py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, PluginBase) and attr is not PluginBase:
                        instance = attr()
                        info = instance.info()
                        info.source_file = str(py_file)
                        self.register(instance)
                        loaded.append(info)
            except Exception as exc:
                self._log.warning("failed to load plugin %s: %s", py_file, exc)

        return loaded

    def run_hook(self, hook_name: str, context: dict[str, Any]) -> list[HookResult]:
        """Execute a hook across all registered plugins and update *context*."""
        if hook_name not in VALID_HOOKS:
            raise ValueError(f"Invalid hook: {hook_name}")

        caller_context = context
        results: list[HookResult] = []
        for plugin in self._plugins:
            if not getattr(plugin.info(), "enabled", True):
                continue
            method = getattr(plugin, hook_name, None)
            if method is None:
                continue
            try:
                context = method(context)
                results.append(HookResult(plugin_name=plugin.name, hook=hook_name, success=True))
            except Exception as exc:
                results.append(
                    HookResult(
                        plugin_name=plugin.name,
                        hook=hook_name,
                        success=False,
                        message=str(exc),
                    )
                )
                self._log.warning("plugin %s hook %s failed: %s", plugin.name, hook_name, exc)

        if context is not caller_context:
            caller_context.clear()
            caller_context.update(context)

        return results

    def list_plugins(self) -> list[PluginInfo]:
        """Return metadata for all registered plugins."""
        return [p.info() for p in self._plugins]


def write_plugin_report(infos: list[PluginInfo], output_path: Path) -> Path:
    """Write plugin registry report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "total_plugins": len(infos),
        "plugins": [asdict(i) for i in infos],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage pipeline plugins.")
    parser.add_argument(
        "--plugin-dir",
        type=Path,
        help="Directory containing plugin modules",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_plugins",
        help="List discovered plugins",
    )
    parser.add_argument("--output", type=Path, help="Path for plugin report (JSON)")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("plugins", cfg)

    plugin_dir = (args.plugin_dir or resolve_plugin_dir(cfg)).resolve()

    registry = PluginRegistry()
    infos = registry.discover(plugin_dir)
    log.info("discovered %d plugin(s) from %s", len(infos), plugin_dir)

    if args.list_plugins:
        for info in infos:
            log.info("  %s: %s (hooks: %s)", info.name, info.description, info.hooks)

    if args.output:
        write_plugin_report(infos, args.output)
        log.info("report -> %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
