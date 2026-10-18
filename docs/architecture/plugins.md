# Plugin System

!!! warning "Standalone and experimental — not wired into the pipeline"
    `PluginRegistry` and `PluginBase` (`cortexmark/plugin.py`) are a self-contained hook
    mechanism. Neither `cortexmark run` (`cli.py`) nor the legacy `python -m
    cortexmark.run_pipeline` runner ever constructs a `PluginRegistry` or calls
    `run_hook`. Writing a plugin file has **no effect on a real conversion run** today —
    the hooks below only fire if you call `PluginRegistry().discover(...)` and
    `run_hook(...)` yourself, directly, from your own code. Treat this page as documentation
    for an experimental library surface, not for a live pipeline extension point.

The plugin module provides file-based hook discovery and dispatch that a caller can use without changing core modules.

## Hook Points

Plugins can override these hooks:

- `pre_convert`
- `post_convert`
- `pre_clean`
- `post_clean`
- `pre_chunk`
- `post_chunk`
- `post_pipeline`

Each hook receives and returns a single mutable context object:

```python
dict[str, Any]
```

## Creating a Plugin

```python
from cortexmark.plugin import PluginBase


class StatsPlugin(PluginBase):
    name = "stats"
    description = "Collects simple stage metrics"

    def post_clean(self, context: dict[str, object]) -> dict[str, object]:
        output_path = context.get("output_path")
        context["stats_enabled"] = True
        context["cleaned_output"] = str(output_path) if output_path else ""
        return context
```

## Discovery and Reporting

Plugins are discovered from `plugins/*.py` (excluding files that start with `_`):

```bash
python -m cortexmark.plugin --plugin-dir plugins/ --list
python -m cortexmark.plugin --plugin-dir plugins/ --output plugins.json
```

The JSON report contains plugin metadata and overridden hook names.
