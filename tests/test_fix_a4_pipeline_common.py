"""Regression tests for A4 (run_pipeline.py / common.py) audit fixes.

Covers:
  C05  nested --input under data_raw mis-wired convert vs clean output layout.
  C27  chunk vs semantic-chunk manifest key collision (shared bare-source-path key).
  C29  session-scoped analyze/validate fallback guard was a tautology.
  C30  run_pipeline.main did not catch ImportError (missing docling) like `cortexmark run`.
  C47  run_pipeline.main did not catch the broader failure classes `cortexmark run` catches.
  G16  common.load_config's cache could serve a stale config across in-process calls.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cortexmark.common import Manifest, load_config, reset_config_cache


def _write_pipeline_config(
    tmp_path: Path,
    *,
    idempotency_enabled: bool = False,
    engine: str = "markitdown",
    log_level: str = "WARNING",
) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "source_id: test\n"
        "paths:\n"
        f"  data_raw: {tmp_path / 'data'}\n"
        f"  output_raw_md: {tmp_path / 'raw_md'}\n"
        f"  output_cleaned_md: {tmp_path / 'cleaned_md'}\n"
        f"  output_chunks: {tmp_path / 'chunks'}\n"
        f"  output_quality: {tmp_path / 'quality'}\n"
        f"  output_semantic_chunks: {tmp_path / 'semantic_chunks'}\n"
        "convert:\n"
        f"  engine: {engine}\n"
        "idempotency:\n"
        f"  enabled: {'true' if idempotency_enabled else 'false'}\n"
        f"  manifest_file: {tmp_path / '.manifest.json'}\n"
        "logging:\n"
        f"  level: {log_level}\n",
        encoding="utf-8",
    )
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# C05 -- nested --input under data_raw
# ═══════════════════════════════════════════════════════════════════════════════


class TestNestedInputPathWiring:
    def test_two_level_nested_input_converts_then_cleans(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--input two levels below data_raw must produce a layout clean_tree can read.

        Before the fix, rel_label was the FULL path relative to data_raw ("a/b") while
        convert_tree (include_input_root_name=True) only ever uses the bare directory
        name ("b"), so clean aborted with FileNotFoundError even though convert had
        already written its output successfully.
        """
        from cortexmark import convert as convert_module
        from cortexmark import run_pipeline as pipeline_module

        cfg = _write_pipeline_config(tmp_path)
        nested_input = tmp_path / "data" / "a" / "b"
        nested_input.mkdir(parents=True)
        (nested_input / "paper.pdf").write_bytes(b"%PDF-fake")

        def fake_convert_pdf(input_path: Path, output_path: Path, **_kwargs: object) -> Path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("# Title\n\nBody text.\n", encoding="utf-8")
            return output_path

        monkeypatch.setattr(convert_module, "convert_pdf", fake_convert_pdf)

        rc = pipeline_module.main(
            [
                "--config",
                str(cfg),
                "--input",
                str(nested_input),
                "--stages",
                "convert",
                "clean",
            ]
        )
        assert rc == 0
        # convert writes under raw_md/<bare-name>; clean must read from the same tree.
        assert (tmp_path / "raw_md" / "b" / "paper.md").exists()
        assert (tmp_path / "cleaned_md" / "b" / "paper.md").exists()

    def test_one_level_input_still_works(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Control case: a single-level --input under data_raw was never broken."""
        from cortexmark import convert as convert_module
        from cortexmark import run_pipeline as pipeline_module

        cfg = _write_pipeline_config(tmp_path)
        one_level_input = tmp_path / "data" / "foo"
        one_level_input.mkdir(parents=True)
        (one_level_input / "paper.pdf").write_bytes(b"%PDF-fake")

        def fake_convert_pdf(input_path: Path, output_path: Path, **_kwargs: object) -> Path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("# Title\n\nBody text.\n", encoding="utf-8")
            return output_path

        monkeypatch.setattr(convert_module, "convert_pdf", fake_convert_pdf)

        rc = pipeline_module.main(
            [
                "--config",
                str(cfg),
                "--input",
                str(one_level_input),
                "--stages",
                "convert",
                "clean",
            ]
        )
        assert rc == 0
        assert (tmp_path / "cleaned_md" / "foo" / "paper.md").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# C27 -- chunk vs semantic-chunk manifest key collision
# ═══════════════════════════════════════════════════════════════════════════════


class TestManifestScoping:
    def test_scoped_views_isolate_keys_by_stage(self, tmp_path: Path) -> None:
        """Manifest.scoped() must key records by stage so two stages consuming the
        same source path never overwrite each other's idempotency record."""
        source = tmp_path / "cleaned.md"
        source.write_text("# Doc\n\nBody.\n", encoding="utf-8")
        chunk_out = tmp_path / "chunk_out.md"
        chunk_out.write_text("chunk\n", encoding="utf-8")
        sem_out = tmp_path / "sem_out.md"
        sem_out.write_text("sem\n", encoding="utf-8")

        root_manifest = Manifest(tmp_path / ".manifest.json")
        chunk_view = root_manifest.scoped("chunk")
        semantic_view = root_manifest.scoped("semantic_chunk")

        assert chunk_view.needs_update(source, fingerprint="chunk-fp")
        assert semantic_view.needs_update(source, fingerprint="semantic-fp")

        chunk_view.record(source, fingerprint="chunk-fp", outputs=[chunk_out])
        # Recording under "chunk" must not satisfy the "semantic_chunk" scope for the
        # identical source path (this is exactly the C27 collision).
        assert semantic_view.needs_update(source, fingerprint="semantic-fp")
        assert not chunk_view.needs_update(source, fingerprint="chunk-fp", outputs=[chunk_out])

        semantic_view.record(source, fingerprint="semantic-fp", outputs=[sem_out])
        assert not semantic_view.needs_update(source, fingerprint="semantic-fp", outputs=[sem_out])
        # The chunk scope must remain unaffected by the later semantic_chunk record.
        assert not chunk_view.needs_update(source, fingerprint="chunk-fp", outputs=[chunk_out])

        root_manifest.save()
        saved = json.loads((tmp_path / ".manifest.json").read_text(encoding="utf-8"))
        assert f"chunk:{source}" in saved
        assert f"semantic_chunk:{source}" in saved
        assert str(source) not in saved

    def test_unscoped_manifest_keeps_legacy_bare_path_key(self, tmp_path: Path) -> None:
        """Backward compatibility: a Manifest used without .scoped() keeps the original
        bare-source-path key format (convert/clean stages are unaffected by C27)."""
        source = tmp_path / "input.pdf"
        source.write_bytes(b"%PDF-fake")
        out = tmp_path / "out.md"
        out.write_text("out\n", encoding="utf-8")

        manifest = Manifest(tmp_path / ".manifest.json")
        manifest.record(source, fingerprint="fp", outputs=[out])
        manifest.save()
        saved = json.loads((tmp_path / ".manifest.json").read_text(encoding="utf-8"))
        assert str(source) in saved

    def test_pipeline_chunk_and_analyze_manifest_keys_do_not_collide(self, tmp_path: Path) -> None:
        """Integration check: run_pipeline's chunk and analyze stages must record
        independent manifest entries for the same cleaned .md when both run together."""
        from cortexmark import run_pipeline as pipeline_module

        cfg = _write_pipeline_config(tmp_path, idempotency_enabled=True)
        cleaned_dir = tmp_path / "cleaned_md" / "test"
        cleaned_dir.mkdir(parents=True)
        doc = cleaned_dir / "doc.md"
        doc.write_text("# Intro\n\nAlgorithm 1: Demo\n\nInput: x\nOutput: y\n", encoding="utf-8")

        rc = pipeline_module.main(["--config", str(cfg), "--stages", "chunk", "analyze"])
        assert rc == 0

        manifest_data = json.loads((tmp_path / ".manifest.json").read_text(encoding="utf-8"))
        doc_key = str(doc.resolve())
        assert f"chunk:{doc_key}" in manifest_data
        assert f"semantic_chunk:{doc_key}" in manifest_data
        assert doc_key not in manifest_data


# ═══════════════════════════════════════════════════════════════════════════════
# C29 -- session-scoped analyze/validate fallback guard was a tautology
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionAnalyzeValidateFallback:
    def test_analyze_falls_back_to_base_cleaned_when_session_cleaned_is_empty(self, tmp_path: Path) -> None:
        cfg = _write_pipeline_config(tmp_path, log_level="INFO")
        base_cleaned = tmp_path / "cleaned_md" / "test"
        base_cleaned.mkdir(parents=True)
        (base_cleaned / "doc.md").write_text(
            "# Intro\n\nAlgorithm 1: Demo\n\nInput: x\nOutput: y\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.run_pipeline",
                "--config",
                str(cfg),
                "--stages",
                "analyze",
                "--session-name",
                "sess-fallback",
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PROJECT_ROOT": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr
        assert "falling back to base cleaned path" in result.stderr
        quality_dir = tmp_path / "sessions" / "sess-fallback" / "outputs" / "quality"
        assert (quality_dir / "crossref_report.json").exists()

    def test_validate_falls_back_to_base_cleaned_when_session_cleaned_is_empty(self, tmp_path: Path) -> None:
        cfg = _write_pipeline_config(tmp_path, log_level="INFO")
        base_cleaned = tmp_path / "cleaned_md" / "test"
        base_cleaned.mkdir(parents=True)
        (base_cleaned / "doc.md").write_text("# Intro\n\nBody text with a citation [1].\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.run_pipeline",
                "--config",
                str(cfg),
                "--stages",
                "validate",
                "--session-name",
                "sess-fallback-2",
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PROJECT_ROOT": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr
        assert "falling back to base cleaned path" in result.stderr
        quality_dir = tmp_path / "sessions" / "sess-fallback-2" / "outputs" / "quality"
        assert (quality_dir / "formula_validation.json").exists()

    def test_analyze_stays_in_session_scope_when_session_cleaned_has_content(self, tmp_path: Path) -> None:
        """Control: when the session-scoped cleaned dir is NOT empty, no fallback fires
        and the session's own content is analyzed (not the base source_id content)."""
        cfg = _write_pipeline_config(tmp_path, log_level="INFO")
        base_cleaned = tmp_path / "cleaned_md" / "test"
        base_cleaned.mkdir(parents=True)
        (base_cleaned / "doc.md").write_text("# Base\n\nBase content.\n", encoding="utf-8")

        # Pre-create the session-scoped cleaned dir WITH content so the fallback must
        # not trigger (ensure_directories() alone would leave it present but empty).
        session_cleaned = tmp_path / "sessions" / "sess-has-content" / "outputs" / "cleaned_md"
        session_cleaned.mkdir(parents=True, exist_ok=True)
        (session_cleaned / "own.md").write_text("# Own\n\nSession-scoped content.\n", encoding="utf-8")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.run_pipeline",
                "--config",
                str(cfg),
                "--stages",
                "analyze",
                "--session-name",
                "sess-has-content",
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PROJECT_ROOT": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr
        assert "falling back to base cleaned path" not in result.stderr


# ═══════════════════════════════════════════════════════════════════════════════
# C30 / C47 -- run_pipeline.main under-caught exceptions vs `cortexmark run`
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunPipelineErrorHandling:
    def test_missing_docling_engine_returns_clean_error_not_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before the fix, an ImportError from _resolve_engine (explicit --engine
        docling on a docling-less install) escaped run_pipeline.main unhandled."""
        from cortexmark import convert as convert_module
        from cortexmark import run_pipeline as pipeline_module

        cfg = _write_pipeline_config(tmp_path, engine="docling")
        input_dir = tmp_path / "data" / "test"
        input_dir.mkdir(parents=True)
        (input_dir / "paper.pdf").write_bytes(b"%PDF-fake")

        monkeypatch.setattr(convert_module, "docling_available", lambda: False)

        # Must return 1, not raise -- proves the ImportError is now caught.
        rc = pipeline_module.main(
            [
                "--config",
                str(cfg),
                "--input",
                str(input_dir),
                "--stages",
                "convert",
                "--engine",
                "docling",
            ]
        )
        assert rc == 1

    def test_unexpected_runtime_error_in_a_stage_returns_clean_error_not_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A RuntimeError escaping a stage call (matches cli.run's caught tuple) must
        be reported cleanly by run_pipeline.main instead of propagating."""
        from cortexmark import chunk as chunk_module
        from cortexmark import run_pipeline as pipeline_module

        cfg = _write_pipeline_config(tmp_path)
        cleaned_dir = tmp_path / "cleaned_md" / "test"
        cleaned_dir.mkdir(parents=True)
        (cleaned_dir / "doc.md").write_text("# Doc\n\nBody.\n", encoding="utf-8")

        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated conversion-engine crash")

        monkeypatch.setattr(chunk_module, "chunk_tree", boom)

        rc = pipeline_module.main(["--config", str(cfg), "--stages", "chunk"])
        assert rc == 1


# ═══════════════════════════════════════════════════════════════════════════════
# G16 -- common.load_config global cache could serve a stale config in-process
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadConfigCacheFreshness:
    def test_different_env_resolved_paths_are_not_confused_by_stale_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two default (path=None) calls that resolve to DIFFERENT config files (via a
        changed PROJECT_ROOT) in the same process must not share one cache slot."""
        reset_config_cache()
        try:
            project_a = tmp_path / "project-a"
            project_b = tmp_path / "project-b"
            (project_a / "configs").mkdir(parents=True)
            (project_b / "configs").mkdir(parents=True)
            (project_a / "pyproject.toml").write_text("[project]\nname='a'\n", encoding="utf-8")
            (project_b / "pyproject.toml").write_text("[project]\nname='b'\n", encoding="utf-8")
            (project_a / "configs" / "pipeline.yaml").write_text("source_id: project-a\n", encoding="utf-8")
            (project_b / "configs" / "pipeline.yaml").write_text("source_id: project-b\n", encoding="utf-8")

            monkeypatch.setenv("PROJECT_ROOT", str(project_a))
            cfg_a = load_config()
            assert cfg_a["source_id"] == "project-a"

            # No reset_config_cache() call here: only the resolved environment changed.
            monkeypatch.setenv("PROJECT_ROOT", str(project_b))
            cfg_b = load_config()
            assert cfg_b["source_id"] == "project-b"

            # Switching back must also return the correct (still cached) config, not
            # whatever happened to be loaded last.
            monkeypatch.setenv("PROJECT_ROOT", str(project_a))
            cfg_a_again = load_config()
            assert cfg_a_again["source_id"] == "project-a"
        finally:
            reset_config_cache()

    def test_on_disk_edit_is_picked_up_without_explicit_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An on-disk edit to the resolved default config must be visible on the next
        load_config() call even without an explicit reset_config_cache()."""
        reset_config_cache()
        try:
            root = tmp_path / "portable-project"
            (root / "configs").mkdir(parents=True)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            cfg_file = root / "configs" / "pipeline.yaml"
            cfg_file.write_text("source_id: v1\n", encoding="utf-8")
            monkeypatch.setenv("PROJECT_ROOT", str(root))

            cfg1 = load_config()
            assert cfg1["source_id"] == "v1"

            cfg_file.write_text("source_id: v2\n", encoding="utf-8")
            # Force a distinct mtime: a same-tick rewrite on a coarse filesystem clock
            # must still not be mistaken for "unchanged" by the freshness check.
            bumped = cfg_file.stat().st_mtime + 5
            os.utime(cfg_file, (bumped, bumped))

            cfg2 = load_config()  # no reset_config_cache() call in between
            assert cfg2["source_id"] == "v2"
        finally:
            reset_config_cache()

    def test_explicit_path_calls_still_see_on_disk_edits(self, tmp_path: Path) -> None:
        """An explicit --config path must also reflect on-disk edits on the next call,
        matching the pre-fix "explicit path is never cached" behavior in effect."""
        cfg_file = tmp_path / "custom.yaml"
        cfg_file.write_text("source_id: v1\n", encoding="utf-8")

        cfg1 = load_config(cfg_file)
        assert cfg1["source_id"] == "v1"

        cfg_file.write_text("source_id: v2\n", encoding="utf-8")
        bumped = cfg_file.stat().st_mtime + 5
        os.utime(cfg_file, (bumped, bumped))

        cfg2 = load_config(cfg_file)
        assert cfg2["source_id"] == "v2"
