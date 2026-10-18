from __future__ import annotations

from pathlib import Path

from cortexmark.common import (
    load_config,
    reset_config_cache,
    resolve_configured_path,
    resolve_manifest_path,
    runtime_env_value,
)
from cortexmark.paths import (
    build_path_settings,
    find_project_root,
    resolve_binary,
    resolve_manifest_file,
    resolve_session_settings,
)


class TestProjectRootDiscovery:
    def test_find_project_root_from_nested_dir(self, tmp_path: Path) -> None:
        root = tmp_path / "portable-project"
        nested = root / "docs" / "notes"
        nested.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

        found = find_project_root(nested)
        assert found == root.resolve()

    def test_load_config_default_from_nested_working_dir(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        config_dir = root / "configs"
        config_dir.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        (config_dir / "pipeline.yaml").write_text(
            "source_id: demo\npaths:\n  output_quality: outputs/quality\n", encoding="utf-8"
        )
        nested = root / "subdir" / "deep"
        nested.mkdir(parents=True)

        monkeypatch.chdir(nested)
        reset_config_cache()
        cfg = load_config()
        assert cfg["source_id"] == "demo"
        assert Path(cfg["__project_root__"]) == root.resolve()

    def test_env_config_path_is_repo_relative_outside_repo_root(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        config_dir = root / "configs"
        config_dir.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        (config_dir / "pipeline.yaml").write_text("source_id: env-demo\n", encoding="utf-8")
        outside = tmp_path / "outside"
        outside.mkdir()

        monkeypatch.chdir(outside)
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        monkeypatch.setenv("CORTEXMARK_CONFIG", "configs/pipeline.yaml")
        reset_config_cache()
        cfg = load_config()
        assert cfg["source_id"] == "env-demo"
        assert Path(cfg["__config_file__"]) == (config_dir / "pipeline.yaml").resolve()

    def test_dotenv_config_path_is_repo_relative_from_nested_cwd(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        config_dir = root / "configs"
        config_dir.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        (root / ".env").write_text("CORTEXMARK_CONFIG=configs/pipeline.yaml\n", encoding="utf-8")
        (config_dir / "pipeline.yaml").write_text("source_id: dotenv-demo\n", encoding="utf-8")
        nested = root / "nested" / "deep"
        nested.mkdir(parents=True)

        monkeypatch.chdir(nested)
        reset_config_cache()
        cfg = load_config()
        assert cfg["source_id"] == "dotenv-demo"
        assert Path(cfg["__config_file__"]) == (config_dir / "pipeline.yaml").resolve()


class TestPathSettingsPrecedence:
    def test_dotenv_overrides_config_defaults(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        config_dir = root / "configs"
        config_dir.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        (root / ".env").write_text("OUTPUT_DIR=.portable-output\nCACHE_DIR=.portable-cache\n", encoding="utf-8")
        cfg = {
            "__project_root__": str(root),
            "__config_dir__": str(config_dir),
            "paths": {"output_quality": "quality/from-config"},
        }

        monkeypatch.chdir(root)
        settings = build_path_settings(cfg, project_root=root)
        assert settings.outputs_dir == (root / ".portable-output").resolve()
        assert settings.quality_dir == (root / ".portable-output" / "quality").resolve()
        assert settings.cache_dir == (root / ".portable-cache").resolve()

    def test_environment_overrides_dotenv_and_config(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        config_dir = root / "configs"
        config_dir.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        (root / ".env").write_text("OUTPUT_DIR=.dotenv-output\n", encoding="utf-8")
        monkeypatch.setenv("OUTPUT_DIR", str(root / "env-output"))
        cfg = {
            "__project_root__": str(root),
            "__config_dir__": str(config_dir),
            "paths": {"output_quality": "quality/from-config"},
        }

        settings = build_path_settings(cfg, project_root=root)
        assert settings.outputs_dir == (root / "env-output").resolve()
        assert settings.quality_dir == (root / "env-output" / "quality").resolve()

    def test_resolve_configured_path_uses_output_root_override(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        config_dir = root / "configs"
        config_dir.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        monkeypatch.setenv("OUTPUT_DIR", str(root / "shared-output"))
        cfg = {
            "__project_root__": str(root),
            "__config_dir__": str(config_dir),
            "paths": {"output_cleaned_md": "config-output/cleaned"},
        }

        resolved = resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")
        assert resolved == (root / "shared-output" / "cleaned_md").resolve()


class TestManifestAndDirectories:
    def test_manifest_follows_output_root(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        config_dir = root / "configs"
        config_dir.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        monkeypatch.setenv("OUTPUT_DIR", str(root / "portable-output"))
        cfg = {"__project_root__": str(root), "__config_dir__": str(config_dir)}

        manifest = resolve_manifest_file(cfg, project_root=root)
        assert manifest == (root / "portable-output" / ".manifest.json").resolve()
        assert (
            resolve_manifest_path(cfg, session_name="s1")
            == (root / "sessions" / "s1" / "outputs" / ".manifest.json").resolve()
        )

    def test_session_settings_use_repo_root_sessions_dir(self, tmp_path: Path) -> None:
        root = tmp_path / "portable-project"
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

        session = resolve_session_settings("Research Batch", project_root=root)
        assert session.session_root == (root / "sessions" / "Research Batch").resolve()
        assert session.raw_data_dir == (root / "sessions" / "Research Batch" / "data" / "raw").resolve()
        assert session.cleaned_md_dir == (root / "sessions" / "Research Batch" / "outputs" / "cleaned_md").resolve()

    def test_ensure_directories_creates_portable_targets(self, tmp_path: Path) -> None:
        root = tmp_path / "portable-project"
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        settings = build_path_settings(project_root=root)

        settings.ensure_directories(settings.logs_dir, settings.checkpoints_dir, settings.tests_resources_dir)
        assert settings.logs_dir.is_dir()
        assert settings.checkpoints_dir.is_dir()
        assert settings.tests_resources_dir.is_dir()


class TestBinaryResolution:
    def test_resolve_binary_uses_external_bin_dir(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        bin_dir = root / "bin"
        bin_dir.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        tool = bin_dir / "demo-tool"
        tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)
        monkeypatch.setenv("EXTERNAL_BIN_DIR", str(bin_dir))

        resolved = resolve_binary("demo-tool", project_root=root)
        assert resolved == str(tool.resolve())


class TestRuntimeEnvPrecedence:
    def test_runtime_env_value_reads_dotenv_when_process_env_missing(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        (root / ".env").write_text("PIPELINE_ENGINE=docling\n", encoding="utf-8")
        cfg = {"__project_root__": str(root)}

        monkeypatch.delenv("PIPELINE_ENGINE", raising=False)
        monkeypatch.delenv("CORTEXMARK_ENGINE", raising=False)
        assert runtime_env_value("PIPELINE_ENGINE", "CORTEXMARK_ENGINE", cfg=cfg) == "docling"

    def test_runtime_env_value_prefers_process_env_over_dotenv(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        (root / ".env").write_text("PIPELINE_ENGINE=docling\n", encoding="utf-8")
        cfg = {"__project_root__": str(root)}

        monkeypatch.setenv("PIPELINE_ENGINE", "markitdown")
        assert runtime_env_value("PIPELINE_ENGINE", "CORTEXMARK_ENGINE", cfg=cfg) == "markitdown"

    def test_resolve_binary_ignores_non_executable_candidate(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "portable-project"
        bin_dir = root / "bin"
        bin_dir.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        tool = bin_dir / "demo-tool"
        tool.write_text("not executable\n", encoding="utf-8")
        tool.chmod(0o644)
        monkeypatch.setenv("EXTERNAL_BIN_DIR", str(bin_dir))

        assert resolve_binary("demo-tool", project_root=root) is None
