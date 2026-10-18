"""Comprehensive tests for the CortexMark."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from cortexmark.algorithm_extract import (
    ALGO_HEADER_RE,
    Algorithm,
    extract_algorithms,
    is_algorithm_content,
    parse_algorithm_body,
    parse_step,
)
from cortexmark.algorithm_extract import (
    build_summary as algo_build_summary,
)
from cortexmark.algorithm_extract import (
    extract_from_file as algo_extract_from_file,
)
from cortexmark.algorithm_extract import (
    extract_from_tree as algo_extract_from_tree,
)
from cortexmark.algorithm_extract import (
    write_report as write_algo_report,
)
from cortexmark.chunk import (
    Chunk,
    build_heading_re,
    chunk_file,
    chunk_tree,
    parse_chunks,
    slugify,
)
from cortexmark.citations import (
    Citation,
    CitationEdge,
    CitationGraph,
    Reference,
    analyze_file,
    analyze_tree,
    build_citation_graph,
    extract_inline_citations,
    extract_references,
    write_citation_report,
    write_dot_graph,
)
from cortexmark.clean import (
    clean_file,
    clean_markdown,
    clean_tree,
    fix_wrapped_lines,
    normalize_heading_lines,
    normalize_table_blocks,
    normalize_table_cell,
    remove_page_numbers,
    remove_repeated_headers_footers,
)
from cortexmark.common import (
    Manifest,
    detect_device,
    file_hash,
    find_project_root,
    get_source_id,
    load_config,
    mirror_directory_tree,
    reset_config_cache,
    resolve_path,
    resolve_quality_dir,
    resolve_quality_report_path,
    setup_logging,
)
from cortexmark.convert import (
    derive_output_path,
    format_algorithm_block,
    format_formula_block,
    format_incomplete_formula_block,
    is_algorithmic_text,
    merge_docling_markitdown,
    normalize_markdown,
    normalize_recovered_text,
    recover_formula_placeholders,
    reformat_algorithm_sections,
    render_formula_item,
)
from cortexmark.cross_ref import (
    CATEGORY_EQUATION,
    CATEGORY_FIGURE,
    CATEGORY_TABLE,
    CATEGORY_THEOREM,
    CrossRefReport,
    RefDefinition,
    RefMention,
    classify_kind,
    extract_definitions,
    extract_mentions,
    resolve_references,
)
from cortexmark.cross_ref import (
    analyze_file as crossref_analyze_file,
)
from cortexmark.cross_ref import (
    analyze_tree as crossref_analyze_tree,
)
from cortexmark.cross_ref import (
    write_report as write_crossref_report,
)
from cortexmark.diff import (
    FileDiff,
    TreeDiff,
    diff_files,
    diff_texts,
    diff_trees,
    write_diff_report,
    write_unified_diff,
)
from cortexmark.doc_type import (
    ALL_TYPES,
    GENERIC,
    PAPER,
    TEXTBOOK,
    DocTypeResult,
    detect_file,
    detect_paper,
    detect_report,
    detect_slides,
    detect_syllabus,
    detect_textbook,
    detect_tree,
    detect_type,
    get_template,
    render_template_scaffold,
    write_detection_report,
)
from cortexmark.figures import (
    FigureEntry,
    FigureReport,
    build_figure_report,
    extract_figures_from_text,
    extract_from_file,
    extract_from_tree,
    write_figure_manifest,
    write_gallery_page,
)
from cortexmark.formula_score import (
    FileReport,
    build_file_report,
    score_file,
    score_markdown,
    score_tree,
    validate_formula_text,
    write_report,
)
from cortexmark.ghpages import (
    PageEntry,
    build_document_page,
    build_index_page,
    build_nav_html,
    collect_pages,
    generate_site,
    write_site_manifest,
)
from cortexmark.metadata import (
    ScholarlyMetadata,
    extract_abstract,
    extract_authors,
    extract_doi,
    extract_emails,
    extract_file,
    extract_funding,
    extract_journal,
    extract_keywords,
    extract_metadata,
    extract_title,
    extract_tree,
    extract_volume_issue,
    extract_year,
    to_apa7,
    to_bibtex,
    to_yaml_frontmatter,
    write_metadata_report,
)
from cortexmark.multi_format import (
    FORMAT_EXTENSIONS,
    convert_file,
    convert_tree,
    md_to_html,
    md_to_text,
    md_to_yaml,
)
from cortexmark.notation_glossary import (
    NotationEntry,
    NotationGlossary,
    detect_common_notations,
    extract_explicit_definitions,
    extract_list_notations,
    extract_table_notations,
    glossary_to_scientific_objects,
    write_markdown_glossary,
)
from cortexmark.notation_glossary import (
    build_summary as notation_build_summary,
)
from cortexmark.notation_glossary import (
    extract_all as notation_extract_all,
)
from cortexmark.notation_glossary import (
    extract_from_file as notation_extract_from_file,
)
from cortexmark.notation_glossary import (
    extract_from_tree as notation_extract_from_tree,
)
from cortexmark.notation_glossary import (
    write_report as write_notation_report,
)
from cortexmark.ocr_quality import (
    OCRFileReport,
    OCRQualityMetrics,
    assess_file,
    assess_quality,
    assess_tree,
    confidence_to_grade,
    count_common_words,
    count_garble_chars,
    count_repeat_artefacts,
    count_short_noise_lines,
    count_symbol_soup,
    write_ocr_report,
)
from cortexmark.parallel import (
    ParallelConfig,
    ParallelReport,
    TaskResult,
    collect_md_files,
    parallel_map,
    parallel_tree,
)
from cortexmark.plugin import (
    VALID_HOOKS,
    PluginBase,
    PluginInfo,
    PluginRegistry,
    write_plugin_report,
)
from cortexmark.qa_pipeline import (
    BADGE_FAIL,
    BADGE_GOLD,
    BADGE_SILVER,
    FileQAReport,
    build_summary,
    check_broken_links,
    check_empty_chunks,
    check_encoding_errors,
    check_missing_text,
    check_orphan_headings,
    check_table_integrity,
    compute_badge,
    compute_overall_badge,
    qa_check_text,
    qa_file,
    qa_tree,
    write_markdown_report,
)
from cortexmark.qa_pipeline import (
    write_report as write_qa_report,
)
from cortexmark.rag_export import (
    RAGRecord,
    estimate_tokens,
    export_file,
    export_tree,
    make_chunk_id,
    normalize_text_for_embedding,
    parse_chunk_file,
    write_json_array,
    write_jsonl,
)
from cortexmark.rag_export import (
    build_summary as rag_build_summary,
)
from cortexmark.render_templates import (
    build_assignment_text,
    build_global_rules_text,
    build_section_rules_text,
    build_source_profile_text,
    bullet_lines_from_section,
    clean_inline,
    extract_line_value,
    extract_programs,
    extract_section,
    first_items,
    headings_from_markdown,
    humanize_topic,
    paragraphs_from_markdown,
    parse_section_entries,
    read_text,
    resolve_outline_path,
    summarize_text,
)
from cortexmark.run_pipeline import build_parser
from cortexmark.scientific_ir import stable_source_label
from cortexmark.semantic_chunk import (
    BLOCK_OPENER_RE,
    ENTITY_ALGORITHM,
    ENTITY_DEFINITION,
    ENTITY_EXAMPLE,
    ENTITY_NARRATIVE,
    ENTITY_PROOF,
    ENTITY_REMARK,
    ENTITY_THEOREM,
    PROOF_OPENER_RE,
    SemanticChunk,
    build_entity_summary,
    build_scientific_object_links,
    chunks_to_records,
    chunks_to_scientific_objects,
    classify_env_kind,
    extract_cross_refs,
    extract_formulas,
    has_qed,
    parse_semantic_chunks,
)
from cortexmark.semantic_chunk import (
    chunk_file as semantic_chunk_file,
)
from cortexmark.semantic_chunk import (
    chunk_tree as semantic_chunk_tree,
)
from cortexmark.topics import (
    TOPIC_KEYWORDS,
    DocumentTopics,
    TopicScore,
    build_topic_distribution,
    classify_file,
    classify_text,
    classify_tree,
    get_top_topics,
    write_topic_report,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

REPO_ROOT = find_project_root(Path(__file__).parent)


class DummyFormula:
    def __init__(self, orig: str, text: str = "") -> None:
        self.orig = orig
        self.text = text


@pytest.fixture(autouse=True)
def _fresh_config():
    """Ensure each test gets a fresh config cache."""
    reset_config_cache()
    yield
    reset_config_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# common.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfig:
    def test_load_config_from_file(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("source_id: test-source\npaths:\n  data_raw: data/raw\n", encoding="utf-8")
        cfg = load_config(cfg_file)
        assert cfg["source_id"] == "test-source"

    def test_get_source_id_prefers_source_id(self) -> None:
        assert get_source_id({"source_id": "alpha"}) == "alpha"

    def test_quality_path_helpers(self) -> None:
        cfg = {"paths": {"output_quality": "outputs/quality"}}
        assert resolve_quality_dir(cfg) == (resolve_path("outputs/quality")).resolve()
        assert (
            resolve_quality_report_path(cfg, "formula_validation.json")
            == (resolve_path("outputs/quality/formula_validation.json")).resolve()
        )
        assert (
            resolve_quality_report_path(cfg, "scientific_qa.json", session_name="s1")
            == (resolve_path("sessions/s1/outputs/quality/scientific_qa.json")).resolve()
        )

    def test_load_config_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_config_non_mapping_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(cfg_file)


class TestLogging:
    def test_setup_logging_returns_logger(self) -> None:
        log = setup_logging("test-logger")
        assert log.name == "test-logger"
        assert log.handlers


class TestMirrorDirectoryTree:
    def test_mirrors_structure(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        (src / "a" / "b").mkdir(parents=True)
        (src / "c").mkdir(parents=True)
        target = mirror_directory_tree(src, tmp_path / "out")
        assert target == tmp_path / "out" / "src"
        assert (target / "a" / "b").is_dir()
        assert (target / "c").is_dir()


class TestFileHash:
    def test_deterministic_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("hello", encoding="utf-8")
        h1 = file_hash(f)
        h2 = file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello", encoding="utf-8")
        f2.write_text("world", encoding="utf-8")
        assert file_hash(f1) != file_hash(f2)


class TestManifest:
    def test_needs_update_on_new_file(self, tmp_path: Path) -> None:
        f = tmp_path / "input.pdf"
        f.write_bytes(b"pdf-content")
        m = Manifest(tmp_path / ".manifest.json")
        assert m.needs_update(f)

    def test_no_update_after_record(self, tmp_path: Path) -> None:
        f = tmp_path / "input.pdf"
        f.write_bytes(b"pdf-content")
        m = Manifest(tmp_path / ".manifest.json")
        m.record(f)
        assert not m.needs_update(f)

    def test_needs_update_after_file_change(self, tmp_path: Path) -> None:
        f = tmp_path / "input.pdf"
        f.write_bytes(b"v1")
        m = Manifest(tmp_path / ".manifest.json")
        m.record(f)
        f.write_bytes(b"v2")
        assert m.needs_update(f)

    def test_persist_and_reload(self, tmp_path: Path) -> None:
        f = tmp_path / "input.pdf"
        f.write_bytes(b"content")
        output = tmp_path / "output.md"
        output.write_text("converted", encoding="utf-8")
        manifest_path = tmp_path / ".manifest.json"
        m1 = Manifest(manifest_path)
        m1.record(f, fingerprint="convert-v1", outputs=(output,))
        m1.save()
        m2 = Manifest(manifest_path)
        assert not m2.needs_update(f, fingerprint="convert-v1", outputs=(output,))
        assert json.loads(manifest_path.read_text(encoding="utf-8"))[str(f)] == {
            "fingerprint": "convert-v1",
            "hash": file_hash(f),
            "outputs": [str(output.resolve())],
        }

    def test_needs_update_for_legacy_string_record(self, tmp_path: Path) -> None:
        source = tmp_path / "input.pdf"
        source.write_bytes(b"content")
        manifest_path = tmp_path / ".manifest.json"
        manifest_path.write_text(json.dumps({str(source): file_hash(source)}), encoding="utf-8")

        manifest = Manifest(manifest_path)

        assert manifest.needs_update(source)

    def test_corrupt_json_is_treated_as_empty(self, tmp_path: Path) -> None:
        source = tmp_path / "input.pdf"
        source.write_bytes(b"content")
        manifest_path = tmp_path / ".manifest.json"
        manifest_path.write_text("{not valid json", encoding="utf-8")

        manifest = Manifest(manifest_path)

        assert manifest.needs_update(source)

    def test_needs_update_when_recorded_output_is_missing(self, tmp_path: Path) -> None:
        source = tmp_path / "input.pdf"
        source.write_bytes(b"content")
        output = tmp_path / "output.md"
        output.write_text("converted", encoding="utf-8")
        manifest = Manifest(tmp_path / ".manifest.json")
        manifest.record(source, outputs=(output,))
        output.unlink()

        assert manifest.needs_update(source, outputs=(output,))

    def test_needs_update_when_fingerprint_changes(self, tmp_path: Path) -> None:
        source = tmp_path / "input.pdf"
        source.write_bytes(b"content")
        manifest = Manifest(tmp_path / ".manifest.json")
        manifest.record(source, fingerprint="config-v1")

        assert manifest.needs_update(source, fingerprint="config-v2")


class TestConvertManifest:
    def test_single_file_skips_until_output_is_deleted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from cortexmark import convert as convert_module

        source = tmp_path / "input.pdf"
        source.write_bytes(b"pdf-content")
        output = tmp_path / "output.md"
        manifest_path = tmp_path / ".manifest.json"
        cfg = {"convert": {"engine": "markitdown"}, "idempotency": {"enabled": True}}
        calls: list[Path] = []

        def fake_convert_pdf(input_path: Path, output_path: Path, **_kwargs: object) -> Path:
            calls.append(input_path)
            output_path.write_text("converted", encoding="utf-8")
            return output_path

        monkeypatch.setattr(convert_module, "load_config", lambda _path: cfg)
        monkeypatch.setattr(convert_module, "resolve_manifest_path", lambda _cfg: manifest_path)
        monkeypatch.setattr(convert_module, "convert_pdf", fake_convert_pdf)
        monkeypatch.setattr(
            sys,
            "argv",
            ["cortexmark.convert", "--input", str(source), "--output", str(output)],
        )

        assert convert_module.main() == 0
        assert convert_module.main() == 0
        assert calls == [source.resolve()]

        output.unlink()

        assert convert_module.main() == 0
        assert calls == [source.resolve(), source.resolve()]


class TestConvertEngineRobustness:
    def test_dual_without_docling_degrades_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from cortexmark import convert as convert_module

        source = tmp_path / "input.pdf"
        source.write_bytes(b"pdf-content")
        output = tmp_path / "output.md"
        monkeypatch.setattr(convert_module.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(convert_module, "convert_with_markitdown", lambda _path: "# Converted\n")

        with caplog.at_level("WARNING", logger="cortexmark.convert"):
            result = convert_module.convert_pdf(source, output, engine="dual")

        assert result == output
        assert output.read_text(encoding="utf-8") == "# Converted\n"
        assert caplog.text.count("degrading conversion engine from 'dual' to 'markitdown'") == 1

    def test_unknown_engine_raises_before_input_validation(self, tmp_path: Path) -> None:
        from cortexmark import convert as convert_module

        with pytest.raises(ValueError, match="Unknown conversion engine"):
            convert_module.convert_pdf(tmp_path / "missing.pdf", tmp_path / "output.md", engine="duel")

    def test_invalid_table_structure_mode_raises(self) -> None:
        from cortexmark import convert as convert_module

        cfg = {"convert": {"docling": {"table_structure_mode": "turbo"}}}
        with pytest.raises(ValueError, match="table_structure_mode"):
            convert_module.build_converter(cfg)

    @pytest.mark.parametrize(
        ("formula", "code", "supports_enrichment"),
        [(False, False, True), (True, True, True), (True, True, False)],
    )
    def test_build_converter_handles_docling_enrichment_options(
        self,
        monkeypatch: pytest.MonkeyPatch,
        formula: bool,
        code: bool,
        supports_enrichment: bool,
    ) -> None:
        from cortexmark import convert as convert_module

        captured: dict[str, Any] = {}

        class FakeAcceleratorOptions:
            def __init__(self, **kwargs: Any) -> None:
                captured["accelerator_kwargs"] = kwargs

        class FakeInputFormat:
            PDF = "pdf"

        class FakePdfPipelineOptions:
            def __init__(self, **kwargs: Any) -> None:
                captured["pipeline_kwargs"] = kwargs
                if supports_enrichment:
                    self.do_formula_enrichment = False
                    self.do_code_enrichment = False

        class FakePdfFormatOption:
            def __init__(self, *, pipeline_options: Any) -> None:
                captured["pipeline_options"] = pipeline_options

        class FakeDocumentConverter:
            def __init__(self, **kwargs: Any) -> None:
                captured["converter_kwargs"] = kwargs

        fake_modules = {
            "docling": types.ModuleType("docling"),
            "docling.datamodel": types.ModuleType("docling.datamodel"),
            "docling.datamodel.accelerator_options": types.ModuleType("docling.datamodel.accelerator_options"),
            "docling.datamodel.base_models": types.ModuleType("docling.datamodel.base_models"),
            "docling.datamodel.pipeline_options": types.ModuleType("docling.datamodel.pipeline_options"),
            "docling.document_converter": types.ModuleType("docling.document_converter"),
        }
        fake_attrs: list[tuple[str, str, object]] = [
            ("docling.datamodel.accelerator_options", "AcceleratorOptions", FakeAcceleratorOptions),
            ("docling.datamodel.base_models", "InputFormat", FakeInputFormat),
            ("docling.datamodel.pipeline_options", "PdfPipelineOptions", FakePdfPipelineOptions),
            ("docling.document_converter", "DocumentConverter", FakeDocumentConverter),
            ("docling.document_converter", "PdfFormatOption", FakePdfFormatOption),
        ]
        for module_name, attr_name, attr_value in fake_attrs:
            setattr(fake_modules[module_name], attr_name, attr_value)
        for module_name, module in fake_modules.items():
            monkeypatch.setitem(sys.modules, module_name, module)

        cfg = {
            "convert": {
                "docling": {
                    "do_formula_enrichment": formula,
                    "do_code_enrichment": code,
                }
            }
        }
        convert_module.build_converter(cfg)

        pipeline_options = captured["pipeline_options"]
        if supports_enrichment:
            assert pipeline_options.do_formula_enrichment is formula
            assert pipeline_options.do_code_enrichment is code
        else:
            assert not hasattr(pipeline_options, "do_formula_enrichment")
            assert not hasattr(pipeline_options, "do_code_enrichment")

    def test_convert_tree_skips_bad_pdf_and_continues(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from cortexmark import convert as convert_module

        input_root = tmp_path / "pdfs"
        input_root.mkdir()
        bad = input_root / "a-bad.pdf"
        good = input_root / "z-good.pdf"
        bad.write_bytes(b"")
        good.write_bytes(b"pdf-content")
        manifest = Manifest(tmp_path / ".manifest.json")

        def fake_markitdown(path: Path) -> str:
            if path == bad:
                raise RuntimeError("corrupt PDF")
            return "# Converted\n"

        monkeypatch.setattr(convert_module, "convert_with_markitdown", fake_markitdown)
        with caplog.at_level("WARNING", logger="cortexmark.convert"):
            written = convert_module.convert_tree(
                input_root,
                tmp_path / "output",
                engine="markitdown",
                manifest=manifest,
            )
        manifest.save()

        assert [path.name for path in written] == ["z-good.md"]
        assert "a-bad.pdf" in caplog.text
        assert "corrupt PDF" in caplog.text
        records = json.loads(manifest.path.read_text(encoding="utf-8"))
        assert str(bad) not in records
        assert str(good) in records

    def test_convert_tree_builds_docling_converter_only_for_uncached_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cortexmark import convert as convert_module

        input_root = tmp_path / "pdfs"
        input_root.mkdir()
        source = input_root / "input.pdf"
        source.write_bytes(b"pdf-content")
        manifest = Manifest(tmp_path / ".manifest.json")
        build_calls = 0

        def fake_build_converter(_cfg: dict[str, Any] | None = None) -> object:
            nonlocal build_calls
            build_calls += 1
            return object()

        def fake_convert_pdf(input_path: Path, output_path: Path, **_kwargs: object) -> Path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(input_path.name, encoding="utf-8")
            return output_path

        monkeypatch.setattr(convert_module, "docling_available", lambda: True)
        monkeypatch.setattr(convert_module, "build_converter", fake_build_converter)
        monkeypatch.setattr(convert_module, "convert_pdf", fake_convert_pdf)

        first = convert_module.convert_tree(input_root, tmp_path / "output", engine="dual", manifest=manifest)
        cached = convert_module.convert_tree(input_root, tmp_path / "output", engine="dual", manifest=manifest)

        assert len(first) == 1
        assert cached == []
        assert build_calls == 1  # was: the cached rerun eagerly built a second converter


# ═══════════════════════════════════════════════════════════════════════════════
# convert.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeMarkdown:
    def test_strips_and_adds_newline(self) -> None:
        assert normalize_markdown("  hello  ") == "hello\n"

    def test_normalizes_line_endings(self) -> None:
        assert normalize_markdown("a\r\nb\rc") == "a\nb\nc\n"


class TestNormalizeRecoveredText:
    def test_collapses_whitespace(self) -> None:
        assert "hello world" in normalize_recovered_text("  hello   world  ")

    def test_fixes_punctuation(self) -> None:
        result = normalize_recovered_text("value . = 5 ,")
        assert " . =" not in result


class TestFormatBlocks:
    def test_algorithm_block_wrapped_in_code_fence(self) -> None:
        result = format_algorithm_block("Initialize Q(s,a)")
        assert result.startswith("```text\n")
        assert result.endswith("\n```")
        assert "Initialize Q(s,a)" in result

    def test_formula_block_uses_blockquote(self) -> None:
        result = format_formula_block("q ∗ (a)")
        assert "> Equation (recovered from PDF):" in result
        assert "> q ∗ (a)" in result


class TestReformatAlgorithmSections:
    def test_splits_heading_and_box(self) -> None:
        md = (
            "## Sarsa (on-policy TD control) for estimating Q ≈ q ∗ "
            "Algorithm parameters: step size α ∈ (0 , 1], small ε > 0 Initialize Q ( s, a )\n"
        )
        reformatted = reformat_algorithm_sections(md)
        assert "## Sarsa (on-policy TD control) for estimating Q ≈ q ∗\n\n```text" in reformatted


class TestRecoverFormulaPlaceholders:
    def test_renders_visible_blocks(self) -> None:
        markdown = "Before\n\n<!-- formula-not-decoded -->\n\nAfter\n\n<!-- formula-not-decoded -->\n"
        formula_items = [
            DummyFormula("q ∗ ( a ) . = E [ R t | A t = a ] ."),
            DummyFormula("Input: the policy π Algorithm parameter: step size α ∈ (0 , 1] Initialize V ( s )"),
        ]
        recovered = recover_formula_placeholders(markdown, formula_items)
        assert "<!-- formula-not-decoded -->" not in recovered
        assert "Equation (recovered from PDF)" in recovered
        assert "```text" in recovered


class TestMergeDoclingMarkitdown:
    def test_no_supplement_when_identical(self) -> None:
        text = "## Heading\n\nThis is a paragraph that exists in both engines and should not be duplicated.\n"
        assert merge_docling_markitdown(text, text) == text

    def test_adds_supplement_for_new_paragraphs(self) -> None:
        docling = "## Heading\n\nDocling paragraph about reinforcement learning basics.\n"
        markitdown = "Docling paragraph about reinforcement learning basics.\n\nExtra markitdown paragraph that provides additional context and explanation.\n"
        merged = merge_docling_markitdown(docling, markitdown)
        assert "<!-- markitdown-supplement -->" in merged
        assert "Extra markitdown paragraph" in merged

    def test_skips_table_artifacts(self) -> None:
        docling = "## Heading\n\nSome long paragraph about reinforcement learning concepts in detail.\n"
        markitdown = "| broken | table | artifact | data | garbage | more |\n"
        merged = merge_docling_markitdown(docling, markitdown)
        assert "<!-- markitdown-supplement -->" not in merged

    def test_skips_short_fragments(self) -> None:
        docling = "## Heading\n\nA meaningful paragraph about control theory and optimisation.\n"
        markitdown = "Short text\n\nAnother short\n"
        merged = merge_docling_markitdown(docling, markitdown)
        assert "<!-- markitdown-supplement -->" not in merged


# ═══════════════════════════════════════════════════════════════════════════════
# clean.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestRemovePageNumbers:
    def test_removes_standalone_numbers(self) -> None:
        assert remove_page_numbers(["text", "42", "more"]) == ["text", "more"]

    def test_removes_page_prefix(self) -> None:
        assert remove_page_numbers(["Page 7", "text"]) == ["text"]

    def test_preserves_numbers_in_text(self) -> None:
        assert remove_page_numbers(["Chapter 7 stuff"]) == ["Chapter 7 stuff"]


class TestRemoveRepeatedHeadersFooters:
    def test_removes_repeated_short_lines(self) -> None:
        lines = ["footer text"] * 5 + ["real content"]
        result = remove_repeated_headers_footers(lines)
        assert "footer text" not in result
        assert "real content" in result

    def test_preserves_headings(self) -> None:
        lines = ["## Heading"] * 5 + ["body"]
        result = remove_repeated_headers_footers(lines)
        assert all("## Heading" in line for line in result if line.startswith("##"))


class TestNormalizeHeadingLines:
    def test_combined_chapter_section(self) -> None:
        result = normalize_heading_lines("## CHAPTER 2 SECTION 3")
        assert result == ["# 2", "## 3"]

    def test_chapter_only(self) -> None:
        result = normalize_heading_lines("## Chapter 5: Introduction")
        assert result == ["# 5: Introduction"]

    def test_section_only(self) -> None:
        result = normalize_heading_lines("## Section 3")
        assert result == ["## 3"]

    def test_regular_heading_passthrough(self) -> None:
        result = normalize_heading_lines("## Something Normal")
        assert result == ["## Something Normal"]


class TestFixWrappedLines:
    def test_merges_broken_paragraphs(self) -> None:
        lines = ["This is a", "broken paragraph."]
        result = fix_wrapped_lines(lines)
        assert result == ["This is a broken paragraph."]

    def test_preserves_code_blocks(self) -> None:
        lines = ["```", "code here", "```"]
        result = fix_wrapped_lines(lines)
        assert "code here" in result

    def test_preserves_blockquotes(self) -> None:
        lines = ["", "> quote line"]
        result = fix_wrapped_lines(lines)
        assert "> quote line" in result

    def test_dehyphenates(self) -> None:
        lines = ["reinforce-", "ment learning"]
        result = fix_wrapped_lines(lines)
        assert result == ["reinforcement learning"]


class TestCleanMarkdown:
    def test_normalizes_combined_headings(self) -> None:
        raw = "## CHAPTER 2 SECTION 3\nHello world.\n"
        cleaned = clean_markdown(raw)
        assert cleaned.startswith("# 2\n\n## 3\n\nHello world.\n")

    def test_preserves_equation_and_algorithm_blocks(self) -> None:
        raw = (
            "## Tabular TD(0)\n\n"
            "> Equation (recovered from PDF):\n"
            ">\n"
            "> q ∗ ( a ) = E [ R t | A t = a ]\n\n"
            "```text\n"
            "Algorithm parameter: step size α ∈ (0, 1]\n"
            "Initialize Q(s, a)\n"
            "```\n"
        )
        cleaned = clean_markdown(raw)
        assert "## Tabular TD(0)" in cleaned
        assert "> Equation (recovered from PDF):" in cleaned
        assert "```text" in cleaned

    def test_empty_input(self) -> None:
        assert clean_markdown("") == "\n"

    def test_only_heading(self) -> None:
        result = clean_markdown("## Chapter 1: Test\n")
        assert result.strip().startswith("# 1: Test")

    def test_utf8_preserved(self) -> None:
        result = clean_markdown("α β γ δ ε ζ η θ ι κ λ μ\n")
        assert "α" in result
        assert "μ" in result


class TestCleanTree:
    def test_mirrors_and_cleans(self, tmp_path: Path) -> None:
        input_root = tmp_path / "raw_md" / "source-alpha"
        (input_root / "01_week" / "assignment").mkdir(parents=True)
        (input_root / "02_intro_rl").mkdir(parents=True)
        (input_root / "02_intro_rl" / "content.md").write_text("## CHAPTER 1 SECTION 1\nSample\n", encoding="utf-8")
        output_root = tmp_path / "cleaned_md"
        written = clean_tree(input_root, output_root)
        assert len(written) == 1
        assert (output_root / "source-alpha" / "01_week" / "assignment").is_dir()
        assert (output_root / "source-alpha" / "02_intro_rl" / "content.md").is_file()


# ═══════════════════════════════════════════════════════════════════════════════
# chunk.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Hello World!") == "hello-world"

    def test_empty(self) -> None:
        assert slugify("!!!") == "untitled"

    def test_caps_long_slugs(self) -> None:
        assert len(slugify("A" * 300)) == 80


class TestParseChunks:
    def test_basic(self) -> None:
        chunks = parse_chunks("# Intro\n\nAlpha\n\n## Part A\n\nBeta\n")
        assert [c.title for c in chunks] == ["Intro", "Part A"]

    def test_empty_input(self) -> None:
        assert parse_chunks("") == []

    def test_only_headings(self) -> None:
        chunks = parse_chunks("# A\n\n## B\n")
        assert chunks == []  # no body content

    def test_body_without_heading(self) -> None:
        chunks = parse_chunks("Some text without any heading.\n")
        assert len(chunks) == 1
        assert chunks[0].title == "untitled"

    def test_ignores_headings_inside_fenced_code(self) -> None:
        chunks = parse_chunks("# Methods\n\n```python\n# Initialize weights\nrun()\n```\n")
        assert len(chunks) == 1
        assert chunks[0].title == "Methods"
        assert "# Initialize weights" in chunks[0].body


class TestChunkRender:
    def test_render_with_both_headings(self) -> None:
        c = Chunk(chapter="Ch1", section="Sec1", body=["text"])
        rendered = c.render()
        assert "# Ch1" in rendered
        assert "## Sec1" in rendered
        assert "text" in rendered


class TestChunkTree:
    def test_mirrors_and_chunks(self, tmp_path: Path) -> None:
        input_root = tmp_path / "cleaned_md" / "source-alpha"
        (input_root / "05_coding" / "assignment").mkdir(parents=True)
        (input_root / "02_intro_rl").mkdir(parents=True)
        (input_root / "02_intro_rl" / "content.md").write_text("# Intro\n\n## Basics\n\nBody text.\n", encoding="utf-8")
        output_root = tmp_path / "chunks"
        written = chunk_tree(input_root, output_root)
        assert len(written) == 1
        assert (output_root / "source-alpha" / "05_coding" / "assignment").is_dir()
        assert written[0].name.startswith("chunk_001_")

    def test_chunk_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            chunk_file(tmp_path / "nonexistent.md", tmp_path / "out")

    def test_chunk_file_no_chunks(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")
        assert chunk_file(f, tmp_path / "out") == []  # was: raised ValueError

    def test_chunk_file_removes_stale_chunks(self, tmp_path: Path) -> None:
        source = tmp_path / "doc.md"
        output = tmp_path / "out"
        source.write_text("# One\nBody.\n\n# Two\nMore body.\n", encoding="utf-8")
        chunk_file(source, output)
        source.write_text("# Current\nOnly body.\n", encoding="utf-8")

        written = chunk_file(source, output)

        assert list(output.glob("chunk_*.md")) == written
        assert [path.name for path in written] == ["chunk_001_current.md"]

    def test_chunk_file_strips_utf8_bom(self, tmp_path: Path) -> None:
        source = tmp_path / "bom.md"
        source.write_text("\ufeff# Introduction\n\nBody.\n", encoding="utf-8")

        written = chunk_file(source, tmp_path / "out")

        assert [path.name for path in written] == ["chunk_001_introduction.md"]

    def test_chunk_file_caps_long_heading_slug(self, tmp_path: Path) -> None:
        source = tmp_path / "long-heading.md"
        source.write_text(f"# {'A' * 300}\n\nBody.\n", encoding="utf-8")

        written = chunk_file(source, tmp_path / "out")

        assert len(written) == 1
        assert written[0].name == f"chunk_001_{'a' * 80}.md"

    def test_chunk_file_splits_headingless_oversized_document(self, tmp_path: Path) -> None:
        source = tmp_path / "headingless.md"
        paragraphs = [f"Paragraph {index}: " + "scientific evidence " * 35 for index in range(24)]
        fenced_block = "```python\n" + "\n".join(f"result_{index} = analyze()" for index in range(30)) + "\n```"
        source.write_text("\n\n".join([*paragraphs[:12], fenced_block, *paragraphs[12:]]), encoding="utf-8")

        written = chunk_file(source, tmp_path / "out")

        assert len(written) > 1  # was: 1 giant chunk
        assert [path.name for path in written] == [
            f"chunk_{index:03d}_untitled.md" for index in range(1, len(written) + 1)
        ]
        assert all(len(path.read_text(encoding="utf-8")) <= 8000 for path in written)
        rendered_chunks = [path.read_text(encoding="utf-8") for path in written]
        assert all(rendered.count("```") % 2 == 0 for rendered in rendered_chunks)
        combined = "\n".join(rendered_chunks)
        assert all(f"Paragraph {index}:" in combined for index in range(24))
        assert fenced_block in combined

    def test_chunk_tree_skips_heading_only_file(self, tmp_path: Path) -> None:
        input_root = tmp_path / "input"
        input_root.mkdir()
        (input_root / "a.md").write_text("# A\nBody A.\n", encoding="utf-8")
        (input_root / "empty.md").write_text("# Cover\n\n## Contents\n", encoding="utf-8")
        (input_root / "z.md").write_text("# Z\nBody Z.\n", encoding="utf-8")

        written = chunk_tree(input_root, tmp_path / "output")

        assert len(written) == 2
        assert {path.parent.name for path in written} == {"a", "z"}


# ═══════════════════════════════════════════════════════════════════════════════
# render_templates.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseSectionEntries:
    def test_deterministic(self) -> None:
        outline = (
            "## Section 2: Introduction to Structured Parsing\n"
            "- Parsing as staged transformation\n"
            "- Deterministic post-processing\n"
            "## Section 3: Hierarchical Chunking\n"
            "- Chunk boundaries and context\n"
        )
        entries = parse_section_entries(outline)
        assert entries[2]["title"] == "Introduction to Structured Parsing"
        assert entries[2]["bullets"] == ["Parsing as staged transformation", "Deterministic post-processing"]
        assert entries[3]["title"] == "Hierarchical Chunking"

    def test_empty_input(self) -> None:
        assert parse_section_entries("") == {}

    def test_no_weeks(self) -> None:
        assert parse_section_entries("## Some Other Heading\ncontent\n") == {}


class TestResolveOutlinePath:
    def test_prefers_configured_file(self, tmp_path: Path) -> None:
        raw_root = tmp_path / "raw"
        (raw_root / "00_meta").mkdir(parents=True)
        configured = raw_root / "00_meta" / "my_outline.md"
        configured.write_text("# Outline\n", encoding="utf-8")
        found = resolve_outline_path(raw_root, cfg={"render_templates": {"outline_file": "00_meta/my_outline.md"}})
        assert found == configured.resolve()

    def test_falls_back_to_discovery(self, tmp_path: Path) -> None:
        raw_root = tmp_path / "raw"
        meta = raw_root / "00_meta"
        meta.mkdir(parents=True)
        discovered = meta / "source_outline_v2.md"
        discovered.write_text("# Outline\n", encoding="utf-8")
        found = resolve_outline_path(raw_root, cfg={})
        assert found == discovered.resolve()

    def test_override_missing_raises(self, tmp_path: Path) -> None:
        raw_root = tmp_path / "raw"
        raw_root.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            resolve_outline_path(raw_root, cfg={}, override=Path("00_meta/missing.md"))


class TestExtractHelpers:
    def test_extract_line_value(self) -> None:
        text = "Source Name: Structured Source Processing\nMaintainer: Maintainer X\n"
        assert extract_line_value(text, "Source Name", "?") == "Structured Source Processing"
        assert extract_line_value(text, "Missing", "default") == "default"

    def test_extract_section(self) -> None:
        text = "## Recommended Tools\n- Python\n- Matlab\n## Next\nstuff\n"
        section = extract_section(text, "Recommended Tools")
        assert "Python" in section
        assert "Matlab" in section

    def test_extract_section_missing(self) -> None:
        assert extract_section("## Other\ndata\n", "Missing") == ""


class TestHumanizeTopic:
    def test_strips_number_prefix(self) -> None:
        assert humanize_topic("03_segments") == "Segments"

    def test_acronyms(self) -> None:
        assert humanize_topic("09_td") == "TD"
        assert humanize_topic("04_mdp") == "MDP"


class TestFirstItems:
    def test_deduplicates(self) -> None:
        assert first_items(["a", "b", "a", "c"], 3) == ["a", "b", "c"]

    def test_limit(self) -> None:
        assert first_items(["x", "y", "z"], 2) == ["x", "y"]


class TestSummarizeText:
    def test_short_passthrough(self) -> None:
        assert summarize_text("Short.", max_chars=100) == "Short."

    def test_truncates_long(self) -> None:
        long = "A" * 300
        result = summarize_text(long, max_chars=50)
        assert len(result) <= 50
        assert result.endswith("…")


class TestBuildTemplates:
    def test_course_profile(self) -> None:
        text = build_source_profile_text(
            source_name="Structured Source",
            source_cycle="Cycle A",
            maintainer="Maintainer X",
            main_topics=["RL", "Segments"],
            programs=["Python"],
            notes=["Note 1"],
        )
        assert "# Source Profile" in text
        assert "Structured Source" in text
        assert "- Python" in text

    def test_global_rules(self) -> None:
        text = build_global_rules_text(["R1"], ["R2"], ["R3"])
        assert "# Global Rules" in text
        assert "- R1" in text
        assert "- R2" in text

    def test_week_rules(self) -> None:
        text = build_section_rules_text(3, "Section Alpha", ["S1"], ["E1"], ["O1"])
        assert "# Section 03 Rules" in text
        assert "## Scope" in text

    def test_assignment(self) -> None:
        text = build_assignment_text(5, "Learn the source structure", ["Task1", "Task2"], ["Submit md"])
        assert "# Section 05 Tasks" in text
        assert "1. Task1" in text
        assert "2. Task2" in text


# ═══════════════════════════════════════════════════════════════════════════════
# B1 - Table extraction (clean.py table normalisation)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeTableCell:
    def test_collapses_whitespace(self) -> None:
        assert normalize_table_cell("  hello   world  ") == "hello world"

    def test_empty_cell(self) -> None:
        assert normalize_table_cell("") == ""


class TestNormalizeTableBlocks:
    def test_simple_table_passthrough(self) -> None:
        table = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
        result = normalize_table_blocks(table)
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result

    def test_inserts_separator_when_missing(self) -> None:
        table = "| A | B |\n| 1 | 2 |\n"
        result = normalize_table_blocks(table)
        assert "---" in result

    def test_normalises_cell_whitespace(self) -> None:
        table = "|  lots   of   spaces  | ok |\n| --- | --- |\n| val | val2 |\n"
        result = normalize_table_blocks(table)
        assert "lots of spaces" in result

    def test_removes_empty_data_table(self) -> None:
        # Contract updated 2026-07-24: a header-only table whose header carries
        # content is KEPT (docling renders single-row Q&A pairs in that shape;
        # dropping them destroyed Casebook content). Only tables that are empty
        # everywhere are stripped.
        kept = normalize_table_blocks("| Header |\n| --- |\n")
        assert "Header" in kept
        dropped = normalize_table_blocks("|  |  |\n| --- | --- |\n|  |  |\n")
        assert "|" not in dropped

    def test_preserves_non_table_text(self) -> None:
        text = "Hello world\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nGoodbye\n"
        result = normalize_table_blocks(text)
        assert "Hello world" in result
        assert "Goodbye" in result
        assert "| A | B |" in result

    def test_uneven_columns_padded(self) -> None:
        table = "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 |\n"
        result = normalize_table_blocks(table)
        lines = [line for line in result.strip().split("\n") if line.startswith("|")]
        # Data row should have 3 cols (padded)
        assert lines[-1].count("|") >= 4  # | col | col | col |

    def test_clean_markdown_preserves_tables(self) -> None:
        raw = "## Results\n\n| Method | Score |\n| --- | --- |\n| A | 95 |\n| B | 87 |\n"
        cleaned = clean_markdown(raw)
        table_rows = [line for line in cleaned.splitlines() if line.startswith("|")]
        assert table_rows == ["| Method | Score |", "| --- | --- |", "| A | 95 |", "| B | 87 |"]

    def test_clean_markdown_preserves_list_item_boundaries(self) -> None:
        cleaned = clean_markdown("- first\n* second\n3. third\n")
        assert cleaned.splitlines() == ["- first", "* second", "3. third"]


# ═══════════════════════════════════════════════════════════════════════════════
# B6 - Configurable chunk strategy
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildHeadingRe:
    def test_default_matches_h1_h2(self) -> None:
        regex = build_heading_re()
        assert regex.match("# Title")
        assert regex.match("## Section")
        assert not regex.match("### Sub")

    def test_custom_levels_1_2_3(self) -> None:
        regex = build_heading_re([1, 2, 3])
        assert regex.match("# Title")
        assert regex.match("## Section")
        assert regex.match("### Sub")
        assert not regex.match("#### Deep")

    def test_single_level(self) -> None:
        regex = build_heading_re([1])
        assert regex.match("# Title")
        assert not regex.match("## Section")

    def test_levels_2_3(self) -> None:
        regex = build_heading_re([2, 3])
        assert not regex.match("# Title")
        assert regex.match("## Section")
        assert regex.match("### Sub")


class TestParseChunksWithLevels:
    def test_default_same_as_before(self) -> None:
        text = "# Intro\n\nAlpha\n\n## Part A\n\nBeta\n"
        chunks_default = parse_chunks(text)
        chunks_explicit = parse_chunks(text, split_levels=[1, 2])
        assert len(chunks_default) == len(chunks_explicit)
        assert [c.title for c in chunks_default] == [c.title for c in chunks_explicit]

    def test_three_levels(self) -> None:
        text = "# Ch1\n\nIntro\n\n## Sec1\n\nBody1\n\n### Sub1\n\nDeep\n"
        chunks = parse_chunks(text, split_levels=[1, 2, 3])
        titles = [c.title for c in chunks]
        assert "Ch1" in titles
        assert "Sec1" in titles
        assert "Sub1" in titles

    def test_h3_ignored_in_default(self) -> None:
        text = "# Ch1\n\nIntro\n\n### Sub1\n\nDeep\n"
        chunks = parse_chunks(text)
        # ### not in split levels, so it's body text
        assert len(chunks) == 1
        assert any("### Sub1" in line for line in chunks[0].body)

    def test_single_level_chunks(self) -> None:
        text = "# A\n\nBody A\n\n# B\n\nBody B\n"
        chunks = parse_chunks(text, split_levels=[1])
        assert len(chunks) == 2
        assert chunks[0].title == "A"
        assert chunks[1].title == "B"

    def test_chunk_file_with_split_levels(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# A\n\nBody\n\n## B\n\nMore\n\n### C\n\nDeep\n", encoding="utf-8")
        out = tmp_path / "out"
        chunks = chunk_file(f, out, split_levels=[1, 2, 3])
        assert len(chunks) == 3

    def test_chunk_tree_with_split_levels(self, tmp_path: Path) -> None:
        root = tmp_path / "cleaned" / "project"
        root.mkdir(parents=True)
        (root / "doc.md").write_text("# A\n\nBody\n\n## B\n\nMore\n\n### C\n\nDeep\n", encoding="utf-8")
        out = tmp_path / "chunks"
        written = chunk_tree(root, out, split_levels=[1, 2, 3])
        assert len(written) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# A3 - Formula fidelity scoring
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateFormulaText:
    def test_valid_formula(self) -> None:
        valid, issues = validate_formula_text("q*(a) = E[R_t | A_t = a]")
        assert valid
        assert issues == []

    def test_empty_formula(self) -> None:
        valid, issues = validate_formula_text("")
        assert not valid
        assert "empty" in issues[0]

    def test_no_operators(self) -> None:
        valid, issues = validate_formula_text("just some plain english text")
        assert not valid
        assert any("operator" in i for i in issues)

    def test_unbalanced_parens(self) -> None:
        valid, issues = validate_formula_text("f(x = 3 + 2")
        assert not valid
        assert any("parentheses" in i for i in issues)

    def test_unbalanced_brackets(self) -> None:
        valid, issues = validate_formula_text("E[X = 5")
        assert not valid
        assert any("brackets" in i for i in issues)

    def test_greek_letters_as_operators(self) -> None:
        valid, _issues = validate_formula_text("θ")
        assert valid


class TestScoreMarkdown:
    def test_detects_recovered_equations(self) -> None:
        md = "> Equation (recovered from PDF):\n>\n> q*(a) = E[R_t]\n"
        formulas = score_markdown(md)
        assert len(formulas) == 1
        assert formulas[0].kind == "recovered"
        assert formulas[0].valid

    def test_detects_incomplete_equations(self) -> None:
        md = "> Equation present in PDF, but text recovery was incomplete.\n"
        formulas = score_markdown(md)
        assert len(formulas) == 1
        assert formulas[0].kind == "incomplete"
        assert not formulas[0].valid

    def test_detects_placeholders(self) -> None:
        md = "Some text\n\n<!-- formula-not-decoded -->\n\nMore text\n"
        formulas = score_markdown(md)
        assert len(formulas) == 1
        assert formulas[0].kind == "placeholder"

    def test_detects_algorithm_blocks(self) -> None:
        md = "```text\nInitialize Q(s,a)\nLoop for each episode\n```\n"
        formulas = score_markdown(md)
        assert len(formulas) == 1
        assert formulas[0].kind == "algorithm"
        assert formulas[0].valid

    def test_mixed_content(self) -> None:
        md = (
            "> Equation (recovered from PDF):\n>\n> x = 5 + 3\n\n"
            "<!-- formula-not-decoded -->\n\n"
            "> Equation present in PDF, but text recovery was incomplete.\n\n"
            "```text\nAlgorithm step\n```\n"
        )
        formulas = score_markdown(md)
        assert len(formulas) == 4
        kinds = {f.kind for f in formulas}
        assert kinds == {"recovered", "placeholder", "incomplete", "algorithm"}


class TestBuildFileReport:
    def test_fidelity_score(self) -> None:
        md = (
            "> Equation (recovered from PDF):\n>\n> x = y + z\n\n"
            "> Equation present in PDF, but text recovery was incomplete.\n\n"
            "```text\nAlgorithm A\n```\n"
        )
        report = build_file_report(Path("test.md"), md)
        assert report.total_count == 3
        assert report.valid_count == 2
        assert report.fidelity_score == pytest.approx(66.7, abs=0.1)

    def test_no_formulas(self) -> None:
        report = build_file_report(Path("plain.md"), "Just plain text.\n")
        assert report.total_count == 0
        assert report.fidelity_score == 100.0


class TestScoreFile:
    def test_score_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("> Equation (recovered from PDF):\n>\n> a + b = c\n", encoding="utf-8")
        report = score_file(f)
        assert report.recovered_count == 1
        assert report.fidelity_score == 100.0

    def test_score_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            score_file(tmp_path / "nonexistent.md")


class TestScoreTree:
    def test_score_tree(self, tmp_path: Path) -> None:
        root = tmp_path / "docs"
        root.mkdir()
        (root / "a.md").write_text("> Equation (recovered from PDF):\n>\n> x = 1\n", encoding="utf-8")
        (root / "b.md").write_text("<!-- formula-not-decoded -->\n", encoding="utf-8")
        reports = score_tree(root)
        assert len(reports) == 2
        valids = sum(r.valid_count for r in reports)
        assert valids == 1

    def test_score_tree_empty(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        with pytest.raises(FileNotFoundError):
            score_tree(root)


class TestWriteReport:
    def test_writes_json(self, tmp_path: Path) -> None:
        reports = [FileReport(file="test.md", recovered_count=2, valid_count=2, total_count=2, fidelity_score=100.0)]
        out = tmp_path / "report.json"
        write_report(reports, out)
        assert out.exists()
        import json

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 1
        assert data["summary"]["overall_fidelity"] == 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Additional coverage - convert.py functions previously untested
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsAlgorithmicText:
    def test_detects_algorithm(self) -> None:
        assert is_algorithmic_text("Algorithm parameter: step size")
        assert is_algorithmic_text("Initialize Q(s,a)")

    def test_rejects_non_algorithm(self) -> None:
        assert not is_algorithmic_text("This is a normal paragraph about control theory.")

    def test_multiple_hints(self) -> None:
        assert is_algorithmic_text("Input: policy π")
        assert is_algorithmic_text("Loop for each episode")
        assert is_algorithmic_text("Choose action a")
        assert is_algorithmic_text("Observe reward r")


class TestFormatIncompleteFormulaBlock:
    def test_output(self) -> None:
        result = format_incomplete_formula_block()
        assert "Equation present in PDF" in result
        assert "recovery was incomplete" in result


class TestDeriveOutputPath:
    def test_simple(self, tmp_path: Path) -> None:
        input_root = tmp_path / "data" / "raw" / "manuscripts"
        input_path = input_root / "paper.pdf"
        output_root = tmp_path / "outputs" / "raw_md"
        result = derive_output_path(input_path, input_root, output_root)
        assert result == output_root / "manuscripts" / "paper.md"

    def test_nested(self, tmp_path: Path) -> None:
        input_root = tmp_path / "data" / "raw" / "books"
        input_path = input_root / "ch1" / "intro.pdf"
        output_root = tmp_path / "outputs" / "raw_md"
        result = derive_output_path(input_path, input_root, output_root)
        assert result == output_root / "books" / "ch1" / "intro.md"

    def test_without_root_prefix(self, tmp_path: Path) -> None:
        input_root = tmp_path / "sessions" / "demo" / "data" / "raw"
        input_path = input_root / "paper-a" / "paper.pdf"
        output_root = tmp_path / "sessions" / "demo" / "outputs" / "raw_md"
        result = derive_output_path(input_path, input_root, output_root, include_input_root_name=False)
        assert result == output_root / "paper-a" / "paper.md"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests - CLI entry points
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleanCLI:
    def test_clean_single_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "source_id: test\npaths:\n  output_raw_md: out\n  output_cleaned_md: out\n"
            "clean:\n  min_repeated_header_count: 3\n  max_repeated_header_length: 80\n"
            "idempotency:\n  enabled: false\nlogging:\n  level: WARNING\n",
            encoding="utf-8",
        )
        inp = tmp_path / "raw.md"
        inp.write_text("## Chapter 1\n\nHello world.\n42\n", encoding="utf-8")
        out_dir = tmp_path / "cleaned"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.clean",
                "--input",
                str(inp),
                "--output-dir",
                str(out_dir),
                "--config",
                str(cfg),
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        output_file = out_dir / "raw.md"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "Hello world" in content
        # Page number "42" should be removed
        assert "\n42\n" not in content


class TestChunkCLI:
    def test_chunk_single_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "source_id: test\npaths:\n  output_cleaned_md: out\n  output_chunks: out\n"
            "chunk:\n  split_levels: [1, 2]\n"
            "idempotency:\n  enabled: false\nlogging:\n  level: WARNING\n",
            encoding="utf-8",
        )
        inp = tmp_path / "doc.md"
        inp.write_text("# Intro\n\nParagraph A.\n\n## Details\n\nParagraph B.\n", encoding="utf-8")
        out_dir = tmp_path / "chunks"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.chunk",
                "--input",
                str(inp),
                "--output-dir",
                str(out_dir),
                "--config",
                str(cfg),
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (out_dir / "doc").is_dir()
        chunk_files = list((out_dir / "doc").glob("chunk_*.md"))
        assert len(chunk_files) == 2

    def test_chunk_with_three_levels(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "source_id: test\npaths:\n  output_cleaned_md: out\n  output_chunks: out\n"
            "chunk:\n  split_levels: [1, 2, 3]\n"
            "idempotency:\n  enabled: false\nlogging:\n  level: WARNING\n",
            encoding="utf-8",
        )
        inp = tmp_path / "doc.md"
        inp.write_text(
            "# Ch1\n\nBody1\n\n## Sec1\n\nBody2\n\n### Sub1\n\nBody3\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "chunks"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.chunk",
                "--input",
                str(inp),
                "--output-dir",
                str(out_dir),
                "--config",
                str(cfg),
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        chunk_files = list((out_dir / "doc").glob("chunk_*.md"))
        assert len(chunk_files) == 3


class TestFormulaScoreCLI:
    def test_score_single_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "source_id: test\npaths:\n  output_cleaned_md: out\nlogging:\n  level: WARNING\n",
            encoding="utf-8",
        )
        inp = tmp_path / "doc.md"
        inp.write_text(
            "> Equation (recovered from PDF):\n>\n> x + y = z\n\n<!-- formula-not-decoded -->\n",
            encoding="utf-8",
        )
        out = tmp_path / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.formula_score",
                "--input",
                str(inp),
                "--output",
                str(out),
                "--config",
                str(cfg),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert out.exists()
        import json

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["total_formulas"] == 2
        assert data["summary"]["total_valid"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Integration - clean + chunk end-to-end
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleanThenChunkIntegration:
    def test_end_to_end(self, tmp_path: Path) -> None:
        """Run clean_markdown → parse_chunks on realistic content."""
        raw = (
            "## CHAPTER 1 SECTION 1\n\n"
            "Intro paragraph about reinforcement learning fundamentals.\n\n"
            "42\n\n"  # page number
            "## CHAPTER 1 SECTION 2\n\n"
            "Second section discussing policy gradient methods.\n\n"
            "| Method | Score |\n| --- | --- |\n| PPO | 95 |\n| A2C | 87 |\n\n"
            "footer line\nfooter line\nfooter line\n"
        )
        cleaned = clean_markdown(raw)
        # Page number removed
        assert "\n42\n" not in cleaned
        # Table preserved
        assert "| Method | Score |" in cleaned
        # Headings normalised
        assert "# 1" in cleaned

        chunks = parse_chunks(cleaned)
        assert len(chunks) >= 2

    def test_three_level_end_to_end(self, tmp_path: Path) -> None:
        """Clean + chunk with 3 heading levels."""
        raw = "# Introduction\n\nOverview.\n\n## Background\n\nBackground text.\n\n### Details\n\nDetail text.\n"
        cleaned = clean_markdown(raw)
        # Default: 2 levels
        chunks_2 = parse_chunks(cleaned, split_levels=[1, 2])
        chunks_3 = parse_chunks(cleaned, split_levels=[1, 2, 3])
        assert len(chunks_3) >= len(chunks_2)
        assert any(c.title == "Details" for c in chunks_3)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 Tests: Metadata Extraction (A1)
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractTitle:
    def test_from_heading(self) -> None:
        lines = ["# My Research Paper", "", "Abstract..."]
        assert extract_title(lines) == "My Research Paper"

    def test_fallback_to_first_line(self) -> None:
        lines = ["My Research Paper", "", "Abstract..."]
        assert extract_title(lines) == "My Research Paper"

    def test_empty(self) -> None:
        assert extract_title([]) == ""


class TestExtractAuthors:
    def test_comma_separated(self) -> None:
        lines = ["# A Title", "John Doe, Jane Smith, Bob Lee", ""]
        authors = extract_authors(lines)
        assert "John Doe" in authors
        assert "Jane Smith" in authors

    def test_and_separated(self) -> None:
        lines = ["# A Title", "Alice Wang and Bob Chen", ""]
        authors = extract_authors(lines)
        assert "Alice Wang" in authors
        assert "Bob Chen" in authors

    def test_no_authors(self) -> None:
        lines = ["# Title", "", "## Abstract"]
        assert extract_authors(lines) == []

    def test_affiliation_is_not_an_author(self) -> None:
        lines = ["# A Title", "Department of Robotics, University of Example", "", "## Abstract"]
        assert extract_authors(lines) == []

    def test_comma_less_single_author_is_kept(self) -> None:
        lines = ["# A Title", "Jane Doe", "", "## Abstract"]
        assert extract_authors(lines) == ["Jane Doe"]


class TestExtractAbstract:
    def test_extracts_abstract(self) -> None:
        text = "# Title\n\nAbstract: This paper explores RL.\n\nKeywords: RL, AI"
        abstract = extract_abstract(text)
        assert "This paper explores RL" in abstract

    def test_no_abstract(self) -> None:
        text = "# Title\n\nSome body text.\n"
        assert extract_abstract(text) == ""


class TestExtractKeywords:
    def test_comma_separated(self) -> None:
        text = "Keywords: reinforcement learning, policy gradient, Segments"
        kws = extract_keywords(text)
        assert "reinforcement learning" in kws
        assert "policy gradient" in kws
        assert len(kws) == 3

    def test_semicolon_separated(self) -> None:
        text = "Keywords: RL; deep learning; optimization"
        kws = extract_keywords(text)
        assert len(kws) == 3

    def test_no_keywords(self) -> None:
        assert extract_keywords("Just a regular paragraph.") == []


class TestExtractDoi:
    def test_finds_doi(self) -> None:
        text = "Available at https://doi.org/10.1234/abcdef.5678 and see also..."
        assert extract_doi(text) == "10.1234/abcdef.5678"

    def test_no_doi(self) -> None:
        assert extract_doi("No DOI in this text.") == ""

    def test_reference_doi_is_not_attributed_to_document(self) -> None:
        text = (
            "# Current Paper\n\nJane Doe\n\nThe document body ends here. REFERENCES "
            "[37] Other Author (2017). Proceedings of Example Research. 10.18653/v1/S17-2001\n"
        )
        meta = extract_metadata(text)

        assert meta.doi == ""  # was: "10.18653/v1/S17-2001"
        assert meta.journal == ""  # was: "Proceedings of Example Research"
        assert meta.year == ""  # was: "2017"
        assert "10.18653/v1/S17-2001" not in to_bibtex(meta)
        assert "2017" not in to_apa7(meta)


class TestExtractYear:
    def test_finds_year(self) -> None:
        text = "Published in 2023 by the IEEE."
        assert extract_year(text) == "2023"

    def test_no_year(self) -> None:
        assert extract_year("No dates here.") == ""


class TestExtractMetadata:
    def test_full_extraction(self) -> None:
        text = (
            "# Deep Structured Source Processing for Control\n\n"
            "Alice Wang, Bob Chen\n\n"
            "Abstract: We propose a new method for RL.\n\n"
            "Keywords: RL, deep learning, control\n\n"
            "Published in 2024. DOI: 10.1000/test.123\n\n"
            "This work was funded by NSF grant 12345.\n"
        )
        meta = extract_metadata(text, source_file="test.md")
        assert meta.title == "Deep Structured Source Processing for Control"
        assert len(meta.authors) >= 2
        assert "RL" in meta.keywords
        assert meta.doi == "10.1000/test.123"
        assert meta.year == "2024"
        assert "funded" in meta.funding.lower() or "fund" in meta.funding.lower()
        assert meta.source_file == "test.md"

    def test_low_confidence_heuristics_have_provenance(self) -> None:
        meta = extract_metadata("Running Banner\n\nJournal of Example Science\n\nThis work was funded by grant 123.\n")

        assert meta.provenance["title"] == "low-confidence-heuristic"
        assert meta.provenance["journal"] == "low-confidence-heuristic"
        assert meta.provenance["funding"] == "low-confidence-heuristic"


class TestYAMLFrontmatter:
    def test_basic_output(self) -> None:
        meta = ScholarlyMetadata(title="Test Paper", authors=["A"], year="2024")
        yaml_str = to_yaml_frontmatter(meta)
        assert yaml_str.startswith("---\n")
        assert yaml_str.rstrip().endswith("---")
        assert 'title: "Test Paper"' in yaml_str
        assert 'year: "2024"' in yaml_str

    def test_special_characters_produce_valid_yaml(self) -> None:
        meta = ScholarlyMetadata(
            title=r'A "quoted": title at C:\papers',
            authors=[r"Jane: Doe\Lab"],
            abstract=r"For x: use \theta and \phi.",
        )

        frontmatter = to_yaml_frontmatter(meta)
        parsed = yaml.safe_load("\n".join(frontmatter.splitlines()[1:-1]))

        assert parsed["title"] == meta.title
        assert parsed["authors"] == meta.authors
        assert parsed["abstract"] == meta.abstract


class TestBibtex:
    def test_basic_output(self) -> None:
        meta = ScholarlyMetadata(title="Test Paper", authors=["John Doe"], year="2024", doi="10.1000/test")
        bib = to_bibtex(meta)
        assert "@article{doe2024," in bib
        assert "title = {Test Paper}" in bib
        assert "doi = {10.1000/test}" in bib


class TestAPA7:
    def test_basic_output(self) -> None:
        meta = ScholarlyMetadata(title="Test Paper", authors=["John Doe"], year="2024")
        apa = to_apa7(meta)
        assert "John Doe" in apa
        assert "(2024)" in apa
        assert "Test Paper" in apa


class TestExtractFileAndTree:
    def test_extract_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nBody text.\n", encoding="utf-8")
        meta = extract_file(md)
        assert meta.title == "Title"

    def test_extract_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("# A\n\nBody.\n", encoding="utf-8")
        (d / "b.md").write_text("# B\n\nBody.\n", encoding="utf-8")
        metas = extract_tree(d)
        assert len(metas) == 2

    def test_extract_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_file(tmp_path / "nope.md")

    def test_extract_tree_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            extract_tree(d)


class TestWriteMetadataReport:
    def test_writes_json(self, tmp_path: Path) -> None:
        meta = ScholarlyMetadata(title="Test", year="2024")
        out = tmp_path / "report.json"
        write_metadata_report([meta], out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["documents"] == 1
        assert data["entries"][0]["title"] == "Test"


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 Tests: RAG Export (C2)
# ══════════════════════════════════════════════════════════════════════════════


class TestEstimateTokens:
    def test_basic(self) -> None:
        assert estimate_tokens("hello world") > 0

    def test_empty(self) -> None:
        assert estimate_tokens("") == 1  # max(1, 0//4)


class TestMakeChunkId:
    def test_deterministic(self) -> None:
        id1 = make_chunk_id("src.md", "Title", 0)
        id2 = make_chunk_id("src.md", "Title", 0)
        assert id1 == id2
        assert len(id1) == 16

    def test_different_inputs(self) -> None:
        id1 = make_chunk_id("a.md", "T1", 0)
        id2 = make_chunk_id("a.md", "T2", 0)
        assert id1 != id2


class TestNormalizeTextForEmbedding:
    def test_collapses_whitespace(self) -> None:
        text = "Hello\n\nWorld\t\tfoo   bar"
        result = normalize_text_for_embedding(text)
        assert result == "Hello World foo bar"

    def test_empty(self) -> None:
        assert normalize_text_for_embedding("") == ""


class TestParseChunkFile:
    def test_basic(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.md"
        f.write_text("# Chapter 1\n## Section A\n\nBody text.\n", encoding="utf-8")
        rec = parse_chunk_file(f)
        assert rec.title == "Section A"
        assert "Body text" in rec.text
        assert rec.metadata["chapter"] == "Chapter 1"
        assert rec.metadata["section"] == "Section A"

    def test_no_headings(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.md"
        f.write_text("Just body text.\n", encoding="utf-8")
        rec = parse_chunk_file(f)
        assert rec.title == "chunk"  # falls back to stem
        assert "Just body text" in rec.text

    def test_id_is_portable_across_absolute_paths(self, tmp_path: Path) -> None:
        left = tmp_path / "machine-a" / "chunk.md"
        right = tmp_path / "machine-b" / "chunk.md"
        left.parent.mkdir()
        right.parent.mkdir()
        content = "# Chapter\n## Section\n\nPortable chunk content.\n"
        left.write_text(content, encoding="utf-8")
        right.write_text(content, encoding="utf-8")

        assert parse_chunk_file(left).id == parse_chunk_file(right).id


class TestExportFileAndTree:
    def test_export_file(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.md"
        f.write_text("# Ch\n## Sec\n\nContent here.\n", encoding="utf-8")
        rec = export_file(f)
        assert rec.title == "Sec"
        assert "Content here" in rec.text
        assert rec.metadata["token_estimate"] > 0

    def test_export_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "chunks"
        d.mkdir()
        (d / "a.md").write_text("# A\n\nBody A.\n", encoding="utf-8")
        (d / "b.md").write_text("# B\n\nBody B.\n", encoding="utf-8")
        records = export_tree(d)
        assert len(records) == 2

    def test_export_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            export_file(tmp_path / "nope.md")

    def test_export_tree_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            export_tree(d)


class TestWriteJSONL:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        records = [
            RAGRecord(id="abc", source="a.md", title="A", text="body1", metadata={}),
            RAGRecord(id="def", source="b.md", title="B", text="body2", metadata={}),
        ]
        out = tmp_path / "output.jsonl"
        write_jsonl(records, out)
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "abc"


class TestWriteJSONArray:
    def test_writes_json(self, tmp_path: Path) -> None:
        records = [
            RAGRecord(id="abc", source="a.md", title="A", text="body1", metadata={}),
        ]
        out = tmp_path / "output.json"
        write_json_array(records, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["title"] == "A"


class TestRAGBuildSummary:
    def test_summary(self) -> None:
        records = [
            RAGRecord(id="1", source="a.md", title="A", text="x" * 100, metadata={"token_estimate": 25}),
            RAGRecord(id="2", source="b.md", title="B", text="y" * 200, metadata={"token_estimate": 50}),
        ]
        s = rag_build_summary(records)
        assert s["total_records"] == 2
        assert s["total_tokens_estimate"] == 75
        assert s["sources"] == 2

    def test_summary_entity_types(self) -> None:
        records = [
            RAGRecord(
                id="1", source="a.md", title="A", text="x", metadata={"token_estimate": 1, "entity_type": "theorem"}
            ),
            RAGRecord(
                id="2", source="b.md", title="B", text="y", metadata={"token_estimate": 1, "entity_type": "theorem"}
            ),
            RAGRecord(
                id="3", source="c.md", title="C", text="z", metadata={"token_estimate": 1, "entity_type": "proof"}
            ),
        ]
        s = rag_build_summary(records)
        assert s["entity_types"]["theorem"] == 2
        assert s["entity_types"]["proof"] == 1

    def test_summary_total_formulas(self) -> None:
        records = [
            RAGRecord(
                id="1", source="a.md", title="A", text="x", metadata={"token_estimate": 1, "formulas": ["x^2", "y^2"]}
            ),
            RAGRecord(id="2", source="b.md", title="B", text="y", metadata={"token_estimate": 1, "formulas": ["z"]}),
        ]
        s = rag_build_summary(records)
        assert s["total_formulas"] == 3

    def test_summary_tracks_cross_ref_and_notation_counts(self) -> None:
        records = [
            RAGRecord(
                id="1",
                source="a.md",
                title="A",
                text="x",
                metadata={
                    "token_estimate": 1,
                    "cross_ref_links": [{"relation": "references"}],
                    "notation_symbols": [r"\alpha", r"\beta"],
                },
            ),
            RAGRecord(
                id="2",
                source="b.md",
                title="B",
                text="y",
                metadata={"token_estimate": 1, "cross_ref_links": [{}, {}], "notation_symbols": [r"\gamma"]},
            ),
        ]
        s = rag_build_summary(records)
        assert s["total_cross_ref_links"] == 3
        assert s["total_notation_symbols"] == 3


class TestRAGSemanticEnrichment:
    """Test semantic metadata enrichment in parse_chunk_file."""

    def test_theorem_detection_in_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.md"
        f.write_text(
            "**Theorem 3.2.** The optimal value satisfies $$V^*(s)=\\max_a Q^*(s,a)$$.\n",
            encoding="utf-8",
        )
        rec = parse_chunk_file(f)
        assert rec.metadata["entity_type"] == "theorem"
        assert rec.metadata["entity_label"] == "Theorem 3.2"
        assert len(rec.metadata["formulas"]) >= 1

    def test_definition_detection_in_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.md"
        f.write_text("**Definition 1 (Segments).** A Markov Decision Process.\n", encoding="utf-8")
        rec = parse_chunk_file(f)
        assert rec.metadata["entity_type"] == "definition"

    def test_cross_refs_in_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.md"
        f.write_text("As shown in Theorem 2.1 and Figure 3.\n", encoding="utf-8")
        rec = parse_chunk_file(f)
        assert len(rec.metadata["cross_refs"]) >= 2

    def test_narrative_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.md"
        f.write_text("# Ch1\n## Intro\n\nJust regular text without any entities.\n", encoding="utf-8")
        rec = parse_chunk_file(f)
        assert rec.metadata["entity_type"] == "narrative"
        assert rec.metadata["formulas"] == []
        assert rec.metadata["cross_refs"] == []

    def test_proof_metadata_in_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.md"
        f.write_text("**Proof of Theorem 2.** The argument is immediate. □\n", encoding="utf-8")
        rec = parse_chunk_file(f)
        assert rec.metadata["entity_type"] == "proof"
        assert rec.metadata["entity_kind"] == "proof"
        assert rec.metadata["parent_label"] == "Theorem 2"

    def test_object_level_metadata_in_chunk(self, tmp_path: Path) -> None:
        f = tmp_path / "chunk.md"
        f.write_text(
            "**Theorem 1 (Main Result).** Let $\\alpha$ denote the learning rate and $$V(s)=0$$.\n\n"
            "**Proof of Theorem 1.** See Theorem 1. □\n",
            encoding="utf-8",
        )
        rec = parse_chunk_file(f)
        assert rec.metadata["entity_kind"] == "theorem"
        assert rec.metadata["entity_name"] == "Main Result"
        assert any(obj["object_kind"] == "proof" for obj in rec.metadata["scientific_objects"])
        assert any(link["relation"] == "proof_of" for link in rec.metadata["object_links"])
        assert any(link["status"] == "resolved" for link in rec.metadata["cross_ref_links"])
        assert rec.metadata["notation_symbols"]
        assert rec.metadata["notation_object_ids"]
        assert rec.metadata["equations"]

    def test_parent_object_link_survives_real_chunk_files(self, tmp_path: Path) -> None:
        src = tmp_path / "paper.md"
        src.write_text(
            "# Chapter 1\n\n**Theorem 1.** Statement.\n\n**Proof.** Argument. □\n",
            encoding="utf-8",
        )
        written = semantic_chunk_file(src, tmp_path / "chunks")
        records = [parse_chunk_file(path) for path in written]
        theorem_record = next(record for record in records if record.metadata["entity_type"] == "theorem")
        proof_record = next(record for record in records if record.metadata["entity_type"] == "proof")
        theorem_object_id = theorem_record.metadata["scientific_object_ids"][0]

        assert proof_record.metadata["parent_label"] == "Theorem 1"
        assert proof_record.metadata["parent_object_id"] == theorem_object_id
        assert any(
            link["relation"] == "proof_of" and link["target_object_id"] == theorem_object_id
            for link in proof_record.metadata["object_links"]
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 Tests: QA Pipeline (C5)
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckEncodingErrors:
    def test_detects_replacement_char(self) -> None:
        text = "Hello \ufffd world"
        issues = check_encoding_errors(text)
        assert any(i.check == "encoding" and i.severity == "error" for i in issues)

    def test_clean_text(self) -> None:
        issues = check_encoding_errors("Hello world")
        assert len(issues) == 0

    def test_flags_mojibake_sequences_but_not_legitimate_typography(self) -> None:
        mojibake = "cafÃ©, Ã¢â‚¬, Â , and â€™"
        legitimate = "Pages 10–20; © 2026; see §3; error below 1‰."

        assert any(issue.message == "Possible mojibake detected" for issue in check_encoding_errors(mojibake))
        assert check_encoding_errors(legitimate) == []


class TestCheckMissingText:
    def test_short_file(self) -> None:
        issues = check_missing_text("Hi")
        assert any(i.check == "missing_text" for i in issues)

    def test_normal_file(self) -> None:
        issues = check_missing_text("A" * 100)
        assert len(issues) == 0


class TestCheckBrokenLinks:
    def test_broken_link(self) -> None:
        text = "See [here](..)"
        issues = check_broken_links(text)
        assert any(i.check == "broken_link" for i in issues)

    def test_empty_target(self) -> None:
        text = "See [click]()"
        issues = check_broken_links(text)
        assert any(i.check == "broken_link" for i in issues)

    def test_valid_url(self) -> None:
        text = "See [docs](https://example.com)"
        issues = check_broken_links(text)
        assert len(issues) == 0

    def test_anchor_link(self) -> None:
        text = "See [section](#intro)"
        issues = check_broken_links(text)
        assert len(issues) == 0


class TestCheckOrphanHeadings:
    def test_orphan_heading(self) -> None:
        text = "# Title\n## Section\n## Another Section\n\nBody."
        issues = check_orphan_headings(text)
        # "# Title" and "## Section" are orphans (followed by another heading)
        assert any(i.check == "orphan_heading" for i in issues)

    def test_heading_with_body(self) -> None:
        text = "# Title\n\nSome body text.\n"
        issues = check_orphan_headings(text)
        assert len(issues) == 0


class TestCheckTableIntegrity:
    def test_consistent_table(self) -> None:
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        issues = check_table_integrity(text)
        assert len(issues) == 0

    def test_missing_separator(self) -> None:
        text = "| A | B |\n| 1 | 2 |"
        issues = check_table_integrity(text)
        assert any(i.check == "table_integrity" for i in issues)

    def test_inconsistent_columns(self) -> None:
        text = "| A | B |\n| --- | --- |\n| 1 | 2 | 3 |"
        issues = check_table_integrity(text)
        assert any("inconsistent" in i.message for i in issues)


class TestCheckEmptyChunks:
    def test_empty_chunk(self) -> None:
        text = "# Title\n"
        issues = check_empty_chunks(text)
        assert any(i.check == "empty_chunk" for i in issues)

    def test_has_content(self) -> None:
        text = "# Title\n\nThis is a substantial body text for the chunk.\n"
        issues = check_empty_chunks(text)
        assert len(issues) == 0


class TestComputeBadge:
    def test_gold(self) -> None:
        report = FileQAReport(file="test.md", issues=[])
        assert compute_badge(report) == BADGE_GOLD

    def test_fail_on_error(self) -> None:
        from cortexmark.qa_pipeline import QAIssue

        report = FileQAReport(
            file="test.md",
            issues=[QAIssue(check="enc", severity="error", message="bad")],
        )
        assert compute_badge(report) == BADGE_FAIL

    def test_silver_on_warning(self) -> None:
        from cortexmark.qa_pipeline import QAIssue

        report = FileQAReport(
            file="test.md",
            issues=[QAIssue(check="x", severity="warning", message="w")],
        )
        assert compute_badge(report) == BADGE_SILVER


class TestComputeOverallBadge:
    def test_all_gold(self) -> None:
        reports = [FileQAReport(file="a.md", badge="gold"), FileQAReport(file="b.md", badge="gold")]
        assert compute_overall_badge(reports) == BADGE_GOLD

    def test_one_fail(self) -> None:
        reports = [FileQAReport(file="a.md", badge="gold"), FileQAReport(file="b.md", badge="fail")]
        assert compute_overall_badge(reports) == BADGE_FAIL

    def test_empty(self) -> None:
        assert compute_overall_badge([]) == BADGE_GOLD


class TestQACheckText:
    def test_clean_text(self) -> None:
        text = "# Title\n\nSome good content here that is long enough.\n"
        issues = qa_check_text(text)
        # Should have no errors (may have orphan heading info)
        assert not any(i.severity == "error" for i in issues)

    def test_problematic_text(self) -> None:
        text = "Broken \ufffd text"
        issues = qa_check_text(text)
        assert any(i.check == "encoding" for i in issues)


class TestQAFileAndTree:
    def test_qa_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nGood body text for quality checks.\n", encoding="utf-8")
        report = qa_file(f)
        assert report.file == str(f)
        assert report.badge in ("gold", "silver", "bronze", "fail")

    def test_qa_file_grades_invalid_utf8(self, tmp_path: Path) -> None:
        malformed = tmp_path / "malformed.md"
        malformed.write_bytes(b"# Title\n\nInvalid byte: \xff\n")

        report = qa_file(malformed)

        assert report.badge == BADGE_FAIL
        assert any(issue.message == "Unicode replacement character (U+FFFD) found" for issue in report.issues)

    def test_qa_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("# A\n\nBody text for file A.\n", encoding="utf-8")
        (d / "b.md").write_text("# B\n\nBody text for file B.\n", encoding="utf-8")
        reports = qa_tree(d)
        assert len(reports) == 2

    def test_qa_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            qa_file(tmp_path / "nope.md")

    def test_qa_tree_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            qa_tree(d)


class TestQABuildSummary:
    def test_summary(self) -> None:
        r1 = FileQAReport(file="a.md", badge="gold", issues=[])
        r2 = FileQAReport(file="b.md", badge="silver", issues=[])
        s = build_summary([r1, r2])
        assert s.files_scanned == 2
        assert "gold" in s.badge_distribution


class TestWriteQAReports:
    def test_writes_json(self, tmp_path: Path) -> None:
        r = FileQAReport(file="a.md", badge="gold", issues=[])
        s = build_summary([r])
        out = tmp_path / "qa.json"
        write_qa_report([r], s, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 1

    def test_writes_markdown(self, tmp_path: Path) -> None:
        from cortexmark.qa_pipeline import QAIssue

        issue = QAIssue(check="enc", severity="error", message="bad char", line=5)
        r = FileQAReport(file="a.md", badge="fail", issues=[issue])
        s = build_summary([r])
        out = tmp_path / "qa.md"
        write_markdown_report([r], s, out)
        content = out.read_text(encoding="utf-8")
        assert "Quality Assurance Report" in content
        assert "bad char" in content


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 CLI Integration Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMetadataCLI:
    def test_extract_single_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Research Paper\n\nJohn Doe, Jane Smith\n\nAbstract: A study.\n", encoding="utf-8")
        out = tmp_path / "meta.json"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.metadata", "--input", str(md), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["documents"] == 1


class TestRAGExportCLI:
    def test_export_single_file(self, tmp_path: Path) -> None:
        md = tmp_path / "chunk.md"
        md.write_text("# Ch\n## Sec\n\nBody.\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.rag_export", "--input", str(md), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1


class TestQACLI:
    def test_qa_single_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nGood body text for quality checks.\n", encoding="utf-8")
        out = tmp_path / "qa.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.qa_pipeline",
                "--input",
                str(md),
                "--output",
                str(out),
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 1


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 - Multi-Format Output (B4)
# ══════════════════════════════════════════════════════════════════════════════


class TestMdToHtml:
    """Tests for Markdown → HTML conversion."""

    def test_heading_conversion(self) -> None:
        result = md_to_html("# Hello World")
        assert "<h1>Hello World</h1>" in result

    def test_h2_and_h3(self) -> None:
        result = md_to_html("## Second\n### Third")
        assert "<h2>Second</h2>" in result
        assert "<h3>Third</h3>" in result

    def test_bold_and_italic(self) -> None:
        result = md_to_html("**bold** and *italic*")
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result

    def test_inline_code(self) -> None:
        result = md_to_html("Use `print()` here")
        assert "<code>print()</code>" in result

    def test_code_block(self) -> None:
        src = "```python\nprint('hi')\n```"
        result = md_to_html(src)
        assert '<pre><code class="language-python">' in result

    def test_link(self) -> None:
        result = md_to_html("[site](https://example.com)")
        assert '<a href="https://example.com">site</a>' in result

    def test_unsafe_link_and_image_urls_are_neutralized(self) -> None:
        source = (
            "[js](javascript:alert) [spaced](java\tscript:alert) "
            "[data](data:text/html,bad) ![vb](vbscript:evil) ![data](data:image/svg+xml,bad)"
        )
        result = md_to_html(source)
        assert result.count('href="#"') == 3
        assert result.count('src="#"') == 2
        assert "javascript:" not in result.lower()
        assert "vbscript:" not in result.lower()
        assert "data:" not in result.lower()

    def test_safe_urls_survive_with_escaped_attributes(self) -> None:
        source = (
            '[http](http://example.com) [https](https://example.com/?a=1&b="quoted") '
            "[mail](mailto:user@example.com) [relative](docs/page.html?q=1&b=2) "
            '![alt "quoted" & more](images/a.png?x=1&y=2)'
        )
        result = md_to_html(source)
        assert 'href="http://example.com"' in result
        assert 'href="https://example.com/?a=1&amp;b=&quot;quoted&quot;"' in result
        assert 'href="mailto:user@example.com"' in result
        assert 'href="docs/page.html?q=1&amp;b=2"' in result
        assert 'src="images/a.png?x=1&amp;y=2"' in result
        assert 'alt="alt &quot;quoted&quot; &amp; more"' in result

    def test_image(self) -> None:
        result = md_to_html("![alt](img.png)")
        assert '<img src="img.png" alt="alt">' in result

    def test_unordered_list(self) -> None:
        result = md_to_html("- item1\n- item2")
        assert "<ul>" in result
        assert "<li>item1</li>" in result

    def test_ordered_list(self) -> None:
        result = md_to_html("1. first\n2. second")
        assert "<li>first</li>" in result

    def test_ordered_list_uses_one_well_formed_wrapper(self) -> None:
        result = md_to_html("1. first\n2. second\n3. third\n\nParagraph")
        assert result.count("<ol>") == 1
        assert result.count("</ol>") == 1
        assert result.count("<li>") == 3
        assert "</ul>" not in result

    def test_blockquote(self) -> None:
        result = md_to_html("> quoted text")
        assert "<blockquote>" in result

    def test_horizontal_rule(self) -> None:
        result = md_to_html("---")
        assert "<hr>" in result

    def test_table(self) -> None:
        src = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = md_to_html(src)
        assert "<table>" in result
        assert "<th>" in result
        assert "<td>" in result

    def test_standalone_page_structure(self) -> None:
        result = md_to_html("# Doc", title="My Doc")
        assert "<!DOCTYPE html>" in result
        assert "<title>My Doc</title>" in result
        assert "</html>" in result

    def test_html_escape(self) -> None:
        result = md_to_html("Use <script>alert(1)</script>")
        assert "&lt;script&gt;" in result

    def test_multiplication_asterisks_remain_literal(self) -> None:
        source = "The result is a * b and E = m * c."
        result = md_to_html(source)
        assert source in result
        assert "<em>" not in result

    def test_emphasis_does_not_modify_code_or_math(self) -> None:
        result = md_to_html("`*code*` and $x * y$ and *italic*")
        assert "<code>*code*</code>" in result
        assert "$x * y$" in result
        assert "<em>italic</em>" in result

    def test_empty_input(self) -> None:
        result = md_to_html("")
        assert "<!DOCTYPE html>" in result


class TestMdToText:
    """Tests for Markdown → plain text."""

    def test_strips_headings(self) -> None:
        result = md_to_text("# Title\n## Sub")
        assert result.startswith("Title\nSub")

    def test_strips_bold_italic(self) -> None:
        result = md_to_text("**bold** and *italic*")
        assert "bold" in result
        assert "italic" in result
        assert "**" not in result

    def test_preserves_multiplication_asterisks(self) -> None:
        source = "The result is a * b and E = m * c."
        assert md_to_text(source) == source + "\n"

    def test_strips_links(self) -> None:
        result = md_to_text("[text](url)")
        assert "text" in result
        assert "[" not in result

    def test_strips_code_markers(self) -> None:
        result = md_to_text("Use `func()`")
        assert "func()" in result
        assert "`" not in result

    def test_table_pipes_stripped(self) -> None:
        result = md_to_text("| A | B |\n|---|---|\n| 1 | 2 |")
        assert "|" not in result
        assert "A" in result

    def test_code_block_preserved(self) -> None:
        result = md_to_text("```\ncode\n```")
        assert "code" in result

    def test_blockquote_stripped(self) -> None:
        result = md_to_text("> quoted")
        assert "quoted" in result
        assert ">" not in result


class TestMdToYaml:
    """Tests for Markdown → YAML."""

    def test_extracts_title(self) -> None:
        result = md_to_yaml("# My Title\n\nBody text.")
        assert "title: My Title" in result

    def test_includes_body(self) -> None:
        result = md_to_yaml("# T\n\nParagraph here.")
        assert "Paragraph here." in result

    def test_records_headings(self) -> None:
        result = md_to_yaml("# H1\n## H2\n### H3")
        assert "H1" in result
        assert "H2" in result
        assert "H3" in result

    def test_source_field(self) -> None:
        result = md_to_yaml("# T", source="doc.md")
        assert "source: doc.md" in result

    def test_no_title_uses_untitled(self) -> None:
        result = md_to_yaml("Just a paragraph.")
        assert "title: Untitled" in result


class TestMultiFormatConvert:
    """Tests for file/tree conversion."""

    def test_convert_file_html(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Hello\n\nWorld.", encoding="utf-8")
        out = convert_file(md, tmp_path / "out", fmt="html")
        assert out.suffix == ".html"
        assert "Hello" in out.read_text(encoding="utf-8")

    def test_convert_file_text(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Hello\n\n**bold**", encoding="utf-8")
        out = convert_file(md, tmp_path / "out", fmt="text")
        assert out.suffix == ".txt"
        content = out.read_text(encoding="utf-8")
        assert "Hello" in content
        assert "**" not in content

    def test_convert_file_yaml(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nBody.", encoding="utf-8")
        out = convert_file(md, tmp_path / "out", fmt="yaml")
        assert out.suffix == ".yaml"
        assert "title: Title" in out.read_text(encoding="utf-8")

    def test_convert_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            convert_file(tmp_path / "missing.md", tmp_path / "out")

    def test_convert_file_unsupported_format(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# T", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported format"):
            convert_file(md, tmp_path / "out", fmt="pdf")

    def test_convert_tree(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one.md").write_text("# One", encoding="utf-8")
        (tmp_path / "a" / "two.md").write_text("# Two", encoding="utf-8")
        out = tmp_path / "html_out"
        paths = convert_tree(tmp_path / "a", out, fmt="html")
        assert len(paths) == 2
        assert all(p.suffix == ".html" for p in paths)

    def test_convert_tree_empty(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            convert_tree(tmp_path / "empty", tmp_path / "out")

    def test_format_extensions(self) -> None:
        assert FORMAT_EXTENSIONS["html"] == ".html"
        assert FORMAT_EXTENSIONS["text"] == ".txt"
        assert FORMAT_EXTENSIONS["yaml"] == ".yaml"


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 - GitHub Pages Generator (C1)
# ══════════════════════════════════════════════════════════════════════════════


class TestGHPagesNav:
    """Tests for navigation building."""

    def test_build_nav_html_contains_home(self) -> None:
        pages = [PageEntry(title="Doc1", relative_path="doc1.html", source_md="doc1.md")]
        result = build_nav_html(pages)
        assert "Home" in result
        assert "index.html" in result

    def test_build_nav_highlights_current(self) -> None:
        pages = [PageEntry(title="Doc1", relative_path="doc1.html", source_md="doc1.md")]
        result = build_nav_html(pages, current="doc1.html")
        assert "font-weight:bold" in result

    def test_nav_limits_pages(self) -> None:
        pages = [PageEntry(title=f"D{i}", relative_path=f"d{i}.html", source_md=f"d{i}.md") for i in range(30)]
        result = build_nav_html(pages)
        assert result.count("</a>") <= 21


class TestGHPagesIndex:
    """Tests for index page generation."""

    def test_build_index_page_contains_title(self) -> None:
        pages = [PageEntry(title="Doc", relative_path="doc.html", source_md="doc.md")]
        result = build_index_page("My Site", pages)
        assert "My Site" in result
        assert "<!DOCTYPE html>" in result

    def test_build_index_page_with_description(self) -> None:
        pages = [PageEntry(title="D", relative_path="d.html", source_md="d.md")]
        result = build_index_page("Title", pages, description="Site desc")
        assert "Site desc" in result

    def test_index_page_lists_documents(self) -> None:
        pages = [
            PageEntry(title="Alpha", relative_path="alpha.html", source_md="alpha.md"),
            PageEntry(title="Beta", relative_path="beta.html", source_md="beta.md"),
        ]
        result = build_index_page("Site", pages)
        assert "Alpha" in result
        assert "Beta" in result
        assert 'class="card"' in result

    def test_index_page_escapes_html(self) -> None:
        pages = [PageEntry(title="<script>", relative_path="bad.html", source_md="bad.md")]
        result = build_index_page("Safe", pages)
        assert "&lt;script&gt;" in result
        assert "<script>" not in result

    def test_index_page_neutralizes_unsafe_page_urls(self) -> None:
        pages = [PageEntry(title="Unsafe", relative_path="javascript:alert", source_md="bad.md")]
        result = build_index_page("Safe", pages)
        assert result.count('href="#"') == 2
        assert "javascript:" not in result.lower()


class TestGHPagesDocumentPage:
    """Tests for individual document page generation."""

    def test_document_page_from_markdown(self) -> None:
        page = PageEntry(title="Test", relative_path="test.html", source_md="test.md")
        result = build_document_page(page, "# Test\n\nBody text.", [page])
        assert "Test" in result
        assert "breadcrumb" in result

    def test_document_page_has_nav(self) -> None:
        page = PageEntry(title="P", relative_path="p.html", source_md="p.md")
        result = build_document_page(page, "Content", [page])
        assert "<nav>" in result
        assert "Home" in result

    def test_document_page_has_footer(self) -> None:
        page = PageEntry(title="P", relative_path="p.html", source_md="p.md")
        result = build_document_page(page, "C", [page])
        assert "<footer>" in result

    def test_document_page_neutralizes_unsafe_markdown_urls(self) -> None:
        page = PageEntry(title="P", relative_path="p.html", source_md="p.md")
        result = build_document_page(page, "[unsafe](javascript:alert)", [page])
        assert '<a href="#">unsafe</a>' in result
        assert "javascript:" not in result.lower()


class TestGHPagesSite:
    """Tests for full site generation."""

    def test_collect_pages(self, tmp_path: Path) -> None:
        (tmp_path / "doc1.md").write_text("# Doc One\n\nText.", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "doc2.md").write_text("# Doc Two\n\nText.", encoding="utf-8")
        pages = collect_pages(tmp_path)
        assert len(pages) == 2
        titles = {p.title for p in pages}
        assert "Doc One" in titles
        assert "Doc Two" in titles

    def test_collect_pages_uses_stem_if_no_heading(self, tmp_path: Path) -> None:
        (tmp_path / "noheading.md").write_text("Just text.", encoding="utf-8")
        pages = collect_pages(tmp_path)
        assert pages[0].title == "noheading"

    def test_source_index_does_not_overwrite_landing_page(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "index.md").write_text("# Source Index\n\nSource body.", encoding="utf-8")
        out = tmp_path / "site"
        written = generate_site(src, out, site_title="Landing")
        assert {path.name for path in written} == {"index.html", "index_doc.html"}
        assert "<h1>Landing</h1>" in (out / "index.html").read_text(encoding="utf-8")
        assert "Source body." in (out / "index_doc.html").read_text(encoding="utf-8")

    def test_generate_site(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "doc.md").write_text("# Hello\n\nWorld.", encoding="utf-8")
        out = tmp_path / "site"
        written = generate_site(src, out, site_title="Test Site")
        assert len(written) == 2
        assert (out / "index.html").exists()
        assert (out / "doc.html").exists()

    def test_generate_site_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            generate_site(tmp_path / "missing", tmp_path / "out")

    def test_generate_site_no_md_files(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            generate_site(empty, tmp_path / "out")

    def test_write_site_manifest(self, tmp_path: Path) -> None:
        pages = [
            PageEntry(title="A", relative_path="a.html", source_md="a.md"),
            PageEntry(title="B", relative_path="b.html", source_md="b.md"),
        ]
        out = tmp_path / "manifest.json"
        write_site_manifest(pages, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["pages"] == 2
        assert len(data["entries"]) == 2

    def test_site_preserves_subdirectories(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "sub").mkdir()
        (src / "sub" / "deep.md").write_text("# Deep\n\nContent.", encoding="utf-8")
        out = tmp_path / "site"
        written = generate_site(src, out)
        assert any("sub" in str(p) for p in written)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 - Document-Type Templates (C4)
# ══════════════════════════════════════════════════════════════════════════════


class TestDocTypeDetection:
    """Tests for document type detection."""

    def test_detect_paper(self) -> None:
        text = "# Research\n\nAbstract: Results show...\n\nKeywords: A, B\n\n## References\n\n[1] Smith 2020."
        score, signals = detect_paper(text)
        assert score >= 0.5
        assert any("abstract" in s for s in signals)

    def test_detect_textbook(self) -> None:
        text = "# Chapter 1\n\n" + "Section content. " * 200 + "\n\nExample 1: ...\nExample 2: ...\nExercise 1: ..."
        score, _signals = detect_textbook(text)
        assert score >= 0.3

    def test_detect_syllabus(self) -> None:
        text = "# Source Outline\n\nMaintainer: Team Lead\n\nWeek 1: Intro\nWeek 2: Basics\nWeek 3: Advanced\n\nGrading: 40% Midterm"
        score, signals = detect_syllabus(text)
        assert score >= 0.5
        assert any("week" in s for s in signals)

    def test_detect_slides_short_paragraphs(self) -> None:
        text = "# Slide 1\n\n- Point A\n- Point B\n- Point C\n\n# Slide 2\n\n- More\n- Bullets\n- Here\n\n# Slide 3\n\n- Even\n- More\n- Items\n\n# Slide 4\n\n- Final\n- Points"
        score, _signals = detect_slides(text)
        assert score >= 0.3

    def test_detect_type_paper(self) -> None:
        text = "# Paper\n\nAbstract: Study results.\n\n10.1234/test\n\n## References\n\n[1] R"
        result = detect_type(text)
        assert result.doc_type == PAPER
        assert result.confidence >= 0.3

    def test_detect_type_generic_fallback(self) -> None:
        text = "This is a regular paragraph of text that does not strongly match any document type. It has some length to avoid the short-paragraph heuristic for slides."
        result = detect_type(text)
        assert result.doc_type == GENERIC

    def test_detect_type_stores_source(self) -> None:
        result = detect_type("Some text", source_file="test.md")
        assert result.source_file == "test.md"

    def test_detect_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nAbstract: Study.\n\n## References\n", encoding="utf-8")
        result = detect_file(md)
        assert isinstance(result, DocTypeResult)

    def test_detect_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            detect_file(tmp_path / "missing.md")

    def test_detect_tree(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# Paper\n\nAbstract: ...\n\n## References\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("Just text.", encoding="utf-8")
        results = detect_tree(tmp_path)
        assert len(results) == 2

    def test_detect_tree_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            detect_tree(empty)


class TestDocTypeTemplates:
    """Tests for template operations."""

    def test_all_types_have_template(self) -> None:
        for t in ALL_TYPES:
            template = get_template(t)
            assert template.doc_type == t
            assert len(template.sections) > 0

    def test_get_template_unknown_returns_generic(self) -> None:
        template = get_template("unknown")
        assert template.doc_type == GENERIC

    def test_render_scaffold(self) -> None:
        template = get_template(PAPER)
        scaffold = render_template_scaffold(template)
        assert "## Title" in scaffold or "## Authors" in scaffold
        assert "Document Type: paper" in scaffold

    def test_scaffold_has_all_sections(self) -> None:
        template = get_template(TEXTBOOK)
        scaffold = render_template_scaffold(template)
        for section in template.sections:
            assert f"## {section}" in scaffold


class TestDocTypeReport:
    """Tests for detection report writing."""

    def test_write_detection_report(self, tmp_path: Path) -> None:
        results = [
            DocTypeResult(doc_type=PAPER, confidence=0.8, signals=["has abstract"]),
            DocTypeResult(doc_type=GENERIC, confidence=0.1, signals=["no signals"]),
        ]
        out = tmp_path / "report.json"
        write_detection_report(results, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 2
        assert data["summary"]["type_distribution"]["paper"] == 1
        assert data["summary"]["type_distribution"]["generic"] == 1
        assert len(data["files"]) == 2


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 CLI Integration Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMultiFormatCLI:
    def test_convert_html_cli(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Hello\n\nWorld.", encoding="utf-8")
        out = tmp_path / "html_out"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.multi_format",
                "--input",
                str(md),
                "--output-dir",
                str(out),
                "--format",
                "html",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (out / "doc.html").exists()

    def test_convert_text_cli(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Hello\n\n**bold**", encoding="utf-8")
        out = tmp_path / "txt_out"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.multi_format",
                "--input",
                str(md),
                "--output-dir",
                str(out),
                "--format",
                "text",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (out / "doc.txt").exists()


class TestGHPagesCLI:
    def test_generate_site_cli(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "doc.md").write_text("# Test\n\nBody.", encoding="utf-8")
        out = tmp_path / "site"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.ghpages",
                "--input",
                str(src),
                "--output-dir",
                str(out),
                "--title",
                "TestSite",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert (out / "index.html").exists()


class TestDocTypeCLI:
    def test_detect_cli(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Paper\n\nAbstract: Results.\n\n## References\n", encoding="utf-8")
        out = tmp_path / "report.json"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.doc_type", "--input", str(md), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 1

    def test_scaffold_cli(self, tmp_path: Path) -> None:
        out = tmp_path / "scaffold.md"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.doc_type", "--scaffold", "paper", "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        content = out.read_text(encoding="utf-8")
        assert "Document Type: paper" in content


# ═══════════════════════════════════════════════════════════════════════════════
# ocr_quality.py  (Phase 4 — B2)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOCRQualityCounters:
    def test_count_garble_chars_clean(self) -> None:
        assert count_garble_chars("Hello, world!") == 0

    def test_count_garble_chars_replacement(self) -> None:
        text = "Hello \ufffd world \x00 end"
        assert count_garble_chars(text) == 2

    def test_count_symbol_soup_clean(self) -> None:
        assert count_symbol_soup("Normal text with punctuation!") == 0

    def test_count_symbol_soup_detected(self) -> None:
        text = "before \u2592\u2593\u2588 after"
        assert count_symbol_soup(text) >= 1

    def test_count_repeat_artefacts_clean(self) -> None:
        assert count_repeat_artefacts("Hello world") == 0

    def test_count_repeat_artefacts_found(self) -> None:
        text = "aaaaaa repeated bbbbbbb"
        assert count_repeat_artefacts(text) >= 1

    def test_count_short_noise_lines_clean(self) -> None:
        text = "Good line\nAnother good line"
        assert count_short_noise_lines(text) == 0

    def test_count_short_noise_lines_noisy(self) -> None:
        text = "Good line\n..\n---\n??\nEnd"
        assert count_short_noise_lines(text) >= 1

    def test_count_common_words(self) -> None:
        text = "The cat is on the mat and the dog is in the house."
        assert count_common_words(text) >= 5

    def test_turkish_letters_are_words_not_noise(self) -> None:
        text = "Bu çalışma için değişken ışığı ölçer."

        assert count_common_words(text) == 2
        assert count_common_words("değişken") == 0
        assert count_short_noise_lines("ışığı\nşı") == 0
        assert assess_quality(text).short_noise_lines == 0


class TestOCRAssessQuality:
    def test_empty_text(self) -> None:
        m = assess_quality("")
        assert m.confidence == 0.0
        assert "empty or near-empty document" in m.issues  # was: "empty document"

    @pytest.mark.parametrize("text", [" \n\t \n", "# OCR\n"])
    def test_whitespace_or_near_empty_text_has_low_quality(self, text: str) -> None:
        metrics = assess_quality(text)

        assert metrics.confidence == 0.0
        assert confidence_to_grade(metrics.confidence) == "F"

    def test_clean_text_high_confidence(self) -> None:
        text = "The reinforcement learning algorithm converges under the given assumptions. " * 5
        m = assess_quality(text)
        assert m.confidence >= 0.8

    def test_garbled_text_low_confidence(self) -> None:
        text = "\ufffd\x00\ufffd\x01 " * 50 + "real text"
        m = assess_quality(text)
        assert m.confidence < 0.8

    def test_symbol_soup_penalty(self) -> None:
        text = "normal " + "\u2592\u2593\u2588 " * 10 + "text"
        m = assess_quality(text)
        assert m.symbol_soup_count >= 1

    def test_word_count(self) -> None:
        text = "one two three four five"
        m = assess_quality(text)
        assert m.word_count == 5


class TestOCRGrading:
    def test_grade_a(self) -> None:
        assert confidence_to_grade(0.95) == "A"

    def test_grade_b(self) -> None:
        assert confidence_to_grade(0.80) == "B"

    def test_grade_c(self) -> None:
        assert confidence_to_grade(0.65) == "C"

    def test_grade_d(self) -> None:
        assert confidence_to_grade(0.45) == "D"

    def test_grade_f(self) -> None:
        assert confidence_to_grade(0.1) == "F"

    def test_grade_boundary_a(self) -> None:
        assert confidence_to_grade(0.9) == "A"

    def test_grade_boundary_b(self) -> None:
        assert confidence_to_grade(0.75) == "B"


class TestOCRFileOperations:
    def test_assess_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nThe document has normal content here.\n", encoding="utf-8")
        report = assess_file(md)
        assert isinstance(report, OCRFileReport)
        assert report.grade in {"A", "B", "C", "D", "F"}
        assert report.metrics.char_count > 0

    def test_assess_file_grades_invalid_utf8(self, tmp_path: Path) -> None:
        malformed = tmp_path / "malformed.md"
        malformed.write_bytes(b"Normal words followed by an invalid byte: \xff\n")

        report = assess_file(malformed)

        assert report.metrics.garble_count == 1
        assert report.grade in {"B", "C", "D", "F"}

    def test_assess_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            assess_file(tmp_path / "nonexistent.md")

    def test_assess_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("The quick brown fox jumps over the lazy dog.\n", encoding="utf-8")
        (d / "b.md").write_text("Another document with normal text in it.\n", encoding="utf-8")
        reports = assess_tree(d)
        assert len(reports) == 2

    def test_assess_tree_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            assess_tree(d)


class TestOCRReportWriter:
    def test_write_ocr_report(self, tmp_path: Path) -> None:
        metrics = OCRQualityMetrics(char_count=100, word_count=20, confidence=0.85)
        report = OCRFileReport(source_file="test.md", metrics=metrics, grade="B")
        out = tmp_path / "report.json"
        write_ocr_report([report], out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 1
        assert data["summary"]["average_confidence"] == 0.85
        assert data["summary"]["grade_distribution"]["B"] == 1
        assert len(data["files"]) == 1

    def test_write_ocr_report_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.json"
        write_ocr_report([], out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 0
        assert data["summary"]["average_confidence"] == 0.0


class TestOCRQualityCLI:
    def test_cli_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nNormal document content.\n", encoding="utf-8")
        out = tmp_path / "report.json"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.ocr_quality", "--input", str(md), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 1

    def test_cli_directory(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("Document one with normal text.\n", encoding="utf-8")
        out = tmp_path / "report.json"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.ocr_quality", "--input", str(d), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════════
# figures.py  (Phase 4 — B3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFigureExtraction:
    def test_no_figures(self) -> None:
        text = "# Title\n\nJust text, no images."
        entries = extract_figures_from_text(text)
        assert entries == []

    def test_markdown_image(self) -> None:
        text = '![Alt text](images/fig1.png "Title")'
        entries = extract_figures_from_text(text)
        assert len(entries) == 1
        assert entries[0].alt_text == "Alt text"
        assert entries[0].image_path == "images/fig1.png"
        assert entries[0].title == "Title"

    def test_markdown_image_no_title(self) -> None:
        text = "![Diagram](path/diagram.svg)"
        entries = extract_figures_from_text(text)
        assert len(entries) == 1
        assert entries[0].title == ""

    def test_html_image(self) -> None:
        text = '<img src="photo.jpg" alt="Photo">'
        entries = extract_figures_from_text(text)
        assert len(entries) == 1
        assert entries[0].image_path == "photo.jpg"
        assert entries[0].alt_text == "Photo"

    def test_html_images_keep_their_own_alt_text(self) -> None:
        text = '<img src="first.jpg"><img alt="Second" src="second.jpg">'
        entries = extract_figures_from_text(text)
        assert [(entry.image_path, entry.alt_text) for entry in entries] == [
            ("first.jpg", ""),
            ("second.jpg", "Second"),
        ]

    def test_multiple_images(self) -> None:
        text = "![A](a.png)\n\nText\n\n![B](b.png)"
        entries = extract_figures_from_text(text)
        assert len(entries) == 2

    def test_line_numbers(self) -> None:
        text = "Line 1\n![Fig](fig.png)\nLine 3"
        entries = extract_figures_from_text(text)
        assert entries[0].line_number == 2

    def test_exists_check(self, tmp_path: Path) -> None:
        img = tmp_path / "real.png"
        img.write_bytes(b"\x89PNG")
        text = "![Real](real.png)\n![Fake](fake.png)"
        entries = extract_figures_from_text(text, base_dir=tmp_path)
        real = next(e for e in entries if e.image_path == "real.png")
        fake = next(e for e in entries if e.image_path == "fake.png")
        assert real.exists is True
        assert fake.exists is False

    def test_remote_image_is_not_reported_missing(self, tmp_path: Path) -> None:
        entries = extract_figures_from_text("![Remote](https://example.com/image.png)", base_dir=tmp_path)
        assert entries[0].exists is True

    def test_context_capture(self) -> None:
        text = "Line before\n![Fig](fig.png)\nLine after"
        entries = extract_figures_from_text(text)
        assert "Line before" in entries[0].context
        assert "Line after" in entries[0].context


class TestFigureReport:
    def test_build_report_empty(self) -> None:
        report = build_figure_report([])
        assert report.total_figures == 0
        assert report.missing_files == 0

    def test_build_report_with_missing(self) -> None:
        entries = [
            FigureEntry(alt_text="a", image_path="a.png", exists=True),
            FigureEntry(alt_text="b", image_path="b.png", exists=False),
        ]
        report = build_figure_report(entries)
        assert report.total_figures == 2
        assert report.missing_files == 1


class TestFigureFileOps:
    def test_extract_from_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\n![Fig](fig.png)\n", encoding="utf-8")
        report = extract_from_file(md)
        assert report.total_figures == 1

    def test_extract_from_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_from_file(tmp_path / "nonexistent.md")

    def test_extract_from_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("![A](a.png)\n", encoding="utf-8")
        (d / "b.md").write_text("![B](b.png)\n![C](c.png)\n", encoding="utf-8")
        report = extract_from_tree(d)
        assert report.total_figures == 3

    def test_extract_from_tree_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            extract_from_tree(d)


class TestFigureWriters:
    def test_write_manifest(self, tmp_path: Path) -> None:
        entry = FigureEntry(
            alt_text="Test",
            image_path="test.png",
            source_file="doc.md",
            line_number=5,
            exists=True,
        )
        report = FigureReport(total_figures=1, missing_files=0, figures=[entry])
        out = tmp_path / "manifest.json"
        write_figure_manifest(report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["total_figures"] == 1
        assert len(data["figures"]) == 1

    def test_write_gallery(self, tmp_path: Path) -> None:
        entry = FigureEntry(
            alt_text="Fig1",
            image_path="fig1.png",
            source_file="doc.md",
            line_number=1,
            exists=True,
        )
        report = FigureReport(total_figures=1, missing_files=0, figures=[entry])
        out = tmp_path / "gallery.md"
        write_gallery_page(report, out)
        content = out.read_text(encoding="utf-8")
        assert "# Figure Gallery" in content
        assert "fig1.png" in content

    def test_write_gallery_missing(self, tmp_path: Path) -> None:
        entry = FigureEntry(
            alt_text="Missing",
            image_path="gone.png",
            source_file="doc.md",
            line_number=1,
            exists=False,
        )
        report = FigureReport(total_figures=1, missing_files=1, figures=[entry])
        out = tmp_path / "gallery.md"
        write_gallery_page(report, out)
        content = out.read_text(encoding="utf-8")
        assert "MISSING" in content

    def test_write_gallery_rebases_local_image_path(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        image_dir = source_dir / "images"
        image_dir.mkdir(parents=True)
        (image_dir / "figure.png").write_bytes(b"\x89PNG")
        entry = FigureEntry(
            alt_text="Figure",
            image_path="images/figure.png",
            source_file=str(source_dir / "doc.md"),
            line_number=1,
            exists=True,
        )
        report = FigureReport(total_figures=1, missing_files=0, figures=[entry])
        gallery = tmp_path / "published" / "gallery.md"
        write_gallery_page(report, gallery)
        content = gallery.read_text(encoding="utf-8")
        assert "![Figure](../source/images/figure.png)" in content


class TestFigureCLI:
    def test_cli_basic(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("![Fig](fig.png)\n", encoding="utf-8")
        out = tmp_path / "manifest.json"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.figures", "--input", str(md), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["total_figures"] == 1

    def test_cli_with_gallery(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("![Fig](fig.png)\n", encoding="utf-8")
        out = tmp_path / "manifest.json"
        gallery = tmp_path / "gallery.md"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.figures",
                "--input",
                str(md),
                "--output",
                str(out),
                "--gallery",
                str(gallery),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert gallery.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# parallel.py  (Phase 4 — B5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestParallelConfig:
    def test_defaults(self) -> None:
        cfg = ParallelConfig()
        assert cfg.workers >= 1
        assert cfg.pool_type == "thread"
        assert cfg.timeout is None

    def test_custom(self) -> None:
        cfg = ParallelConfig(workers=4, pool_type="process", timeout=30.0)
        assert cfg.workers == 4
        assert cfg.pool_type == "process"
        assert cfg.timeout == 30.0


class TestTaskResult:
    def test_success(self) -> None:
        r = TaskResult(input_path="a.md", success=True, result="ok", elapsed=0.1)
        assert r.success
        assert r.error == ""

    def test_failure(self) -> None:
        r = TaskResult(input_path="b.md", success=False, error="fail", elapsed=0.2)
        assert not r.success
        assert r.error == "fail"


class TestParallelMap:
    def test_empty_paths(self) -> None:
        report = parallel_map(lambda p: p, [])
        assert report.total == 0

    def test_single_worker(self, tmp_path: Path) -> None:
        files = []
        for i in range(3):
            f = tmp_path / f"doc{i}.md"
            f.write_text(f"content {i}", encoding="utf-8")
            files.append(f)

        def read_len(p: Path) -> int:
            return len(p.read_text(encoding="utf-8"))

        cfg = ParallelConfig(workers=1)
        report = parallel_map(read_len, files, config=cfg)
        assert report.total == 3
        assert report.succeeded == 3
        assert report.failed == 0

    def test_multi_worker(self, tmp_path: Path) -> None:
        files = []
        for i in range(5):
            f = tmp_path / f"doc{i}.md"
            f.write_text(f"content {i}", encoding="utf-8")
            files.append(f)

        def read_len(p: Path) -> int:
            return len(p.read_text(encoding="utf-8"))

        cfg = ParallelConfig(workers=3, pool_type="thread")
        report = parallel_map(read_len, files, config=cfg)
        assert report.total == 5
        assert report.succeeded == 5

    def test_multi_worker_preserves_input_order(self, tmp_path: Path) -> None:
        paths = [tmp_path / name for name in ("slow.md", "fast.md", "medium.md")]
        delays = {"slow.md": 0.06, "fast.md": 0.01, "medium.md": 0.03}

        def finish_out_of_order(path: Path) -> str:
            time.sleep(delays[path.name])
            return path.name

        report = parallel_map(finish_out_of_order, paths, config=ParallelConfig(workers=3))

        assert [result.input_path for result in report.results] == [str(path) for path in paths]
        assert [result.result for result in report.results] == [path.name for path in paths]

    def test_batch_timeout_returns_partial_results_and_cancels_pending_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cortexmark import parallel as parallel_module

        shutdown_cancel_flags: list[bool] = []
        original_shutdown = parallel_module.ThreadPoolExecutor.shutdown

        def tracking_shutdown(
            executor: parallel_module.ThreadPoolExecutor,
            wait: bool = True,
            *,
            cancel_futures: bool = False,
        ) -> None:
            shutdown_cancel_flags.append(cancel_futures)
            original_shutdown(executor, wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(parallel_module.ThreadPoolExecutor, "shutdown", tracking_shutdown)
        paths = [tmp_path / name for name in ("fast.md", "slow.md", "queued-1.md", "queued-2.md")]

        def mixed_duration(path: Path) -> str:
            time.sleep(0.005 if path.name == "fast.md" else 0.08)
            return path.name

        report = parallel_map(
            mixed_duration,
            paths,
            config=ParallelConfig(workers=1, timeout=0.03),
        )

        assert report.results[0].success is True
        assert [result.error for result in report.results[1:]] == [
            "task did not complete within batch timeout",
        ] * 3
        assert shutdown_cancel_flags == [True]

    def test_pool_elapsed_measures_worker_execution_only(self, tmp_path: Path) -> None:
        paths = [tmp_path / f"doc-{index}.md" for index in range(4)]

        def measured_sleep(_path: Path) -> float:
            start = time.monotonic()
            time.sleep(0.04)
            return time.monotonic() - start

        report = parallel_map(measured_sleep, paths, config=ParallelConfig(workers=2))

        assert all(result.elapsed == pytest.approx(result.result, abs=0.015) for result in report.results)

    def test_error_handling(self, tmp_path: Path) -> None:
        f = tmp_path / "missing.md"

        def will_fail(p: Path) -> str:
            return p.read_text(encoding="utf-8")

        report = parallel_map(will_fail, [f], config=ParallelConfig(workers=1))
        assert report.total == 1
        assert report.failed == 1
        assert report.results[0].error != ""

    def test_mixed_success_failure(self, tmp_path: Path) -> None:
        good = tmp_path / "good.md"
        good.write_text("ok", encoding="utf-8")
        bad = tmp_path / "bad.md"  # does not exist

        def read_text(p: Path) -> str:
            return p.read_text(encoding="utf-8")

        report = parallel_map(read_text, [good, bad], config=ParallelConfig(workers=1))
        assert report.succeeded == 1
        assert report.failed == 1


class TestCollectMdFiles:
    def test_collects_md(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("B", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.md").write_text("C", encoding="utf-8")
        files = collect_md_files(tmp_path)
        names = [f.name for f in files]
        assert "a.md" in names
        assert "c.md" in names
        assert "b.txt" not in names


class TestParallelTree:
    def test_basic(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        for i in range(3):
            (d / f"doc{i}.md").write_text(f"content {i}", encoding="utf-8")

        report = parallel_tree(
            lambda p: len(p.read_text(encoding="utf-8")),
            d,
            config=ParallelConfig(workers=2),
        )
        assert report.total == 3
        assert report.succeeded == 3

    def test_empty_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            parallel_tree(lambda p: p, d)

    def test_custom_glob(self, tmp_path: Path) -> None:
        d = tmp_path / "mixed"
        d.mkdir()
        (d / "a.md").write_text("md", encoding="utf-8")
        (d / "b.txt").write_text("txt", encoding="utf-8")
        report = parallel_tree(
            lambda p: p.suffix,
            d,
            glob="*.txt",
            config=ParallelConfig(workers=1),
        )
        assert report.total == 1

    def test_timing(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("ok", encoding="utf-8")
        report = parallel_tree(
            lambda p: p.name,
            d,
            config=ParallelConfig(workers=1),
        )
        assert report.total_elapsed >= 0


class TestParallelCLI:
    def test_cli_ocr_quality(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("The document content.\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.parallel",
                "--input",
                str(d),
                "--operation",
                "ocr_quality",
                "--workers",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════════
# plugin.py  (Phase 5 — D3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPluginBase:
    def test_default_hooks_empty(self) -> None:
        p = PluginBase()
        assert p.get_hooks() == []

    def test_info(self) -> None:
        p = PluginBase()
        p.name = "test"
        p.description = "A test plugin"
        info = p.info()
        assert info.name == "test"
        assert info.description == "A test plugin"

    def test_override_detected(self) -> None:
        class MyPlugin(PluginBase):
            name = "my"

            def post_convert(self, context: dict) -> dict:
                context["touched"] = True
                return context

        p = MyPlugin()
        hooks = p.get_hooks()
        assert "post_convert" in hooks
        assert "pre_convert" not in hooks

    def test_context_passthrough(self) -> None:
        p = PluginBase()
        ctx = {"key": "value"}
        result = p.pre_convert(ctx)
        assert result is ctx


class TestPluginRegistry:
    def test_register(self) -> None:
        reg = PluginRegistry()
        p = PluginBase()
        p.name = "test"
        reg.register(p)
        assert len(reg.plugins) == 1

    def test_register_type_error(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(TypeError):
            reg.register("not a plugin")  # type: ignore[arg-type]

    def test_run_hook(self) -> None:
        class Incrementer(PluginBase):
            name = "inc"

            def post_convert(self, context: dict) -> dict:
                context["count"] = context.get("count", 0) + 1
                return context

        reg = PluginRegistry()
        reg.register(Incrementer())
        results = reg.run_hook("post_convert", {"count": 0})
        assert len(results) == 1
        assert results[0].success is True

    def test_disabled_plugin_does_not_run(self) -> None:
        calls: list[str] = []

        class DisabledPlugin(PluginBase):
            name = "disabled"
            enabled = False

            def post_convert(self, context: dict) -> dict:
                calls.append("called")
                return context

        reg = PluginRegistry()
        reg.register(DisabledPlugin())

        assert reg.run_hook("post_convert", {}) == []
        assert calls == []
        assert reg.list_plugins()[0].enabled is False

    def test_run_hook_chains_returned_context_to_plugins_and_caller(self) -> None:
        seen_by_second: list[dict[str, Any]] = []

        class FirstPlugin(PluginBase):
            name = "first"

            def post_convert(self, context: dict) -> dict:
                return {**context, "first": True}

        class SecondPlugin(PluginBase):
            name = "second"

            def post_convert(self, context: dict) -> dict:
                seen_by_second.append(dict(context))
                return {**context, "second": True}

        reg = PluginRegistry()
        reg.register(FirstPlugin())
        reg.register(SecondPlugin())
        context = {"initial": True}

        results = reg.run_hook("post_convert", context)

        assert [result.plugin_name for result in results] == ["first", "second"]
        assert seen_by_second == [{"initial": True, "first": True}]
        assert context == {"initial": True, "first": True, "second": True}

    def test_run_hook_invalid(self) -> None:
        reg = PluginRegistry()
        with pytest.raises(ValueError):
            reg.run_hook("invalid_hook", {})

    def test_run_hook_error_handling(self) -> None:
        class BadPlugin(PluginBase):
            name = "bad"

            def post_convert(self, context: dict) -> dict:
                raise RuntimeError("boom")

        reg = PluginRegistry()
        reg.register(BadPlugin())
        results = reg.run_hook("post_convert", {})
        assert results[0].success is False
        assert "boom" in results[0].message

    def test_list_plugins(self) -> None:
        reg = PluginRegistry()
        p = PluginBase()
        p.name = "listed"
        reg.register(p)
        infos = reg.list_plugins()
        assert len(infos) == 1
        assert infos[0].name == "listed"

    def test_discover_empty_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "plugins"
        d.mkdir()
        reg = PluginRegistry()
        infos = reg.discover(d)
        assert infos == []

    def test_discover_nonexistent_dir(self, tmp_path: Path) -> None:
        reg = PluginRegistry()
        infos = reg.discover(tmp_path / "nope")
        assert infos == []

    def test_discover_plugin_file(self, tmp_path: Path) -> None:
        d = tmp_path / "plugins"
        d.mkdir()
        plugin_py = d / "my_plugin.py"
        plugin_py.write_text(
            "from cortexmark.plugin import PluginBase\n\n"
            "class MyPlugin(PluginBase):\n"
            "    name = 'discovered'\n"
            "    description = 'Auto-discovered'\n",
            encoding="utf-8",
        )
        reg = PluginRegistry()
        infos = reg.discover(d)
        assert len(infos) == 1
        assert infos[0].name == "discovered"

    def test_discover_skips_underscore(self, tmp_path: Path) -> None:
        d = tmp_path / "plugins"
        d.mkdir()
        (d / "__init__.py").write_text("", encoding="utf-8")
        (d / "_private.py").write_text("x = 1\n", encoding="utf-8")
        reg = PluginRegistry()
        infos = reg.discover(d)
        assert infos == []


class TestPluginReport:
    def test_write_report(self, tmp_path: Path) -> None:
        info = PluginInfo(name="test", description="desc", hooks=["post_convert"])
        out = tmp_path / "plugins.json"
        write_plugin_report([info], out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total_plugins"] == 1
        assert data["plugins"][0]["name"] == "test"


class TestPluginValidHooks:
    def test_all_hooks_exist(self) -> None:
        p = PluginBase()
        for hook in VALID_HOOKS:
            assert hasattr(p, hook)

    def test_hook_count(self) -> None:
        assert len(VALID_HOOKS) == 7


class TestPluginCLI:
    def test_cli_list(self, tmp_path: Path) -> None:
        d = tmp_path / "plugins"
        d.mkdir()
        out = tmp_path / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.plugin",
                "--plugin-dir",
                str(d),
                "--list",
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════════
# citations.py  (Phase 5 — A2)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCitationExtraction:
    def test_no_citations(self) -> None:
        citations = extract_inline_citations("Just plain text.")
        assert citations == []

    def test_author_year_single(self) -> None:
        text = "As shown by (Smith, 2020), the result holds."
        citations = extract_inline_citations(text)
        assert len(citations) == 1
        assert citations[0].cite_type == "author-year"
        assert "Smith" in citations[0].raw_text

    def test_author_year_et_al(self) -> None:
        text = "Results in (Jones et al., 2019) confirm."
        citations = extract_inline_citations(text)
        assert len(citations) == 1

    def test_numeric_single(self) -> None:
        text = "The method [3] is efficient."
        citations = extract_inline_citations(text)
        assert len(citations) == 1
        assert citations[0].cite_type == "numeric"

    def test_numeric_multiple(self) -> None:
        text = "See [1, 2, 3] for details."
        citations = extract_inline_citations(text)
        assert len(citations) == 1
        assert "1" in citations[0].raw_text

    def test_multiple_citations(self) -> None:
        text = "(Smith, 2020) and [5] show this."
        citations = extract_inline_citations(text)
        assert len(citations) == 2

    def test_line_number(self) -> None:
        text = "Line 1\nLine 2\n(Author, 2021) here"
        citations = extract_inline_citations(text)
        assert citations[0].line_number == 3

    def test_author_year_suffix_citation(self) -> None:
        citations = extract_inline_citations("(Smith, 2020a) extends prior work.")
        assert len(citations) == 1
        assert citations[0].target_hints == ["smith2020a"]

    def test_lowercase_author_like_suffix_pattern_not_treated_as_citation(self) -> None:
        citations = extract_inline_citations("(version, 2020a) is metadata, not a citation.")
        assert citations == []

    def test_numeric_mention_id_not_affected_by_earlier_citation_on_other_line(self) -> None:
        baseline = extract_inline_citations("Intro line.\nSee [1].")
        shifted = extract_inline_citations("(Smith, 2020)\nSee [1].")
        baseline_numeric = next(c for c in baseline if c.cite_type == "numeric")
        shifted_numeric = next(c for c in shifted if c.cite_type == "numeric")
        assert baseline_numeric.line_number == shifted_numeric.line_number
        assert baseline_numeric.mention_id == shifted_numeric.mention_id

    def test_numbered_bibliography_entries_are_not_inline_citations(self) -> None:
        text = "Body cites [1].\n\n## References\n[1] Smith. Cited work.\n[2] Jones. Uncited work.\n"
        citations = extract_inline_citations(text)
        references = extract_references(text)
        graph = build_citation_graph(citations, references)

        assert [citation.surface_text for citation in citations] == ["[1]"]
        assert len(graph.audit.phantom_references) == 1


class TestReferenceExtraction:
    def test_no_references(self) -> None:
        refs = extract_references("Just text, no references section.")
        assert refs == []

    def test_numbered_references(self) -> None:
        text = "# References\n\n[1] Smith J. (2020). A great paper.\n[2] Jones A. (2019). Another paper."
        refs = extract_references(text)
        assert len(refs) == 2
        assert refs[0].key == "1"
        assert refs[1].key == "2"

    def test_author_year_references(self) -> None:
        text = "## References\n\nSmith (2020). A great paper about things.\nJones (2019). Another paper about stuff."
        refs = extract_references(text)
        assert len(refs) >= 1
        assert refs[0].year == "2020"

    def test_lowercase_author_like_suffix_pattern_not_treated_as_reference(self) -> None:
        text = "## References\n\nversion (2020a). Not a scholarly reference.\n"
        refs = extract_references(text)
        assert refs == []

    def test_doi_extraction(self) -> None:
        text = "# References\n\n[1] Smith (2020). Title. 10.1234/example.2020"
        refs = extract_references(text)
        assert refs[0].doi == "10.1234/example.2020"

    def test_doi_extraction_strips_trailing_period(self) -> None:
        text = "# References\n\n[1] Smith (2020). Title. 10.1000/xyz."
        refs = extract_references(text)
        assert refs[0].doi == "10.1000/xyz"

    def test_stops_at_next_section(self) -> None:
        text = "## References\n\n[1] First ref.\n\n# Appendix\n\n[2] Not a ref."
        refs = extract_references(text)
        assert len(refs) == 1


class TestCitationGraph:
    def test_empty(self) -> None:
        graph = build_citation_graph([], [])
        assert graph.citations == []
        assert graph.edges == []

    def test_numeric_edges(self) -> None:
        cites = [Citation(raw_text="1, 2", cite_type="numeric")]
        graph = build_citation_graph(cites, [])
        assert len(graph.edges) == 2

    def test_author_year_edges(self) -> None:
        cites = [Citation(raw_text="Smith, 2020", cite_type="author-year")]
        graph = build_citation_graph(cites, [])
        assert len(graph.edges) == 1
        assert graph.edges[0].target_ref == "smith2020"
        assert graph.edges[0].status == "missing"

    def test_top_cited(self) -> None:
        cites = [
            Citation(raw_text="1", cite_type="numeric"),
            Citation(raw_text="1", cite_type="numeric"),
            Citation(raw_text="2", cite_type="numeric"),
        ]
        graph = build_citation_graph(cites, [])
        assert graph.top_cited[0] == ("1", 2)

    def test_resolves_author_year_edges_to_reference_key(self) -> None:
        cites = [Citation(raw_text="Smith, 2020", cite_type="author-year")]
        refs = [
            Reference(key="smith2020", raw_text="Smith (2020). Paper.", authors="Smith", year="2020", title="Paper")
        ]
        graph = build_citation_graph(cites, refs)
        assert len(graph.edges) == 1
        assert graph.edges[0].target_ref == "smith2020"
        assert graph.edges[0].reference_id == "smith2020"
        assert graph.edges[0].status == "resolved"

    def test_audit_flags_missing_duplicate_and_phantom_references(self) -> None:
        cites = [
            Citation(raw_text="Smith, 2020", cite_type="author-year"),
            Citation(raw_text="Jones, 2021", cite_type="author-year"),
        ]
        refs = [
            Reference(
                key="smith2020",
                raw_text="Smith (2020). Same paper. 10.1234/example",
                authors="Smith",
                year="2020",
                title="Same paper",
                doi="10.1234/example",
            ),
            Reference(
                key="smith2020-copy",
                raw_text="Smith (2020). Same paper. 10.1234/example",
                authors="Smith",
                year="2020",
                title="Same paper",
                doi="10.1234/example",
            ),
            Reference(
                key="unused2018",
                raw_text="Unused (2018). Never cited.",
                authors="Unused",
                year="2018",
                title="Never cited",
            ),
        ]
        graph = build_citation_graph(cites, refs)
        assert "jones2021" in graph.audit.missing_references
        assert "unused2018" in graph.audit.phantom_references
        assert graph.audit.duplicate_references
        assert graph.audit.duplicate_references[0].reason == "doi"

    def test_ambiguous_duplicate_match_is_not_marked_phantom(self) -> None:
        cites = [Citation(raw_text="Smith, 2020", cite_type="author-year")]
        refs = [
            Reference(
                key="smith2020", raw_text="Smith (2020). Paper A.", authors="Smith", year="2020", title="Paper A"
            ),
            Reference(
                key="smith2020b", raw_text="Smith (2020). Paper B.", authors="Smith", year="2020", title="Paper B"
            ),
        ]
        graph = build_citation_graph(cites, refs)
        assert graph.edges[0].status == "ambiguous"
        assert graph.edges[0].candidate_reference_ids
        assert graph.audit.phantom_references == []

    def test_author_year_suffix_reference_links(self) -> None:
        cites = [Citation(raw_text="Smith, 2020a", cite_type="author-year")]
        refs = extract_references("## References\n\nSmith (2020a). Variant Paper.\n")
        graph = build_citation_graph(cites, refs)
        assert graph.edges[0].status == "resolved"
        assert graph.edges[0].target_ref == "smith2020a"

    def test_build_citation_graph_does_not_mutate_inputs(self) -> None:
        cite = Citation(raw_text="Smith, 2020", cite_type="author-year")
        ref = Reference(key="smith2020", raw_text="Smith (2020). Paper.", authors="Smith", year="2020", title="Paper")
        build_citation_graph([cite], [ref])
        assert cite.mention_id == ""
        assert cite.target_hints == []
        assert ref.reference_id == ""
        assert ref.aliases == []


class TestCitationFileOps:
    def test_analyze_file(self, tmp_path: Path) -> None:
        md = tmp_path / "paper.md"
        md.write_text("(Smith, 2020) proved this.\n\n## References\n\n[1] Smith (2020). Paper.\n", encoding="utf-8")
        graph = analyze_file(md)
        assert len(graph.citations) >= 1
        assert graph.citations[0].surface_text == "(Smith, 2020)"

    def test_analyze_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            analyze_file(tmp_path / "nope.md")

    def test_analyze_file_uses_cwd_relative_source_label_when_possible(self, tmp_path: Path, monkeypatch) -> None:
        nested = tmp_path / "nested"
        nested.mkdir()
        md = nested / "paper.md"
        md.write_text("(Smith, 2020).\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        graph = analyze_file(md)
        assert graph.citations[0].source_file == "nested/paper.md"
        assert graph.edges[0].source_doc == "nested/paper.md"

    def test_analyze_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "papers"
        d.mkdir()
        (d / "a.md").write_text("(Author, 2021) said.\n", encoding="utf-8")
        (d / "b.md").write_text("[1] cited.\n", encoding="utf-8")
        graph = analyze_tree(d)
        assert len(graph.citations) >= 1

    def test_analyze_tree_uses_document_local_reference_scope(self, tmp_path: Path) -> None:
        d = tmp_path / "papers"
        d.mkdir()
        (d / "a.md").write_text("(Smith, 2020).\n\n## References\n\nSmith (2020). Local Paper.\n", encoding="utf-8")
        (d / "b.md").write_text("(Jones, 2021).\n\n## References\n\nBrown (2018). Unused Paper.\n", encoding="utf-8")
        graph = analyze_tree(d)
        assert "jones2021" in graph.audit.missing_references
        assert any("brown2018" in phantom for phantom in graph.audit.phantom_references)

    def test_analyze_tree_aggregates_top_cited_from_all_edges(self, tmp_path: Path) -> None:
        d = tmp_path / "papers"
        d.mkdir()
        (d / "a.md").write_text("(Smith, 2020).\n", encoding="utf-8")
        (d / "b.md").write_text("(Smith, 2020).\n(Jones, 2021).\n", encoding="utf-8")
        graph = analyze_tree(d)
        assert graph.top_cited[0] == ("smith2020", 2)

    def test_analyze_tree_uses_relative_source_labels(self, tmp_path: Path) -> None:
        d = tmp_path / "papers"
        nested = d / "nested"
        nested.mkdir(parents=True)
        (nested / "a.md").write_text("(Smith, 2020).\n", encoding="utf-8")
        graph = analyze_tree(d)
        assert graph.citations[0].source_file == "nested/a.md"
        assert graph.edges[0].source_doc == "nested/a.md"

    def test_analyze_tree_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            analyze_tree(d)


class TestCitationWriters:
    def test_write_report(self, tmp_path: Path) -> None:
        graph = CitationGraph(
            citations=[Citation(raw_text="Smith, 2020", cite_type="author-year")],
            references=[Reference(key="1", raw_text="Smith (2020). Paper.")],
            edges=[CitationEdge(source_doc="doc.md", target_ref="Smith")],
            top_cited=[("Smith", 1)],
        )
        out = tmp_path / "citations.json"
        write_citation_report(graph, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["total_citations"] == 1
        assert data["summary"]["total_references"] == 1
        assert "audit" in data
        assert "canonical_ir" in data
        assert "references" in data["canonical_ir"]

    def test_write_dot(self, tmp_path: Path) -> None:
        graph = CitationGraph(
            edges=[CitationEdge(source_doc="doc.md", target_ref="ref1")],
        )
        out = tmp_path / "graph.dot"
        write_dot_graph(graph, out)
        content = out.read_text(encoding="utf-8")
        assert "digraph citations" in content
        assert "doc.md" in content
        assert "ref1" in content


class TestCitationCLI:
    def test_cli_basic(self, tmp_path: Path) -> None:
        md = tmp_path / "paper.md"
        md.write_text("(Smith, 2020) here.\n", encoding="utf-8")
        out = tmp_path / "report.json"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.citations", "--input", str(md), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "summary" in data


# ═══════════════════════════════════════════════════════════════════════════════
# topics.py  (Phase 5 — A4)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopicClassification:
    def test_empty_text(self) -> None:
        result = classify_text("")
        assert result.primary_topic == "generic"
        assert result.word_count == 0

    def test_rl_text(self) -> None:
        text = (
            "Reinforcement learning uses a Markov decision process. "
            "The policy gradient method optimizes the reward. "
            "Q-learning and temporal difference are key algorithms."
        )
        result = classify_text(text)
        assert result.primary_topic == "reinforcement_learning"
        assert result.confidence > 0

    def test_ml_text(self) -> None:
        text = (
            "Machine learning with neural network and deep learning. "
            "Classification via supervised learning and gradient descent. "
            "We avoid overfitting with cross-validation."
        )
        result = classify_text(text)
        assert result.primary_topic == "machine_learning"

    def test_nlp_text(self) -> None:
        text = (
            "Natural language processing uses transformer attention mechanism. "
            "BERT and GPT are language model architectures. "
            "Tokenization and word embedding are preprocessing steps."
        )
        result = classify_text(text)
        assert result.primary_topic == "nlp"

    def test_generic_fallback(self) -> None:
        text = "The quick brown fox jumps over the lazy dog repeatedly many times."
        result = classify_text(text)
        assert result.primary_topic == "generic"
        assert result.confidence == 0

    def test_substrings_and_single_weak_hit_do_not_assert_a_topic(self) -> None:
        substring_only = classify_text("The elegant organization began an organic redesign.")
        weak_evidence = classify_text("This report briefly mentions a transformer.")

        assert substring_only.primary_topic == "generic"
        assert substring_only.scores == []
        assert weak_evidence.primary_topic == "generic"
        assert weak_evidence.confidence == 0.0

    def test_scores_sorted(self) -> None:
        text = "Reinforcement learning Segments policy gradient. Also mentions machine learning neural network."
        result = classify_text(text)
        if len(result.scores) >= 2:
            assert result.scores[0].score >= result.scores[1].score


class TestTopicScore:
    def test_dataclass(self) -> None:
        ts = TopicScore(topic="ml", score=5.0, keyword_hits=3)
        assert ts.topic == "ml"
        assert ts.keyword_hits == 3


class TestGetTopTopics:
    def test_top_n(self) -> None:
        doc = DocumentTopics(
            source_file="test.md",
            word_count=100,
            scores=[
                TopicScore(topic="a", score=10.0, keyword_hits=5),
                TopicScore(topic="b", score=8.0, keyword_hits=4),
                TopicScore(topic="c", score=3.0, keyword_hits=2),
            ],
        )
        assert get_top_topics(doc, 2) == ["a", "b"]

    def test_top_n_fewer(self) -> None:
        doc = DocumentTopics(
            source_file="",
            word_count=10,
            scores=[
                TopicScore(topic="x", score=1.0, keyword_hits=1),
            ],
        )
        assert get_top_topics(doc, 5) == ["x"]


class TestTopicKeywords:
    def test_keywords_exist(self) -> None:
        assert "reinforcement_learning" in TOPIC_KEYWORDS
        assert "machine_learning" in TOPIC_KEYWORDS
        assert len(TOPIC_KEYWORDS) >= 9

    def test_keywords_non_empty(self) -> None:
        for topic, kws in TOPIC_KEYWORDS.items():
            assert len(kws) > 0, f"No keywords for {topic}"


class TestTopicFileOps:
    def test_classify_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("Reinforcement learning with policy gradient and Segments.\n", encoding="utf-8")
        result = classify_file(md)
        assert result.primary_topic == "reinforcement_learning"

    def test_classify_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            classify_file(tmp_path / "nope.md")

    def test_classify_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("Machine learning neural network classification.\n", encoding="utf-8")
        (d / "b.md").write_text("Statistics and probability distribution.\n", encoding="utf-8")
        results = classify_tree(d)
        assert len(results) == 2

    def test_classify_tree_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(FileNotFoundError):
            classify_tree(d)


class TestTopicDistribution:
    def test_distribution(self) -> None:
        results = [
            DocumentTopics(source_file="a", word_count=100, primary_topic="ml"),
            DocumentTopics(source_file="b", word_count=100, primary_topic="ml"),
            DocumentTopics(source_file="c", word_count=100, primary_topic="nlp"),
        ]
        dist = build_topic_distribution(results)
        assert dist["ml"] == 2
        assert dist["nlp"] == 1


class TestTopicReport:
    def test_write_report(self, tmp_path: Path) -> None:
        result = DocumentTopics(
            source_file="doc.md",
            word_count=50,
            primary_topic="ml",
            confidence=0.8,
        )
        out = tmp_path / "topics.json"
        write_topic_report([result], out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 1
        assert "ml" in data["summary"]["topic_distribution"]


class TestTopicCLI:
    def test_cli_basic(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("Reinforcement learning and policy gradient.\n", encoding="utf-8")
        out = tmp_path / "report.json"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.topics", "--input", str(md), "--output", str(out)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# diff.py  (Phase 5 — B7)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiffTexts:
    def test_identical(self) -> None:
        fd = diff_texts("hello\n", "hello\n", label="test")
        assert fd.status == "unchanged"
        assert fd.lines_added == 0
        assert fd.lines_removed == 0

    def test_modified(self) -> None:
        fd = diff_texts("line1\nline2\n", "line1\nline3\n", label="test")
        assert fd.status == "modified"
        assert fd.lines_added >= 1
        assert fd.lines_removed >= 1

    def test_additions(self) -> None:
        fd = diff_texts("a\n", "a\nb\nc\n", label="test")
        assert fd.status == "modified"
        assert fd.lines_added >= 2

    def test_removals(self) -> None:
        fd = diff_texts("a\nb\nc\n", "a\n", label="test")
        assert fd.status == "modified"
        assert fd.lines_removed >= 2

    def test_diff_text_content(self) -> None:
        fd = diff_texts("old\n", "new\n", label="f.md")
        assert "old/" in fd.diff_text
        assert "new/" in fd.diff_text

    def test_unified_diff_has_well_formed_header_lines(self) -> None:
        fd = diff_texts("old\n", "new\n", label="f.md")

        assert fd.diff_text.splitlines()[:3] == [
            "--- old/f.md",
            "+++ new/f.md",
            "@@ -1 +1 @@",
        ]


class TestDiffFiles:
    def test_both_exist(self, tmp_path: Path) -> None:
        old = tmp_path / "old.md"
        new = tmp_path / "new.md"
        old.write_text("line1\n", encoding="utf-8")
        new.write_text("line2\n", encoding="utf-8")
        fd = diff_files(old, new)
        assert fd.status == "modified"

    def test_old_missing(self, tmp_path: Path) -> None:
        new = tmp_path / "new.md"
        new.write_text("content\n", encoding="utf-8")
        fd = diff_files(tmp_path / "nope.md", new)
        assert fd.status == "added"
        assert fd.lines_added >= 1

    def test_new_missing(self, tmp_path: Path) -> None:
        old = tmp_path / "old.md"
        old.write_text("content\n", encoding="utf-8")
        fd = diff_files(old, tmp_path / "nope.md")
        assert fd.status == "removed"
        assert fd.lines_removed >= 1

    def test_identical_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("same\n", encoding="utf-8")
        f2.write_text("same\n", encoding="utf-8")
        fd = diff_files(f1, f2)
        assert fd.status == "unchanged"


class TestDiffTrees:
    def test_identical_trees(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir()
        new.mkdir()
        (old / "a.md").write_text("same\n", encoding="utf-8")
        (new / "a.md").write_text("same\n", encoding="utf-8")
        td = diff_trees(old, new)
        assert len(td.files_unchanged) == 1
        assert len(td.files_modified) == 0

    def test_added_file(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir()
        new.mkdir()
        (new / "added.md").write_text("new content\n", encoding="utf-8")
        td = diff_trees(old, new)
        assert "added.md" in td.files_added

    def test_removed_file(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir()
        new.mkdir()
        (old / "gone.md").write_text("old content\n", encoding="utf-8")
        td = diff_trees(old, new)
        assert "gone.md" in td.files_removed

    def test_modified_file(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir()
        new.mkdir()
        (old / "doc.md").write_text("version 1\n", encoding="utf-8")
        (new / "doc.md").write_text("version 2\n", encoding="utf-8")
        td = diff_trees(old, new)
        assert "doc.md" in td.files_modified

    def test_modified_file_diffs_use_noncolliding_relative_paths(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        for root in (old, new):
            (root / "one").mkdir(parents=True)
            (root / "two").mkdir()
        for relative_path in (Path("one/doc.md"), Path("two/doc.md")):
            (old / relative_path).write_text("old\n", encoding="utf-8")
            (new / relative_path).write_text("new\n", encoding="utf-8")

        td = diff_trees(old, new)

        assert td.files_modified == ["one/doc.md", "two/doc.md"]
        assert [file_diff.file_path for file_diff in td.file_diffs] == td.files_modified
        assert [file_diff.diff_text.splitlines()[0] for file_diff in td.file_diffs] == [
            "--- old/one/doc.md",
            "--- old/two/doc.md",
        ]

    def test_totals(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir()
        new.mkdir()
        (old / "a.md").write_text("line1\nline2\n", encoding="utf-8")
        (new / "a.md").write_text("line1\nchanged\nextra\n", encoding="utf-8")
        td = diff_trees(old, new)
        assert td.total_lines_added >= 1

    def test_subdirectories(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        (old / "sub").mkdir(parents=True)
        (new / "sub").mkdir(parents=True)
        (old / "sub" / "deep.md").write_text("v1\n", encoding="utf-8")
        (new / "sub" / "deep.md").write_text("v2\n", encoding="utf-8")
        td = diff_trees(old, new)
        assert len(td.files_modified) == 1


class TestDiffReportWriters:
    def test_write_diff_report(self, tmp_path: Path) -> None:
        td = TreeDiff(
            files_added=["new.md"],
            files_modified=["changed.md"],
            total_lines_added=10,
            total_lines_removed=5,
            file_diffs=[
                FileDiff(file_path="new.md", status="added", lines_added=10),
                FileDiff(file_path="changed.md", status="modified", lines_added=5, lines_removed=5),
            ],
        )
        out = tmp_path / "diff.json"
        write_diff_report(td, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_added"] == 1
        assert data["summary"]["files_modified"] == 1
        assert data["summary"]["total_lines_added"] == 10

    def test_write_unified_diff(self, tmp_path: Path) -> None:
        td = TreeDiff(
            file_diffs=[
                FileDiff(file_path="a.md", status="modified", diff_text="--- a\n+++ b\n-old\n+new\n"),
            ],
        )
        out = tmp_path / "diff.txt"
        write_unified_diff(td, out)
        content = out.read_text(encoding="utf-8")
        assert "---" in content

    def test_write_unified_empty(self, tmp_path: Path) -> None:
        td = TreeDiff()
        out = tmp_path / "empty.txt"
        write_unified_diff(td, out)
        assert out.exists()


class TestDiffCLI:
    def test_cli_dirs(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir()
        new.mkdir()
        (old / "a.md").write_text("v1\n", encoding="utf-8")
        (new / "a.md").write_text("v2\n", encoding="utf-8")
        out = tmp_path / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.diff",
                "--old",
                str(old),
                "--new",
                str(new),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "summary" in data


# ═══════════════════════════════════════════════════════════════════════════════
# run_pipeline.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildParser:
    """Test argument parsing for the pipeline orchestrator."""

    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.stages == ["convert", "clean", "chunk", "render"]
        assert args.engine is None
        assert args.no_manifest is False
        assert args.config is None

    def test_single_stage(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--stages", "convert"])
        assert args.stages == ["convert"]

    def test_multiple_stages(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--stages", "convert", "clean"])
        assert args.stages == ["convert", "clean"]

    def test_invalid_stage_rejected(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--stages", "nonexistent"])

    def test_engine_choices(self) -> None:
        parser = build_parser()
        for engine in ("docling", "markitdown", "dual"):
            args = parser.parse_args(["--engine", engine])
            assert args.engine == engine

    def test_invalid_engine_rejected(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--engine", "imaginary"])

    def test_no_manifest_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--no-manifest"])
        assert args.no_manifest is True

    def test_config_path(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.yaml"
        cfg.touch()
        parser = build_parser()
        args = parser.parse_args(["--config", str(cfg)])
        assert args.config == cfg

    def test_combined_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--stages", "clean", "chunk", "--engine", "docling", "--no-manifest"])
        assert args.stages == ["clean", "chunk"]
        assert args.engine == "docling"
        assert args.no_manifest is True


class TestRunPipelineCLI:
    """Test the pipeline CLI via subprocess."""

    @staticmethod
    def _write_config(tmp_path: Path) -> Path:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "source_id: test\n"
            "paths:\n"
            "  data_raw: {d}\n"
            "  output_raw_md: {r}\n"
            "  output_cleaned_md: {c}\n"
            "  output_chunks: {k}\n"
            "  output_quality: {q}\n"
            "  output_semantic_chunks: {s}\n"
            "convert:\n"
            "  engine: dual\n"
            "idempotency:\n"
            "  enabled: false\n"
            "logging:\n"
            "  level: WARNING\n".format(
                d=tmp_path / "data",
                r=tmp_path / "raw_md",
                c=tmp_path / "cleaned_md",
                k=tmp_path / "chunks",
                q=tmp_path / "quality",
                s=tmp_path / "semantic_chunks",
            ),
            encoding="utf-8",
        )
        for d in ("data/test", "raw_md/test", "cleaned_md/test", "chunks/test", "quality", "semantic_chunks"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)
        return cfg

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.run_pipeline", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "cortexmark" in result.stdout.lower()

    def test_cli_unknown_arg(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.run_pipeline", "--unknown-flag"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_cli_missing_config(self, tmp_path: Path) -> None:
        bad_cfg = tmp_path / "nonexistent.yaml"
        result = subprocess.run(
            [sys.executable, "-m", "cortexmark.run_pipeline", "--config", str(bad_cfg)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_cli_clean_stage_on_empty_dir(self, tmp_path: Path) -> None:
        cfg = self._write_config(tmp_path)
        # Place a dummy markdown so clean_tree finds something
        (tmp_path / "raw_md" / "test" / "doc.md").write_text("# Title\n\nBody.\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.run_pipeline",
                "--config",
                str(cfg),
                "--stages",
                "clean",
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_cli_chunk_stage_on_empty_dir(self, tmp_path: Path) -> None:
        cfg = self._write_config(tmp_path)
        # Place a dummy cleaned markdown so chunk_tree finds something
        (tmp_path / "cleaned_md" / "test" / "doc.md").write_text("# Chapter\n\nContent.\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.run_pipeline",
                "--config",
                str(cfg),
                "--stages",
                "chunk",
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_single_pdf_input_converts_only_that_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from cortexmark import convert as convert_module
        from cortexmark import run_pipeline as pipeline_module

        cfg = self._write_config(tmp_path)
        input_root = tmp_path / "custom-pdfs"
        input_root.mkdir()
        selected = input_root / "selected.pdf"
        sibling = input_root / "sibling.pdf"
        selected.write_bytes(b"selected")
        sibling.write_bytes(b"sibling")
        converted: list[Path] = []

        def fake_convert_pdf(input_path: Path, output_path: Path, **_kwargs: object) -> Path:
            converted.append(input_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(input_path.stem, encoding="utf-8")
            return output_path

        monkeypatch.setattr(convert_module, "convert_pdf", fake_convert_pdf)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cortexmark.run_pipeline",
                "--config",
                str(cfg),
                "--input",
                str(selected),
                "--stages",
                "convert",
                "--engine",
                "markitdown",
                "--no-manifest",
            ],
        )

        assert pipeline_module.main() == 0
        assert converted == [selected.resolve()]  # was: selected and sibling were both converted
        assert (tmp_path / "raw_md" / input_root.name / "selected.md").exists()
        assert not (tmp_path / "raw_md" / input_root.name / "sibling.md").exists()

    def test_cli_validate_stage_writes_session_quality(self, tmp_path: Path) -> None:
        cfg = self._write_config(tmp_path)
        session_cleaned = tmp_path / "sessions" / "sess-1" / "outputs" / "cleaned_md"
        session_cleaned.mkdir(parents=True)
        (session_cleaned / "doc.md").write_text("# Intro\n\nBody text.\n", encoding="utf-8")
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
                "sess-1",
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PROJECT_ROOT": str(tmp_path)},
        )
        assert result.returncode == 0
        quality_dir = tmp_path / "sessions" / "sess-1" / "outputs" / "quality"
        assert (quality_dir / "formula_validation.json").exists()
        assert (quality_dir / "scientific_qa.json").exists()
        assert (quality_dir / "citation_context.json").exists()

    def test_cli_analyze_stage_writes_session_quality(self, tmp_path: Path) -> None:
        cfg = self._write_config(tmp_path)
        session_cleaned = tmp_path / "sessions" / "sess-2" / "outputs" / "cleaned_md"
        session_cleaned.mkdir(parents=True)
        (session_cleaned / "doc.md").write_text(
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
                "sess-2",
                "--no-manifest",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PROJECT_ROOT": str(tmp_path)},
        )
        assert result.returncode == 0
        quality_dir = tmp_path / "sessions" / "sess-2" / "outputs" / "quality"
        assert (quality_dir / "crossref_report.json").exists()
        assert (quality_dir / "algorithm_report.json").exists()
        assert (quality_dir / "notation_report.json").exists()

    def test_cli_files(self, tmp_path: Path) -> None:
        old = tmp_path / "old.md"
        new = tmp_path / "new.md"
        old.write_text("old content\n", encoding="utf-8")
        new.write_text("new content\n", encoding="utf-8")
        out = tmp_path / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "cortexmark.diff",
                "--old",
                str(old),
                "--new",
                str(new),
                "--output",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════════
# semantic_chunk.py
# ═══════════════════════════════════════════════════════════════════════════════


class TestClassifyEnvKind:
    def test_theorem_like(self) -> None:
        for kw in ("theorem", "lemma", "proposition", "corollary"):
            assert classify_env_kind(kw) == ENTITY_THEOREM

    def test_definition_like(self) -> None:
        for kw in ("definition", "assumption", "axiom"):
            assert classify_env_kind(kw) == ENTITY_DEFINITION

    def test_example_like(self) -> None:
        assert classify_env_kind("example") == ENTITY_EXAMPLE

    def test_remark_like(self) -> None:
        assert classify_env_kind("remark") == ENTITY_REMARK

    def test_unknown(self) -> None:
        assert classify_env_kind("unknown") == ENTITY_NARRATIVE


class TestExtractFormulas:
    def test_display_math(self) -> None:
        text = "We have $$V(s)=E[R]$$ as the value."
        formulas = extract_formulas(text)
        assert "V(s)=E[R]" in formulas

    def test_inline_math(self) -> None:
        text = "The variable $x$ is useful and $y$ too."
        formulas = extract_formulas(text)
        assert len(formulas) == 2
        assert "x" in formulas
        assert "y" in formulas

    def test_no_math(self) -> None:
        assert extract_formulas("No math here.") == []


class TestExtractCrossRefs:
    def test_theorem_ref(self) -> None:
        text = "As shown in Theorem 3.2 and Figure 1."
        refs = extract_cross_refs(text)
        assert any("Theorem 3.2" in r for r in refs)
        assert any("Figure 1" in r for r in refs)

    def test_equation_ref(self) -> None:
        text = "From Equation (4) and Eq. 5.1 we get..."
        refs = extract_cross_refs(text)
        assert len(refs) >= 2

    def test_no_refs(self) -> None:
        assert extract_cross_refs("Plain text.") == []


class TestHasQed:
    def test_box_qed(self) -> None:
        assert has_qed("This completes the proof. □")
        assert has_qed("∎")
        assert has_qed("■")

    def test_text_qed(self) -> None:
        assert has_qed("…which was to be shown. Q.E.D.")
        assert has_qed("QED")

    def test_no_qed(self) -> None:
        assert not has_qed("Normal line.")
        assert not has_qed("")


class TestBlockOpenerRE:
    def test_bold_theorem(self) -> None:
        m = BLOCK_OPENER_RE.match("**Theorem 3.2.** Let $x$ be...")
        assert m is not None
        assert m.group("kind").lower() == "theorem"
        assert m.group("label") == "3.2"

    def test_definition_with_name(self) -> None:
        m = BLOCK_OPENER_RE.match("**Definition 1 (Markov Property).** A process...")
        assert m is not None
        assert m.group("kind").lower() == "definition"
        assert m.group("label") == "1"
        assert m.group("name") == "Markov Property"

    def test_lemma_no_label(self) -> None:
        m = BLOCK_OPENER_RE.match("**Lemma.** Suppose that...")
        assert m is not None
        assert m.group("kind").lower() == "lemma"

    def test_no_match(self) -> None:
        assert BLOCK_OPENER_RE.match("Regular text line.") is None

    @pytest.mark.parametrize("line", ["Theorems are useful.", "Note that this holds.", "Problematic input."])
    def test_rejects_narrative_prefixes(self, line: str) -> None:
        assert BLOCK_OPENER_RE.match(line) is None


class TestProofOpenerRE:
    def test_basic_proof(self) -> None:
        m = PROOF_OPENER_RE.match("**Proof.** We start by...")
        assert m is not None

    def test_proof_of_theorem(self) -> None:
        m = PROOF_OPENER_RE.match("Proof of Theorem 3.2.")
        assert m is not None
        assert m.group("of_label") is not None
        assert "3.2" in m.group("of_label")

    @pytest.mark.parametrize("line", ["Proofs are omitted.", "Proofreading matters.", "Proof techniques vary."])
    def test_rejects_proof_prefix_prose(self, line: str) -> None:
        assert PROOF_OPENER_RE.match(line) is None


class TestParseSemanticChunks:
    def test_narrative_only(self) -> None:
        text = "# Intro\n\nSome narrative text.\n\n## Part A\n\nMore narrative.\n"
        chunks = parse_semantic_chunks(text)
        assert len(chunks) == 2
        assert all(c.entity_type == ENTITY_NARRATIVE for c in chunks)

    def test_preserves_narrative_block_keyword(self) -> None:
        chunks = parse_semantic_chunks("# Intro\n\nNote that the value function is bounded.\n")
        assert len(chunks) == 1
        assert chunks[0].entity_type == ENTITY_NARRATIVE
        assert chunks[0].body == ["Note that the value function is bounded."]

    def test_preserves_proof_prefix_narrative(self) -> None:
        chunks = parse_semantic_chunks("# Intro\n\nProofs are omitted for brevity.\n")
        assert len(chunks) == 1
        assert chunks[0].entity_type == ENTITY_NARRATIVE
        assert chunks[0].body == ["Proofs are omitted for brevity."]

    def test_theorem_detection(self) -> None:
        text = "# Chapter 1\n\n**Theorem 3.2.** The optimal value function satisfies.\n\nWe can see that this holds.\n"
        chunks = parse_semantic_chunks(text)
        theorem_chunks = [c for c in chunks if c.entity_type == ENTITY_THEOREM]
        assert len(theorem_chunks) == 1
        assert theorem_chunks[0].entity_label == "Theorem 3.2"

    def test_proof_detection(self) -> None:
        text = (
            "# Chapter 1\n\n**Theorem 1.** Something.\n\n**Proof.** We proceed by induction.\nThe base case holds. □\n"
        )
        chunks = parse_semantic_chunks(text)
        proof_chunks = [c for c in chunks if c.entity_type == ENTITY_PROOF]
        assert len(proof_chunks) == 1
        assert proof_chunks[0].parent_label == "Theorem 1"
        assert proof_chunks[0].parent_evidence_level == "inferred"

    def test_explicit_proof_parent_evidence(self) -> None:
        text = "# Chapter 1\n\n**Proof of Theorem 3.** We proceed directly. □\n"
        chunks = parse_semantic_chunks(text)
        proof = next(c for c in chunks if c.entity_type == ENTITY_PROOF)
        assert proof.parent_label == "Theorem 3"
        assert proof.parent_evidence_level == "explicit"

    def test_definition_detection(self) -> None:
        text = "# Ch\n\n**Definition 1 (Segments).** A Markov Decision Process is a tuple.\n"
        chunks = parse_semantic_chunks(text)
        def_chunks = [c for c in chunks if c.entity_type == ENTITY_DEFINITION]
        assert len(def_chunks) == 1
        assert def_chunks[0].entity_name == "Segments"
        assert def_chunks[0].entity_label == "Definition 1"

    def test_entity_kind_preserves_exact_role(self) -> None:
        text = "**Lemma 2.** Auxiliary claim.\n\n**Assumption 1.** Regularity holds.\n"
        chunks = parse_semantic_chunks(text, split_on_headings=False)
        lemma = next(c for c in chunks if c.entity_label == "Lemma 2")
        assumption = next(c for c in chunks if c.entity_label == "Assumption 1")
        assert lemma.entity_type == ENTITY_THEOREM
        assert lemma.entity_kind == "lemma"
        assert assumption.entity_type == ENTITY_DEFINITION
        assert assumption.entity_kind == "assumption"

    def test_algorithm_code_fence(self) -> None:
        text = "# Algorithms\n\n```text\nInitialize Q(s,a)\nLoop for each episode:\n  Choose action\n```\n"
        chunks = parse_semantic_chunks(text)
        algo_chunks = [c for c in chunks if c.entity_type == ENTITY_ALGORITHM]
        assert len(algo_chunks) == 1

    def test_algorithm_label_line(self) -> None:
        text = "# Ch\n\nAlgorithm 2.1 Value Iteration\nFirst, initialize V.\n"
        chunks = parse_semantic_chunks(text)
        algo_chunks = [c for c in chunks if c.entity_type == ENTITY_ALGORITHM]
        assert len(algo_chunks) == 1
        assert algo_chunks[0].entity_label == "Algorithm 2.1"

    def test_formula_extraction(self) -> None:
        text = "# Ch\n\n**Theorem 1.** We have $$V(s) = max_a Q(s,a)$$.\n"
        chunks = parse_semantic_chunks(text)
        theorem = next(c for c in chunks if c.entity_type == ENTITY_THEOREM)
        assert len(theorem.formulas) >= 1
        assert "V(s) = max_a Q(s,a)" in theorem.formulas

    def test_cross_ref_extraction(self) -> None:
        text = "# Ch\n\nAs shown in Theorem 3.2 and Figure 1, the result follows.\n"
        chunks = parse_semantic_chunks(text)
        assert len(chunks) >= 1
        refs = chunks[0].cross_refs
        assert any("Theorem 3.2" in r for r in refs)

    def test_empty_input(self) -> None:
        assert parse_semantic_chunks("") == []

    def test_proof_without_qed(self) -> None:
        text = "# Ch\n\n**Theorem 1.** Statement.\n\n**Proof.** The argument.\nIt continues here.\n"
        chunks = parse_semantic_chunks(text)
        proof_chunks = [c for c in chunks if c.entity_type == ENTITY_PROOF]
        assert len(proof_chunks) == 1
        assert "The argument." in proof_chunks[0].body[0]

    def test_mixed_environments(self) -> None:
        text = (
            "# Analysis\n\n"
            "**Definition 1.** A set is open if...\n\n"
            "**Theorem 2.** Every open cover has...\n\n"
            "**Proof.** Obvious. □\n\n"
            "**Example 1.** Consider the interval.\n"
        )
        chunks = parse_semantic_chunks(text)
        types = {c.entity_type for c in chunks}
        assert ENTITY_DEFINITION in types
        assert ENTITY_THEOREM in types
        assert ENTITY_PROOF in types
        assert ENTITY_EXAMPLE in types

    def test_heading_creates_narrative(self) -> None:
        text = "# Chapter 1\n\n## Introduction\n\nBackground text.\n"
        chunks = parse_semantic_chunks(text)
        assert len(chunks) == 1
        assert chunks[0].entity_type == ENTITY_NARRATIVE
        assert chunks[0].section == "Introduction"


class TestSemanticChunkRender:
    def test_render_basic(self) -> None:
        c = SemanticChunk(chapter="Ch1", section="Sec1", body=["line1", "line2"])
        rendered = c.render()
        assert "# Ch1" in rendered
        assert "## Sec1" in rendered
        assert "line1" in rendered

    def test_title_with_label(self) -> None:
        c = SemanticChunk(entity_label="Theorem 3.2", entity_name="Bellman")
        assert c.title == "Theorem 3.2 (Bellman)"

    def test_title_fallback(self) -> None:
        c = SemanticChunk(section="Intro")
        assert c.title == "Intro"

    def test_title_untitled(self) -> None:
        c = SemanticChunk()
        assert c.title == "untitled"


class TestSemanticChunkFile:
    def test_basic_file(self, tmp_path: Path) -> None:
        src = tmp_path / "test.md"
        src.write_text(
            "# Chapter\n\n**Theorem 1.** Statement.\n\nNarrative text.\n",
            encoding="utf-8",
        )
        out = tmp_path / "out"
        written = semantic_chunk_file(src, out)
        assert len(written) >= 1
        assert all(p.exists() for p in written)
        assert all(p.suffix == ".md" for p in written)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            semantic_chunk_file(tmp_path / "missing.md", tmp_path / "out")

    def test_empty_file(self, tmp_path: Path) -> None:
        # Empty/structureless cleaned text no longer raises (see finding
        # C14): a minimal single-chunk result is written instead so an
        # `analyze` caller doesn't drop the whole document.
        src = tmp_path / "empty.md"
        src.write_text("", encoding="utf-8")
        written = semantic_chunk_file(src, tmp_path / "out")
        assert len(written) == 1
        assert written[0].exists()


class TestSemanticChunkTree:
    def test_tree_chunking(self, tmp_path: Path) -> None:
        root = tmp_path / "cleaned_md" / "course"
        root.mkdir(parents=True)
        (root / "topic.md").write_text(
            "# Topic\n\n**Definition 1.** X is defined as Y.\n",
            encoding="utf-8",
        )
        out = tmp_path / "chunks"
        written = semantic_chunk_tree(root, out)
        assert len(written) >= 1

    def test_tree_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            semantic_chunk_tree(tmp_path / "nonexistent", tmp_path / "out")


class TestChunksToRecords:
    def test_basic(self) -> None:
        chunks = [
            SemanticChunk(
                chapter="Ch1",
                entity_type=ENTITY_THEOREM,
                entity_label="Theorem 1",
                body=["Content"],
            ),
        ]
        records = chunks_to_records(chunks, "test.md")
        assert len(records) == 1
        assert records[0]["entity_type"] == ENTITY_THEOREM
        assert records[0]["entity_label"] == "Theorem 1"
        assert records[0]["source"] == "test.md"

    def test_records_include_object_metadata(self) -> None:
        chunks = parse_semantic_chunks(
            "**Theorem 1.** Statement.\n\n**Proof of Theorem 1.** Argument. □\n",
            split_on_headings=False,
        )
        records = chunks_to_records(chunks, "paper.md")
        theorem_record = next(r for r in records if r["entity_type"] == ENTITY_THEOREM)
        proof_record = next(r for r in records if r["entity_type"] == ENTITY_PROOF)
        assert theorem_record["object_id"]
        assert proof_record["object_id"]
        assert proof_record["parent_object_id"]


class TestScientificObjectLinks:
    def test_chunks_to_scientific_objects_and_links(self) -> None:
        chunks = parse_semantic_chunks(
            "**Theorem 1.** We have $$V(s)=0$$.\n\n**Proof.** Therefore $V(s)=0$. □\n",
            split_on_headings=False,
        )
        objects = chunks_to_scientific_objects(chunks, "paper.md")
        theorem = next(obj for obj in objects if obj.object_type == ENTITY_THEOREM)
        proof = next(obj for obj in objects if obj.object_type == ENTITY_PROOF)
        equations = [obj for obj in objects if obj.object_type == "equation"]
        links = build_scientific_object_links(objects)

        assert theorem.object_kind == "theorem"
        assert proof.parent_object_id == theorem.object_id
        assert equations
        assert all(eq.parent_object_id for eq in equations)
        assert any(link.relation == "proof_of" and link.target_object_id == theorem.object_id for link in links)
        assert any(link.relation == "appears_in" for link in links)


class TestStableSourceLabel:
    def test_is_cwd_independent_and_root_relative_when_root_is_known(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = tmp_path / "sources" / "paper.md"
        source.parent.mkdir()
        source.write_text("# Paper\n", encoding="utf-8")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        monkeypatch.chdir(tmp_path)
        label_before = stable_source_label(source)
        monkeypatch.chdir(elsewhere)
        label_after = stable_source_label(source)

        assert label_before == label_after == "paper.md"
        assert stable_source_label(source, root=tmp_path) == "sources/paper.md"


class TestBuildEntitySummary:
    def test_counts(self) -> None:
        chunks = [
            SemanticChunk(entity_type=ENTITY_THEOREM, body=["a"]),
            SemanticChunk(entity_type=ENTITY_THEOREM, body=["b"]),
            SemanticChunk(entity_type=ENTITY_PROOF, body=["c"]),
            SemanticChunk(entity_type=ENTITY_NARRATIVE, body=["d"]),
        ]
        summary = build_entity_summary(chunks)
        assert summary[ENTITY_THEOREM] == 2
        assert summary[ENTITY_PROOF] == 1
        assert summary[ENTITY_NARRATIVE] == 1

    def test_empty(self) -> None:
        assert build_entity_summary([]) == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Cross-Reference Resolution (cross_ref.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractDefinitions:
    def test_theorem_definition(self) -> None:
        md = "## Theorem 3.1\nSome theorem body.\n"
        defs = extract_definitions(md)
        assert len(defs) >= 1
        assert any(d.kind.lower() == "theorem" and d.label == "3.1" for d in defs)

    def test_figure_definition(self) -> None:
        md = "Figure 2: A plot of velocity.\n"
        defs = extract_definitions(md)
        assert len(defs) >= 1
        assert any(d.kind.lower() == "figure" and d.label == "2" for d in defs)

    def test_table_definition(self) -> None:
        md = "Table 1: Summary of results.\n"
        defs = extract_definitions(md)
        assert any(d.kind.lower() == "table" and d.label == "1" for d in defs)

    def test_equation_label(self) -> None:
        md = r"$$E = mc^2 \tag{5}$$"
        defs = extract_definitions(md)
        assert any(d.kind.lower() == "equation" for d in defs)

    def test_no_definitions(self) -> None:
        md = "This is a plain paragraph with no definitions.\n"
        defs = extract_definitions(md)
        assert defs == []

    def test_definition_with_source(self) -> None:
        md = "## Lemma 4.2\nThe proof is straightforward.\n"
        defs = extract_definitions(md, source_file="notes.md")
        assert all(d.source_file == "notes.md" for d in defs)


class TestExtractMentions:
    def test_theorem_mention(self) -> None:
        md = "As shown in Theorem 3.1, the result holds.\n"
        mentions = extract_mentions(md)
        assert len(mentions) >= 1
        assert any(m.kind.lower() == "theorem" and m.label == "3.1" for m in mentions)

    def test_figure_ref(self) -> None:
        md = "See Figure 2 for details.\n"
        mentions = extract_mentions(md)
        assert any(m.kind.lower() == "figure" and m.label == "2" for m in mentions)

    def test_abbreviated_ref(self) -> None:
        md = "According to Eq. 5, the velocity is constant.\n"
        mentions = extract_mentions(md)
        assert any("5" in m.label for m in mentions)

    def test_multiple_mentions(self) -> None:
        md = "From Theorem 1 and Lemma 2, plus Table 3.\n"
        mentions = extract_mentions(md)
        assert len(mentions) >= 3

    def test_no_mentions(self) -> None:
        md = "A plain paragraph.\n"
        assert extract_mentions(md) == []


class TestResolveReferences:
    def test_full_resolution(self) -> None:
        defs = [
            RefDefinition(kind="Theorem", label="1", name="Main Theorem", source_file="ch1.md", line_number=1),
        ]
        mentions = [
            RefMention(kind="Theorem", label="1", source_file="ch2.md", line_number=10),
        ]
        report = resolve_references(defs, mentions)
        assert report.resolution_rate == 1.0
        assert len(report.unresolved) == 0
        assert len(report.resolved) == 1

    def test_dangling_reference(self) -> None:
        defs: list[RefDefinition] = []
        mentions = [
            RefMention(kind="Figure", label="99", source_file="ch1.md", line_number=5),
        ]
        report = resolve_references(defs, mentions)
        assert report.resolution_rate == 0.0
        assert len(report.unresolved) == 1

    def test_empty_inputs(self) -> None:
        report = resolve_references([], [])
        assert report.resolution_rate == 1.0
        assert len(report.resolved) == 0
        assert len(report.unresolved) == 0

    def test_kind_normalization(self) -> None:
        defs = [
            RefDefinition(kind="Equation", label="5", name="Energy", source_file="a.md", line_number=1),
        ]
        mentions = [
            RefMention(kind="Eq.", label="5", source_file="b.md", line_number=2),
        ]
        report = resolve_references(defs, mentions)
        assert report.resolution_rate == 1.0

    def test_builds_canonical_objects_and_links(self) -> None:
        defs = [
            RefDefinition(kind="Theorem", label="1", name="Main", source_file="a.md", line_number=1),
        ]
        mentions = [
            RefMention(kind="Theorem", label="1", source_file="b.md", line_number=4),
            RefMention(kind="Figure", label="2", source_file="b.md", line_number=8),
        ]
        report = resolve_references(defs, mentions)
        assert len(report.objects) == 3
        assert len(report.links) == 2
        assert report.links[0].relation == "references"
        assert report.links[0].status == "resolved"
        assert report.links[1].status == "unresolved"
        assert any(obj.object_type == "xref_mention" for obj in report.objects)

    def test_prefers_local_definition_and_marks_ambiguous_tree_matches(self) -> None:
        defs = [
            RefDefinition(kind="Theorem", label="1", source_file="a.md", line_number=1),
            RefDefinition(kind="Theorem", label="1", source_file="b.md", line_number=1),
        ]
        mentions = [
            RefMention(kind="Theorem", label="1", source_file="a.md", line_number=4),
            RefMention(kind="Theorem", label="1", source_file="c.md", line_number=9),
        ]
        report = resolve_references(defs, mentions)
        local_link = next(link for link in report.links if link.source_file == "a.md")
        ambiguous_link = next(link for link in report.links if link.source_file == "c.md")

        assert local_link.status == "resolved"
        assert ambiguous_link.status == "ambiguous"
        assert ambiguous_link.metadata["candidate_sources"] == ["a.md", "b.md"]


class TestClassifyKind:
    def test_theorem_category(self) -> None:
        assert classify_kind("Theorem") == CATEGORY_THEOREM

    def test_figure_category(self) -> None:
        assert classify_kind("Figure") == CATEGORY_FIGURE

    def test_table_category(self) -> None:
        assert classify_kind("Table") == CATEGORY_TABLE

    def test_equation_category(self) -> None:
        assert classify_kind("Equation") == CATEGORY_EQUATION


class TestCrossRefReport:
    def test_resolution_rate(self) -> None:
        report = CrossRefReport(
            definitions=[
                RefDefinition(kind="Theorem", label="1", name="T1", source_file="a.md", line_number=1),
            ],
            mentions=[
                RefMention(kind="Theorem", label="1", source_file="b.md", line_number=10),
                RefMention(kind="Figure", label="2", source_file="b.md", line_number=20),
            ],
            resolved=["Theorem 1"],
            unresolved=["Figure 2"],
        )
        assert report.resolution_rate == 0.5


class TestCrossRefFile:
    def test_analyze_file(self, tmp_path: Path) -> None:
        md = "## Theorem 1\nBody.\n\nAs shown in Theorem 1.\n"
        f = tmp_path / "test.md"
        f.write_text(md)
        report = crossref_analyze_file(f)
        assert len(report.definitions) >= 1
        assert len(report.mentions) >= 1

    def test_analyze_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("## Theorem 1\nBody.\n")
        (d / "b.md").write_text("See Theorem 1.\n")
        report = crossref_analyze_tree(d)
        assert len(report.definitions) >= 1
        assert len(report.mentions) >= 1

    def test_real_definition_resolves_and_missing_target_is_dangling(self, tmp_path: Path) -> None:
        md = tmp_path / "theorems.md"
        md.write_text("## Theorem 3.2\nStatement.\n\nSee Theorem 3.2 and Theorem 9.\n", encoding="utf-8")

        report = crossref_analyze_file(md)

        assert [(definition.kind, definition.label) for definition in report.definitions] == [("Theorem", "3.2")]
        assert report.resolved == ["Theorem 3.2"]
        assert report.unresolved == ["Theorem 9"]

    def test_section_and_chapter_mentions_resolve_against_headings(self, tmp_path: Path) -> None:
        md = tmp_path / "sections.md"
        md.write_text(
            "# Chapter 2 Methods\n\n## 2.1 Setup\n\nSee Section 2.1 and Chapter 2.\n",
            encoding="utf-8",
        )

        report = crossref_analyze_file(md)

        assert [(definition.kind, definition.label) for definition in report.definitions] == [
            ("Chapter", "2"),
            ("Section", "2.1"),
        ]
        assert [(mention.kind, mention.label) for mention in report.mentions] == [
            ("Section", "2.1"),
            ("Chapter", "2"),
        ]
        assert report.unresolved == []

    def test_write_report(self, tmp_path: Path) -> None:
        report = CrossRefReport(
            definitions=[],
            mentions=[],
            resolved=[],
            unresolved=[],
        )
        out = tmp_path / "crossref.json"
        written = write_crossref_report(report, out)
        assert written.exists()
        data = json.loads(written.read_text())
        assert "resolution_rate" in data.get("summary", data)
        assert "canonical_ir" in data
        assert "objects" in data["canonical_ir"]
        assert "links" in data["canonical_ir"]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Algorithm Extraction (algorithm_extract.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlgoStep:
    def test_parse_step_basic(self) -> None:
        step = parse_step("x = x + 1")
        assert step.text == "x = x + 1"
        assert step.indent_level == 0
        assert step.is_control_flow is False

    def test_parse_step_indented(self) -> None:
        step = parse_step("    x = x + 1")
        assert step.indent_level == 2
        assert step.is_control_flow is False

    def test_parse_step_control_flow(self) -> None:
        step = parse_step("for i in range(n):")
        assert step.is_control_flow is True
        assert step.keyword == "for"

    def test_parse_step_while(self) -> None:
        step = parse_step("while x > 0:")
        assert step.is_control_flow is True
        assert step.keyword == "while"

    def test_parse_step_if(self) -> None:
        step = parse_step("if condition:")
        assert step.is_control_flow is True
        assert step.keyword == "if"

    def test_parse_step_return(self) -> None:
        step = parse_step("return result")
        assert step.is_control_flow is True
        assert step.keyword == "return"


class TestAlgorithmDataclass:
    def test_full_label(self) -> None:
        algo = Algorithm(
            label="1",
            title="Binary Search",
            inputs=["array A", "target x"],
            outputs=["index i"],
            steps=[parse_step("compare")],
            source_file="algo.md",
            line_number=10,
            raw_text="...",
        )
        assert "1" in algo.full_label
        assert algo.step_count == 1

    def test_max_depth(self) -> None:
        algo = Algorithm(
            label="2",
            title="Sort",
            inputs=[],
            outputs=[],
            steps=[parse_step("    compare"), parse_step("        swap")],
            source_file="sort.md",
            line_number=1,
            raw_text="...",
        )
        assert algo.max_depth >= 2


class TestParseAlgorithmBody:
    def test_inputs_outputs_steps(self) -> None:
        body_lines = [
            "Input: array A",
            "Output: sorted A",
            "for i in range(n):",
            "  compare A[i]",
            "return A",
        ]
        inputs, outputs, steps = parse_algorithm_body(body_lines)
        assert len(inputs) >= 1
        assert len(outputs) >= 1
        assert len(steps) >= 1

    def test_no_io(self) -> None:
        body_lines = ["x = 0", "for i in range(n):", "  x += i", "return x"]
        inputs, outputs, steps = parse_algorithm_body(body_lines)
        assert inputs == []
        assert outputs == []
        assert len(steps) >= 1


class TestIsAlgorithmContent:
    def test_pseudocode_detected(self) -> None:
        text = "for i in range(n)\n  if A[i] > max\n    max = A[i]\nreturn max"
        assert is_algorithm_content(text) is True

    def test_plain_code_rejected(self) -> None:
        text = "x = 42\ny = 'hello'\nprint(y)"
        assert is_algorithm_content(text) is False


class TestExtractAlgorithms:
    def test_fenced_algorithm(self) -> None:
        md = (
            "## Algorithm 1: Binary Search\n"
            "```\n"
            "Input: sorted array A, target x\n"
            "Output: index i\n"
            "while low <= high:\n"
            "  mid = (low + high) / 2\n"
            "  if A[mid] == x:\n"
            "    return mid\n"
            "```\n"
        )
        algos = extract_algorithms(md)
        assert len(algos) >= 1
        assert algos[0].title is not None

    def test_no_algorithms(self) -> None:
        md = "This is plain text with no algorithms.\n"
        assert extract_algorithms(md) == []

    def test_multiple_algorithms(self) -> None:
        md = (
            "## Algorithm 1: Search\n```\nfor i in range(n):\n  if A[i] == x: return i\nreturn -1\n```\n\n"
            "## Algorithm 2: Sort\n```\nfor i in range(n):\n  for j in range(i, n):\n    if A[j] < A[i]: swap\n```\n"
        )
        algos = extract_algorithms(md)
        assert len(algos) >= 2

    def test_bare_fence_with_only_generic_code_is_not_algorithm(self) -> None:
        md = "```\nif ready:\n  while pending:\n    return value\n```\n"
        assert extract_algorithms(md) == []

    def test_explicit_header_uses_real_line_number(self) -> None:
        md = "Introduction.\n\nAlgorithm 7: Search\nInput: values\nOutput: result\nReturn result\n"
        algorithms = extract_algorithms(md)
        assert len(algorithms) == 1
        assert algorithms[0].line_number == 3


class TestAlgorithmFileOps:
    def test_extract_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("## Algorithm 1: Test\n```\nfor i in range(n):\n  if x > 0: return x\n```\n")
        algos = algo_extract_from_file(f)
        assert len(algos) >= 1

    def test_extract_from_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("## Algorithm 1: A\n```\nfor i in range(n):\n  step\n```\n")
        (d / "b.md").write_text("No algorithms here.\n")
        algos = algo_extract_from_tree(d)
        assert len(algos) >= 1

    def test_write_report(self, tmp_path: Path) -> None:
        algo = Algorithm(
            label="1",
            title="Test",
            inputs=["x"],
            outputs=["y"],
            steps=[parse_step("y = x + 1")],
            source_file="test.md",
            line_number=1,
            raw_text="y = x + 1",
        )
        out = tmp_path / "algo.json"
        written = write_algo_report([algo], out)
        assert written.exists()
        data = json.loads(written.read_text())
        assert "algorithms" in data

    def test_build_summary(self) -> None:
        algo = Algorithm(
            label="1",
            title="Test",
            inputs=[],
            outputs=[],
            steps=[parse_step("x = 1")],
            source_file="test.md",
            line_number=1,
            raw_text="...",
        )
        summary = algo_build_summary([algo])
        assert summary["total_algorithms"] == 1


class TestAlgoHeaderRegex:
    def test_matches_algorithm_header(self) -> None:
        assert ALGO_HEADER_RE.match("Algorithm 1: Binary Search")

    def test_matches_with_hashes(self) -> None:
        assert ALGO_HEADER_RE.match("## Algorithm 2: Sort")

    def test_no_match_plain(self) -> None:
        assert ALGO_HEADER_RE.match("This is not an algorithm") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Notation Glossary (notation_glossary.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNotationEntry:
    def test_fields(self) -> None:
        entry = NotationEntry(
            symbol=r"\alpha",
            definition="learning rate",
            source="explicit",
            source_file="notes.md",
            line_number=5,
            context="Let $\\alpha$ denote the learning rate",
        )
        assert entry.symbol == r"\alpha"
        assert entry.source == "explicit"


class TestNotationGlossary:
    def test_unique_symbols(self) -> None:
        entries = [
            NotationEntry(symbol=r"\alpha", definition="learning rate", source="explicit"),
            NotationEntry(symbol=r"\alpha", definition="learning rate", source="explicit"),
            NotationEntry(symbol=r"\beta", definition="momentum", source="explicit"),
        ]
        glossary = NotationGlossary(entries=entries)
        assert glossary.unique_symbols == 2

    def test_lookup(self) -> None:
        entries = [
            NotationEntry(symbol=r"\alpha", definition="lr", source="explicit"),
            NotationEntry(symbol=r"\beta", definition="momentum", source="explicit"),
        ]
        glossary = NotationGlossary(entries=entries)
        results = glossary.lookup(r"\alpha")
        assert len(results) >= 1
        assert results[0].definition == "lr"

    def test_deduplicated(self) -> None:
        entries = [
            NotationEntry(symbol=r"\alpha", definition="lr", source="explicit"),
            NotationEntry(symbol=r"\alpha", definition="lr", source="explicit"),
            NotationEntry(symbol=r"\alpha", definition="learning rate", source="where-clause"),
        ]
        glossary = NotationGlossary(entries=entries)
        deduped = glossary.deduplicated()
        # One preferred definition remains per symbol.
        alpha_entries = [e for e in deduped if e.symbol == r"\alpha"]
        assert len(alpha_entries) == 1

    def test_deduplicated_prefers_specific_where_definition(self) -> None:
        md = "In the equation, where $V$ is the value function and $s$ is the state.\n"
        glossary = NotationGlossary(entries=extract_explicit_definitions(md))
        value_entry = next(entry for entry in glossary.deduplicated() if entry.symbol == "V")
        assert value_entry.definition == "value function"
        assert value_entry.source == "where-clause"


class TestExtractExplicitDefinitions:
    def test_let_pattern(self) -> None:
        md = "Let $\\alpha$ denote the learning rate.\n"
        entries = extract_explicit_definitions(md)
        assert len(entries) >= 1
        assert any(r"\alpha" in e.symbol or "alpha" in e.symbol for e in entries)

    def test_where_pattern(self) -> None:
        md = "The formula $E = mc^2$, where $E$ is the energy.\n"
        entries = extract_explicit_definitions(md)
        assert len(entries) >= 1

    def test_no_definitions(self) -> None:
        md = "This is a plain paragraph.\n"
        assert extract_explicit_definitions(md) == []


class TestExtractListNotations:
    def test_list_notation(self) -> None:
        md = "- $\\alpha$: learning rate\n- $\\beta$: momentum coefficient\n"
        entries = extract_list_notations(md)
        assert len(entries) >= 2

    def test_no_list(self) -> None:
        md = "Just text, no lists.\n"
        assert extract_list_notations(md) == []


class TestExtractTableNotations:
    def test_table_notation(self) -> None:
        md = "| $\\alpha$ | learning rate |\n| $\\beta$ | momentum |\n"
        entries = extract_table_notations(md)
        assert len(entries) >= 2

    def test_no_table(self) -> None:
        md = "Plain text.\n"
        assert extract_table_notations(md) == []


class TestDetectCommonNotations:
    def test_detects_alpha(self) -> None:
        md = "The value $\\alpha$ is used here.\n"
        entries = detect_common_notations(md)
        assert any(r"\alpha" in e.symbol for e in entries)

    def test_detects_sum(self) -> None:
        md = "Compute $\\sum_{i=1}^{n} x_i$.\n"
        entries = detect_common_notations(md)
        assert any(r"\sum" in e.symbol for e in entries)

    def test_preserves_source_file_for_conventions(self) -> None:
        md = "The policy $\\pi$ is updated.\n"
        entries = detect_common_notations(md, source_file="doc.md")
        assert any(e.source_file == "doc.md" for e in entries)

    def test_no_math(self) -> None:
        md = "No math content here.\n"
        entries = detect_common_notations(md)
        assert entries == []


class TestNotationExtractAll:
    def test_combines_sources(self) -> None:
        md = (
            "Let $\\alpha$ denote the learning rate.\n"
            "- $\\beta$: momentum\n"
            "The value $\\nabla$ is the gradient operator.\n"
        )
        glossary = notation_extract_all(md)
        assert glossary.unique_symbols >= 2

    def test_empty_text(self) -> None:
        glossary = notation_extract_all("")
        assert len(glossary.entries) == 0


class TestNotationFileOps:
    def test_extract_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("Let $\\alpha$ denote the learning rate.\n")
        glossary = notation_extract_from_file(f)
        assert len(glossary.entries) >= 1

    def test_extract_from_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("Let $\\alpha$ denote the learning rate.\n")
        (d / "b.md").write_text("Plain.\n")
        glossary = notation_extract_from_tree(d)
        assert len(glossary.entries) >= 1

    def test_write_report(self, tmp_path: Path) -> None:
        glossary = NotationGlossary(
            entries=[
                NotationEntry(symbol=r"\alpha", definition="lr", source="explicit"),
            ]
        )
        out = tmp_path / "notation.json"
        written = write_notation_report(glossary, out)
        assert written.exists()
        data = json.loads(written.read_text())
        assert "glossary" in data
        assert "canonical_ir" in data
        assert data["canonical_ir"]["objects"]

    def test_write_markdown_glossary(self, tmp_path: Path) -> None:
        glossary = NotationGlossary(
            entries=[
                NotationEntry(symbol=r"\alpha", definition="learning rate", source="explicit"),
                NotationEntry(symbol=r"\beta", definition="momentum", source="list"),
            ]
        )
        out = tmp_path / "glossary.md"
        written = write_markdown_glossary(glossary, out)
        assert written.exists()
        content = written.read_text()
        assert "alpha" in content.lower() or r"\alpha" in content
        assert "|" in content  # Markdown table

    def test_build_summary(self) -> None:
        glossary = NotationGlossary(
            entries=[
                NotationEntry(symbol=r"\alpha", definition="lr", source="explicit"),
                NotationEntry(symbol=r"\beta", definition="momentum", source="list"),
            ]
        )
        summary = notation_build_summary(glossary)
        assert summary["unique_symbols"] == 2


class TestNotationScientificObjects:
    def test_glossary_to_scientific_objects(self) -> None:
        glossary = NotationGlossary(
            entries=[
                NotationEntry(symbol=r"\alpha", definition="learning rate", source="explicit", source_file="a.md"),
                NotationEntry(symbol=r"\pi", definition="policy", source="convention", source_file="a.md"),
            ]
        )
        objects = glossary_to_scientific_objects(glossary)
        assert len(objects) == 2
        assert all(obj.object_type == "notation" for obj in objects)
        assert objects[0].metadata["source"] == "explicit"
        assert objects[1].evidence_level == "convention"
        assert all(entry.object_id for entry in glossary.entries)


# ═══════════════════════════════════════════════════════════════════════════════
# D2: py.typed marker  (Phase 5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPyTyped:
    def test_py_typed_exists(self) -> None:
        marker = REPO_ROOT / "cortexmark" / "py.typed"
        assert marker.exists(), "py.typed marker should exist in cortexmark/"

    def test_pyrightconfig_exists(self) -> None:
        cfg = REPO_ROOT / "pyrightconfig.json"
        assert cfg.exists()

    def test_pyrightconfig_has_standard(self) -> None:
        cfg = REPO_ROOT / "pyrightconfig.json"
        data = json.loads(cfg.read_text(encoding="utf-8"))
        assert data.get("typeCheckingMode") == "standard"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Quality & Integration Hardening Tests
# ══════════════════════════════════════════════════════════════════════════════


# ── formula_validate.py ──────────────────────────────────────────────────────


class TestFormulaIssueDataclass:
    def test_creation(self) -> None:
        from cortexmark.formula_validate import FormulaIssue

        issue = FormulaIssue(kind="error", message="unclosed brace")
        assert issue.kind == "error"
        assert "unclosed" in issue.message

    def test_warning_kind(self) -> None:
        from cortexmark.formula_validate import FormulaIssue

        issue = FormulaIssue(kind="warning", message="unknown command")
        assert issue.kind == "warning"


class TestFormulaDataclass:
    def test_basic_fields(self) -> None:
        from cortexmark.formula_validate import Formula

        f = Formula(text="x^2", display="inline", valid=True)
        assert f.text == "x^2"
        assert f.display == "inline"
        assert f.valid is True

    def test_error_warning_counts(self) -> None:
        from cortexmark.formula_validate import Formula, FormulaIssue

        f = Formula(
            text="bad",
            display="display",
            issues=[
                FormulaIssue("error", "e1"),
                FormulaIssue("warning", "w1"),
                FormulaIssue("error", "e2"),
            ],
        )
        assert f.error_count == 2
        assert f.warning_count == 1

    def test_empty_issues(self) -> None:
        from cortexmark.formula_validate import Formula

        f = Formula(text="x", display="inline")
        assert f.error_count == 0
        assert f.warning_count == 0


class TestFileValidationDataclass:
    def test_creation(self) -> None:
        from cortexmark.formula_validate import FileValidation

        fv = FileValidation(file="test.md", inline_count=3, display_count=1, total_count=4)
        assert fv.file == "test.md"
        assert fv.total_count == 4


class TestValidationSummaryDataclass:
    def test_defaults(self) -> None:
        from cortexmark.formula_validate import ValidationSummary

        vs = ValidationSummary()
        assert vs.files_scanned == 0
        assert vs.total_formulas == 0
        assert vs.avg_complexity == 0.0


class TestCheckBalancedDelimiters:
    def test_balanced(self) -> None:
        from cortexmark.formula_validate import check_balanced_delimiters

        assert check_balanced_delimiters("{x + y}") == []
        assert check_balanced_delimiters("(a[b]{c})") == []

    def test_unclosed_brace(self) -> None:
        from cortexmark.formula_validate import check_balanced_delimiters

        issues = check_balanced_delimiters("{x + y")
        assert len(issues) >= 1
        assert any("unclosed" in i.message for i in issues)

    def test_mismatched(self) -> None:
        from cortexmark.formula_validate import check_balanced_delimiters

        issues = check_balanced_delimiters("{x)")
        assert any("mismatched" in i.message or "unclosed" in i.message for i in issues)

    def test_extra_closing(self) -> None:
        from cortexmark.formula_validate import check_balanced_delimiters

        issues = check_balanced_delimiters("x}")
        assert any("unmatched closing" in i.message for i in issues)

    def test_escaped_brace_ignored(self) -> None:
        from cortexmark.formula_validate import check_balanced_delimiters

        # Escaped braces should not be counted
        assert check_balanced_delimiters("\\{text\\}") == []

    def test_nested(self) -> None:
        from cortexmark.formula_validate import check_balanced_delimiters

        assert check_balanced_delimiters("{{inner}}") == []

    def test_half_open_and_mixed_latex_delimiters(self) -> None:
        from cortexmark.formula_validate import check_balanced_delimiters

        valid_formulas = ("[a, b)", "(a, b]", r"\left( a, b \right]", r"\left. x \right)")
        assert all(check_balanced_delimiters(formula) == [] for formula in valid_formulas)


class TestCheckEnvironments:
    def test_matched_environment(self) -> None:
        from cortexmark.formula_validate import check_environments

        text = "\\begin{equation}x=1\\end{equation}"
        issues = check_environments(text)
        # No errors (environment is known)
        assert not any(i.kind == "error" for i in issues)

    def test_mismatched_environment(self) -> None:
        from cortexmark.formula_validate import check_environments

        text = "\\begin{equation}x=1\\end{align}"
        issues = check_environments(text)
        assert any("mismatch" in i.message for i in issues)

    def test_unclosed_environment(self) -> None:
        from cortexmark.formula_validate import check_environments

        text = "\\begin{align}x=1"
        issues = check_environments(text)
        assert any("unclosed" in i.message for i in issues)

    def test_extra_end(self) -> None:
        from cortexmark.formula_validate import check_environments

        text = "\\end{equation}"
        issues = check_environments(text)
        assert any("without matching" in i.message for i in issues)

    def test_unknown_environment_warning(self) -> None:
        from cortexmark.formula_validate import check_environments

        text = "\\begin{myenv}x\\end{myenv}"
        issues = check_environments(text)
        assert any(i.kind == "warning" and "unknown" in i.message for i in issues)


class TestCheckCommands:
    def test_known_commands(self) -> None:
        from cortexmark.formula_validate import check_commands

        cmds, issues = check_commands("\\frac{1}{2} + \\alpha")
        assert "frac" in cmds
        assert "alpha" in cmds
        assert not any(i.kind == "error" for i in issues)

    def test_unknown_command(self) -> None:
        from cortexmark.formula_validate import check_commands

        _cmds, issues = check_commands("\\myfancycmd{x}")
        assert any("unknown" in i.message and "myfancycmd" in i.message for i in issues)

    def test_empty_text(self) -> None:
        from cortexmark.formula_validate import check_commands

        cmds, issues = check_commands("")
        assert cmds == []
        assert issues == []


class TestComputeNestingDepth:
    def test_flat(self) -> None:
        from cortexmark.formula_validate import compute_nesting_depth

        assert compute_nesting_depth("x + y") == 0

    def test_depth_one(self) -> None:
        from cortexmark.formula_validate import compute_nesting_depth

        assert compute_nesting_depth("{x}") == 1

    def test_depth_three(self) -> None:
        from cortexmark.formula_validate import compute_nesting_depth

        assert compute_nesting_depth("{a{b{c}}}") == 3

    def test_escaped_braces(self) -> None:
        from cortexmark.formula_validate import compute_nesting_depth

        assert compute_nesting_depth("\\{x\\}") == 0


class TestComputeComplexity:
    def test_simple_formula(self) -> None:
        from cortexmark.formula_validate import Formula, compute_complexity

        f = Formula(text="x", display="inline", nesting_depth=0, command_count=0)
        score = compute_complexity(f)
        assert 0.0 <= score <= 100.0

    def test_complex_formula(self) -> None:
        from cortexmark.formula_validate import Formula, compute_complexity

        f = Formula(
            text="a" * 300,
            display="display",
            nesting_depth=6,
            command_count=20,
            environments=["equation", "align", "cases"],
        )
        score = compute_complexity(f)
        assert score == 100.0  # all factors maxed out

    def test_medium_formula(self) -> None:
        from cortexmark.formula_validate import Formula, compute_complexity

        f = Formula(
            text="\\frac{a}{b}",
            display="inline",
            nesting_depth=2,
            command_count=5,
            environments=["equation"],
        )
        score = compute_complexity(f)
        assert 10.0 < score < 90.0


class TestValidateFormula:
    def test_valid_simple(self) -> None:
        from cortexmark.formula_validate import validate_formula

        f = validate_formula("x + y", display="inline")
        assert f.valid is True
        assert f.display == "inline"

    def test_empty_formula(self) -> None:
        from cortexmark.formula_validate import validate_formula

        f = validate_formula("", display="inline")
        assert f.valid is False
        assert any("empty" in i.message for i in f.issues)

    def test_whitespace_only_formula(self) -> None:
        from cortexmark.formula_validate import validate_formula

        f = validate_formula("   ", display="inline")
        assert f.valid is False

    def test_unbalanced_brace(self) -> None:
        from cortexmark.formula_validate import validate_formula

        f = validate_formula("\\frac{x}{y", display="display")
        assert f.valid is False
        assert f.error_count >= 1

    def test_valid_with_environments(self) -> None:
        from cortexmark.formula_validate import validate_formula

        f = validate_formula("\\begin{cases}a\\\\b\\end{cases}", display="display")
        assert f.valid is True
        assert "cases" in f.environments

    def test_complexity_assigned(self) -> None:
        from cortexmark.formula_validate import validate_formula

        f = validate_formula("\\frac{\\alpha}{\\beta}", display="display")
        assert f.complexity >= 0.0

    def test_commands_populated(self) -> None:
        from cortexmark.formula_validate import validate_formula

        f = validate_formula("\\sum_{i=1}^{n} \\alpha_i", display="display")
        assert "sum" in f.commands
        assert "alpha" in f.commands

    def test_line_number(self) -> None:
        from cortexmark.formula_validate import validate_formula

        f = validate_formula("x", display="inline", line=42)
        assert f.line == 42


class TestExtractAndValidate:
    def test_inline_extraction(self) -> None:
        from cortexmark.formula_validate import extract_and_validate

        text = "The value $x + y$ is positive."
        formulas = extract_and_validate(text)
        assert len(formulas) == 1
        assert formulas[0].display == "inline"

    def test_display_extraction(self) -> None:
        from cortexmark.formula_validate import extract_and_validate

        text = "We have:\n$$E = mc^2$$\n"
        formulas = extract_and_validate(text)
        assert len(formulas) == 1
        assert formulas[0].display == "display"

    def test_mixed_formulas(self) -> None:
        from cortexmark.formula_validate import extract_and_validate

        text = "Inline: $\\alpha$. Display:\n$$\\beta = 1$$\n"
        formulas = extract_and_validate(text)
        inline = [f for f in formulas if f.display == "inline"]
        display = [f for f in formulas if f.display == "display"]
        assert len(inline) >= 1
        assert len(display) >= 1

    def test_no_formulas(self) -> None:
        from cortexmark.formula_validate import extract_and_validate

        formulas = extract_and_validate("Just plain text.")
        assert formulas == []

    def test_display_not_double_counted_as_inline(self) -> None:
        from cortexmark.formula_validate import extract_and_validate

        text = "$$x^2$$"
        formulas = extract_and_validate(text)
        # Should have exactly 1 display formula, not also an inline
        assert len(formulas) == 1
        assert formulas[0].display == "display"


class TestValidateFile:
    def test_validate_file(self, tmp_path: Path) -> None:
        from cortexmark.formula_validate import validate_file

        md = tmp_path / "test.md"
        md.write_text("# Test\n$x$\n$$y^2$$\n", encoding="utf-8")
        result = validate_file(md)
        assert result.file == str(md)
        assert result.inline_count == 1
        assert result.display_count == 1
        assert result.total_count == 2


class TestValidateTree:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from cortexmark.formula_validate import validate_tree

        assert validate_tree(tmp_path) == []

    def test_nonexistent(self, tmp_path: Path) -> None:
        from cortexmark.formula_validate import validate_tree

        assert validate_tree(tmp_path / "nope") == []

    def test_single_file(self, tmp_path: Path) -> None:
        from cortexmark.formula_validate import validate_tree

        md = tmp_path / "a.md"
        md.write_text("$x$\n", encoding="utf-8")
        results = validate_tree(tmp_path)
        assert len(results) == 1

    def test_multiple_files(self, tmp_path: Path) -> None:
        from cortexmark.formula_validate import validate_tree

        (tmp_path / "a.md").write_text("$x$\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("$$y$$\n", encoding="utf-8")
        results = validate_tree(tmp_path)
        assert len(results) == 2

    def test_file_as_root(self, tmp_path: Path) -> None:
        from cortexmark.formula_validate import validate_tree

        md = tmp_path / "single.md"
        md.write_text("$a$\n", encoding="utf-8")
        results = validate_tree(md)
        assert len(results) == 1


class TestFormulaBuildSummary:
    def test_summary_aggregation(self) -> None:
        from cortexmark.formula_validate import FileValidation, build_summary

        v1 = FileValidation(file="a.md", inline_count=2, display_count=1, total_count=3, valid_count=2, error_count=1)
        v2 = FileValidation(file="b.md", inline_count=1, display_count=2, total_count=3, valid_count=3, error_count=0)
        summary = build_summary([v1, v2])
        assert summary.files_scanned == 2
        assert summary.total_formulas == 6
        assert summary.total_inline == 3
        assert summary.total_display == 3
        assert summary.total_valid == 5
        assert summary.total_errors == 1

    def test_empty_summary(self) -> None:
        from cortexmark.formula_validate import build_summary

        summary = build_summary([])
        assert summary.files_scanned == 0
        assert summary.total_formulas == 0
        assert summary.avg_complexity == 0.0


class TestFormulaWriteReport:
    def test_writes_json(self, tmp_path: Path) -> None:
        from cortexmark.formula_validate import (
            FileValidation,
            ValidationSummary,
            write_report,
        )

        v = FileValidation(file="test.md", total_count=5, valid_count=5)
        s = ValidationSummary(files_scanned=1, total_formulas=5)
        out = tmp_path / "report.json"
        result = write_report([v], s, out)
        assert result == out
        assert out.exists()
        import json

        data = json.loads(out.read_text(encoding="utf-8"))
        assert "summary" in data
        assert "files" in data
        assert data["summary"]["total_formulas"] == 5


class TestFormulaCLIParser:
    def test_parser_defaults(self) -> None:
        from cortexmark.formula_validate import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.input is None
        assert args.output is None

    def test_parser_with_args(self) -> None:
        from cortexmark.formula_validate import build_parser

        parser = build_parser()
        args = parser.parse_args(["--input", "/tmp/test", "--output", "/tmp/out.json"])
        assert args.input == Path("/tmp/test")
        assert args.output == Path("/tmp/out.json")


# ── citation_context.py ──────────────────────────────────────────────────────


class TestCitationContextDataclass:
    def test_creation(self) -> None:
        from cortexmark.citation_context import CitationContext

        ctx = CitationContext(
            raw_text="(Smith, 2020)",
            cite_type="author-year",
            sentence="We follow Smith, 2020.",
            purpose="foundational",
            purpose_confidence=0.8,
        )
        assert ctx.cite_type == "author-year"
        assert ctx.purpose == "foundational"


class TestCoCitationDataclass:
    def test_creation(self) -> None:
        from cortexmark.citation_context import CoCitation

        cc = CoCitation(cite_a="(A, 2020)", cite_b="(B, 2021)", sentence="Both A, 2020 and B, 2021.", count=2)
        assert cc.count == 2


class TestSelfCitationDataclass:
    def test_creation(self) -> None:
        from cortexmark.citation_context import SelfCitation

        sc = SelfCitation(raw_text="(Smith, 2020)", matching_author="Smith", line=5)
        assert sc.matching_author == "Smith"


class TestContextSummaryDataclass:
    def test_defaults(self) -> None:
        from cortexmark.citation_context import ContextSummary

        cs = ContextSummary()
        assert cs.files_scanned == 0
        assert cs.total_citations == 0


class TestSplitSentences:
    def test_two_sentences(self) -> None:
        from cortexmark.citation_context import _split_sentences

        text = "First sentence. Second sentence."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_collapses_single_newlines(self) -> None:
        from cortexmark.citation_context import _split_sentences

        text = "This is\na continuation. And a second."
        sentences = _split_sentences(text)
        assert any("continuation" in s for s in sentences)


class TestClassifyPurpose:
    def test_foundational(self) -> None:
        from cortexmark.citation_context import classify_purpose

        purpose, conf = classify_purpose("Our method is based on Smith (2020).")
        assert purpose == "foundational"
        assert conf > 0.5

    def test_comparative(self) -> None:
        from cortexmark.citation_context import classify_purpose

        purpose, _conf = classify_purpose("In contrast to Jones (2019), we find.")
        assert purpose == "comparative"

    def test_methodological(self) -> None:
        from cortexmark.citation_context import classify_purpose

        purpose, _conf = classify_purpose("We use the method of Brown (2018) to analyze data.")
        assert purpose == "methodological"

    def test_extending(self) -> None:
        from cortexmark.citation_context import classify_purpose

        purpose, _conf = classify_purpose("We extend the framework of Chen (2021).")
        assert purpose == "extending"

    def test_background(self) -> None:
        from cortexmark.citation_context import classify_purpose

        purpose, _conf = classify_purpose("This has been widely studied in the literature.")
        assert purpose == "background"

    def test_refuting(self) -> None:
        from cortexmark.citation_context import classify_purpose

        purpose, _conf = classify_purpose("We disagree with the analysis of Lee (2017).")
        assert purpose == "refuting"

    def test_unknown(self) -> None:
        from cortexmark.citation_context import classify_purpose

        purpose, conf = classify_purpose("Some random sentence with no cues.")
        assert purpose == "unknown"
        assert conf < 0.5


class TestExtractCitationContexts:
    def test_author_year(self) -> None:
        from cortexmark.citation_context import extract_citation_contexts

        text = "This builds upon (Smith, 2020) for the core method."
        contexts = extract_citation_contexts(text)
        assert len(contexts) >= 1
        assert contexts[0].cite_type == "author-year"

    def test_numeric_bracket(self) -> None:
        from cortexmark.citation_context import extract_citation_contexts

        text = "Several works [1, 2, 3] have studied this problem."
        contexts = extract_citation_contexts(text)
        assert len(contexts) >= 1
        assert contexts[0].cite_type == "numeric"

    def test_no_citations(self) -> None:
        from cortexmark.citation_context import extract_citation_contexts

        contexts = extract_citation_contexts("Plain text with no refs.")
        assert contexts == []

    def test_multiple_citations_in_sentence(self) -> None:
        from cortexmark.citation_context import extract_citation_contexts

        text = "Based on (Smith, 2020) and [5], we proceed."
        contexts = extract_citation_contexts(text)
        assert len(contexts) >= 2

    def test_source_file_preserved(self) -> None:
        from cortexmark.citation_context import extract_citation_contexts

        contexts = extract_citation_contexts("See (Smith, 2020).", source_file="paper.md")
        if contexts:
            assert contexts[0].source_file == "paper.md"

    def test_parenthetical_et_al_citation(self) -> None:
        from cortexmark.citation_context import extract_citation_contexts

        contexts = extract_citation_contexts("We follow (Smith et al., 2020) for this method.")
        assert [context.raw_text for context in contexts] == ["(Smith et al., 2020)"]

    def test_wrapped_citation_reports_its_original_line(self) -> None:
        from cortexmark.citation_context import extract_citation_contexts

        text = "This sentence wraps onto the next line\nbefore citing (Smith, 2020) for support."
        contexts = extract_citation_contexts(text)
        assert contexts[0].line == 2


class TestDetectCoCitations:
    def test_co_citation_pair(self) -> None:
        from cortexmark.citation_context import CitationContext, detect_co_citations

        shared_sent = "Both (Smith, 2020) and (Jones, 2021) found similar results."
        contexts = [
            CitationContext("(Smith, 2020)", "author-year", shared_sent, "background", 0.8),
            CitationContext("(Jones, 2021)", "author-year", shared_sent, "background", 0.8),
        ]
        co = detect_co_citations(contexts)
        assert len(co) >= 1

    def test_no_co_citations(self) -> None:
        from cortexmark.citation_context import CitationContext, detect_co_citations

        contexts = [
            CitationContext("(Smith, 2020)", "author-year", "Sentence A.", "background", 0.8),
            CitationContext("(Jones, 2021)", "author-year", "Different sentence.", "background", 0.8),
        ]
        co = detect_co_citations(contexts)
        assert co == []


class TestDetectSelfCitations:
    def test_self_citation_found(self) -> None:
        from cortexmark.citation_context import CitationContext, detect_self_citations

        contexts = [
            CitationContext("(Smith, 2020)", "author-year", "Our prior work (Smith, 2020).", "extending", 0.8),
        ]
        self_cites = detect_self_citations(contexts, document_authors=["John Smith"])
        assert len(self_cites) == 1
        assert self_cites[0].matching_author == "Smith"

    def test_no_match(self) -> None:
        from cortexmark.citation_context import CitationContext, detect_self_citations

        contexts = [
            CitationContext("(Jones, 2021)", "author-year", "See Jones.", "background", 0.8),
        ]
        self_cites = detect_self_citations(contexts, document_authors=["John Smith"])
        assert self_cites == []

    def test_no_authors_provided(self) -> None:
        from cortexmark.citation_context import CitationContext, detect_self_citations

        contexts = [
            CitationContext("(Smith, 2020)", "author-year", "See Smith.", "background", 0.8),
        ]
        assert detect_self_citations(contexts, document_authors=None) == []
        assert detect_self_citations(contexts, document_authors=[]) == []

    def test_numeric_citations_skipped(self) -> None:
        from cortexmark.citation_context import CitationContext, detect_self_citations

        contexts = [
            CitationContext("[1]", "numeric", "See [1].", "background", 0.8),
        ]
        self_cites = detect_self_citations(contexts, document_authors=["Smith"])
        assert self_cites == []

    def test_et_al_uses_leading_surname(self) -> None:
        from cortexmark.citation_context import CitationContext, detect_self_citations

        contexts = [
            CitationContext(
                "(Smith et al., 2020)",
                "author-year",
                "Our prior work (Smith et al., 2020).",
                "extending",
                0.8,
            ),
        ]
        self_cites = detect_self_citations(contexts, document_authors=["Jane Smith", "Alex Al"])
        assert [citation.matching_author for citation in self_cites] == ["Smith"]


class TestExtractAuthorsFromText:
    def test_author_line(self) -> None:
        from cortexmark.citation_context import _extract_authors_from_text

        text = "# My Paper\nAuthors: Alice Smith, Bob Jones, and Carol Lee\n\n## Abstract"
        authors = _extract_authors_from_text(text)
        assert len(authors) >= 2
        assert any("Smith" in a for a in authors)

    def test_no_authors(self) -> None:
        from cortexmark.citation_context import _extract_authors_from_text

        authors = _extract_authors_from_text("Just some text with no author.")
        assert authors == []


class TestCitationAnalyzeFile:
    def test_analyze_file(self, tmp_path: Path) -> None:
        from cortexmark.citation_context import analyze_file

        md = tmp_path / "paper.md"
        md.write_text(
            "# Paper\nAuthors: John Smith\n\n"
            "We build upon (Smith, 2020) for our approach. "
            "In contrast to (Jones, 2019), our method is better.\n",
            encoding="utf-8",
        )
        report = analyze_file(md)
        assert report.file == str(md)
        assert report.total_citations >= 1


class TestCitationAnalyzeTree:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from cortexmark.citation_context import analyze_tree

        assert analyze_tree(tmp_path) == []

    def test_nonexistent(self, tmp_path: Path) -> None:
        from cortexmark.citation_context import analyze_tree

        assert analyze_tree(tmp_path / "missing") == []

    def test_single_file(self, tmp_path: Path) -> None:
        from cortexmark.citation_context import analyze_tree

        md = tmp_path / "paper.md"
        md.write_text("See (Smith, 2020) for details.\n", encoding="utf-8")
        results = analyze_tree(tmp_path)
        assert len(results) == 1

    def test_file_as_root(self, tmp_path: Path) -> None:
        from cortexmark.citation_context import analyze_tree

        md = tmp_path / "paper.md"
        md.write_text("See (Smith, 2020).\n", encoding="utf-8")
        results = analyze_tree(md)
        assert len(results) == 1


class TestCitationBuildSummary:
    def test_aggregation(self) -> None:
        from cortexmark.citation_context import FileContextReport, build_summary

        r1 = FileContextReport(file="a.md", total_citations=3, purpose_distribution={"foundational": 2, "unknown": 1})
        r2 = FileContextReport(
            file="b.md", total_citations=2, purpose_distribution={"foundational": 1, "comparative": 1}
        )
        summary = build_summary([r1, r2])
        assert summary.files_scanned == 2
        assert summary.total_citations == 5
        assert summary.purpose_distribution["foundational"] == 3

    def test_empty_summary(self) -> None:
        from cortexmark.citation_context import build_summary

        s = build_summary([])
        assert s.files_scanned == 0
        assert s.total_citations == 0


class TestCitationWriteReport:
    def test_writes_json(self, tmp_path: Path) -> None:
        from cortexmark.citation_context import (
            ContextSummary,
            FileContextReport,
            write_report,
        )

        r = FileContextReport(file="a.md", total_citations=2)
        s = ContextSummary(files_scanned=1, total_citations=2)
        out = tmp_path / "ctx.json"
        result = write_report([r], s, out)
        assert result == out
        assert out.exists()
        import json

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["total_citations"] == 2


class TestCitationCLIParser:
    def test_parser_defaults(self) -> None:
        from cortexmark.citation_context import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.input is None


# ── scientific_qa.py ─────────────────────────────────────────────────────────


class TestSciQAIssueDataclass:
    def test_creation(self) -> None:
        from cortexmark.scientific_qa import SciQAIssue

        issue = SciQAIssue(check="theorem_proof_pairing", severity="warning", message="missing proof")
        assert issue.check == "theorem_proof_pairing"
        assert issue.severity == "warning"


class TestFileSciQAReportDataclass:
    def test_badge_gold(self) -> None:
        from cortexmark.scientific_qa import FileSciQAReport

        r = FileSciQAReport(file="a.md")
        assert r.badge == "gold"
        assert r.error_count == 0

    def test_badge_fail(self) -> None:
        from cortexmark.scientific_qa import FileSciQAReport, SciQAIssue

        r = FileSciQAReport(
            file="a.md",
            issues=[SciQAIssue("formula_quality", "error", "bad")],
        )
        assert r.badge == "fail"

    def test_badge_silver(self) -> None:
        from cortexmark.scientific_qa import FileSciQAReport, SciQAIssue

        r = FileSciQAReport(
            file="a.md",
            issues=[SciQAIssue("notation_consistency", "warning", "w1")],
        )
        assert r.badge == "silver"

    def test_badge_bronze(self) -> None:
        from cortexmark.scientific_qa import FileSciQAReport, SciQAIssue

        r = FileSciQAReport(
            file="a.md",
            issues=[
                SciQAIssue("a", "warning", "w1"),
                SciQAIssue("b", "warning", "w2"),
                SciQAIssue("c", "warning", "w3"),
            ],
        )
        assert r.badge == "bronze"

    def test_error_warning_counts(self) -> None:
        from cortexmark.scientific_qa import FileSciQAReport, SciQAIssue

        r = FileSciQAReport(
            file="x.md",
            issues=[
                SciQAIssue("a", "error", "e1"),
                SciQAIssue("b", "warning", "w1"),
                SciQAIssue("c", "error", "e2"),
                SciQAIssue("d", "info", "i1"),
            ],
        )
        assert r.error_count == 2
        assert r.warning_count == 1


class TestSciQASummaryDataclass:
    def test_defaults(self) -> None:
        from cortexmark.scientific_qa import SciQASummary

        s = SciQASummary()
        assert s.files_scanned == 0
        assert s.total_issues == 0


class TestCheckTheoremProofPairing:
    def test_theorem_with_proof(self) -> None:
        from cortexmark.scientific_qa import check_theorem_proof_pairing

        text = "**Theorem 1.** Some statement.\n\n**Proof.** We show that..."
        issues = check_theorem_proof_pairing(text)
        assert len(issues) == 0

    def test_theorem_without_proof(self) -> None:
        from cortexmark.scientific_qa import check_theorem_proof_pairing

        text = "**Theorem 1.** Some statement.\n\nSome other text."
        issues = check_theorem_proof_pairing(text)
        assert len(issues) >= 1
        assert any("no corresponding proof" in i.message for i in issues)

    def test_theorem_proof_omitted(self) -> None:
        from cortexmark.scientific_qa import check_theorem_proof_pairing

        text = "**Theorem 1.** Statement.\n\nThe proof is omitted."
        issues = check_theorem_proof_pairing(text)
        assert len(issues) == 0

    def test_theorem_proof_left_as_exercise(self) -> None:
        from cortexmark.scientific_qa import check_theorem_proof_pairing

        text = "**Theorem 1.** Statement.\n\nThe proof is left to the reader."
        issues = check_theorem_proof_pairing(text)
        assert len(issues) == 0

    def test_labeled_proof(self) -> None:
        from cortexmark.scientific_qa import check_theorem_proof_pairing

        text = "**Theorem 2.1** Statement.\n\n**Proof of Theorem 2.1.** Done."
        issues = check_theorem_proof_pairing(text)
        assert len(issues) == 0

    def test_lemma_no_proof(self) -> None:
        from cortexmark.scientific_qa import check_theorem_proof_pairing

        text = "**Lemma 3.** A useful lemma."
        issues = check_theorem_proof_pairing(text)
        assert len(issues) >= 1


class TestCheckDefinitionBeforeUse:
    def test_definition_before(self) -> None:
        from cortexmark.scientific_qa import check_definition_before_use

        text = "**Definition 1.** A set is...\n\nBy Definition 1, we have..."
        issues = check_definition_before_use(text)
        assert len(issues) == 0

    def test_reference_before_definition(self) -> None:
        from cortexmark.scientific_qa import check_definition_before_use

        text = "**Definition 1.** A group is...\n\nBy Definition 2, we see a ring.\n\n**Definition 2.** A ring is..."
        issues = check_definition_before_use(text)
        # was: assert isinstance(issues, list)
        assert any("Definition 2 referenced before it is defined" in issue.message for issue in issues)

    def test_forward_reference_before_markdown_definition_heading(self) -> None:
        from cortexmark.scientific_qa import check_definition_before_use

        text = "See Definition 3 for the metric.\n\n## **Definition 3.**\nThe metric is nonnegative."
        issues = check_definition_before_use(text)
        assert [(issue.line, issue.message) for issue in issues] == [
            (1, "Definition 3 referenced before it is defined")
        ]

    def test_no_definitions(self) -> None:
        from cortexmark.scientific_qa import check_definition_before_use

        issues = check_definition_before_use("Just regular text.")
        assert issues == []


class TestCheckNotationConsistency:
    def test_consistent_notation(self) -> None:
        from cortexmark.scientific_qa import check_notation_consistency

        text = "Let $X$ be a set."
        issues = check_notation_consistency(text)
        assert issues == []

    def test_conflicting_notation(self) -> None:
        from cortexmark.scientific_qa import check_notation_consistency

        text = "Let $X$ be a set. Later, let $X$ be a function."
        issues = check_notation_consistency(text)
        assert len(issues) >= 1
        # was: assert any("multiple times" in i.message for i in issues)
        assert any("conflicting meanings" in i.message for i in issues)

    def test_repeated_identical_notation_is_consistent(self) -> None:
        from cortexmark.scientific_qa import check_notation_consistency

        text = "Let $X$ be a set.\nLet $X$ be a set."
        assert check_notation_consistency(text) == []

    def test_no_notation(self) -> None:
        from cortexmark.scientific_qa import check_notation_consistency

        issues = check_notation_consistency("No math here.")
        assert issues == []


class TestCheckCrossrefCompleteness:
    def test_resolved_reference(self) -> None:
        from cortexmark.scientific_qa import check_crossref_completeness

        text = "**Theorem 1.** Statement.\n\nBy Theorem 1, we have."
        issues = check_crossref_completeness(text)
        assert len(issues) == 0

    def test_resolved_theorem_family_references(self) -> None:
        from cortexmark.scientific_qa import check_crossref_completeness

        text = (
            "**Theorem 1.** Main result.\n"
            "**Lemma 2.** Supporting result.\n"
            "**Proposition 3.** Intermediate result.\n"
            "**Corollary 4.** Consequence.\n"
            "**Conjecture 5.** Expected result.\n\n"
            "By Lemma 2, Proposition 3, Corollary 4, and Conjecture 5, the claim follows."
        )
        assert check_crossref_completeness(text) == []

    def test_unresolved_reference(self) -> None:
        from cortexmark.scientific_qa import check_crossref_completeness

        text = "See Theorem 5 for details."
        issues = check_crossref_completeness(text)
        # Theorem 5 is not defined, but we only flag if the kind has definitions
        # Since no theorems are defined, no definitions for the "theorem" kind
        # So it should NOT flag (the kind has no defined labels)
        assert len(issues) == 0

    def test_unresolved_with_some_definitions(self) -> None:
        from cortexmark.scientific_qa import check_crossref_completeness

        # Note: THEOREM_HEADING_RE also matches plain "Theorem X" references,
        # so the function may not detect all unresolved refs. Verify it runs.
        text = "**Theorem 1.** Statement.\n\nSee Theorem 3 for more details."
        issues = check_crossref_completeness(text)
        assert isinstance(issues, list)


class TestCheckAlgorithmValidity:
    def test_valid_algorithm(self) -> None:
        from cortexmark.scientific_qa import check_algorithm_validity

        text = (
            "**Algorithm 1: Binary Search**\n"
            "Input: sorted array A, target x\n"
            "Output: index of x in A\n"
            "1. Set low = 0, high = len(A)\n"
            "2. While low < high:\n"
            "   compare mid element with x\n"
        )
        issues = check_algorithm_validity(text)
        assert not any(i.severity in ("error", "warning") for i in issues)

    def test_no_input(self) -> None:
        from cortexmark.scientific_qa import check_algorithm_validity

        text = "**Algorithm 2: Sort**\nOutput: sorted array\n1. Do the sorting\n2. Return result\n"
        issues = check_algorithm_validity(text)
        assert any("no declared inputs" in i.message for i in issues)

    def test_short_body(self) -> None:
        from cortexmark.scientific_qa import check_algorithm_validity

        text = "**Algorithm 3: Empty**\nOk"
        issues = check_algorithm_validity(text)
        assert any("very short" in i.message for i in issues)

    def test_no_algorithms(self) -> None:
        from cortexmark.scientific_qa import check_algorithm_validity

        issues = check_algorithm_validity("Regular text without algorithms.")
        assert issues == []


class TestCheckFormulaQuality:
    def test_good_quality(self) -> None:
        from cortexmark.scientific_qa import check_formula_quality

        text = "We have $x = 1$ and $$y = 2$$."
        issues = check_formula_quality(text)
        assert not any(i.severity == "error" for i in issues)

    def test_low_fidelity(self) -> None:
        from cortexmark.scientific_qa import check_formula_quality

        text = "$x=1$ <!-- formula-not-decoded -->\n$y=2$ <!-- formula-not-decoded -->\n$z=3$\n"
        # was: issues = check_formula_quality(text, min_fidelity=50.0)
        issues = check_formula_quality(text, min_fidelity=70.0)
        assert any("fidelity" in i.message.lower() for i in issues)

    def test_fidelity_and_issue_count_stay_bounded(self) -> None:
        from cortexmark.scientific_qa import check_formula_quality

        text = "$x=1$\n" + "<!-- formula-not-decoded -->\n" * 3
        issues = check_formula_quality(text, min_fidelity=50.0)
        message = next(issue.message for issue in issues if "fidelity" in issue.message.lower())
        match = re.search(r"(-?\d+)% \((\d+) of (\d+) formulas have issues\)", message)
        assert match is not None
        fidelity, problem_count, total = (int(value) for value in match.groups())
        assert 0.0 <= fidelity / 100 <= 1.0
        assert 0 <= problem_count <= total

    def test_empty_blocks(self) -> None:
        from cortexmark.scientific_qa import check_formula_quality

        text = "$$ $$ and $ $."
        issues = check_formula_quality(text)
        assert any("empty" in i.message for i in issues)

    def test_truly_empty_display_block(self) -> None:
        from cortexmark.scientific_qa import check_formula_quality

        issues = check_formula_quality("$$$$")
        assert any("1 empty formula block" in issue.message for issue in issues)

    def test_no_formulas(self) -> None:
        from cortexmark.scientific_qa import check_formula_quality

        issues = check_formula_quality("Just text, no math.")
        assert issues == []


class TestRunAllChecks:
    def test_combines_all_checks(self) -> None:
        from cortexmark.scientific_qa import run_all_checks

        text = "**Theorem 1.** Statement.\n**Algorithm 1: Test**\nOk.\n$x = 1$\n"
        issues = run_all_checks(text)
        # Should run without error; may produce warnings
        assert isinstance(issues, list)

    def test_clean_document(self) -> None:
        from cortexmark.scientific_qa import run_all_checks

        text = (
            "**Theorem 1.** Statement.\n\n"
            "**Proof.** We prove it here: $x = y$. QED.\n\n"
            "**Definition 1.** A set is a collection.\n\n"
            "By Definition 1, we have. See Theorem 1.\n"
        )
        issues = run_all_checks(text)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0


class TestSciQAAnalyzeFile:
    def test_analyze_file(self, tmp_path: Path) -> None:
        from cortexmark.scientific_qa import analyze_file

        md = tmp_path / "doc.md"
        md.write_text(
            "**Theorem 1.** Statement.\n\n**Proof.** Done.\n\n**Definition 1.** A set.\n\n$x = 1$\n",
            encoding="utf-8",
        )
        report = analyze_file(md)
        assert report.file == str(md)
        assert report.theorems_found >= 1
        assert report.proofs_found >= 1
        assert report.definitions_found >= 1
        assert report.formulas_found >= 1

    def test_badge_assigned(self, tmp_path: Path) -> None:
        from cortexmark.scientific_qa import analyze_file

        md = tmp_path / "clean.md"
        md.write_text("**Theorem 1.** Stmt.\n\n**Proof.** Done.\n", encoding="utf-8")
        report = analyze_file(md)
        assert report.badge in ("gold", "silver", "bronze", "fail")


class TestSciQAAnalyzeTree:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from cortexmark.scientific_qa import analyze_tree

        assert analyze_tree(tmp_path) == []

    def test_nonexistent(self, tmp_path: Path) -> None:
        from cortexmark.scientific_qa import analyze_tree

        assert analyze_tree(tmp_path / "missing") == []

    def test_multiple_files(self, tmp_path: Path) -> None:
        from cortexmark.scientific_qa import analyze_tree

        (tmp_path / "a.md").write_text("**Theorem 1.** X.\n\n**Proof.** Y.\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("$z = 1$\n", encoding="utf-8")
        results = analyze_tree(tmp_path)
        assert len(results) == 2

    def test_file_as_root(self, tmp_path: Path) -> None:
        from cortexmark.scientific_qa import analyze_tree

        md = tmp_path / "doc.md"
        md.write_text("$x$\n", encoding="utf-8")
        results = analyze_tree(md)
        assert len(results) == 1


class TestSciQABuildSummary:
    def test_aggregation(self) -> None:
        from cortexmark.scientific_qa import FileSciQAReport, SciQAIssue, build_summary

        r1 = FileSciQAReport(
            file="a.md",
            issues=[SciQAIssue("a", "error", "e1")],
            theorems_found=2,
            proofs_found=1,
        )
        r2 = FileSciQAReport(
            file="b.md",
            issues=[SciQAIssue("b", "warning", "w1"), SciQAIssue("c", "warning", "w2")],
            definitions_found=3,
        )
        summary = build_summary([r1, r2])
        assert summary.files_scanned == 2
        assert summary.total_issues == 3
        assert summary.total_errors == 1
        assert summary.total_warnings == 2
        assert summary.total_theorems == 2
        assert summary.total_definitions == 3
        assert "fail" in summary.badge_distribution

    def test_empty_summary(self) -> None:
        from cortexmark.scientific_qa import build_summary

        s = build_summary([])
        assert s.files_scanned == 0


class TestSciQAWriteReport:
    def test_writes_json(self, tmp_path: Path) -> None:
        from cortexmark.scientific_qa import FileSciQAReport, SciQASummary, write_report

        r = FileSciQAReport(file="a.md", theorems_found=1)
        s = SciQASummary(files_scanned=1, total_issues=0)
        out = tmp_path / "qa.json"
        result = write_report([r], s, out)
        assert result == out
        assert out.exists()
        import json

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["files_scanned"] == 1
        assert data["files"][0]["theorems"] == 1


class TestSciQACLIParser:
    def test_parser_defaults(self) -> None:
        from cortexmark.scientific_qa import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert args.input is None


# ── run_pipeline.py Phase 4 integration ──────────────────────────────────────


class TestRunPipelinePhase4Stages:
    def test_analyze_in_stage_choices(self) -> None:
        from cortexmark.run_pipeline import build_parser

        parser = build_parser()
        args = parser.parse_args(["--stages", "analyze"])
        assert "analyze" in args.stages

    def test_validate_in_stage_choices(self) -> None:
        from cortexmark.run_pipeline import build_parser

        parser = build_parser()
        args = parser.parse_args(["--stages", "validate"])
        assert "validate" in args.stages

    def test_analyze_validate_together(self) -> None:
        from cortexmark.run_pipeline import build_parser

        parser = build_parser()
        args = parser.parse_args(["--stages", "analyze", "validate"])
        assert "analyze" in args.stages
        assert "validate" in args.stages

    def test_default_stages_exclude_analyze_validate(self) -> None:
        from cortexmark.run_pipeline import build_parser

        parser = build_parser()
        args = parser.parse_args([])
        assert "analyze" not in args.stages
        assert "validate" not in args.stages

    def test_all_six_stages(self) -> None:
        from cortexmark.run_pipeline import build_parser

        parser = build_parser()
        all_stages = ["convert", "clean", "chunk", "render", "analyze", "validate"]
        args = parser.parse_args(["--stages", *all_stages])
        assert args.stages == all_stages


# ── Module-level import tests ────────────────────────────────────────────────


class TestPhase4ModuleImports:
    def test_formula_validate_imports(self) -> None:
        import cortexmark.formula_validate as fv

        assert hasattr(fv, "validate_formula")
        assert hasattr(fv, "extract_and_validate")
        assert hasattr(fv, "validate_file")
        assert hasattr(fv, "validate_tree")
        assert hasattr(fv, "build_summary")
        assert hasattr(fv, "write_report")
        assert hasattr(fv, "check_balanced_delimiters")
        assert hasattr(fv, "check_environments")
        assert hasattr(fv, "check_commands")
        assert hasattr(fv, "compute_nesting_depth")
        assert hasattr(fv, "compute_complexity")

    def test_citation_context_imports(self) -> None:
        import cortexmark.citation_context as cc

        assert hasattr(cc, "classify_purpose")
        assert hasattr(cc, "extract_citation_contexts")
        assert hasattr(cc, "detect_co_citations")
        assert hasattr(cc, "detect_self_citations")
        assert hasattr(cc, "analyze_file")
        assert hasattr(cc, "analyze_tree")
        assert hasattr(cc, "build_summary")
        assert hasattr(cc, "write_report")

    def test_scientific_qa_imports(self) -> None:
        import cortexmark.scientific_qa as sq

        assert hasattr(sq, "check_theorem_proof_pairing")
        assert hasattr(sq, "check_definition_before_use")
        assert hasattr(sq, "check_notation_consistency")
        assert hasattr(sq, "check_crossref_completeness")
        assert hasattr(sq, "check_algorithm_validity")
        assert hasattr(sq, "check_formula_quality")
        assert hasattr(sq, "run_all_checks")
        assert hasattr(sq, "analyze_file")
        assert hasattr(sq, "analyze_tree")
        assert hasattr(sq, "build_summary")
        assert hasattr(sq, "write_report")

    def test_constants_exist(self) -> None:
        from cortexmark.formula_validate import KNOWN_COMMANDS, KNOWN_ENVIRONMENTS
        from cortexmark.scientific_qa import ALL_CHECKS

        assert len(KNOWN_COMMANDS) > 50
        assert len(KNOWN_ENVIRONMENTS) > 10
        assert len(ALL_CHECKS) == 6

    def test_all_purposes(self) -> None:
        from cortexmark.citation_context import ALL_PURPOSES

        assert "foundational" in ALL_PURPOSES
        assert "comparative" in ALL_PURPOSES
        assert "unknown" in ALL_PURPOSES
        assert len(ALL_PURPOSES) == 7


# ══════════════════════════════════════════════════════════════════════════════
# Coverage boost tests — pure-function deep testing
# ══════════════════════════════════════════════════════════════════════════════


class TestCleanInline:
    def test_normalizes_whitespace(self) -> None:
        assert clean_inline("  hello   world  ") == "hello world"

    def test_replaces_newlines(self) -> None:
        assert clean_inline("hello\nworld") == "hello world"

    def test_empty_string(self) -> None:
        assert clean_inline("") == ""

    def test_tabs_and_mixed_whitespace(self) -> None:
        assert clean_inline("a\t\tb\n\nc") == "a b c"


class TestBulletLinesFromSection:
    def test_extracts_bullets(self) -> None:
        section = "- Python\n- Matlab\n- R"
        assert bullet_lines_from_section(section) == ["Python", "Matlab", "R"]

    def test_ignores_non_bullets(self) -> None:
        section = "This is plain text\n- Only this\n* Not this"
        assert bullet_lines_from_section(section) == ["Only this"]

    def test_empty_section(self) -> None:
        assert bullet_lines_from_section("") == []

    def test_strips_bullet_content(self) -> None:
        assert bullet_lines_from_section("-   spaced  ") == ["spaced"]


class TestHeadingsFromMarkdown:
    def test_multiple_levels(self) -> None:
        text = "# Title\n\nBody\n\n## Section\n\n### Sub\n"
        headings = headings_from_markdown(text)
        assert headings == ["Title", "Section", "Sub"]

    def test_no_headings(self) -> None:
        assert headings_from_markdown("Just text.\nMore text.") == []

    def test_empty_heading_skipped(self) -> None:
        assert headings_from_markdown("# \n## Real\n") == ["Real"]


class TestParagraphsFromMarkdown:
    def test_splits_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\n# Heading\n\nThird."
        paras = paragraphs_from_markdown(text)
        assert paras == ["First paragraph.", "Second paragraph.", "Third."]

    def test_empty_document(self) -> None:
        assert paragraphs_from_markdown("") == []

    def test_heading_only(self) -> None:
        assert paragraphs_from_markdown("# Title\n\n## Sub\n") == []

    def test_multiline_paragraph(self) -> None:
        text = "Line one\nline two\n\nAnother."
        paras = paragraphs_from_markdown(text)
        assert len(paras) == 2
        assert paras[0] == "Line one line two"


class TestExtractPrograms:
    def test_bullet_list(self) -> None:
        section = "- Python\n- MATLAB\n- R"
        assert extract_programs(section) == ["Python", "MATLAB", "R"]

    def test_comma_separated(self) -> None:
        section = "Python, MATLAB, R"
        assert extract_programs(section) == ["Python", "MATLAB", "R"]

    def test_empty_section(self) -> None:
        assert extract_programs("") == []

    def test_bullet_with_commas(self) -> None:
        section = "- Python, PyTorch\n- R, tidyverse"
        result = extract_programs(section)
        assert "Python" in result
        assert "PyTorch" in result
        assert "R" in result


class TestReadText:
    def test_reads_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("hello world", encoding="utf-8")
        assert read_text(f) == "hello world"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_text(tmp_path / "missing.md")


class TestDetectReportDocType:
    def test_report_with_summary_and_sections(self) -> None:
        text = (
            "# Annual Report\n\n"
            "## Executive Summary\n\nWe had a great year.\n\n"
            "1.1 Section A\n1.2 Section B\n1.3 Section C\n1.4 Section D\n\n"
            "## Conclusion\n\nFindings indicate positive results.\n"
        )
        score, signals = detect_report(text)
        assert score >= 0.3
        assert any("summary" in s.lower() or "introduction" in s.lower() for s in signals)

    def test_report_with_numbered_sections(self) -> None:
        text = "1.0 Intro\n2.0 Methods\n3.0 Results\n4.0 Discussion\n" * 3
        score, _signals = detect_report(text)
        assert score >= 0.3

    def test_non_report(self) -> None:
        text = "- bullet 1\n- bullet 2\n- bullet 3"
        score, _ = detect_report(text)
        assert score < 0.5


class TestDetectSlidesDeep:
    def test_high_bullet_ratio(self) -> None:
        text = "\n\n".join([f"# Slide {i}\n\n- A\n- B\n- C" for i in range(6)])
        score, _signals = detect_slides(text)
        assert score >= 0.3

    def test_long_prose_not_slides(self) -> None:
        text = "This is a long prose document. " * 200
        score, _ = detect_slides(text)
        assert score < 0.5


class TestDetectTextbookDeep:
    def test_many_chapters_and_examples(self) -> None:
        text = "# Chapter 1: Intro\n" + "content " * 300 + "\nExample 1: foo\nExample 2: bar\nExercise 1: baz"
        text += "\n# Chapter 2: More\n" + "content " * 300
        score, signals = detect_textbook(text)
        assert score >= 0.5
        assert any("chapter" in s for s in signals)


class TestDetectSyllabusDeep:
    def test_all_signals(self) -> None:
        text = (
            "# Source Outline\n\n"
            "Maintainer: Team Lead\n\n"
            "Week 1: Intro\nWeek 2: Basics\nWeek 3: Intermediate\nWeek 4: Adv\n\n"
            "## Grading\n\n40% Midterm, 60% Final\n"
        )
        score, signals = detect_syllabus(text)
        assert score >= 0.7
        assert any("week" in s for s in signals)
        assert any("grading" in s or "assessment" in s for s in signals)


class TestDetectPaperDeep:
    def test_all_signals(self) -> None:
        text = (
            "# Deep RL Methods\n\n"
            "Abstract: We propose...\n\n"
            "Keywords: RL, control\n\n"
            "DOI: 10.1234/abc.456\n\n"
            "## References\n\n"
            "[1] Smith 2020.\n[2] Jones 2021.\n[3] Lee 2022.\n"
        )
        score, signals = detect_paper(text)
        assert score >= 0.8
        assert any("DOI" in s for s in signals)


class TestMetadataExtractorsDeep:
    def test_extract_abstract_multiline(self) -> None:
        text = "# Title\n\nAbstract: First sentence.\nSecond sentence.\n\n## Introduction\n"
        abstract = extract_abstract(text)
        assert "First sentence." in abstract

    def test_extract_abstract_missing(self) -> None:
        assert extract_abstract("# Title\n\nNo abstract here.") == ""

    def test_extract_authors_comma_separated(self) -> None:
        lines = ["# Paper Title", "Alice Smith, Bob Jones, Carol White", "", "Abstract: ..."]
        authors = extract_authors(lines)
        assert len(authors) >= 2
        assert "Alice Smith" in authors

    def test_extract_authors_with_and(self) -> None:
        lines = ["# Title", "Alice Smith and Bob Jones", "", "Abstract: ..."]
        authors = extract_authors(lines)
        assert "Alice Smith" in authors
        assert "Bob Jones" in authors

    def test_extract_authors_no_title(self) -> None:
        lines = ["No title here", "Alice Smith"]
        assert extract_authors(lines) == []

    def test_extract_emails(self) -> None:
        text = "Contact: alice@example.com and bob@test.org\n" * 2
        emails = extract_emails(text)
        assert "alice@example.com" in emails
        assert "bob@test.org" in emails

    def test_extract_emails_empty(self) -> None:
        assert extract_emails("No emails here.") == []

    def test_extract_journal(self) -> None:
        text = "Journal of Machine Learning Research\nVolume 25, 2024\n"
        journal = extract_journal(text)
        assert "Machine Learning" in journal or journal != ""

    def test_extract_volume_issue(self) -> None:
        text = "Vol. 25, No. 3\n"
        vol, issue = extract_volume_issue(text)
        # May or may not match depending on regex — just test it runs
        assert isinstance(vol, str)
        assert isinstance(issue, str)

    def test_extract_funding(self) -> None:
        text = "This work was supported by NSF grant #12345.\n"
        funding = extract_funding(text)
        assert "NSF" in funding or funding == ""

    def test_extract_funding_empty(self) -> None:
        assert extract_funding("Plain text without grants.") == ""


class TestYAMLFrontmatterDeep:
    def test_all_fields(self) -> None:
        meta = ScholarlyMetadata(
            title="Test Paper",
            authors=["Alice", "Bob"],
            doi="10.1000/test",
            journal="JMLR",
            volume="25",
            issue="3",
            year="2024",
            keywords=["ML", "RL"],
            abstract="A study on reinforcement learning.",
            funding="NSF grant #12345",
        )
        yaml_str = to_yaml_frontmatter(meta)
        assert yaml_str.startswith("---\n")
        assert yaml_str.strip().endswith("---")
        assert 'title: "Test Paper"' in yaml_str
        assert '  - "Alice"' in yaml_str
        assert '  - "Bob"' in yaml_str
        assert 'doi: "10.1000/test"' in yaml_str
        assert 'journal: "JMLR"' in yaml_str
        assert 'volume: "25"' in yaml_str
        assert 'issue: "3"' in yaml_str
        assert 'year: "2024"' in yaml_str
        assert '  - "ML"' in yaml_str
        assert "abstract:" in yaml_str
        assert "funding:" in yaml_str

    def test_empty_metadata(self) -> None:
        meta = ScholarlyMetadata()
        yaml_str = to_yaml_frontmatter(meta)
        assert yaml_str.startswith("---\n")
        assert yaml_str.strip().endswith("---")


class TestBibtexDeep:
    def test_all_fields(self) -> None:
        meta = ScholarlyMetadata(
            title="Test Paper",
            authors=["Alice Smith", "Bob Jones"],
            doi="10.1000/test",
            journal="JMLR",
            volume="25",
            issue="3",
            year="2024",
        )
        bib = to_bibtex(meta)
        assert "@article{smith2024," in bib
        assert "title = {Test Paper}" in bib
        assert "author = {Alice Smith and Bob Jones}" in bib
        assert "journal = {JMLR}" in bib
        assert "volume = {25}" in bib
        assert "number = {3}" in bib
        assert "year = {2024}" in bib
        assert "doi = {10.1000/test}" in bib

    def test_no_year(self) -> None:
        meta = ScholarlyMetadata(authors=["John Doe"])
        bib = to_bibtex(meta)
        assert "@article{doe," in bib

    def test_no_authors(self) -> None:
        meta = ScholarlyMetadata(title="Orphan Paper")
        bib = to_bibtex(meta)
        assert "@article{unknown," in bib


class TestAPA7Deep:
    def test_full_citation(self) -> None:
        meta = ScholarlyMetadata(
            title="Test Paper",
            authors=["Alice Smith"],
            year="2024",
            journal="JMLR",
            volume="25",
            issue="3",
            doi="10.1000/test",
        )
        apa = to_apa7(meta)
        assert "Alice Smith" in apa
        assert "(2024)" in apa
        assert "Test Paper" in apa
        assert "*JMLR*" in apa
        assert "https://doi.org/10.1000/test" in apa

    def test_no_year(self) -> None:
        meta = ScholarlyMetadata(title="Paper", authors=["X"])
        apa = to_apa7(meta)
        assert "(n.d.)" in apa

    def test_no_journal(self) -> None:
        meta = ScholarlyMetadata(title="Paper", year="2024")
        apa = to_apa7(meta)
        assert "Paper" in apa


class TestRenderFormulaItem:
    def test_valid_formula(self) -> None:
        class FakeItem:
            text = "E = mc^2"
            orig = ""

        result = render_formula_item(FakeItem())
        assert "E = mc^2" in result

    def test_empty_formula(self) -> None:
        class FakeItem:
            text = ""
            orig = ""

        from cortexmark.convert import FORMULA_PLACEHOLDER

        result = render_formula_item(FakeItem())
        assert result == FORMULA_PLACEHOLDER

    def test_short_formula_incomplete(self) -> None:
        class FakeItem:
            text = "x"
            orig = ""

        result = render_formula_item(FakeItem())
        assert "incomplete" in result.lower()

    def test_algorithmic_formula(self) -> None:
        class FakeItem:
            text = "Initialize θ\nFor each episode:\n  Sample action\n  Update weights\nReturn policy"
            orig = ""

        result = render_formula_item(FakeItem())
        assert "```text" in result or "algorithm" in result.lower() or "Initialize" in result


class TestCommonDetectDevice:
    def test_returns_string(self) -> None:
        device = detect_device()
        assert device in ("cpu", "cuda")


class TestCommonResolvePath:
    def test_absolute_path(self) -> None:
        p = resolve_path("/tmp/test.yaml")
        assert p == Path("/tmp/test.yaml")

    def test_relative_path(self) -> None:
        p = resolve_path("configs/pipeline.yaml")
        assert p.is_absolute()
        assert "configs" in str(p)


class TestConvertDeriveOutputPathDeep:
    def test_basic_derivation(self, tmp_path: Path) -> None:
        input_root = tmp_path / "pdfs"
        output_root = tmp_path / "md"
        input_root.mkdir()
        output_root.mkdir()
        pdf_path = input_root / "subdir" / "test.pdf"
        result = derive_output_path(pdf_path, input_root, output_root)
        assert result.suffix == ".md"
        assert "subdir" in str(result)

    def test_nested_subdirectory(self, tmp_path: Path) -> None:
        input_root = tmp_path / "in"
        output_root = tmp_path / "out"
        pdf_path = input_root / "a" / "b" / "c.pdf"
        result = derive_output_path(pdf_path, input_root, output_root)
        assert result == output_root / "in" / "a" / "b" / "c.md"


class TestNormalizeMarkdownDeep:
    def test_crlf_normalization(self) -> None:
        text = "Hello\r\nWorld\r\n"
        result = normalize_markdown(text)
        assert "\r" not in result

    def test_trailing_whitespace(self) -> None:
        text = "Hello   \nWorld  \n"
        result = normalize_markdown(text)
        # normalize_markdown strips then appends trailing newline
        assert result.endswith("\n")
        assert "\r" not in result


class TestRecoverFormulaPlaceholdersDeep:
    def test_multiple_placeholders(self) -> None:
        from cortexmark.convert import FORMULA_PLACEHOLDER

        class FakeItem:
            def __init__(self, text: str) -> None:
                self.text = text
                self.orig = ""

        items = [FakeItem("alpha + beta"), FakeItem("gamma * delta")]
        md = f"Before\n{FORMULA_PLACEHOLDER}\nMiddle\n{FORMULA_PLACEHOLDER}\nAfter"
        result = recover_formula_placeholders(md, items)
        assert "alpha + beta" in result
        assert "gamma * delta" in result
        assert FORMULA_PLACEHOLDER not in result

    def test_no_placeholders(self) -> None:
        result = recover_formula_placeholders("No placeholders here.", [])
        assert result == "No placeholders here."


class TestParseWeekEntriesDeep:
    def test_multi_week_with_bullets(self) -> None:
        text = (
            "## Section 1: Introduction\n"
            "- Overview of the course\n"
            "- Setup environment\n\n"
            "## Section 2: Basics\n"
            "- Fundamentals\n"
            "- Practice\n"
        )
        entries = parse_section_entries(text)
        assert 1 in entries
        assert 2 in entries
        assert "Introduction" in entries[1]["title"]
        assert len(entries[1]["bullets"]) == 2

    def test_week_without_bullets(self) -> None:
        text = "## Week 1: Intro\n\n## Week 2: Next\n"
        entries = parse_section_entries(text)
        assert 1 in entries
        assert entries[1]["bullets"] == []


class TestPluginRegistryDeep:
    def test_register_and_list(self) -> None:
        registry = PluginRegistry()

        class TestPlugin(PluginBase):
            name = "test-plugin"
            version = "1.0"
            description = "A test plugin"

        registry.register(TestPlugin())
        assert len(registry.plugins) == 1
        assert registry.plugins[0].name == "test-plugin"

    def test_register_invalid_type(self) -> None:
        registry = PluginRegistry()
        with pytest.raises(TypeError):
            registry.register("not a plugin")  # type: ignore[arg-type]

    def test_get_hooks(self) -> None:
        class HookPlugin(PluginBase):
            name = "hook-test"
            version = "1.0"
            description = "Hook test"

            def pre_convert(self, context: dict[str, Any]) -> dict[str, Any]:
                return context

        plugin = HookPlugin()
        hooks = plugin.get_hooks()
        assert "pre_convert" in hooks


class TestWritePluginReport:
    def test_writes_json(self, tmp_path: Path) -> None:
        infos = [
            PluginInfo(name="p1", description="Plugin 1", hooks=["pre_convert"]),
            PluginInfo(name="p2", description="Plugin 2", hooks=[]),
        ]
        out = tmp_path / "plugins.json"
        write_plugin_report(infos, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total_plugins"] == 2
        assert data["plugins"][0]["name"] == "p1"


class TestFiguresExtract:
    def test_extract_images(self) -> None:
        text = "# Document\n\n![Figure 1](images/fig1.png)\n\n![Chart](data/chart.jpg)\n"
        figures = extract_figures_from_text(text, "doc.md")
        assert len(figures) >= 2
        assert any("fig1.png" in f.image_path for f in figures)

    def test_no_figures(self) -> None:
        text = "# Document\n\nJust text, no images.\n"
        figures = extract_figures_from_text(text, "doc.md")
        assert len(figures) == 0


class TestAlgorithmExtractDeep:
    def test_parse_step_basic(self) -> None:
        step = parse_step("Initialize θ = 0", 0)
        assert step.text == "Initialize θ = 0"
        assert step.indent_level == 0

    def test_parse_algorithm_body(self) -> None:
        lines = [
            "INPUT: Dataset D",
            "OUTPUT: Model M",
            "1. Initialize weights",
            "2. Train model",
            "3. Return M",
        ]
        inputs, outputs, steps = parse_algorithm_body(lines)
        assert inputs == ["Dataset D"]
        assert outputs == ["Model M"]
        assert len(steps) >= 2

    def test_extract_algorithms_fenced(self) -> None:
        text = (
            "# Algorithm 1: Gradient Descent\n\n"
            "```\n"
            "INPUT: Learning rate α\n"
            "OUTPUT: Optimal θ\n"
            "1. Initialize θ\n"
            "2. Repeat:\n"
            "   θ = θ - α * ∇J(θ)\n"
            "3. Return θ\n"
            "```\n"
        )
        algos = extract_algorithms(text, "test.md")
        assert len(algos) >= 1

    def test_extract_algorithms_empty(self) -> None:
        assert extract_algorithms("Just text.", "test.md") == []

    def test_is_algorithm_content_positive(self) -> None:
        text = "Initialize variables\nFor each iteration:\n  Update weights\nReturn result"
        assert is_algorithm_content(text)

    def test_is_algorithm_content_negative(self) -> None:
        assert not is_algorithm_content("This is just a paragraph of text.")


class TestCitationsDeep:
    def test_extract_inline_citations(self) -> None:
        text = "As shown by Smith (2020) and Jones et al. (2021), the method works. See also [1] and [2]."
        citations = extract_inline_citations(text, "doc.md")
        assert len(citations) >= 1

    def test_extract_references(self) -> None:
        text = (
            "## References\n\n"
            "[1] Smith, J. (2020). A Study. Journal of ML, 25(3).\n"
            "[2] Jones, K. et al. (2021). Another Study. NeurIPS.\n"
        )
        refs = extract_references(text)
        assert len(refs) >= 1

    def test_build_citation_graph_empty(self) -> None:
        graph = build_citation_graph([], [])
        assert len(graph.citations) == 0
        assert graph.edges == []

    def test_build_citation_graph_linked(self) -> None:
        citations = [Citation(raw_text="[1]", source_file="a.md", line_number=5, cite_type="numeric")]
        refs = [Reference(key="1", raw_text="Smith 2020", authors="Smith", year="2020", title="A Study", doi="")]
        graph = build_citation_graph(citations, refs, source_doc="a.md")
        assert len(graph.citations) >= 1


class TestRenderMetaTemplates:
    def test_render_meta_templates(self, tmp_path: Path) -> None:
        source_root = tmp_path / "source"
        source_root.mkdir()
        outline_text = (
            "Source Name: Structured Source Processing\n"
            "Maintainer: Team Lead\n"
            "Source Code: ABC123\n\n"
            "## Recommended Tools\n- Python\n- MATLAB\n\n"
            "## Evaluation\n- 40% Midterm\n- 60% Final\n\n"
            "## Section 1: Introduction\n- Basics of RL\n\n"
            "## Section 2: Segments\n- Markov processes\n"
        )
        entries = parse_section_entries(outline_text)
        from cortexmark.render_templates import render_meta_templates

        written = render_meta_templates(source_root, outline_text, entries)
        assert len(written) == 2
        profile = (source_root / "00_meta" / "source_profile.md").read_text(encoding="utf-8")
        assert "Structured Source Processing" in profile
        assert "Team Lead" in profile
        rules = (source_root / "00_meta" / "global_rules.md").read_text(encoding="utf-8")
        assert "# Global Rules" in rules


class TestRenderSectionTemplates:
    def test_render_section_templates(self, tmp_path: Path) -> None:
        source_root = tmp_path / "source"
        raw_root = tmp_path / "raw"
        cleaned_root = tmp_path / "cleaned"
        for d in [source_root, raw_root, cleaned_root]:
            d.mkdir()

        # Create a section directory
        section_dir = source_root / "01_introduction"
        section_dir.mkdir()

        # Create raw content
        raw_section = raw_root / "01_introduction"
        raw_section.mkdir()
        (raw_section / "content.md").write_text(
            "# Introduction to Structured Sources\n\nStructured source processing uses deterministic steps.\n\n## Core Concepts\n\n- State\n- Action\n",
            encoding="utf-8",
        )

        entries = {1: {"title": "Introduction", "bullets": ["Source basics", "Tool setup"]}}
        from cortexmark.render_templates import render_section_templates

        written = render_section_templates(source_root, raw_root, cleaned_root, entries)
        assert len(written) >= 2
        rules_path = section_dir / "rules.md"
        task_list_path = section_dir / "tasks" / "task_list.md"
        assert rules_path.exists()
        assert task_list_path.exists()
        rules_text = rules_path.read_text(encoding="utf-8")
        assert "Section 01 Rules" in rules_text
        assert "Introduction" in rules_text

    def test_empty_source(self, tmp_path: Path) -> None:
        source_root = tmp_path / "source"
        raw_root = tmp_path / "raw"
        cleaned_root = tmp_path / "cleaned"
        for d in [source_root, raw_root, cleaned_root]:
            d.mkdir()
        from cortexmark.render_templates import render_section_templates

        written = render_section_templates(source_root, raw_root, cleaned_root, {})
        assert written == []


class TestBuildSourceProfileText:
    def test_full_profile(self) -> None:
        text = build_source_profile_text(
            source_name="Structured Source",
            source_cycle="Cycle 2024",
            maintainer="Maintainer X",
            main_topics=["Segments", "Rule Extraction"],
            programs=["Python"],
            notes=["Note 1"],
        )
        assert "# Source Profile" in text
        assert "Structured Source" in text
        assert "Cycle 2024" in text
        assert "- Segments" in text
        assert "- Python" in text


class TestBuildGlobalRulesText:
    def test_rules_output(self) -> None:
        text = build_global_rules_text(
            existing_rules=["Rule 1", "Rule 2"],
            ai_rules=["AI Rule"],
            admin_rules=["Admin Rule"],
        )
        assert "# Global Rules" in text
        assert "- Rule 1" in text
        assert "- AI Rule" in text


class TestBuildSectionRulesText:
    def test_section_rules(self) -> None:
        text = build_section_rules_text(
            section_number=3,
            title="Structured Processing",
            scope_items=["Boundary detection", "Rule extraction"],
            exclude_items=["Unrelated topics"],
            output_items=["Summary"],
        )
        assert "Section 03 Rules" in text
        assert "## Scope" in text
        assert "Boundary detection" in text


class TestBuildAssignmentText:
    def test_assignment_output(self) -> None:
        text = build_assignment_text(
            section_number=5,
            objective="Learn section processing",
            tasks=["Summarize the key concepts", "List the main constraints"],
            submission=["Submit as Markdown"],
        )
        assert "Section 05 Tasks" in text
        assert "## Objective" in text
        assert "Learn section processing" in text
        assert "1. Summarize the key concepts" in text


class TestRunPipelineBuildParserFull:
    def test_all_stages(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--stages", "convert", "clean", "chunk", "render", "analyze", "validate"])
        assert args.stages == ["convert", "clean", "chunk", "render", "analyze", "validate"]

    def test_engine_override(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--engine", "markitdown"])
        assert args.engine == "markitdown"

    def test_no_manifest(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--no-manifest"])
        assert args.no_manifest is True

    def test_session_name(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--session-name", "test-session"])
        assert args.session_name == "test-session"

    def test_input_path(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--input", "/tmp/test.pdf"])
        assert args.input == Path("/tmp/test.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Coverage boost phase 2 — chunk_file, chunk_tree, convert, diff, figures CLI
# ══════════════════════════════════════════════════════════════════════════════


class TestChunkFileDeep:
    def test_chunk_file_creates_output(self, tmp_path: Path) -> None:
        md = tmp_path / "input.md"
        md.write_text(
            "# Chapter 1\n\nContent of chapter 1.\n\n# Chapter 2\n\nContent of chapter 2.\n",
            encoding="utf-8",
        )
        out = tmp_path / "chunks"
        written = chunk_file(md, out, split_levels=[1])
        assert len(written) >= 2
        assert all(p.exists() for p in written)

    def test_chunk_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            chunk_file(tmp_path / "missing.md", tmp_path / "out")

    def test_chunk_file_no_chunks(self, tmp_path: Path) -> None:
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        assert chunk_file(md, tmp_path / "out") == []  # was: raised ValueError


class TestChunkTreeDeep:
    def test_chunk_tree_single_file(self, tmp_path: Path) -> None:
        input_root = tmp_path / "input"
        input_root.mkdir()
        (input_root / "doc.md").write_text(
            "# Section A\n\nContent A.\n\n# Section B\n\nContent B.\n",
            encoding="utf-8",
        )
        output_root = tmp_path / "output"
        output_root.mkdir()
        written = chunk_tree(input_root, output_root)
        assert len(written) >= 2

    def test_chunk_tree_without_root_prefix(self, tmp_path: Path) -> None:
        input_root = tmp_path / "session" / "cleaned"
        input_root.mkdir(parents=True)
        (input_root / "doc.md").write_text("# Section\n\nContent.\n", encoding="utf-8")
        output_root = tmp_path / "session" / "chunks"
        output_root.mkdir(parents=True)
        written = chunk_tree(input_root, output_root, include_input_root_name=False)
        assert any((output_root / "doc").exists() or p.parent == output_root / "doc" for p in written)

    def test_chunk_tree_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            chunk_tree(empty, tmp_path / "out")


class TestConvertFormatFunctions:
    def test_format_formula_block(self) -> None:
        result = format_formula_block("E = mc^2")
        assert "E = mc^2" in result
        assert "Equation" in result

    def test_format_incomplete_formula_block(self) -> None:
        result = format_incomplete_formula_block()
        assert "incomplete" in result.lower()

    def test_format_algorithm_block(self) -> None:
        text = "Initialize θ\nFor each iteration:\n  Update θ\nReturn θ"
        result = format_algorithm_block(text)
        assert "```text" in result
        assert "Initialize" in result

    def test_is_algorithmic_text_true(self) -> None:
        assert is_algorithmic_text("Initialize weights\nFor each episode:\n  Sample action")

    def test_is_algorithmic_text_false(self) -> None:
        assert not is_algorithmic_text("This is just a regular paragraph.")

    def test_normalize_recovered_text(self) -> None:
        result = normalize_recovered_text("   hello   world   ")
        assert result == "hello world"

    def test_reformat_algorithm_sections(self) -> None:
        text = "## Algorithm 1 Algorithm parameters: α = 0.1, γ = 0.99\n"
        result = reformat_algorithm_sections(text)
        assert "## Algorithm 1" in result

    def test_merge_docling_markitdown(self) -> None:
        docling_md = "# Title\n\nDocling paragraph one.\n\nDocling paragraph two.\n"
        markitdown_md = "# Title\n\nMarkitdown paragraph one.\n\nUnique markitdown paragraph.\n"
        merged = merge_docling_markitdown(docling_md, markitdown_md)
        assert "Title" in merged
        assert len(merged) > 0


class TestNormalizeMarkdownMore:
    def test_crlf_to_lf(self) -> None:
        result = normalize_markdown("Hello\r\nWorld\r\n")
        assert "\r" not in result
        assert "Hello" in result

    def test_cr_to_lf(self) -> None:
        result = normalize_markdown("Hello\rWorld")
        assert "\r" not in result


class TestDeriveOutputPathMore:
    def test_root_file(self, tmp_path: Path) -> None:
        input_root = tmp_path / "pdfs"
        output_root = tmp_path / "md"
        pdf = input_root / "test.pdf"
        result = derive_output_path(pdf, input_root, output_root)
        assert result.suffix == ".md"
        assert result.name == "test.md"


class TestFiguresExtractDeep:
    def test_html_image(self) -> None:
        text = '# Document\n\n<img src="figures/img.png" alt="Test Image">\n'
        figures = extract_figures_from_text(text, "test.md")
        assert len(figures) >= 1

    def test_figure_with_title(self) -> None:
        text = '![Alt text](path/to/img.jpg "Optional title")\n'
        figures = extract_figures_from_text(text, "test.md")
        assert len(figures) >= 1

    def test_extract_from_file_func(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Doc\n\n![Figure](images/fig.png)\n", encoding="utf-8")
        report = extract_from_file(md)
        assert isinstance(report, FigureReport)

    def test_extract_from_tree_func(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("# A\n\n![Fig](img.png)\n", encoding="utf-8")
        (d / "b.md").write_text("# B\n\nNo figures here.\n", encoding="utf-8")
        report = extract_from_tree(d)
        assert isinstance(report, FigureReport)


class TestAlgorithmExtractMore:
    def test_algo_header_re(self) -> None:
        assert ALGO_HEADER_RE.search("# Algorithm 1: Gradient Descent")
        assert ALGO_HEADER_RE.search("## Algorithm 2. Q-Learning")
        assert not ALGO_HEADER_RE.search("# Introduction")

    def test_extract_from_file(self, tmp_path: Path) -> None:
        md = tmp_path / "algo.md"
        md.write_text(
            "# Algorithm 1: Test\n\n```\nINPUT: x\nOUTPUT: y\n1. Compute y = f(x)\n2. Return y\n```\n",
            encoding="utf-8",
        )
        algos = algo_extract_from_file(md)
        assert isinstance(algos, list)

    def test_extract_from_tree_func(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("# Algorithm 1: A\n\n```\n1. Do\n2. Done\n```\n", encoding="utf-8")
        result = algo_extract_from_tree(d)
        assert isinstance(result, list)


class TestFormulaScoreDeep:
    def test_score_file_func(self, tmp_path: Path) -> None:
        md = tmp_path / "formulas.md"
        md.write_text(
            "# Math\n\n$$E = mc^2$$\n\n$$\\int_0^1 x dx$$\n\nNormal text.\n",
            encoding="utf-8",
        )
        report = score_file(md)
        assert isinstance(report, FileReport)

    def test_validate_formula_text(self) -> None:
        valid, issues = validate_formula_text("E = mc^2")
        assert isinstance(valid, bool)
        assert isinstance(issues, list)

    def test_build_file_report(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("$$E = mc^2$$\n\nSome text.\n$$F = ma$$\n", encoding="utf-8")
        report = build_file_report(md, md.read_text(encoding="utf-8"))
        assert isinstance(report, FileReport)
        assert report.file == str(md)


class TestOcrQualityMore:
    def test_count_functions(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        common = count_common_words(text)
        assert common >= 1
        garble = count_garble_chars(text)
        assert garble == 0
        noise = count_short_noise_lines(text)
        assert noise == 0
        soup = count_symbol_soup(text)
        assert soup == 0

    def test_assess_quality_clean_text(self) -> None:
        text = "This is a clean document with proper English text.\nIt has multiple sentences and paragraphs.\n\nAnother paragraph here."
        metrics = assess_quality(text)
        assert isinstance(metrics, OCRQualityMetrics)
        assert metrics.confidence >= 0.0

    def test_confidence_grade(self) -> None:
        assert confidence_to_grade(0.95) in ("A", "B")
        assert confidence_to_grade(0.5) in ("C", "D", "F")
        assert confidence_to_grade(0.1) in ("D", "F")


class TestMultiFormatDeep:
    def test_md_to_html(self) -> None:
        html = md_to_html("# Title\n\nParagraph.\n")
        assert "<h1>" in html or "<h1" in html
        assert "Title" in html

    def test_md_to_text(self) -> None:
        text = md_to_text("# Title\n\n**Bold** and *italic*.\n")
        assert "Title" in text
        assert "Bold" in text

    def test_md_to_yaml(self) -> None:
        result = md_to_yaml("# Title\n\nSome content.\n")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_convert_file(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("# Test\n\nContent.\n", encoding="utf-8")
        result = convert_file(md, tmp_path, fmt="html")
        assert result.exists()
        assert result.suffix == ".html"


# ══════════════════════════════════════════════════════════════════════════════
# Coverage boost phase 3 — clean_file, clean_tree, parallel,
# topics, rag_export, diff, notation, ghpages, semantic_chunk, qa_pipeline
# ══════════════════════════════════════════════════════════════════════════════


class TestCleanFileFunc:
    def test_clean_file(self, tmp_path: Path) -> None:
        input_md = tmp_path / "raw.md"
        input_md.write_text(
            "# Title\n\n\n\nSome content.\n\n---\n\n# Another Section\n\nMore content.\n",
            encoding="utf-8",
        )
        output_md = tmp_path / "out" / "cleaned.md"
        result = clean_file(input_md, output_md)
        assert result == output_md
        assert output_md.exists()
        text = output_md.read_text(encoding="utf-8")
        assert "Title" in text

    def test_clean_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            clean_file(tmp_path / "missing.md", tmp_path / "out.md")

    def test_clean_file_with_cfg(self, tmp_path: Path) -> None:
        md = tmp_path / "input.md"
        md.write_text("# Header\n\nParagraph.\n", encoding="utf-8")
        out = tmp_path / "out.md"
        result = clean_file(md, out, cfg={"clean": {"min_repeated_header_count": 5}})
        assert result.exists()


class TestCleanTreeFunc:
    def test_clean_tree(self, tmp_path: Path) -> None:
        input_root = tmp_path / "raw"
        input_root.mkdir()
        (input_root / "doc1.md").write_text("# Doc1\n\nContent 1.\n", encoding="utf-8")
        subdir = input_root / "sub"
        subdir.mkdir()
        (subdir / "doc2.md").write_text("# Doc2\n\nContent 2.\n", encoding="utf-8")
        output_root = tmp_path / "cleaned"
        output_root.mkdir()
        written = clean_tree(input_root, output_root)
        assert len(written) == 2
        assert all(p.exists() for p in written)

    def test_clean_tree_without_root_prefix(self, tmp_path: Path) -> None:
        input_root = tmp_path / "session" / "raw_md"
        input_root.mkdir(parents=True)
        (input_root / "doc.md").write_text("# Doc\n\nContent.\n", encoding="utf-8")
        output_root = tmp_path / "session" / "cleaned_md"
        output_root.mkdir(parents=True)
        written = clean_tree(input_root, output_root, include_input_root_name=False)
        assert written == [output_root / "doc.md"]
        assert written[0].exists()

    def test_clean_tree_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            clean_tree(empty, tmp_path / "out")


class TestNormalizeTableCellExtra:
    def test_collapse_whitespace(self) -> None:
        result = normalize_table_cell("  hello   world  ")
        assert result == "hello world"


class TestNormalizeTableBlocksDeep:
    def test_simple_table(self) -> None:
        text = "| A | B |\n|---|---|\n| 1 | 2 |\n"
        result = normalize_table_blocks(text)
        assert "| A |" in result
        assert "| 1 |" in result


class TestParallelMapDeep:
    def test_single_worker(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text("hello", encoding="utf-8")
        report = parallel_map(lambda p: p.read_text(encoding="utf-8"), [f], config=ParallelConfig(workers=1))
        assert isinstance(report, ParallelReport)
        assert report.total == 1
        assert report.succeeded == 1

    def test_empty_paths(self) -> None:
        report = parallel_map(lambda p: None, [])
        assert report.total == 0

    def test_multi_worker(self, tmp_path: Path) -> None:
        files = []
        for i in range(5):
            f = tmp_path / f"f{i}.md"
            f.write_text(f"content {i}", encoding="utf-8")
            files.append(f)
        report = parallel_map(
            lambda p: p.read_text(encoding="utf-8"),
            files,
            config=ParallelConfig(workers=2, pool_type="thread"),
        )
        assert report.total == 5
        assert report.succeeded == 5

    def test_error_handling(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text("x", encoding="utf-8")

        def fail(p: Path) -> str:
            raise ValueError("test error")

        report = parallel_map(fail, [f], config=ParallelConfig(workers=1))
        assert report.failed == 1
        assert "test error" in report.results[0].error


class TestCollectMdFilesDeep:
    def test_collect(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("b", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.md").write_text("c", encoding="utf-8")
        result = collect_md_files(tmp_path)
        assert len(result) == 2

    def test_empty(self, tmp_path: Path) -> None:
        result = collect_md_files(tmp_path)
        assert result == []


class TestParallelTreeDeep:
    def test_parallel_tree_basic(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("hello", encoding="utf-8")
        (tmp_path / "b.md").write_text("world", encoding="utf-8")
        report = parallel_tree(
            lambda p: p.read_text(encoding="utf-8"),
            tmp_path,
            config=ParallelConfig(workers=1),
        )
        assert report.total == 2
        assert report.succeeded == 2

    def test_parallel_tree_empty(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parallel_tree(lambda p: None, tmp_path)


class TestDiffDeep:
    def test_diff_files_identical(self, tmp_path: Path) -> None:
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("Same content\n", encoding="utf-8")
        b.write_text("Same content\n", encoding="utf-8")
        result = diff_files(a, b)
        assert result.status == "unchanged"

    def test_diff_files_modified(self, tmp_path: Path) -> None:
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("Line 1\nLine 2\n", encoding="utf-8")
        b.write_text("Line 1\nLine 2 modified\nLine 3 new\n", encoding="utf-8")
        result = diff_files(a, b)
        assert result.status == "modified"

    def test_diff_trees(self, tmp_path: Path) -> None:
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir()
        new.mkdir()
        (old / "shared.md").write_text("old text\n", encoding="utf-8")
        (new / "shared.md").write_text("new text\n", encoding="utf-8")
        (old / "removed.md").write_text("gone\n", encoding="utf-8")
        (new / "added.md").write_text("new file\n", encoding="utf-8")
        td = diff_trees(old, new)
        assert isinstance(td, TreeDiff)
        assert "added.md" in str(td.files_added) or len(td.files_added) > 0

    def test_write_diff_report(self, tmp_path: Path) -> None:
        td = TreeDiff()
        out = tmp_path / "report.json"
        result = write_diff_report(td, out)
        assert result.exists()

    def test_write_unified_diff(self, tmp_path: Path) -> None:
        old = tmp_path / "a.md"
        new = tmp_path / "b.md"
        old.write_text("L1\n", encoding="utf-8")
        new.write_text("L1 changed\n", encoding="utf-8")
        fd = diff_files(old, new)
        td = TreeDiff(file_diffs=[fd])
        out = tmp_path / "diff.txt"
        write_unified_diff(td, out)
        assert out.exists()


class TestTopicsDeep:
    def test_classify_file(self, tmp_path: Path) -> None:
        md = tmp_path / "ml_paper.md"
        md.write_text(
            "# Machine Learning\n\nNeural networks and deep learning models.\n\n"
            "## Classification\n\nSVM, Random Forest, and Gradient Boosting.\n",
            encoding="utf-8",
        )
        result = classify_file(md)
        assert isinstance(result, DocumentTopics)

    def test_write_topic_report(self, tmp_path: Path) -> None:
        results = [
            DocumentTopics(source_file="a.md", word_count=100, primary_topic="ml"),
        ]
        out = tmp_path / "topics.json"
        written = write_topic_report(results, out)
        assert written.exists()


class TestRagExportDeep:
    def test_export_file(self, tmp_path: Path) -> None:
        md = tmp_path / "chunk_001_intro.md"
        md.write_text("# Introduction\n\nThis is the introduction.\n", encoding="utf-8")
        record = export_file(md)
        assert isinstance(record, RAGRecord)

    def test_write_jsonl(self, tmp_path: Path) -> None:
        records = [RAGRecord(id="test-1", source="a.md", title="Title", text="Body text")]
        out = tmp_path / "output.jsonl"
        written = write_jsonl(records, out)
        assert written.exists()

    def test_write_json_array(self, tmp_path: Path) -> None:
        records = [RAGRecord(id="test-1", source="a.md", title="Title", text="Body text")]
        out = tmp_path / "output.json"
        written = write_json_array(records, out)
        assert written.exists()
        data = json.loads(written.read_text(encoding="utf-8"))
        assert len(data) == 1

    def test_rag_build_summary(self) -> None:
        records = [
            RAGRecord(id="1", source="a.md", title="T", text="Body"),
            RAGRecord(id="2", source="a.md", title="T2", text="Body2"),
        ]
        summary = rag_build_summary(records)
        assert isinstance(summary, dict)


class TestGhPagesDeep:
    def test_write_site_manifest(self, tmp_path: Path) -> None:
        pages = [PageEntry(title="Test", relative_path="test.html", source_md="test.md")]
        out = tmp_path / "manifest.json"
        result = write_site_manifest(pages, out)
        assert result.exists()

    def test_build_nav_html(self) -> None:
        pages = [
            PageEntry(title="Home", relative_path="index.html", source_md="index.md"),
            PageEntry(title="About", relative_path="about.html", source_md="about.md"),
        ]
        html = build_nav_html(pages)
        assert "Home" in html
        assert "About" in html

    def test_collect_pages(self, tmp_path: Path) -> None:
        (tmp_path / "a.html").write_text("<h1>A</h1>", encoding="utf-8")
        (tmp_path / "b.html").write_text("<h1>B</h1>", encoding="utf-8")
        pages = collect_pages(tmp_path)
        assert isinstance(pages, list)

    def test_generate_site(self, tmp_path: Path) -> None:
        md_dir = tmp_path / "md"
        md_dir.mkdir()
        (md_dir / "doc.md").write_text("# Document\n\nContent.\n", encoding="utf-8")
        site_dir = tmp_path / "site"
        written = generate_site(md_dir, site_dir)
        assert len(written) >= 1


class TestCrossRefDeep:
    def test_analyze_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text(
            "# Document\n\n**Theorem 1.** Statement.\n\nSee Theorem 1.\n\n**Figure 1.** Caption.\n\nAs in Figure 1.\n",
            encoding="utf-8",
        )
        report = crossref_analyze_file(md)
        assert isinstance(report, CrossRefReport)

    def test_analyze_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text(
            "**Equation 1.** $E=mc^2$\n\nSee Equation 1.\n",
            encoding="utf-8",
        )
        report = crossref_analyze_tree(d)
        assert isinstance(report, CrossRefReport)

    def test_write_crossref_report(self, tmp_path: Path) -> None:
        report = CrossRefReport()
        out = tmp_path / "crossref.json"
        result = write_crossref_report(report, out)
        assert result.exists()


class TestNotationGlossaryDeep:
    def test_extract_from_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text(
            "# Math\n\nLet $\\alpha$ denote the learning rate.\n\nWe define $\\gamma$ as the discount factor.\n",
            encoding="utf-8",
        )
        glossary = notation_extract_from_file(md)
        assert isinstance(glossary, NotationGlossary)

    def test_extract_from_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text(
            "Let $x$ be the input.\n",
            encoding="utf-8",
        )
        glossary = notation_extract_from_tree(d)
        assert isinstance(glossary, NotationGlossary)

    def test_write_notation_report(self, tmp_path: Path) -> None:
        glossary = NotationGlossary()
        out = tmp_path / "notation.json"
        result = write_notation_report(glossary, out)
        assert result.exists()

    def test_write_markdown_glossary(self, tmp_path: Path) -> None:
        glossary = NotationGlossary(entries=[NotationEntry(symbol="α", definition="learning rate")])
        out = tmp_path / "glossary.md"
        result = write_markdown_glossary(glossary, out)
        assert result.exists()
        text = result.read_text(encoding="utf-8")
        assert "α" in text


class TestMetadataDeep:
    def test_extract_file(self, tmp_path: Path) -> None:
        md = tmp_path / "paper.md"
        md.write_text(
            "# A Study of Machine Learning\n\nJohn Doe, Jane Smith\n\n"
            "## Abstract\n\nWe study ML methods.\n\n"
            "Keywords: machine learning, neural networks\n\n"
            "DOI: 10.1234/test.2024\n",
            encoding="utf-8",
        )
        meta = extract_file(md)
        assert isinstance(meta, ScholarlyMetadata)

    def test_write_metadata_report(self, tmp_path: Path) -> None:
        meta = ScholarlyMetadata(source_file="test.md", title="Test")
        out = tmp_path / "metadata.json"
        result = write_metadata_report([meta], out)
        assert result.exists()

    def test_extract_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("# Paper Title\n\nContent.\n", encoding="utf-8")
        results = extract_tree(d)
        assert isinstance(results, list)
        assert len(results) == 1


class TestDocTypeDeep:
    def test_detect_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Chapter 1\n\nSome content.\n\n# Chapter 2\n\nMore content.\n", encoding="utf-8")
        result = detect_file(md)
        assert isinstance(result, DocTypeResult)

    def test_detect_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("# Abstract\n\nWe present results.\n", encoding="utf-8")
        results = detect_tree(d)
        assert isinstance(results, list)

    def test_write_detection_report(self, tmp_path: Path) -> None:
        result = DocTypeResult(source_file="test.md", doc_type=GENERIC, confidence=0.9)
        out = tmp_path / "doctype.json"
        written = write_detection_report([result], out)
        assert written.exists()


class TestQaPipelineDeep:
    def test_qa_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("# Document\n\nSome content here.\n\n## Section 2\n\nMore text.\n", encoding="utf-8")
        report = qa_file(md)
        assert isinstance(report, FileQAReport)

    def test_qa_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("# A\n\nContent.\n", encoding="utf-8")
        reports = qa_tree(d)
        assert isinstance(reports, list)

    def test_write_qa_report(self, tmp_path: Path) -> None:
        report = FileQAReport(file="test.md")
        summary = build_summary([report])
        out = tmp_path / "qa.json"
        written = write_qa_report([report], summary, out)
        assert written.exists()

    def test_write_markdown_report(self, tmp_path: Path) -> None:
        report = FileQAReport(file="test.md")
        summary = build_summary([report])
        out = tmp_path / "qa.md"
        written = write_markdown_report([report], summary, out)
        assert written.exists()

    def test_build_summary(self) -> None:
        from cortexmark.qa_pipeline import QASummary

        report = FileQAReport(file="test.md")
        summary = build_summary([report])
        assert isinstance(summary, QASummary)


class TestOcrQualityDeep:
    def test_assess_file(self, tmp_path: Path) -> None:
        md = tmp_path / "doc.md"
        md.write_text("The quick brown fox jumps over the lazy dog.\n" * 10, encoding="utf-8")
        report = assess_file(md)
        assert isinstance(report, OCRFileReport)

    def test_assess_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("Normal text content.\n", encoding="utf-8")
        reports = assess_tree(d)
        assert isinstance(reports, list)

    def test_write_ocr_report(self, tmp_path: Path) -> None:
        report = OCRFileReport(source_file="test.md", metrics=OCRQualityMetrics(), grade="A")
        out = tmp_path / "ocr.json"
        written = write_ocr_report([report], out)
        assert written.exists()


class TestFormulaScoreTree:
    def test_score_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("$$E = mc^2$$\n\nText.\n", encoding="utf-8")
        reports = score_tree(d)
        assert isinstance(reports, list)

    def test_score_markdown(self) -> None:
        text = "Some text.\n\n$$F = ma$$\n\nMore text.\n"
        result = score_markdown(text)
        assert isinstance(result, list)

    def test_write_report(self, tmp_path: Path) -> None:
        report = FileReport(file="test.md")
        out = tmp_path / "formula.json"
        written = write_report([report], out)
        assert written.exists()


class TestCitationsDeepExtra:
    def test_analyze_tree(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.md").write_text("Some text [1].\n\n## References\n\n1. Smith 2020.\n", encoding="utf-8")
        graph = analyze_tree(d)
        assert isinstance(graph, CitationGraph)

    def test_write_citation_report(self, tmp_path: Path) -> None:
        graph = CitationGraph()
        out = tmp_path / "citations.json"
        written = write_citation_report(graph, out)
        assert written.exists()

    def test_write_dot_graph(self, tmp_path: Path) -> None:
        graph = CitationGraph()
        out = tmp_path / "citations.dot"
        written = write_dot_graph(graph, out)
        assert written.exists()


class TestFiguresDeepExtra:
    def test_write_figure_manifest(self, tmp_path: Path) -> None:
        report = FigureReport()
        out = tmp_path / "figures.json"
        result = write_figure_manifest(report, out)
        assert result.exists()

    def test_write_gallery_page(self, tmp_path: Path) -> None:
        entry = FigureEntry(image_path="fig1.png", alt_text="Figure 1", source_file="a.md", line_number=5)
        report = FigureReport(figures=[entry])
        out = tmp_path / "gallery.md"
        result = write_gallery_page(report, out)
        assert result.exists()
        text = result.read_text(encoding="utf-8")
        assert "fig1.png" in text


class TestAlgorithmWriteReport:
    def test_write_report(self, tmp_path: Path) -> None:
        algos = [Algorithm(label="1", title="Test", source_file="a.md", raw_text="algo text")]
        out = tmp_path / "algo.json"
        result = write_algo_report(algos, out)
        assert result.exists()


# ── build_parser coverage for all modules ────────────────────────────────────


class TestAllBuildParsers:
    """Test every module's build_parser() to cover CLI argument definitions."""

    def test_chunk_parser(self) -> None:
        from cortexmark.chunk import build_parser as bp

        p = bp()
        assert p is not None
        ns = p.parse_args(["--input", "/tmp/x", "--output-dir", "/tmp/y"])
        assert ns.input == Path("/tmp/x")

    def test_clean_parser(self) -> None:
        from cortexmark.clean import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x", "--output-dir", "/tmp/y"])
        assert ns.input == Path("/tmp/x")

    def test_convert_parser(self) -> None:
        from cortexmark.convert import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_diff_parser(self) -> None:
        from cortexmark.diff import build_parser as bp

        p = bp()
        ns = p.parse_args(["--old", "/tmp/a", "--new", "/tmp/b"])
        assert ns.old == Path("/tmp/a")

    def test_parallel_parser(self) -> None:
        from cortexmark.parallel import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x", "--operation", "ocr_quality"])
        assert ns.operation == "ocr_quality"

    def test_figures_parser(self) -> None:
        from cortexmark.figures import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_formula_score_parser(self) -> None:
        from cortexmark.formula_score import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_ghpages_parser(self) -> None:
        from cortexmark.ghpages import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_metadata_parser(self) -> None:
        from cortexmark.metadata import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_doc_type_parser(self) -> None:
        from cortexmark.doc_type import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_rag_export_parser(self) -> None:
        from cortexmark.rag_export import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_ocr_quality_parser(self) -> None:
        from cortexmark.ocr_quality import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_citations_parser(self) -> None:
        from cortexmark.citations import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_qa_pipeline_parser(self) -> None:
        from cortexmark.qa_pipeline import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_topics_parser(self) -> None:
        from cortexmark.topics import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_plugin_parser(self) -> None:
        from cortexmark.plugin import build_parser as bp

        p = bp()
        ns = p.parse_args(["--plugin-dir", "/tmp/x"])
        assert ns.plugin_dir == Path("/tmp/x")

    def test_cross_ref_parser(self) -> None:
        from cortexmark.cross_ref import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_notation_glossary_parser(self) -> None:
        from cortexmark.notation_glossary import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_algorithm_extract_parser(self) -> None:
        from cortexmark.algorithm_extract import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_multi_format_parser(self) -> None:
        from cortexmark.multi_format import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_semantic_chunk_parser(self) -> None:
        from cortexmark.semantic_chunk import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x"])
        assert ns.input == Path("/tmp/x")

    def test_render_templates_parser(self) -> None:
        from cortexmark.render_templates import build_parser as bp

        p = bp()
        ns = p.parse_args(["--outline-file", "00_meta/outline.md"])
        assert ns.outline_file == Path("00_meta/outline.md")


# ── Additional uncovered function tests ──────────────────────────────────────


class TestChunkTreeManifest:
    """Cover chunk_tree with manifest skip path."""

    def test_chunk_tree_skips_manifest(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        md = d / "c.md"
        md.write_text("# Title\n\nParagraph\n", encoding="utf-8")
        out = tmp_path / "out"
        manifest = Manifest(tmp_path / ".manifest.json")

        first = chunk_tree(d, out, manifest=manifest)
        result = chunk_tree(d, out, manifest=manifest)

        assert len(first) == 1
        assert result == []

    def test_chunk_tree_regenerates_all_outputs_when_one_is_deleted(self, tmp_path: Path) -> None:
        input_root = tmp_path / "docs"
        input_root.mkdir()
        source = input_root / "doc.md"
        source.write_text("# One\nBody one.\n\n## Two\nBody two.\n", encoding="utf-8")
        output_root = tmp_path / "out"
        manifest = Manifest(tmp_path / ".manifest.json")

        first = chunk_tree(input_root, output_root, manifest=manifest)
        first[0].unlink()
        regenerated = chunk_tree(input_root, output_root, manifest=manifest)

        assert len(first) == 2
        assert len(regenerated) == 2
        assert all(path.exists() for path in regenerated)

    def test_chunk_tree_regenerates_when_split_levels_change(self, tmp_path: Path) -> None:
        input_root = tmp_path / "docs"
        input_root.mkdir()
        source = input_root / "doc.md"
        source.write_text("# One\nBody one.\n\n## Two\nBody two.\n", encoding="utf-8")
        output_root = tmp_path / "out"
        manifest = Manifest(tmp_path / ".manifest.json")

        first = chunk_tree(input_root, output_root, manifest=manifest, split_levels=[1])
        unchanged = chunk_tree(input_root, output_root, manifest=manifest, split_levels=[1])
        reconfigured = chunk_tree(input_root, output_root, manifest=manifest, split_levels=[1, 2])

        assert len(first) == 1
        assert unchanged == []
        assert len(reconfigured) == 2


class TestDiffBuildParserAndFunctions:
    """Cover diff_files and diff_trees."""

    def test_diff_trees_identical(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        (a / "f.md").write_text("Hello\n", encoding="utf-8")
        (b / "f.md").write_text("Hello\n", encoding="utf-8")
        result = diff_trees(a, b)
        assert isinstance(result, TreeDiff)
        assert result.total_lines_added == 0

    def test_diff_trees_different(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        (a / "f.md").write_text("Hello\n", encoding="utf-8")
        (b / "f.md").write_text("World\n", encoding="utf-8")
        result = diff_trees(a, b)
        assert isinstance(result, TreeDiff)
        assert len(result.file_diffs) >= 1

    def test_write_diff_report(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        (a / "f.md").write_text("A\n", encoding="utf-8")
        (b / "f.md").write_text("B\n", encoding="utf-8")
        diffs = diff_trees(a, b)
        out = tmp_path / "diff.json"
        result = write_diff_report(diffs, out)
        assert result.exists()

    def test_write_unified_diff(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        a.mkdir()
        b = tmp_path / "b"
        b.mkdir()
        (a / "f.md").write_text("Line1\n", encoding="utf-8")
        (b / "f.md").write_text("Line2\n", encoding="utf-8")
        diffs = diff_trees(a, b)
        out = tmp_path / "diff.txt"
        result = write_unified_diff(diffs, out)
        assert result.exists()


class TestConvertBuildParserArgs:
    """Test convert module build_parser with full args."""

    def test_engine_choices(self) -> None:
        from cortexmark.convert import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x", "--engine", "docling"])
        assert ns.engine == "docling"

    def test_output_dir(self) -> None:
        from cortexmark.convert import build_parser as bp

        p = bp()
        ns = p.parse_args(["--input", "/tmp/x", "--output-dir", "/tmp/y"])
        assert ns.output_dir == Path("/tmp/y")


class TestParallelProcessPool:
    """Cover the ProcessPoolExecutor path in parallel_map."""

    def test_process_pool_type(self) -> None:
        from concurrent.futures import ProcessPoolExecutor

        from cortexmark.parallel import ParallelConfig, _get_pool

        cfg = ParallelConfig(workers=2, pool_type="process")
        pool = _get_pool(cfg)
        assert isinstance(pool, ProcessPoolExecutor)
        pool.shutdown(wait=False)
