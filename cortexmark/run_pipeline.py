"""Run all pipeline stages in order: convert → clean → chunk → render → analyze → validate."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from cortexmark.common import (
    Manifest,
    get_session_path_settings,
    get_source_id,
    load_config,
    resolve_configured_path,
    resolve_manifest_path,
    resolve_quality_dir,
    runtime_env_value,
    setup_logging,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full CortexMark.")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    parser.add_argument("--input", type=Path, help="Input directory or PDF file (overrides config source_id)")
    parser.add_argument("--session-name", type=str, help="Session name used to scope output sub-directories")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["convert", "clean", "chunk", "render", "analyze", "validate"],
        default=["convert", "clean", "chunk", "render"],
        help="Which stages to run (default: convert clean chunk render). "
        "Use 'analyze' for semantic-chunk / cross-ref / algorithm / notation analysis, "
        "and 'validate' for formula validation / scientific QA / citation context.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "docling", "markitdown", "dual"],
        help="Conversion engine override",
    )
    parser.add_argument("--no-manifest", action="store_true", help="Disable idempotency manifest")
    return parser


def _has_markdown_files(root: Path) -> bool:
    """Return True if *root* contains at least one .md file (recursively)."""
    return any(p.is_file() for p in root.rglob("*.md"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    log = setup_logging("pipeline", cfg)

    source_id = get_source_id(cfg)
    session_paths = get_session_path_settings(cfg, args.session_name) if args.session_name else None
    if session_paths:
        session_paths.ensure_directories()

    data_raw = session_paths.raw_data_dir if session_paths else resolve_configured_path(cfg, "data_raw", "data/raw")
    raw_md = (
        session_paths.raw_md_dir if session_paths else resolve_configured_path(cfg, "output_raw_md", "outputs/raw_md")
    )
    cleaned_md = (
        session_paths.cleaned_md_dir
        if session_paths
        else resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")
    )
    chunks_dir = (
        session_paths.chunks_dir if session_paths else resolve_configured_path(cfg, "output_chunks", "outputs/chunks")
    )
    semantic_chunks_dir = (
        session_paths.semantic_chunks_dir
        if session_paths
        else resolve_configured_path(cfg, "output_semantic_chunks", "outputs/semantic_chunks")
    )
    session_mode = session_paths is not None

    # --input overrides the default data_raw/source_id path
    input_is_file = False
    single_input: Path | None = None
    if args.input:
        custom_input = Path(args.input).resolve()
        input_is_file = custom_input.is_file()
        convert_input_root = custom_input.parent if input_is_file else custom_input
        if input_is_file:
            single_input = custom_input
        # Must match convert.derive_output_path's include_input_root_name=True layout
        # (output_root / input_root.name). Using the full path relative to data_raw
        # here would diverge from what the convert stage actually writes whenever
        # custom_input is nested two or more levels under data_raw, and the clean
        # stage would then look for markdown in a directory convert never wrote to.
        base_rel_label = convert_input_root.name
    else:
        convert_input_root = data_raw.resolve() if session_mode else (data_raw / source_id).resolve()
        base_rel_label = source_id
    rel_label = "" if session_mode else base_rel_label

    raw_stage_root = raw_md.resolve() if session_mode else (raw_md / rel_label).resolve()
    cleaned_stage_root = cleaned_md.resolve() if session_mode else (cleaned_md / rel_label).resolve()

    manifest = None
    idem_cfg = cfg.get("idempotency", {})
    if idem_cfg.get("enabled", True) and not args.no_manifest:
        manifest = Manifest(resolve_manifest_path(cfg, session_name=args.session_name))

    engine = (
        args.engine
        or runtime_env_value("PIPELINE_ENGINE", "CORTEXMARK_ENGINE", cfg=cfg)
        or cfg.get("convert", {}).get("engine", "dual")
    )
    stages = args.stages
    start = time.monotonic()

    try:
        if "convert" in stages:
            from cortexmark.convert import convert_tree

            log.info("── stage: convert [engine=%s] ──", engine)
            written = convert_tree(
                convert_input_root,
                raw_md.resolve(),
                engine=engine,
                cfg=cfg,
                manifest=manifest,
                include_input_root_name=not session_mode,
                only_files=(single_input,) if single_input else None,
            )
            log.info("converted %d file(s)", len(written))

        if "clean" in stages:
            from cortexmark.clean import clean_tree

            log.info("── stage: clean ──")
            written = clean_tree(
                raw_stage_root,
                cleaned_md.resolve(),
                cfg=cfg,
                manifest=manifest,
                include_input_root_name=not session_mode,
            )
            log.info("cleaned %d file(s)", len(written))

        if "chunk" in stages:
            from cortexmark.chunk import DEFAULT_SPLIT_LEVELS, chunk_tree

            log.info("── stage: chunk ──")
            chunk_cfg = cfg.get("chunk", {})
            split_levels = chunk_cfg.get("split_levels", DEFAULT_SPLIT_LEVELS)
            # Scoped by stage: chunk and analyze/semantic-chunk both read the same
            # cleaned .md files and must not overwrite each other's manifest record.
            written = chunk_tree(
                cleaned_stage_root,
                chunks_dir.resolve(),
                manifest=manifest.scoped("chunk") if manifest else None,
                split_levels=split_levels,
                include_input_root_name=not session_mode,
            )
            log.info("wrote %d chunk(s)", len(written))

        if "render" in stages:
            if args.input:
                log.info("── stage: render_templates ── skipped (custom --input)")
            else:
                from cortexmark.render_templates import (
                    parse_section_entries,
                    read_text,
                    render_meta_templates,
                    render_section_templates,
                    resolve_outline_path,
                )

                log.info("── stage: render_templates ──")
                source_root = data_raw.resolve() if session_mode else (data_raw / rel_label).resolve()
                raw_root = raw_stage_root
                cleaned_root = cleaned_stage_root
                outline_text = ""
                section_entries: dict[int, dict[str, list[str] | str]] = {}
                outline_path = resolve_outline_path(raw_root, cfg=cfg)
                if outline_path:
                    outline_text = read_text(outline_path)
                    section_entries = parse_section_entries(outline_text)
                else:
                    log.warning(
                        "No outline file found under %s; render stage will use folder/content heuristics.",
                        raw_root,
                    )
                written_list: list[Path] = []
                written_list.extend(render_meta_templates(source_root, outline_text, section_entries))
                written_list.extend(
                    render_section_templates(source_root, raw_root, cleaned_root, section_entries, cfg=cfg)
                )
                log.info("populated %d template(s)", len(written_list))

        if "analyze" in stages:
            analysis_input = cleaned_stage_root
            if session_mode and not _has_markdown_files(analysis_input):
                fallback_input = (
                    resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md") / base_rel_label
                ).resolve()
                if _has_markdown_files(fallback_input):
                    log.info(
                        "analyze input empty in session scope (%s); falling back to base cleaned path (%s)",
                        analysis_input,
                        fallback_input,
                    )
                    analysis_input = fallback_input
            quality_dir = resolve_quality_dir(cfg, session_name=args.session_name)

            from cortexmark.semantic_chunk import chunk_tree as semantic_chunk_tree

            log.info("── stage: analyze / semantic-chunk ──")
            # Scoped by stage: see the chunk stage above for why this must not share
            # the bare-source-path manifest key with chunk_tree.
            sem_written = semantic_chunk_tree(
                analysis_input,
                semantic_chunks_dir.resolve(),
                manifest=manifest.scoped("semantic_chunk") if manifest else None,
                include_input_root_name=not session_mode,
            )
            log.info("wrote %d semantic chunk(s)", len(sem_written))

            from cortexmark.cross_ref import analyze_tree as crossref_tree
            from cortexmark.cross_ref import write_report as crossref_write

            log.info("── stage: analyze / cross-ref ──")
            cr_report = crossref_tree(analysis_input)
            crossref_write(cr_report, quality_dir / "crossref_report.json")
            log.info(
                "cross-ref: %d definitions, %.0f%% resolved",
                len(cr_report.definitions),
                cr_report.resolution_rate * 100,
            )

            from cortexmark.algorithm_extract import build_summary as algo_summary
            from cortexmark.algorithm_extract import extract_from_tree as algo_tree
            from cortexmark.algorithm_extract import write_report as algo_write

            log.info("── stage: analyze / algorithm-extract ──")
            al_algos = algo_tree(analysis_input)
            al_summ = algo_summary(al_algos)
            algo_write(al_algos, quality_dir / "algorithm_report.json")
            log.info(
                "algorithms: %d found",
                al_summ.get("total_algorithms", 0),
            )

            from cortexmark.notation_glossary import build_summary as notation_summary
            from cortexmark.notation_glossary import extract_from_tree as notation_tree
            from cortexmark.notation_glossary import write_report as notation_write

            log.info("── stage: analyze / notation-glossary ──")
            nt_glossary = notation_tree(analysis_input)
            nt_summ = notation_summary(nt_glossary)
            notation_write(nt_glossary, quality_dir / "notation_report.json")
            log.info(
                "notation: %d unique symbols, %d entries",
                nt_summ.get("unique_symbols", 0),
                nt_summ.get("total_entries", 0),
            )

        if "validate" in stages:
            validate_input = cleaned_stage_root
            if session_mode and not _has_markdown_files(validate_input):
                fallback_input = (
                    resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md") / base_rel_label
                ).resolve()
                if _has_markdown_files(fallback_input):
                    log.info(
                        "validate input empty in session scope (%s); falling back to base cleaned path (%s)",
                        validate_input,
                        fallback_input,
                    )
                    validate_input = fallback_input
            quality_dir = resolve_quality_dir(cfg, session_name=args.session_name)

            from cortexmark.formula_validate import build_summary as fv_summary
            from cortexmark.formula_validate import validate_tree as fv_tree
            from cortexmark.formula_validate import write_report as fv_write

            log.info("── stage: validate / formula-validate ──")
            fv_results = fv_tree(validate_input)
            fv_summ = fv_summary(fv_results)
            fv_write(fv_results, fv_summ, quality_dir / "formula_validation.json")
            log.info(
                "formulas: %d total (%d valid, %d errors)",
                fv_summ.total_formulas,
                fv_summ.total_valid,
                fv_summ.total_errors,
            )

            from cortexmark.scientific_qa import analyze_tree as sciqa_tree
            from cortexmark.scientific_qa import build_summary as sciqa_summary
            from cortexmark.scientific_qa import write_report as sciqa_write

            log.info("── stage: validate / scientific-qa ──")
            sq_reports = sciqa_tree(validate_input)
            sq_summ = sciqa_summary(sq_reports)
            sciqa_write(sq_reports, sq_summ, quality_dir / "scientific_qa.json")
            log.info(
                "scientific QA: %d issues (%d errors, %d warnings)",
                sq_summ.total_issues,
                sq_summ.total_errors,
                sq_summ.total_warnings,
            )

            from cortexmark.citation_context import analyze_tree as citctx_tree
            from cortexmark.citation_context import build_summary as citctx_summary
            from cortexmark.citation_context import write_report as citctx_write

            log.info("── stage: validate / citation-context ──")
            cc_reports = citctx_tree(validate_input)
            cc_summ = citctx_summary(cc_reports)
            citctx_write(cc_reports, cc_summ, quality_dir / "citation_context.json")
            log.info(
                "citations: %d total (%d self, %d co-citation pairs)",
                cc_summ.total_citations,
                cc_summ.total_self_citations,
                cc_summ.total_co_citation_pairs,
            )

        if manifest:
            manifest.save()

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1
    except ImportError as exc:
        log.error("missing dependency: %s", exc)
        return 1
    except (OSError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    except ValueError as exc:
        log.error("invalid input: %s", exc)
        return 1

    elapsed = time.monotonic() - start
    log.info("pipeline finished in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
