"""Top-level CortexMark command line interface."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cortexmark.common import DEFAULT_SOURCE_ID, setup_logging

KNOWN_SUBCOMMANDS = {"run", "pipeline", "convert", "clean", "chunk", "render", "analyze", "validate"}
RUN_STAGES = ("convert", "clean", "chunk", "analyze", "validate")
log = logging.getLogger("cortexmark.run")


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortexmark run", description="Convert PDFs into self-contained CortexMark bundles."
    )
    parser.add_argument("input", type=Path, help="A PDF file or a directory containing PDFs")
    parser.add_argument("--out", type=Path, default=Path("cortexmark-output"), help="Output directory")
    parser.add_argument(
        "--engine",
        choices=("auto", "docling", "markitdown", "dual"),
        default="auto",
        help="Conversion engine (default: auto)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=RUN_STAGES,
        default=["convert", "clean", "chunk"],
        help="Stages to run (default: convert clean chunk)",
    )
    parser.add_argument("--rag", action="store_true", help="Also emit RAG-ready JSONL")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary")
    parser.add_argument("--ocr", choices=("auto", "on", "off"), default="auto", help="Local OCR mode")
    return parser


def _default_config(*, engine: str, ocr_mode: str, project_root: Path) -> dict[str, Any]:
    """Build the config subset needed by the one-command path without reading a file."""
    return {
        "source_id": DEFAULT_SOURCE_ID,
        "__project_root__": str(project_root),
        "__config_dir__": str(project_root),
        "convert": {
            "engine": engine,
            "ocr": ocr_mode,
            "docling": {
                "do_table_structure": True,
                "table_structure_mode": "accurate",
                # Formula/code enrichment run a model per block and are ~8x slower on CPU
                # (>8 min/paper) for a marginal unicode->LaTeX gain; keep them opt-in (config).
                "do_formula_enrichment": False,
                "do_code_enrichment": False,
                "do_ocr": ocr_mode == "on",
            },
        },
        "clean": {},
        "chunk": {"split_levels": [1, 2]},
        "idempotency": {"enabled": False},
        "logging": {"level": "INFO"},
    }


def _pdf_inputs(input_path: Path) -> list[Path]:
    resolved = input_path.resolve()
    if resolved.is_file():
        if resolved.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {resolved}")
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"Input not found: {resolved}")
    pdfs = sorted(path for path in resolved.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under: {resolved}")
    stem_counts = Counter(path.stem for path in pdfs)
    duplicate_stems = sorted(stem for stem, count in stem_counts.items() if count > 1)
    if duplicate_stems:
        raise ValueError(f"PDF names must have unique stems; duplicates: {', '.join(duplicate_stems)}")
    return pdfs


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_document(
    pdf_path: Path,
    output_root: Path,
    *,
    engine: str,
    cfg: dict[str, Any],
    stages: set[str],
    emit_rag: bool,
) -> dict[str, Any]:
    from cortexmark.chunk import chunk_file
    from cortexmark.clean import clean_file
    from cortexmark.convert import convert_pdf, markdown_uses_ocr

    def _guarded(label: str, action: Callable[[], object]) -> None:
        """Run an optional enrichment/analysis step; degrade to a logged warning on failure.

        Isolation keeps a single fallible artifact (semantic chunks, a quality report)
        from discarding an otherwise-complete bundle. The warning carries the exception
        class so a real subsystem regression is distinguishable from an empty-but-valid
        result.
        """
        try:
            action()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            log.warning("%s failed for %s: %s: %s", label, pdf_path, type(exc).__name__, exc)

    bundle_dir = output_root / pdf_path.stem
    bundle_dir.mkdir(parents=True, exist_ok=True)
    raw_path = bundle_dir / "raw.md"
    cleaned_path = bundle_dir / f"{pdf_path.stem}.md"
    chunks_dir = bundle_dir / "chunks"

    if "convert" in stages:
        convert_pdf(pdf_path, raw_path, engine=engine, cfg=cfg)
    elif not raw_path.exists():
        raise FileNotFoundError(f"Convert stage omitted and prior output is missing: {raw_path}")

    raw_markdown = raw_path.read_text(encoding="utf-8")
    if "clean" in stages:
        clean_file(raw_path, cleaned_path, cfg=cfg)
    elif not cleaned_path.exists():
        raise FileNotFoundError(f"Clean stage omitted and prior output is missing: {cleaned_path}")

    if "chunk" in stages:
        chunk_paths = chunk_file(cleaned_path, chunks_dir, split_levels=cfg["chunk"]["split_levels"])
    else:
        chunk_paths = sorted(chunks_dir.glob("chunk_*.md"))

    if "analyze" in stages:
        from cortexmark.cross_ref import analyze_file, write_report

        def _write_semantic_chunks() -> None:
            from cortexmark.semantic_chunk import chunk_file as semantic_chunk_file

            semantic_chunk_file(cleaned_path, bundle_dir / "semantic_chunks")

        def _write_crossref_report() -> None:
            write_report(analyze_file(cleaned_path), bundle_dir / "crossref_report.json")

        def _write_algorithm_report() -> None:
            from cortexmark.algorithm_extract import extract_from_file
            from cortexmark.algorithm_extract import write_report as algorithm_write

            algorithm_write(extract_from_file(cleaned_path), bundle_dir / "algorithm_report.json")

        def _write_notation_report() -> None:
            from cortexmark.notation_glossary import extract_from_file
            from cortexmark.notation_glossary import write_report as notation_write

            notation_write(extract_from_file(cleaned_path), bundle_dir / "notation_report.json")

        # Same per-document isolation as the enrichment blocks below: semantic_chunk raises
        # (unlike the plain chunker) on empty/structureless text, and it must not drop the doc.
        _guarded("semantic chunking", _write_semantic_chunks)
        _guarded("cross-reference analysis", _write_crossref_report)
        _guarded("algorithm extraction", _write_algorithm_report)
        _guarded("notation glossary", _write_notation_report)

    qa_note: str | None = None
    if "validate" in stages:

        def _write_qa_report() -> None:
            nonlocal qa_note
            from cortexmark.qa_pipeline import build_summary, qa_file, write_report

            qa_report = qa_file(cleaned_path)
            qa_summary = build_summary([qa_report])
            write_report([qa_report], qa_summary, bundle_dir / "qa.json")
            qa_note = f"{qa_summary.overall_badge}; {qa_summary.total_issues} issue(s)"

        def _write_formula_validation() -> None:
            from cortexmark.formula_validate import build_summary as formula_summary
            from cortexmark.formula_validate import validate_file
            from cortexmark.formula_validate import write_report as formula_write

            result = validate_file(cleaned_path)
            formula_write([result], formula_summary([result]), bundle_dir / "formula_validation.json")

        def _write_scientific_qa() -> None:
            from cortexmark.scientific_qa import analyze_file as sciqa_file
            from cortexmark.scientific_qa import build_summary as sciqa_summary
            from cortexmark.scientific_qa import write_report as sciqa_write

            report = sciqa_file(cleaned_path)
            sciqa_write([report], sciqa_summary([report]), bundle_dir / "scientific_qa.json")

        def _write_citation_context() -> None:
            from cortexmark.citation_context import analyze_file as citation_file
            from cortexmark.citation_context import build_summary as citation_summary
            from cortexmark.citation_context import write_report as citation_write

            report = citation_file(cleaned_path)
            citation_write([report], citation_summary([report]), bundle_dir / "citation_context.json")

        _guarded("scientific QA (qa.json)", _write_qa_report)
        _guarded("formula validation", _write_formula_validation)
        _guarded("scientific QA report", _write_scientific_qa)
        _guarded("citation context", _write_citation_context)

    if emit_rag:
        from cortexmark.rag_export import export_tree, write_jsonl

        if not chunk_paths:
            raise FileNotFoundError(f"RAG export requires chunks under: {chunks_dir}")
        write_jsonl(export_tree(chunks_dir), bundle_dir / "rag.jsonl")

    cleaned_markdown = cleaned_path.read_text(encoding="utf-8")
    page_types: dict[str, int] = {}
    n_ir_blocks = 0
    page_count: int | None = None
    try:
        from cortexmark.page_ir import build_document_ir, write_ir

        ocr_mode = str(cfg.get("convert", {}).get("ocr", "auto"))
        document_ir = build_document_ir(pdf_path, ocr_mode=ocr_mode)
        write_ir(document_ir, bundle_dir / "ir.json")
        page_types = dict(Counter(page["page_type"] for page in document_ir["pages"]))
        n_ir_blocks = len(document_ir["blocks"])
        # Reuse the page list already parsed here instead of re-opening the PDF a fourth
        # time just to count pages (len == PDF page count == len(pdf_embedded_text_chars)).
        page_count = len(document_ir["pages"])
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        log.warning("page IR generation failed for %s: %s: %s", pdf_path, type(exc).__name__, exc)

    consensus_confidence = 0.0
    low_agreement_pages = 0
    try:
        from cortexmark.consensus import document_consensus

        # cfg carries the resolved engine (run() sets cfg["convert"]["engine"]); consensus
        # honors it so a docling-only run is not forced through a second MarkItDown pass.
        consensus_report = document_consensus(pdf_path, cfg=cfg)
        _write_summary(bundle_dir / "consensus.json", consensus_report)
        consensus_confidence = float(consensus_report["doc_consensus_confidence"])
        low_agreement_pages = sum(bool(page["low_agreement"]) for page in consensus_report["per_page"])
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        log.warning("consensus generation failed for %s: %s: %s", pdf_path, type(exc).__name__, exc)

    metadata_fields: dict[str, Any] | None = None
    topic_fields: dict[str, Any] | None = None
    try:
        from cortexmark.metadata import extract_metadata

        extracted_metadata = extract_metadata(cleaned_markdown, source_file=str(pdf_path))
        metadata_fields = {
            "title": extracted_metadata.title,
            "authors": extracted_metadata.authors,
            "journal": extracted_metadata.journal,
            "doi": extracted_metadata.doi,
            "keywords": extracted_metadata.keywords,
        }
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        log.warning("metadata extraction for grounding failed for %s: %s: %s", pdf_path, type(exc).__name__, exc)

    try:
        from cortexmark.topics import classify_text

        extracted_topics = classify_text(cleaned_markdown, source_file=str(pdf_path))
        topic_fields = {"primary_topic": extracted_topics.primary_topic}
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        log.warning("topic classification for grounding failed for %s: %s: %s", pdf_path, type(exc).__name__, exc)

    n_generated = 0
    n_ungrounded = 0
    try:
        from cortexmark.grounding import build_grounding_report

        grounding_report = build_grounding_report(
            cleaned_markdown,
            metadata=metadata_fields,
            topics=topic_fields,
        )
        _write_summary(bundle_dir / "grounding.json", grounding_report)
        n_generated = int(grounding_report["n_generated"])
        n_ungrounded = int(grounding_report["n_ungrounded"])
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        log.warning("grounding generation failed for %s: %s: %s", pdf_path, type(exc).__name__, exc)

    if page_count is None:
        # page IR did not yield a count; take it straight from the opened document. len(pdf)
        # avoids the fragile full-text re-extraction (get_text_range) that could raise on a
        # broken-CMap/encrypted page and discard an otherwise-complete bundle, while a
        # genuinely unopenable/corrupt PDF still raises here and is dropped by run().
        from cortexmark.pdfutil import open_pdf

        with open_pdf(pdf_path) as pdf:
            page_count = len(pdf)

    summary = {
        "source": str(pdf_path),
        "engine": engine,
        "ocr_used": markdown_uses_ocr(raw_markdown),
        "pages": page_count if page_count is not None else 0,
        "n_chunks": len(chunk_paths),
        "char_count": len(cleaned_markdown),
        "page_types": page_types,
        "n_ir_blocks": n_ir_blocks,
        "consensus_confidence": consensus_confidence,
        "low_agreement_pages": low_agreement_pages,
        "n_generated": n_generated,
        "n_ungrounded": n_ungrounded,
        "qa_note": qa_note,
        "output_dir": str(bundle_dir),
    }
    _write_summary(bundle_dir / "summary.json", summary)
    return summary


def run(argv: Sequence[str]) -> int:
    args = build_run_parser().parse_args(list(argv))
    output_root = args.out.resolve()
    cfg = _default_config(engine=args.engine, ocr_mode=args.ocr, project_root=Path.cwd().resolve())
    log = setup_logging("cortexmark.run", cfg)

    try:
        from cortexmark.convert import _resolve_engine

        engine = _resolve_engine(args.engine)
        pdf_inputs = list(_pdf_inputs(args.input))
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    # Expose the RESOLVED engine (e.g. auto -> docling) through cfg so consensus and the
    # other cfg-driven stages act on the engine actually used, not the raw --engine token.
    cfg["convert"]["engine"] = engine

    # Isolate per-document failures: one unreadable/corrupt PDF must not abort the batch.
    summaries: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for pdf_path in pdf_inputs:
        try:
            summaries.append(
                _run_document(
                    pdf_path,
                    output_root,
                    engine=engine,
                    cfg=cfg,
                    stages=set(args.stages),
                    emit_rag=args.rag,
                )
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            failed.append({"source": str(pdf_path), "error": f"{type(exc).__name__}: {exc}"})
            log.warning("skipping %s: %s: %s", pdf_path, type(exc).__name__, exc)

    if not summaries and failed:
        log.error("no documents were converted (%d failed)", len(failed))
        return 1

    if args.json:
        # Surface per-document failures in-band: a --json consumer must be able to tell a
        # full success from a partial one without scraping stderr. Exit stays 0 on partial
        # success by contract (README: a failed document is skipped without aborting the run).
        print(
            json.dumps(
                {
                    "documents": summaries,
                    "failures": len(failed),
                    "failed": failed,
                    "output_dir": str(output_root),
                },
                ensure_ascii=False,
            )
        )
    else:
        for summary in summaries:
            ocr_label = "yes" if summary["ocr_used"] else "no"
            print(
                f"{Path(summary['source']).name}: engine={summary['engine']} OCR={ocr_label} "
                f"chunks={summary['n_chunks']} output={summary['output_dir']}"
            )
    return 0


@contextmanager
def _module_argv(command: str, argv: Sequence[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = [f"cortexmark {command}", *argv]
    try:
        yield
    finally:
        sys.argv = previous


def _delegate_stage(command: str, argv: Sequence[str]) -> int:
    module_name = {"render": "render_templates"}.get(command, command)
    if command in {"analyze", "validate"}:
        from cortexmark import run_pipeline

        return run_pipeline.main(["--stages", command, *argv])
    module = importlib.import_module(f"cortexmark.{module_name}")
    with _module_argv(command, argv):
        return int(module.main())


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in KNOWN_SUBCOMMANDS:
        from cortexmark import run_pipeline

        return run_pipeline.main(arguments)
    command, remaining = arguments[0], arguments[1:]
    if command == "run":
        return run(remaining)
    if command == "pipeline":
        from cortexmark import run_pipeline

        return run_pipeline.main(remaining)
    return _delegate_stage(command, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
