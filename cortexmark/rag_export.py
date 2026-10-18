"""RAG-ready export: convert chunks to embedding-friendly JSON Lines.

Each output record contains:
  - id: unique chunk identifier
  - source: original file path
  - title: chunk heading
  - text: chunk body text
  - metadata: chapter, section, token estimate, entity_type, formulas,
    cross_refs, entity_label
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import get_path_settings, load_config, resolve_configured_path, setup_logging
from cortexmark.cross_ref import extract_definitions, extract_mentions, resolve_references
from cortexmark.notation_glossary import extract_all, glossary_to_scientific_objects
from cortexmark.scientific_ir import OBJECT_EQUATION
from cortexmark.semantic_chunk import (
    ENTITY_NARRATIVE,
    build_scientific_object_links,
    chunks_to_scientific_objects,
    extract_cross_refs,
    extract_formulas,
    parse_semantic_chunks,
)

# ── Data structures ──────────────────────────────────────────────────────────

WHITESPACE_RUN_RE = re.compile(r"\s+")
CHUNK_BASENAME_RE = re.compile(r"^chunk_(\d+)_")


@dataclass
class RAGRecord:
    """Single embedding-ready record."""

    id: str
    source: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def make_chunk_id(source: str, title: str, index: int) -> str:
    """Build a deterministic chunk ID from a stable source key, title, and index."""
    raw = f"{source}::{title}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def scientific_source_scope(file_path: Path) -> str:
    """Return a path-independent, document-level scope for chunk files.

    Uses NAMES, never absolute paths: chunk.chunk_tree/chunk_file write standard
    chunk_NNN_slug.md files into a directory named after the source document's
    stem, so that directory name is a stable, portable document identifier
    (identical for the same document across machines/checkouts, distinct across
    documents). Non-standard filenames fall back to their own stem, which keeps
    the scope tied to the file itself rather than to whatever absolute location
    it happens to live under.
    """
    if CHUNK_BASENAME_RE.match(file_path.name):
        return file_path.parent.name
    return file_path.stem


def chunk_file_index(file_path: Path) -> int:
    """Extract the per-document ordinal from a standard chunk_NNN_slug.md filename.

    Sibling chunk files in one document share the same scientific_source_scope()
    (their common parent directory name), so this ordinal is what disambiguates
    them when building a chunk id. Falls back to 0 for non-standard filenames,
    where scientific_source_scope() already returns a file-unique stem.
    """
    match = CHUNK_BASENAME_RE.match(file_path.name)
    return int(match.group(1)) if match else 0


def _resolve_local_cross_ref_links(report: Any, scientific_objects: list[Any]) -> None:
    """Resolve local theorem-like references from semantic objects when raw parsing is ambiguous."""
    label_to_object_ids: dict[str, list[str]] = {}
    for obj in scientific_objects:
        if obj.object_type == OBJECT_EQUATION or not obj.label:
            continue
        label_to_object_ids.setdefault(obj.label, []).append(obj.object_id)

    for link in report.links:
        if link.relation != "references" or link.status not in {"ambiguous", "unresolved"}:
            continue
        candidates = label_to_object_ids.get(link.source_label, [])
        if len(candidates) != 1:
            continue
        link.status = "resolved"
        link.target_object_id = candidates[0]
        link.target_label = link.source_label
        link.metadata = dict(link.metadata)
        link.metadata["resolved_via"] = "semantic_objects"
        if link.source_label in report.unresolved:
            report.unresolved = [label for label in report.unresolved if label != link.source_label]
        if link.source_label not in report.resolved:
            report.resolved.append(link.source_label)


def parse_chunk_file(file_path: Path) -> RAGRecord:
    """Parse a single chunk Markdown file into a RAGRecord.

    Detects semantic entity types (theorem, proof, definition, etc.) and
    extracts formulas and cross-references into metadata.
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.strip().split("\n")

    title = ""
    chapter = ""
    section = ""
    body_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not chapter:
            chapter = stripped.lstrip("# ").strip()
            continue
        if stripped.startswith("## ") and not section:
            section = stripped.lstrip("# ").strip()
            continue
        body_lines.append(stripped)

    title = section or chapter or file_path.stem
    body_text = "\n".join(line for line in body_lines if line).strip()

    # Semantic enrichment: detect entity type from body content
    entity_type = ENTITY_NARRATIVE
    entity_label: str | None = None
    sem_chunks = parse_semantic_chunks(body_text, split_on_headings=False)
    if sem_chunks:
        primary = sem_chunks[0]
        entity_type = primary.entity_type
        entity_label = primary.entity_label
    scientific_objects = chunks_to_scientific_objects(sem_chunks, source=scientific_source_scope(file_path))
    object_links = build_scientific_object_links(scientific_objects)
    notation_glossary = extract_all(body_text, source_file=str(file_path), include_conventions=True)
    notation_objects = glossary_to_scientific_objects(notation_glossary)
    cross_ref_report = resolve_references(
        extract_definitions(body_text, source_file=str(file_path)),
        extract_mentions(body_text, source_file=str(file_path)),
    )
    _resolve_local_cross_ref_links(cross_ref_report, scientific_objects)

    formulas = extract_formulas(body_text)
    cross_refs = extract_cross_refs(body_text)
    primary_parent_label = sem_chunks[0].parent_label if sem_chunks else None
    primary_object = next((obj for obj in scientific_objects if obj.object_type != OBJECT_EQUATION), None)
    primary_parent_object_id = primary_object.parent_object_id if primary_object else ""

    return RAGRecord(
        id=make_chunk_id(scientific_source_scope(file_path), title, chunk_file_index(file_path)),
        source=str(file_path),
        title=title,
        text=body_text,
        metadata={
            "chapter": chapter,
            "section": section,
            "token_estimate": estimate_tokens(body_text),
            "entity_type": entity_type,
            "entity_label": entity_label,
            "entity_kind": sem_chunks[0].entity_kind if sem_chunks else ENTITY_NARRATIVE,
            "entity_name": sem_chunks[0].entity_name if sem_chunks else None,
            "parent_label": primary_parent_label,
            "parent_object_id": primary_parent_object_id,
            "formulas": formulas,
            "equations": [
                {
                    "object_id": obj.object_id,
                    "formula": obj.metadata.get("formula", obj.text),
                    "source_kind": obj.metadata.get("source_kind", ""),
                    "parent_object_id": obj.parent_object_id,
                }
                for obj in scientific_objects
                if obj.object_type == OBJECT_EQUATION
            ],
            "cross_refs": cross_refs,
            "cross_ref_links": [asdict(link) for link in cross_ref_report.links],
            "scientific_object_ids": [obj.object_id for obj in scientific_objects],
            "scientific_objects": [asdict(obj) for obj in scientific_objects],
            "object_links": [asdict(link) for link in object_links],
            "notation_symbols": [entry.symbol for entry in notation_glossary.entries],
            "notation_sources": [entry.source for entry in notation_glossary.entries],
            "notation_object_ids": [obj.object_id for obj in notation_objects],
            "notation_entries": [
                {
                    "symbol": entry.symbol,
                    "definition": entry.definition,
                    "source": entry.source,
                    "object_id": entry.object_id,
                }
                for entry in notation_glossary.entries
            ],
        },
    )


def normalize_text_for_embedding(text: str) -> str:
    """Clean text for embedding: collapse whitespace, strip control chars."""
    cleaned = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return WHITESPACE_RUN_RE.sub(" ", cleaned).strip()


# ── Export functions ─────────────────────────────────────────────────────────


def export_file(file_path: Path, *, normalize: bool = True) -> RAGRecord:
    """Export a single chunk file as a RAGRecord."""
    if not file_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {file_path}")
    record = parse_chunk_file(file_path)
    if normalize:
        record.text = normalize_text_for_embedding(record.text)
        record.metadata["token_estimate"] = estimate_tokens(record.text)
    return record


def export_tree(input_root: Path, *, normalize: bool = True) -> list[RAGRecord]:
    """Export all chunk files under a directory tree."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No chunk files found under: {input_root}")
    return [export_file(f, normalize=normalize) for f in md_files]


def write_jsonl(records: list[RAGRecord], output_path: Path) -> Path:
    """Write records as JSON Lines (one JSON object per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            line = json.dumps(asdict(record), ensure_ascii=False)
            fh.write(line + "\n")
    return output_path


def write_json_array(records: list[RAGRecord], output_path: Path) -> Path:
    """Write records as a JSON array (for smaller datasets)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in records]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def build_summary(records: list[RAGRecord]) -> dict[str, Any]:
    """Build a summary of the RAG export including entity type distribution."""
    total_tokens = sum(r.metadata.get("token_estimate", 0) for r in records)
    entity_counts: dict[str, int] = {}
    for r in records:
        etype = r.metadata.get("entity_type", ENTITY_NARRATIVE)
        entity_counts[etype] = entity_counts.get(etype, 0) + 1
    total_formulas = sum(len(r.metadata.get("formulas", [])) for r in records)
    total_cross_ref_links = sum(len(r.metadata.get("cross_ref_links", [])) for r in records)
    total_notation_symbols = sum(len(r.metadata.get("notation_symbols", [])) for r in records)
    return {
        "total_records": len(records),
        "total_tokens_estimate": total_tokens,
        "avg_tokens_per_record": round(total_tokens / max(len(records), 1)),
        "sources": len({r.source for r in records}),
        "entity_types": entity_counts,
        "total_formulas": total_formulas,
        "total_cross_ref_links": total_cross_ref_links,
        "total_notation_symbols": total_notation_symbols,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export chunks as RAG-ready JSON Lines for embedding pipelines.")
    parser.add_argument("--input", type=Path, help="Chunk file or directory to export")
    parser.add_argument("--output", type=Path, help="Output path for JSONL file")
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format (default: jsonl)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip text normalization for embedding",
    )
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("rag_export", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_chunks", "outputs/chunks")).resolve()
    default_ext = ".jsonl" if args.format == "jsonl" else ".json"
    output_path = (args.output or (get_path_settings(cfg).outputs_dir / "rag" / f"chunks{default_ext}")).resolve()

    normalize = not args.no_normalize

    try:
        if input_path.is_dir():
            records = export_tree(input_path, normalize=normalize)
        else:
            records = [export_file(input_path, normalize=normalize)]

        if args.format == "jsonl":
            written = write_jsonl(records, output_path)
        else:
            written = write_json_array(records, output_path)

        summary = build_summary(records)
        log.info(
            "exported %d record(s), ~%d tokens → %s",
            summary["total_records"],
            summary["total_tokens_estimate"],
            written,
        )

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
