"""Citation graph extraction and analysis from Markdown documents.

Scans converted Markdown for citation patterns and builds:
  - A list of extracted references
  - A citation graph (who cites whom)
  - Canonical citation IR objects (mentions, references, links)
  - Duplicate / missing / phantom reference audits
  - JSON and DOT (Graphviz) output

Works on already-converted Markdown (post-convert stage).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from cortexmark.citation_ir import (
    CitationAudit,
    CitationLink,
    CitationMention,
    DuplicateReferenceCluster,
    Reference,
    build_author_year_key,
    normalize_title,
    parse_author_year_targets,
    parse_numeric_targets,
    slugify_identifier,
)
from cortexmark.common import load_config, resolve_configured_path, resolve_quality_report_path, setup_logging

# ── Patterns ─────────────────────────────────────────────────────────────────

# Inline citation patterns: (Author, Year), (Author et al., Year), [1], [1,2,3]
PAREN_CITE_RE = re.compile(
    r"\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][a-z]+))?,?\s*\d{4}[a-z]?"
    r"(?:;\s*[A-Z][a-z]+(?:\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][a-z]+))?,?\s*\d{4}[a-z]?)*)\)",
)
BRACKET_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")

# Reference list patterns
REF_NUMBERED_RE = re.compile(r"^\[(\d+)\]\s+(.+)")
REF_AUTHOR_YEAR_RE = re.compile(
    r"^([A-Z][a-z]+(?:,?\s+(?:and|&)\s+[A-Z][a-z]+|(?:\s+et\s+al\.?))?)\s*\((\d{4}[a-z]?)\)[.,]?\s*(.+)",
)

# DOI in reference
DOI_IN_REF_RE = re.compile(r"(10\.\d{4,9}/[^\s,;\"')\]]+)")

_REF_HEADINGS = frozenset(
    {
        "# references",
        "## references",
        "### references",
        "# bibliography",
        "## bibliography",
        "**references**",
    }
)


# ── Data structures ──────────────────────────────────────────────────────────


Citation = CitationMention
CitationEdge = CitationLink


@dataclass
class CitationGraph:
    """Complete citation analysis for a document or collection."""

    citations: list[CitationMention] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    edges: list[CitationLink] = field(default_factory=list)
    top_cited: list[tuple[str, int]] = field(default_factory=list)
    audit: CitationAudit = field(default_factory=CitationAudit)
    inventory: dict[str, Any] = field(default_factory=dict)

    @property
    def mentions(self) -> list[CitationMention]:
        """Backward-compatible alias for canonical citation mentions."""
        return self.citations

    @property
    def links(self) -> list[CitationLink]:
        """Backward-compatible alias for canonical citation links."""
        return self.edges


# ── Extraction engine ────────────────────────────────────────────────────────


def _build_mention_id(source_file: str, line_number: int, raw_text: str, ordinal: int) -> str:
    source_part = slugify_identifier(source_file or "document", default="document")
    raw_part = slugify_identifier(raw_text, default="cite")
    return f"{source_part}:L{line_number}:C{ordinal}:{raw_part}"


def _stable_source_label(file_path: Path, *, root: Path | None = None) -> str:
    """Return a source label that avoids machine-specific absolute paths when possible."""
    if root is not None:
        try:
            return file_path.relative_to(root).as_posix()
        except ValueError:
            pass
    try:
        return file_path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return file_path.name


def _populate_target_hints(citation: CitationMention) -> CitationMention:
    if citation.target_hints:
        return citation

    hints: list[str] = []
    if citation.cite_type == "numeric":
        hints = parse_numeric_targets(citation.raw_text)
    elif citation.cite_type == "author-year":
        hints = parse_author_year_targets(citation.raw_text)
    return replace(citation, target_hints=hints)


def _prepare_citations(citations: list[CitationMention]) -> list[CitationMention]:
    prepared: list[CitationMention] = []
    for ordinal, citation in enumerate(citations, start=1):
        current = replace(citation)
        if not current.mention_id:
            current = replace(
                current,
                mention_id=_build_mention_id(current.source_file, current.line_number, current.raw_text, ordinal),
            )
        current = _populate_target_hints(current)
        prepared.append(current)
    return prepared


def _references_start(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line.strip().lower() in _REF_HEADINGS:
            return index + 1
    return len(lines)


def extract_inline_citations(text: str, source_file: str = "") -> list[CitationMention]:
    """Extract inline citations from text."""
    citations: list[CitationMention] = []
    lines = text.split("\n")
    ref_start = _references_start(lines)

    for line_num, line in enumerate(lines, start=1):
        if line_num - 1 >= ref_start:
            break
        # Author-year pattern: (Smith, 2020)
        for match in PAREN_CITE_RE.finditer(line):
            citation = CitationMention(
                raw_text=match.group(1).strip(),
                surface_text=match.group(0),
                source_file=source_file,
                line_number=line_num,
                cite_type="author-year",
                mention_id=_build_mention_id(source_file, line_num, match.group(1).strip(), match.start() + 1),
            )
            citations.append(_populate_target_hints(citation))

        # Numeric pattern: [1], [1,2,3]
        for match in BRACKET_CITE_RE.finditer(line):
            # Skip if this looks like a Markdown link rather than a citation:
            # reference-style "[text][1]" (bracket preceded by "]") or inline
            # "[1](url)" (bracket immediately followed by "(").
            start, end = match.start(), match.end()
            if start > 0 and line[start - 1] == "]":
                continue
            if end < len(line) and line[end] == "(":
                continue
            citation = CitationMention(
                raw_text=match.group(1).strip(),
                surface_text=match.group(0),
                source_file=source_file,
                line_number=line_num,
                cite_type="numeric",
                mention_id=_build_mention_id(source_file, line_num, match.group(1).strip(), match.start() + 1),
            )
            citations.append(_populate_target_hints(citation))

    return citations


def extract_references(text: str) -> list[Reference]:
    """Extract reference entries from the bibliography section."""
    refs: list[Reference] = []
    lines = text.split("\n")

    ref_start = _references_start(lines)
    if ref_start == len(lines):
        return refs

    for line_num, line in enumerate(lines[ref_start:], start=ref_start + 1):
        stripped = line.strip()
        if not stripped:
            continue
        # Stop at next section
        if stripped.startswith("#"):
            break

        # Numbered reference: [1] Author et al. (2020). Title...
        m = REF_NUMBERED_RE.match(stripped)
        if m:
            key = m.group(1)
            raw = m.group(2)
            doi_m = DOI_IN_REF_RE.search(raw)
            refs.append(
                Reference(
                    key=key,
                    raw_text=raw,
                    line_number=line_num,
                    ref_type="numeric",
                    reference_id=key,
                    aliases=[key],
                    doi=doi_m.group(1).rstrip(".") if doi_m else "",
                )
            )
            continue

        # Author-year reference: Smith (2020). Title...
        m = REF_AUTHOR_YEAR_RE.match(stripped)
        if m:
            authors = m.group(1)
            year = m.group(2)
            rest = m.group(3)
            key = build_author_year_key(authors, year) or f"ref-{line_num}"
            doi_m = DOI_IN_REF_RE.search(rest)
            refs.append(
                Reference(
                    key=key,
                    raw_text=stripped,
                    authors=authors,
                    year=year,
                    title=rest.split(".")[0] if "." in rest else rest,
                    line_number=line_num,
                    ref_type="author-year",
                    reference_id=key,
                    aliases=[key, build_author_year_key(authors, year)],
                    normalized_title=normalize_title(rest.split(".")[0] if "." in rest else rest),
                    doi=doi_m.group(1).rstrip(".") if doi_m else "",
                )
            )
            continue

    return refs


def _ensure_reference_identity(references: list[Reference]) -> list[Reference]:
    """Populate stable IDs and aliases on references."""
    seen_ids: Counter[str] = Counter()
    prepared: list[Reference] = []
    for ref in references:
        current = replace(ref)
        source_prefix = slugify_identifier(current.source_file, default="document") if current.source_file else ""
        local_seed = current.reference_id or current.key or current.doi or f"ref-{current.line_number}"
        local_id = slugify_identifier(local_seed, default="ref")
        base_id = f"{source_prefix}-{local_id}" if source_prefix else local_id
        seen_ids[base_id] += 1
        current.reference_id = base_id if seen_ids[base_id] == 1 else f"{base_id}-{seen_ids[base_id]}"

        aliases = list(current.aliases)
        if current.key:
            aliases.append(current.key)
        if current.authors and current.year:
            aliases.append(build_author_year_key(current.authors, current.year))
        if current.doi:
            aliases.append(current.doi.lower())
        current.aliases = list(dict.fromkeys(alias for alias in aliases if alias))
        if not current.normalized_title and current.title:
            current.normalized_title = normalize_title(current.title)
        prepared.append(current)
    return prepared


def _find_duplicate_references(references: list[Reference]) -> list[DuplicateReferenceCluster]:
    """Detect likely duplicate references conservatively."""
    buckets: dict[tuple[str, str], list[str]] = {}

    for ref in references:
        if ref.doi:
            buckets.setdefault(("doi", ref.doi.lower()), []).append(ref.reference_id)
        elif ref.normalized_title and ref.year:
            buckets.setdefault(("title-year", f"{ref.normalized_title}|{ref.year.lower()}"), []).append(
                ref.reference_id
            )

    clusters: list[DuplicateReferenceCluster] = []
    for (reason, signature), ref_ids in sorted(buckets.items()):
        unique_ids = list(dict.fromkeys(ref_ids))
        if len(unique_ids) > 1:
            clusters.append(
                DuplicateReferenceCluster(
                    reason=reason,
                    signature=signature,
                    reference_ids=unique_ids,
                )
            )
    return clusters


def _build_reference_maps(references: list[Reference]) -> tuple[dict[str, list[Reference]], dict[str, list[Reference]]]:
    key_map: dict[str, list[Reference]] = {}
    alias_map: dict[str, list[Reference]] = {}
    for ref in references:
        key_map.setdefault(ref.key, []).append(ref)
        for alias in ref.aliases:
            alias_map.setdefault(alias, []).append(ref)
    return key_map, alias_map


def _resolve_targets(
    cite: CitationMention,
    source_doc: str,
    key_map: dict[str, list[Reference]],
    alias_map: dict[str, list[Reference]],
) -> list[CitationLink]:
    links: list[CitationLink] = []
    target_hints = cite.target_hints or _populate_target_hints(cite).target_hints
    for hint in target_hints:
        if cite.cite_type == "author-year":
            merged_candidates = [*key_map.get(hint, []), *alias_map.get(hint, [])]
            deduped_candidates: list[Reference] = []
            seen_reference_ids: set[str] = set()
            for candidate in merged_candidates:
                if candidate.reference_id in seen_reference_ids:
                    continue
                seen_reference_ids.add(candidate.reference_id)
                deduped_candidates.append(candidate)
            candidates = deduped_candidates
        else:
            candidates = key_map.get(hint, [])
            if not candidates:
                candidates = alias_map.get(hint, [])

        if len(candidates) == 1:
            target = candidates[0]
            links.append(
                CitationLink(
                    source_doc=source_doc,
                    target_ref=target.key or hint,
                    mention_id=cite.mention_id,
                    reference_id=target.reference_id,
                    status="resolved",
                    confidence=1.0,
                )
            )
        elif len(candidates) > 1:
            links.append(
                CitationLink(
                    source_doc=source_doc,
                    target_ref=hint,
                    mention_id=cite.mention_id,
                    candidate_reference_ids=[candidate.reference_id for candidate in candidates],
                    status="ambiguous",
                    confidence=0.0,
                )
            )
        else:
            links.append(
                CitationLink(
                    source_doc=source_doc,
                    target_ref=hint,
                    mention_id=cite.mention_id,
                    status="missing",
                    confidence=0.0,
                )
            )
    return links


def _build_inventory(
    citations: list[CitationMention],
    references: list[Reference],
    links: list[CitationLink],
) -> dict[str, Any]:
    citation_types = Counter(c.cite_type or "unknown" for c in citations)
    reference_types = Counter(r.ref_type or "unknown" for r in references)
    statuses = Counter(link.status for link in links)
    source_files = {c.source_file for c in citations if c.source_file} | {
        r.source_file for r in references if r.source_file
    }
    return {
        "files_scanned": len(source_files),
        "citation_mentions_by_type": dict(citation_types),
        "references_by_type": dict(reference_types),
        "link_statuses": dict(statuses),
    }


def build_citation_graph(
    citations: list[CitationMention],
    references: list[Reference],
    source_doc: str = "",
) -> CitationGraph:
    """Build a citation graph from extracted citations and references."""
    citations = _prepare_citations(citations)
    references = _ensure_reference_identity(references)
    key_map, alias_map = _build_reference_maps(references)

    edges: list[CitationLink] = []
    ref_counter: Counter[str] = Counter()
    missing_refs: list[str] = []
    ambiguous_refs: list[str] = []

    for cite in citations:
        resolved_links = _resolve_targets(cite, source_doc, key_map, alias_map)
        if not resolved_links and cite.cite_type == "numeric":
            resolved_links = [
                CitationLink(
                    source_doc=source_doc,
                    target_ref=num,
                    mention_id=cite.mention_id,
                    status="missing",
                    confidence=0.0,
                )
                for num in parse_numeric_targets(cite.raw_text)
            ]
        if not resolved_links and cite.cite_type == "author-year":
            resolved_links = [
                CitationLink(
                    source_doc=source_doc,
                    target_ref=hint,
                    mention_id=cite.mention_id,
                    status="missing",
                    confidence=0.0,
                )
                for hint in parse_author_year_targets(cite.raw_text)
            ]

        edges.extend(resolved_links)
        for link in resolved_links:
            ref_counter[link.target_ref] += 1
            if link.status == "missing" and link.target_ref not in missing_refs:
                missing_refs.append(link.target_ref)
            elif link.status == "ambiguous" and link.target_ref not in ambiguous_refs:
                ambiguous_refs.append(link.target_ref)

    top_cited = ref_counter.most_common(20)
    cited_reference_ids = {
        ref_id
        for edge in edges
        for ref_id in ([edge.reference_id] if edge.reference_id else edge.candidate_reference_ids)
        if ref_id
    }
    phantom_refs = [ref.reference_id for ref in references if ref.reference_id not in cited_reference_ids]
    audit = CitationAudit(
        missing_references=missing_refs,
        phantom_references=phantom_refs,
        ambiguous_references=ambiguous_refs,
        duplicate_references=_find_duplicate_references(references),
    )

    return CitationGraph(
        citations=citations,
        references=references,
        edges=edges,
        top_cited=top_cited,
        audit=audit,
        inventory=_build_inventory(citations, references, edges),
    )


# ── File / tree operations ───────────────────────────────────────────────────


def analyze_file(file_path: Path, *, source_label: str | None = None) -> CitationGraph:
    """Analyze citations in a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    label = source_label or _stable_source_label(file_path)
    citations = extract_inline_citations(text, source_file=label)
    references = extract_references(text)
    for ref in references:
        ref.source_file = label
    return build_citation_graph(citations, references, source_doc=label)


def analyze_tree(input_root: Path) -> CitationGraph:
    """Analyze citations across all Markdown files in a directory."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")

    all_citations: list[CitationMention] = []
    all_references: list[Reference] = []
    all_edges: list[CitationLink] = []
    top_counter: Counter[str] = Counter()
    missing_refs: list[str] = []
    phantom_refs: list[str] = []
    ambiguous_refs: list[str] = []
    duplicate_clusters: list[DuplicateReferenceCluster] = []

    for md_path in md_files:
        source_label = _stable_source_label(md_path, root=input_root)
        graph = analyze_file(md_path, source_label=source_label)
        all_citations.extend(graph.citations)
        all_references.extend(graph.references)
        all_edges.extend(graph.edges)
        top_counter.update(link.target_ref for link in graph.edges)

        for value in graph.audit.missing_references:
            if value not in missing_refs:
                missing_refs.append(value)
        for value in graph.audit.phantom_references:
            if value not in phantom_refs:
                phantom_refs.append(value)
        for value in graph.audit.ambiguous_references:
            if value not in ambiguous_refs:
                ambiguous_refs.append(value)
        for cluster in graph.audit.duplicate_references:
            if not any(
                existing.reason == cluster.reason
                and existing.signature == cluster.signature
                and existing.reference_ids == cluster.reference_ids
                for existing in duplicate_clusters
            ):
                duplicate_clusters.append(cluster)

    return CitationGraph(
        citations=all_citations,
        references=all_references,
        edges=all_edges,
        top_cited=top_counter.most_common(20),
        audit=CitationAudit(
            missing_references=missing_refs,
            phantom_references=phantom_refs,
            ambiguous_references=ambiguous_refs,
            duplicate_references=duplicate_clusters,
        ),
        inventory=_build_inventory(all_citations, all_references, all_edges),
    )


# ── Output writers ───────────────────────────────────────────────────────────


def write_citation_report(graph: CitationGraph, output_path: Path) -> Path:
    """Write citation graph as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "summary": {
            "total_citations": len(graph.citations),
            "total_references": len(graph.references),
            "total_edges": len(graph.edges),
            "top_cited": graph.top_cited,
        },
        "inventory": graph.inventory,
        "audit": {
            "missing_references": graph.audit.missing_references,
            "phantom_references": graph.audit.phantom_references,
            "ambiguous_references": graph.audit.ambiguous_references,
            "duplicate_references": [asdict(cluster) for cluster in graph.audit.duplicate_references],
        },
        "citations": [asdict(c) for c in graph.citations],
        "references": [asdict(r) for r in graph.references],
        "edges": [asdict(e) for e in graph.edges],
        "canonical_ir": {
            "mentions": [asdict(c) for c in graph.mentions],
            "references": [asdict(r) for r in graph.references],
            "links": [asdict(e) for e in graph.links],
        },
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def write_dot_graph(graph: CitationGraph, output_path: Path) -> Path:
    """Write citation graph in Graphviz DOT format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["digraph citations {", "    rankdir=LR;", "    node [shape=box];"]

    seen_edges: set[tuple[str, str]] = set()
    for edge in graph.edges:
        src = edge.source_doc.replace('"', '\\"')
        tgt = edge.target_ref.replace('"', '\\"')
        pair = (src, tgt)
        if pair not in seen_edges:
            lines.append(f'    "{src}" -> "{tgt}";')
            seen_edges.add(pair)

    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and analyze citations from Markdown files.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory")
    parser.add_argument("--output", type=Path, help="Path for citation report (JSON)")
    parser.add_argument("--dot", type=Path, help="Path for DOT graph output")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("citations", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_raw_md", "outputs/raw_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "citations.json")).resolve()

    try:
        graph = analyze_tree(input_path) if input_path.is_dir() else analyze_file(input_path)

        written = write_citation_report(graph, output_path)
        log.info(
            "found %d citation(s), %d reference(s) -> %s",
            len(graph.citations),
            len(graph.references),
            written,
        )

        if args.dot:
            dot_path = write_dot_graph(graph, args.dot)
            log.info("DOT graph -> %s", dot_path)

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
