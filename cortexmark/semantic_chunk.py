"""Semantic-aware chunking for scientific and mathematical documents.

Extends heading-based chunking with recognition of:
  - Theorem / Lemma / Proposition / Corollary blocks
  - Proof blocks (with QED detection)
  - Definition / Assumption / Remark / Example blocks
  - Algorithm blocks (code-fenced or keyword-detected)
  - Equation labels and internal cross-references

Each chunk carries an ``entity_type`` that downstream consumers (RAG export,
QA, search) can use to filter or weight results.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from cortexmark.common import (
    Manifest,
    config_fingerprint,
    get_source_id,
    load_config,
    mirror_directory_tree,
    resolve_configured_path,
    resolve_manifest_path,
    setup_logging,
)
from cortexmark.scientific_ir import (
    OBJECT_EQUATION,
    ScientificObject,
    ScientificObjectLink,
    make_object_id,
)

# ── Entity types ─────────────────────────────────────────────────────────────

ENTITY_THEOREM = "theorem"
ENTITY_PROOF = "proof"
ENTITY_DEFINITION = "definition"
ENTITY_ALGORITHM = "algorithm"
ENTITY_EXAMPLE = "example"
ENTITY_REMARK = "remark"
ENTITY_NARRATIVE = "narrative"

ALL_ENTITY_TYPES: tuple[str, ...] = (
    ENTITY_THEOREM,
    ENTITY_PROOF,
    ENTITY_DEFINITION,
    ENTITY_ALGORITHM,
    ENTITY_EXAMPLE,
    ENTITY_REMARK,
    ENTITY_NARRATIVE,
)

# ── Patterns ─────────────────────────────────────────────────────────────────

# Theorem-like environments: "Theorem 3.2", "Lemma 1", "Corollary 2.1 (Name)"
_THEOREM_LIKE = (
    "theorem",
    "lemma",
    "proposition",
    "corollary",
)

# Definition-like environments
_DEFINITION_LIKE = (
    "definition",
    "assumption",
    "axiom",
    "conjecture",
    "hypothesis",
    "condition",
)

_EXAMPLE_LIKE = (
    "example",
    "exercise",
    "problem",
    "case",
)

_REMARK_LIKE = (
    "remark",
    "note",
    "observation",
    "comment",
    "convention",
)

# Build combined pattern for block openers
_ALL_ENV_NAMES = _THEOREM_LIKE + _DEFINITION_LIKE + _EXAMPLE_LIKE + _REMARK_LIKE
_ENV_ALTS = "|".join(re.escape(n) for n in _ALL_ENV_NAMES)

# Matches: "**Theorem 3.2.**", "**Theorem 3.2 (Bellman)**", "Theorem 3.2.",
# "**Definition 1:**", "THEOREM 2.1", "Lemma.", etc.
BLOCK_OPENER_RE = re.compile(
    rf"^(?:\*\*)?(?P<kind>{_ENV_ALTS})\b"  # bold-optional + complete kind
    rf"(?=(?:\s+\d|\s*\(|[.:)]|\s*\*\*|$))"  # require block syntax
    rf"(?:\s+(?P<label>[\d]+(?:\.[\d]+)*))?\.?"  # optional numeric label
    rf"(?:\s*\((?P<name>[^)]+)\))?"  # optional parenthesized name
    rf"[.:)]*(?:\*\*)?\s*(?P<rest>.*)$",  # trailing punctuation + rest
    re.IGNORECASE,
)

# Proof opener: "Proof.", "Proof:", "**Proof.**", "Proof of Theorem 3.2."
PROOF_OPENER_RE = re.compile(
    r"^(?:\*\*)?[Pp]roof\b"
    r"(?:\s+of\s+(?P<of_label>\w[\w\s.]*\d[\d.]*))?\.?"
    r"(?:(?:[.:)]+(?:\*\*)?|\s*\*\*)\s*(?P<rest>.*)|\s*)$",
)

# QED markers signalling end of a proof
QED_MARKERS: tuple[str, ...] = ("□", "∎", "■", "q.e.d.", "qed", "▪", "◻")

# Algorithm detection: code fence with "algorithm" or keyword-based
ALGORITHM_FENCE_RE = re.compile(r"^```(?:text|algorithm|pseudocode)?\s*$", re.IGNORECASE)
ALGORITHM_LABEL_RE = re.compile(
    r"^(?:\*\*)?(?:Algorithm)\s+(?P<label>[\d]+(?:\.[\d]+)*)[.:)]*(?:\*\*)?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

# Inline formula detection (for metadata)
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

# Fallback for docling's non-LaTeX equation-recovery convention, which emits
# no "$" delimiters at all:
#   > Equation (recovered from PDF):
#   >
#   > <mangled-unicode-math-text>
# (see convert.format_formula_block / formula_score.RECOVERED_EQUATION_RE).
RECOVERED_EQUATION_RE = re.compile(
    r"^>\s*Equation\s*\(recovered from PDF\):\s*\n>\s*\n>\s*(.+)$",
    re.MULTILINE,
)

# Cross-reference patterns
CROSS_REF_RE = re.compile(
    r"(?:Equation|Eq\.|Figure|Fig\.|Table|Theorem|Lemma|Definition|"
    r"Proposition|Corollary|Algorithm|Section|Sec\.|Chapter|Ch\.)"
    r"\s*[\(]?([\d]+(?:\.[\d]+)*)[\)]?",
    re.IGNORECASE,
)

# Heading pattern (reused from chunk.py logic)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class SemanticChunk:
    """A chunk with scientific entity metadata."""

    chapter: str | None = None
    section: str | None = None
    body: list[str] = field(default_factory=list)
    entity_type: str = ENTITY_NARRATIVE
    entity_kind: str = ENTITY_NARRATIVE
    entity_label: str | None = None  # e.g. "Theorem 3.2"
    entity_name: str | None = None  # e.g. "Bellman Optimality"
    parent_label: str | None = None  # proof → parent theorem label
    parent_evidence_level: str = ""
    line_number: int = 0
    formulas: list[str] = field(default_factory=list)
    cross_refs: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        if self.entity_label:
            name_suffix = f" ({self.entity_name})" if self.entity_name else ""
            return f"{self.entity_label}{name_suffix}"
        return self.section or self.chapter or "untitled"

    def opener(self) -> str | None:
        """Return a normalized opener line that preserves semantic labels."""
        if self.entity_type == ENTITY_PROOF:
            if self.parent_label:
                return f"**Proof of {self.parent_label}.**"
            return "**Proof.**"
        if self.entity_label:
            if self.entity_name:
                return f"**{self.entity_label} ({self.entity_name}).**"
            return f"**{self.entity_label}.**"
        return None

    def render(self) -> str:
        parts: list[str] = []
        if self.chapter:
            parts.append(f"# {self.chapter}")
        if self.section:
            parts.append(f"## {self.section}")
        if parts:
            parts.append("")
        opener = self.opener()
        if opener:
            parts.append(opener)
            if self.body:
                parts.append("")
        parts.extend(self.body)
        return "\n".join(parts).strip() + "\n"


# ── Extraction helpers ───────────────────────────────────────────────────────


def classify_env_kind(kind: str) -> str:
    """Map an environment name to an entity type."""
    kind_lower = kind.lower()
    if kind_lower in _THEOREM_LIKE:
        return ENTITY_THEOREM
    if kind_lower in _DEFINITION_LIKE:
        return ENTITY_DEFINITION
    if kind_lower in _EXAMPLE_LIKE:
        return ENTITY_EXAMPLE
    if kind_lower in _REMARK_LIKE:
        return ENTITY_REMARK
    return ENTITY_NARRATIVE


def extract_formulas(text: str) -> list[str]:
    """Extract display and inline math from text.

    Also recognises docling's dollar-free "recovered from PDF" equation
    blockquotes as a fallback, since docling's math-recovery path never
    emits LaTeX "$" delimiters.
    """
    formulas: list[str] = []
    for m in DISPLAY_MATH_RE.finditer(text):
        formulas.append(m.group(1).strip())
    for m in INLINE_MATH_RE.finditer(text):
        formulas.append(m.group(1).strip())
    for m in RECOVERED_EQUATION_RE.finditer(text):
        formulas.append(m.group(1).strip())
    return formulas


def extract_cross_refs(text: str) -> list[str]:
    """Extract cross-reference labels from text (e.g. 'Theorem 3.2').

    Normalizes away any surrounding parentheses around the numeric label
    (e.g. "Theorem (3.2)") so both the bare and parenthesized spellings of
    the same reference collapse to one token instead of leaving a dangling
    unbalanced "Theorem (3.2" artifact.
    """
    refs: list[str] = []
    for m in CROSS_REF_RE.finditer(text):
        keyword = text[m.start() : m.start(1)].strip().rstrip("(").strip()
        full = f"{keyword} {m.group(1)}"
        if full not in refs:
            refs.append(full)
    return refs


def has_qed(line: str) -> bool:
    """Check if a line ends with a QED marker."""
    stripped = line.rstrip()
    lower = stripped.lower()
    return any(lower.endswith(marker) for marker in QED_MARKERS)


def normalize_entity_label(label: str) -> str:
    """Normalize entity labels while preserving dotted numeric identifiers."""
    collapsed = " ".join(label.strip().split())
    return re.sub(r"[.:;)\]]+$", "", collapsed)


def _slugify(value: str) -> str:
    """URL-safe slug from a title string."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "untitled"


# ── Core parser ──────────────────────────────────────────────────────────────


def parse_semantic_chunks(
    text: str,
    *,
    split_on_headings: bool = True,
    heading_levels: list[int] | None = None,
) -> list[SemanticChunk]:
    """Parse text into semantic chunks aware of scientific environments.

    Recognises theorem/proof/definition/algorithm blocks and assigns
    ``entity_type`` accordingly.  Falls back to heading-based splitting
    for narrative content.
    """
    levels = heading_levels or [1, 2]
    min_level = min(levels) if levels else 1

    chunks: list[SemanticChunk] = []
    chapter: str | None = None
    section: str | None = None

    # Accumulator for the current chunk
    body: list[str] = []
    entity_type: str = ENTITY_NARRATIVE
    entity_label: str | None = None
    entity_name: str | None = None
    entity_kind: str = ENTITY_NARRATIVE
    parent_label: str | None = None
    parent_evidence_level: str = ""
    chunk_line_number: int = 0
    last_theorem_label: str | None = None

    in_code_fence = False
    in_proof = False

    def flush() -> None:
        nonlocal body, entity_type, entity_label, entity_name, entity_kind, parent_label, parent_evidence_level
        nonlocal chunk_line_number
        content = [line for line in body if line.strip()]
        if content:
            full_text = "\n".join(content)
            chunks.append(
                SemanticChunk(
                    chapter=chapter,
                    section=section,
                    body=content,
                    entity_type=entity_type,
                    entity_kind=entity_kind,
                    entity_label=entity_label,
                    entity_name=entity_name,
                    parent_label=parent_label,
                    parent_evidence_level=parent_evidence_level,
                    line_number=chunk_line_number,
                    formulas=extract_formulas(full_text),
                    cross_refs=extract_cross_refs(full_text),
                )
            )
        body = []
        entity_type = ENTITY_NARRATIVE
        entity_kind = ENTITY_NARRATIVE
        entity_label = None
        entity_name = None
        parent_label = None
        parent_evidence_level = ""
        chunk_line_number = 0

    def begin_chunk(line_number: int) -> None:
        nonlocal chunk_line_number
        if chunk_line_number == 0:
            chunk_line_number = line_number

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip()
        stripped = line.strip()

        # Track code fences to avoid false matches inside them
        if stripped.startswith("```"):
            if in_code_fence:
                # Closing fence
                in_code_fence = False
                begin_chunk(line_number)
                body.append(line)
                # If this was an algorithm block, flush after close
                if entity_type == ENTITY_ALGORITHM:
                    flush()
                continue
            # Opening fence — check if it's an algorithm fence
            in_code_fence = True
            if ALGORITHM_FENCE_RE.match(stripped) and entity_type != ENTITY_ALGORITHM:
                flush()
                entity_type = ENTITY_ALGORITHM
                entity_kind = ENTITY_ALGORITHM
                begin_chunk(line_number)
            else:
                begin_chunk(line_number)
            body.append(line)
            continue

        if in_code_fence:
            begin_chunk(line_number)
            body.append(line)
            continue

        # ── Heading detection ────────────────────────────────────────
        if split_on_headings:
            heading_match = HEADING_RE.match(stripped)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                if level in levels:
                    # Flush any in-progress block
                    if in_proof:
                        in_proof = False
                    flush()
                    if level == min_level:
                        chapter = title
                        section = None
                    else:
                        section = title
                    continue

        # ── Proof opener ─────────────────────────────────────────────
        proof_match = PROOF_OPENER_RE.match(stripped)
        if proof_match and not in_proof:
            flush()
            entity_type = ENTITY_PROOF
            entity_kind = ENTITY_PROOF
            begin_chunk(line_number)
            of_label = proof_match.group("of_label")
            if of_label:
                parent_label = normalize_entity_label(of_label)
                parent_evidence_level = "explicit"
            elif last_theorem_label:
                parent_label = last_theorem_label
                parent_evidence_level = "inferred"
            in_proof = True
            rest = proof_match.group("rest")
            if rest:
                body.append(rest)
            continue

        # ── QED detection (end of proof) ─────────────────────────────
        if in_proof and has_qed(stripped):
            begin_chunk(line_number)
            body.append(line)
            in_proof = False
            flush()
            continue

        # ── Block opener (theorem, definition, etc.) ─────────────────
        block_match = BLOCK_OPENER_RE.match(stripped)
        if block_match:
            kind = block_match.group("kind")
            label = block_match.group("label")
            name = block_match.group("name")
            rest = block_match.group("rest")

            new_type = classify_env_kind(kind)
            full_label = f"{kind.capitalize()} {label}" if label else kind.capitalize()

            # Flush previous block
            flush()
            entity_type = new_type
            entity_kind = kind.lower()
            entity_label = full_label
            entity_name = name
            begin_chunk(line_number)
            if new_type == ENTITY_THEOREM:
                last_theorem_label = full_label
            if rest:
                body.append(rest)
            continue

        # ── Algorithm label line ─────────────────────────────────────
        algo_match = ALGORITHM_LABEL_RE.match(stripped)
        if algo_match:
            flush()
            entity_type = ENTITY_ALGORITHM
            entity_kind = ENTITY_ALGORITHM
            entity_label = f"Algorithm {algo_match.group('label')}"
            begin_chunk(line_number)
            rest = algo_match.group("rest")
            if rest:
                body.append(rest)
            continue

        # ── Regular line ─────────────────────────────────────────────
        begin_chunk(line_number)
        body.append(line)

    # Flush remaining
    if in_proof:
        in_proof = False
    flush()

    return [c for c in chunks if c.body or c.entity_label]


# ── File / tree operations ───────────────────────────────────────────────────


def _semantic_chunk_fingerprint(heading_levels: list[int] | None) -> str:
    return config_fingerprint({"heading_levels": heading_levels or [1, 2]})


def chunk_file(
    input_path: Path,
    output_dir: Path,
    *,
    heading_levels: list[int] | None = None,
) -> list[Path]:
    """Parse a file into semantic chunks and write numbered output files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    chunks = parse_semantic_chunks(text, heading_levels=heading_levels)
    if not chunks:
        # Empty or whitespace-only cleaned text yields zero semantic chunks
        # from the parser. Emit one minimal chunk instead of raising, so a
        # structureless document still gets a valid chunk set (and an
        # `analyze` caller doesn't drop the whole document with a
        # half-written bundle); see finding C14.
        stripped_text = text.strip()
        chunks = [SemanticChunk(body=[stripped_text] if stripped_text else [])]

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("chunk_*.md"):
        stale_path.unlink()

    written: list[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        slug = _slugify(chunk.title)
        path = output_dir / f"chunk_{index:03d}_{slug}.md"
        path.write_text(chunk.render(), encoding="utf-8")
        written.append(path)
    return written


def chunk_tree(
    input_root: Path,
    output_root: Path,
    *,
    manifest: Manifest | None = None,
    heading_levels: list[int] | None = None,
    include_input_root_name: bool = True,
) -> list[Path]:
    """Semantically chunk all Markdown files in a directory tree."""
    markdown_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not markdown_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")

    mirror_directory_tree(input_root, output_root)
    fingerprint = _semantic_chunk_fingerprint(heading_levels)
    written: list[Path] = []
    for markdown_path in markdown_files:
        relative = markdown_path.relative_to(input_root)
        base_output = output_root / input_root.name if include_input_root_name else output_root
        target_dir = base_output / relative.parent / markdown_path.stem
        existing_outputs = sorted(target_dir.glob("chunk_*.md"))
        if manifest and not manifest.needs_update(
            markdown_path,
            fingerprint=fingerprint,
            outputs=existing_outputs,
        ):
            continue
        chunk_paths = chunk_file(markdown_path, target_dir, heading_levels=heading_levels)
        written.extend(chunk_paths)
        if manifest:
            manifest.record(markdown_path, fingerprint=fingerprint, outputs=chunk_paths)
    return written


# ── Metadata export ──────────────────────────────────────────────────────────


def chunks_to_records(chunks: list[SemanticChunk], source: str) -> list[dict[str, object]]:
    """Convert semantic chunks to metadata-rich dictionaries for export."""
    records: list[dict[str, object]] = []
    object_map = {
        (obj.label or obj.title, obj.line_number, obj.object_kind): obj
        for obj in chunks_to_scientific_objects(chunks, source)
        if obj.object_type != OBJECT_EQUATION
    }
    for i, chunk in enumerate(chunks):
        linked_object = object_map.get(((chunk.entity_label or chunk.title), chunk.line_number, chunk.entity_kind))
        records.append(
            {
                "index": i,
                "source": source,
                "title": chunk.title,
                "entity_type": chunk.entity_type,
                "entity_kind": chunk.entity_kind,
                "entity_label": chunk.entity_label,
                "entity_name": chunk.entity_name,
                "parent_label": chunk.parent_label,
                "parent_evidence_level": chunk.parent_evidence_level,
                "line_number": chunk.line_number,
                "object_id": linked_object.object_id if linked_object else "",
                "parent_object_id": linked_object.parent_object_id if linked_object else "",
                "formulas": chunk.formulas,
                "cross_refs": chunk.cross_refs,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "line_count": len(chunk.body),
            }
        )
    return records


def build_entity_summary(chunks: list[SemanticChunk]) -> dict[str, int]:
    """Count chunks by entity type."""
    summary: dict[str, int] = {}
    for chunk in chunks:
        summary[chunk.entity_type] = summary.get(chunk.entity_type, 0) + 1
    return summary


def chunks_to_scientific_objects(chunks: list[SemanticChunk], source: str) -> list[ScientificObject]:
    """Convert semantic chunks into stable scholarly objects."""
    objects: list[ScientificObject] = []
    label_to_object_id: dict[str, str] = {}

    for index, chunk in enumerate(chunks, start=1):
        if chunk.entity_type == ENTITY_NARRATIVE:
            continue
        if chunk.entity_label:
            object_id = make_object_id(source, chunk.entity_type, label=chunk.entity_label)
        else:
            object_id = make_object_id(
                source,
                chunk.entity_type,
                label="",
                name=chunk.entity_name or chunk.title,
                line_number=chunk.line_number,
                ordinal=index,
            )
        obj = ScientificObject(
            object_id=object_id,
            object_type=chunk.entity_type,
            object_kind=chunk.entity_kind,
            label=chunk.entity_label or "",
            name=chunk.entity_name or "",
            title=chunk.title,
            source_file=source,
            line_number=chunk.line_number,
            chapter=chunk.chapter or "",
            section=chunk.section or "",
            text="\n".join(chunk.body).strip(),
            evidence_level="explicit",
            metadata={
                "formulas": list(chunk.formulas),
                "cross_refs": list(chunk.cross_refs),
                "parent_label": chunk.parent_label or "",
                "parent_evidence_level": chunk.parent_evidence_level,
            },
        )
        objects.append(obj)
        if chunk.entity_label:
            label_to_object_id[chunk.entity_label] = object_id

    for obj in objects:
        parent_label = str(obj.metadata.get("parent_label", "")).strip()
        if parent_label and parent_label in label_to_object_id:
            obj.parent_object_id = label_to_object_id[parent_label]
        elif parent_label:
            parent_kind = parent_label.split(" ", 1)[0]
            parent_type = classify_env_kind(parent_kind)
            if parent_type != ENTITY_NARRATIVE:
                obj.parent_object_id = make_object_id(source, parent_type, label=parent_label)

    equation_ordinal = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        parent_candidates = [
            obj for obj in objects if obj.line_number == chunk.line_number and obj.source_file == source
        ]
        parent_object_id = parent_candidates[0].object_id if parent_candidates else ""
        for formula_index, formula in enumerate(chunk.formulas, start=1):
            equation_ordinal += 1
            objects.append(
                ScientificObject(
                    object_id=make_object_id(
                        source,
                        OBJECT_EQUATION,
                        label=f"{chunk.entity_label or chunk.title}:{formula_index}",
                        line_number=chunk.line_number,
                        ordinal=equation_ordinal,
                    ),
                    object_type=OBJECT_EQUATION,
                    object_kind=OBJECT_EQUATION,
                    title=f"Equation {equation_ordinal}",
                    source_file=source,
                    line_number=chunk.line_number,
                    chapter=chunk.chapter or "",
                    section=chunk.section or "",
                    text=formula,
                    parent_object_id=parent_object_id,
                    metadata={
                        "formula": formula,
                        "source_kind": "chunk_formula",
                        "chunk_index": chunk_index,
                    },
                )
            )

    return objects


def build_scientific_object_links(objects: list[ScientificObject]) -> list[ScientificObjectLink]:
    """Create typed links between scholarly objects."""
    links: list[ScientificObjectLink] = []
    object_index = {obj.object_id: obj for obj in objects}
    for obj in objects:
        if obj.object_type == OBJECT_EQUATION and obj.parent_object_id:
            parent = object_index.get(obj.parent_object_id)
            links.append(
                ScientificObjectLink(
                    source_object_id=obj.object_id,
                    target_object_id=obj.parent_object_id,
                    relation="appears_in",
                    status="resolved",
                    source_label=obj.title,
                    target_label=(parent.label or parent.title) if parent else "",
                    source_file=obj.source_file,
                    line_number=obj.line_number,
                )
            )
        elif obj.object_type == ENTITY_PROOF and obj.parent_object_id:
            parent = object_index.get(obj.parent_object_id)
            links.append(
                ScientificObjectLink(
                    source_object_id=obj.object_id,
                    target_object_id=obj.parent_object_id,
                    relation="proof_of",
                    status="resolved",
                    source_label=obj.label or obj.title,
                    target_label=(parent.label or parent.title)
                    if parent
                    else str(obj.metadata.get("parent_label", "")),
                    source_file=obj.source_file,
                    line_number=obj.line_number,
                    metadata={"evidence_level": obj.metadata.get("parent_evidence_level", "")},
                )
            )
    return links


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic chunking for scientific Markdown: recognises theorems, proofs, definitions, algorithms."
    )
    parser.add_argument("--input", type=Path, help="Markdown file or directory")
    parser.add_argument("--output-dir", type=Path, help="Chunk output root directory")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    parser.add_argument("--no-manifest", action="store_true", help="Disable idempotency manifest")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("semantic_chunk", cfg)

    chunk_cfg = cfg.get("chunk", {})
    heading_levels = chunk_cfg.get("split_levels", [1, 2])
    fingerprint = _semantic_chunk_fingerprint(heading_levels)

    source_id = get_source_id(cfg)
    input_path = (
        args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md") / source_id
    ).resolve()
    output_dir = (
        args.output_dir or resolve_configured_path(cfg, "output_semantic_chunks", "outputs/semantic_chunks")
    ).resolve()

    manifest = None
    idem_cfg = cfg.get("idempotency", {})
    if idem_cfg.get("enabled", True) and not args.no_manifest:
        manifest = Manifest(resolve_manifest_path(cfg))

    try:
        if input_path.is_dir():
            written = chunk_tree(input_path, output_dir, manifest=manifest, heading_levels=heading_levels)
            if manifest:
                manifest.save()
            log.info("wrote %d semantic chunk(s) from %s", len(written), input_path)
        else:
            written = chunk_file(input_path, output_dir / input_path.stem, heading_levels=heading_levels)
            if manifest:
                manifest.record(input_path, fingerprint=fingerprint, outputs=written)
                manifest.save()
            log.info("wrote %d semantic chunk(s) to %s", len(written), output_dir / input_path.stem)
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1
    except ValueError as exc:
        log.error("invalid input: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
