"""Cross-reference resolution and linking for scientific Markdown.

Detects and indexes internal cross-references such as:
  - Theorem / Lemma / Definition / Proposition references
  - Equation references: Equation (3), Eq. 5.1
  - Figure / Table references: Figure 2, Table 3.1
  - Section / Chapter references: Section 2.3, Chapter 4
  - Algorithm references: Algorithm 1

Produces a reference index mapping labels → locations and detects
unresolved (dangling) references.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import (
    load_config,
    resolve_configured_path,
    resolve_quality_report_path,
    setup_logging,
)
from cortexmark.scientific_ir import ScientificObject, ScientificObjectLink, make_object_id, stable_source_label

# ── Reference categories ─────────────────────────────────────────────────────

CATEGORY_THEOREM = "theorem"
CATEGORY_EQUATION = "equation"
CATEGORY_FIGURE = "figure"
CATEGORY_TABLE = "table"
CATEGORY_SECTION = "section"
CATEGORY_ALGORITHM = "algorithm"
CATEGORY_OTHER = "other"

ALL_CATEGORIES: tuple[str, ...] = (
    CATEGORY_THEOREM,
    CATEGORY_EQUATION,
    CATEGORY_FIGURE,
    CATEGORY_TABLE,
    CATEGORY_SECTION,
    CATEGORY_ALGORITHM,
    CATEGORY_OTHER,
)

# ── Detection patterns ───────────────────────────────────────────────────────

# Definition sites: "**Theorem 3.2.**", "## Theorem 3.2", "Theorem 3.2."
DEFINITION_SITE_RE = re.compile(
    r"^\s{0,3}(?:>\s*)?(?:(?P<marker>#{1,6}\s+(?:\*\*)?|\*\*))?"
    r"(?P<kind>Theorem|Lemma|Proposition|Corollary|Definition|"
    r"Assumption|Axiom|Conjecture|Remark|Example|Algorithm)"
    r"\s+(?P<label>[\d]+(?:\.[\d]+)*)"
    r"(?:\s*\((?P<name>[^)]+)\))?"
    r"(?(marker)[.:)]*|[.:)])(?:\*\*)?",
    re.IGNORECASE,
)

# Equation label site: "$$...$$ (N)" / "\tag{N}" (hand-authored Markdown), or a
# numbered equation recovered from a PDF and rendered as
# "> Equation (recovered from PDF):\n>\n> ... (N)" (see
# convert.py:format_formula_block — this is the form the pipeline actually
# emits for every equation with a source-side number).
EQUATION_LABEL_RE = re.compile(
    r"(?:\$\$.*?\$\$\s*\((?P<label>\d+(?:\.\d+)*)\))"
    r"|(?:\\tag\{(?P<tag>[^}]+)\})"
    r"|(?:Equation \(recovered from PDF\):\s*\n>\s*\n>[^\n]*\((?P<recovered>\d+(?:\.\d+)*)\)[ \t]*(?=\n|$))",
    re.DOTALL,
)

# Figure definition: ![...](...)  or Figure N:
FIGURE_DEF_RE = re.compile(
    r"^\s{0,3}(?:>\s*)?(?:#{1,6}\s+(?:\*\*)?|\*\*)?"
    r"(?:Figure|Fig\.)\s+(?P<label>\d+(?:\.\d+)*)\s*[:.]\s*(?:\*\*)?",
    re.IGNORECASE,
)

# Table definition: Table N:
TABLE_DEF_RE = re.compile(
    r"^\s{0,3}(?:>\s*)?(?:#{1,6}\s+(?:\*\*)?|\*\*)?"
    r"(?:Table|Tbl\.)\s+(?P<label>\d+(?:\.\d+)*)\s*[:.]\s*(?:\*\*)?",
    re.IGNORECASE,
)

# Section definitions: "# Chapter 2", "## Section 2.1", "## 2.1 Methods"
SECTION_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+(?:(?P<kind>Section|Chapter)\s+)?(?P<label>\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)

# Reference sites (mentions in text)
REF_MENTION_RE = re.compile(
    r"(?P<kind>Theorem|Lemma|Proposition|Corollary|Definition|"
    r"Assumption|Axiom|Remark|Example|Algorithm|"
    r"Equation|Eq\.|Figure|Fig\.|Table|Tbl\.|"
    r"Section|Sec\.|Chapter|Ch\.)"
    r"\s*[\(]?(?P<label>\d+(?:\.\d+)*)[\)]?",
    re.IGNORECASE,
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class RefDefinition:
    """A defined label location (where a theorem/figure/etc. is declared)."""

    kind: str
    label: str
    name: str = ""
    source_file: str = ""
    line_number: int = 0
    category: str = CATEGORY_OTHER

    @property
    def full_label(self) -> str:
        return f"{self.kind} {self.label}"


@dataclass
class RefMention:
    """A reference mention in text (e.g. 'see Theorem 3.2')."""

    kind: str
    label: str
    source_file: str = ""
    line_number: int = 0
    category: str = CATEGORY_OTHER

    @property
    def full_label(self) -> str:
        return f"{self.kind} {self.label}"


@dataclass
class CrossRefReport:
    """Complete cross-reference analysis for a document or collection."""

    definitions: list[RefDefinition] = field(default_factory=list)
    mentions: list[RefMention] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    objects: list[ScientificObject] = field(default_factory=list)
    links: list[ScientificObjectLink] = field(default_factory=list)

    @property
    def resolution_rate(self) -> float:
        total = len(self.resolved) + len(self.unresolved)
        if total == 0:
            return 1.0
        return len(self.resolved) / total


# ── Classification ───────────────────────────────────────────────────────────

_KIND_TO_CATEGORY: dict[str, str] = {}
for _k in (
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "assumption",
    "axiom",
    "conjecture",
    "remark",
    "example",
):
    _KIND_TO_CATEGORY[_k] = CATEGORY_THEOREM
for _k in ("equation", "eq."):
    _KIND_TO_CATEGORY[_k] = CATEGORY_EQUATION
for _k in ("figure", "fig."):
    _KIND_TO_CATEGORY[_k] = CATEGORY_FIGURE
for _k in ("table", "tbl."):
    _KIND_TO_CATEGORY[_k] = CATEGORY_TABLE
for _k in ("section", "sec.", "chapter", "ch."):
    _KIND_TO_CATEGORY[_k] = CATEGORY_SECTION
_KIND_TO_CATEGORY["algorithm"] = CATEGORY_ALGORITHM


def classify_kind(kind: str) -> str:
    """Map a reference kind string to a category."""
    return _KIND_TO_CATEGORY.get(kind.lower(), CATEGORY_OTHER)


# ── Extraction ───────────────────────────────────────────────────────────────


def extract_definitions(text: str, source_file: str = "") -> list[RefDefinition]:
    """Extract all label definition sites from text."""
    defs: list[RefDefinition] = []
    lines = text.splitlines()

    for line_num, line in enumerate(lines, start=1):
        for m in DEFINITION_SITE_RE.finditer(line):
            kind = m.group("kind")
            label = m.group("label")
            name = m.group("name") or ""
            defs.append(
                RefDefinition(
                    kind=kind.capitalize(),
                    label=label,
                    name=name,
                    source_file=source_file,
                    line_number=line_num,
                    category=classify_kind(kind),
                )
            )

        for m in FIGURE_DEF_RE.finditer(line):
            label = m.group("label")
            defs.append(
                RefDefinition(
                    kind="Figure",
                    label=label,
                    source_file=source_file,
                    line_number=line_num,
                    category=CATEGORY_FIGURE,
                )
            )

        for m in TABLE_DEF_RE.finditer(line):
            label = m.group("label")
            defs.append(
                RefDefinition(
                    kind="Table",
                    label=label,
                    source_file=source_file,
                    line_number=line_num,
                    category=CATEGORY_TABLE,
                )
            )

        for m in SECTION_HEADING_RE.finditer(line):
            kind = m.group("kind") or "Section"
            defs.append(
                RefDefinition(
                    kind=kind.capitalize(),
                    label=m.group("label"),
                    source_file=source_file,
                    line_number=line_num,
                    category=CATEGORY_SECTION,
                )
            )

    # Equation labels (may span multiple lines)
    for m in EQUATION_LABEL_RE.finditer(text):
        label = m.group("label") or m.group("tag") or m.group("recovered")
        if label:
            offset = text[: m.start()].count("\n") + 1
            defs.append(
                RefDefinition(
                    kind="Equation",
                    label=label,
                    source_file=source_file,
                    line_number=offset,
                    category=CATEGORY_EQUATION,
                )
            )

    return defs


def extract_mentions(text: str, source_file: str = "") -> list[RefMention]:
    """Extract all reference mentions from text."""
    mentions: list[RefMention] = []
    lines = text.splitlines()

    for line_num, line in enumerate(lines, start=1):
        if line.lstrip().startswith("#"):
            continue
        definition_spans = [
            match.span()
            for pattern in (DEFINITION_SITE_RE, FIGURE_DEF_RE, TABLE_DEF_RE)
            for match in pattern.finditer(line)
        ]
        for m in REF_MENTION_RE.finditer(line):
            if any(m.start() < end and start < m.end() for start, end in definition_spans):
                continue
            kind = m.group("kind")
            label = m.group("label")
            mentions.append(
                RefMention(
                    kind=_normalize_kind(kind),
                    label=label,
                    source_file=source_file,
                    line_number=line_num,
                    category=classify_kind(kind),
                )
            )

    return mentions


def _normalize_kind(kind: str) -> str:
    """Normalize abbreviated kinds to canonical form."""
    mapping = {
        "eq.": "Equation",
        "fig.": "Figure",
        "tbl.": "Table",
        "sec.": "Section",
        "ch.": "Chapter",
    }
    return mapping.get(kind.lower(), kind.capitalize())


def _resolution_key(kind: str, label: str) -> str:
    # Keyed on the normalized kind, not the shared category: "Chapter 2" and
    # "Section 2" must resolve independently even though both classify as
    # CATEGORY_SECTION, otherwise a Chapter mention could falsely resolve
    # against an unrelated Section definition sharing the same numeral (and
    # vice versa). A numeric-only heading ("## 2.1") still normalizes to kind
    # "Section" at extraction time, so it remains resolvable by "Section 2.1"
    # mentions as before -- only explicit Chapter/Section cross-matching is cut.
    return f"{_normalize_kind(kind)} {label}".lower()


def resolve_references(
    definitions: list[RefDefinition],
    mentions: list[RefMention],
) -> CrossRefReport:
    """Match mentions against definitions, identify resolved/unresolved."""
    def_object_map: dict[str, list[ScientificObject]] = {}
    objects: list[ScientificObject] = []
    for d in definitions:
        norm_kind = _normalize_kind(d.kind)
        key = _resolution_key(d.kind, d.label)
        obj = ScientificObject(
            object_id=make_object_id(
                d.source_file,
                d.category or norm_kind.lower(),
                label=d.full_label,
                line_number=d.line_number,
            ),
            object_type=d.category or norm_kind.lower(),
            object_kind=norm_kind.lower(),
            label=d.full_label,
            name=d.name,
            title=d.full_label,
            source_file=d.source_file,
            line_number=d.line_number,
            text="",
            metadata={"category": d.category},
        )
        def_object_map.setdefault(key, []).append(obj)
        objects.append(obj)

    resolved: list[str] = []
    unresolved: list[str] = []
    seen_resolved: set[str] = set()
    seen_unresolved: set[str] = set()
    links: list[ScientificObjectLink] = []

    for mention in mentions:
        norm_kind = _normalize_kind(mention.kind)
        key = _resolution_key(mention.kind, mention.label)
        mention_object = ScientificObject(
            object_id=make_object_id(
                mention.source_file,
                "xref_mention",
                label=mention.full_label,
                line_number=mention.line_number,
            ),
            object_type="xref_mention",
            object_kind=norm_kind.lower(),
            label=mention.full_label,
            title=mention.full_label,
            source_file=mention.source_file,
            line_number=mention.line_number,
            text="",
            metadata={"category": mention.category},
        )
        objects.append(mention_object)
        candidates = def_object_map.get(key, [])
        local_candidates = [obj for obj in candidates if obj.source_file == mention.source_file]
        target: ScientificObject | None = None
        status = "unresolved"
        if len(local_candidates) == 1:
            target = local_candidates[0]
            status = "resolved"
        elif len(candidates) == 1:
            target = candidates[0]
            status = "resolved"
        elif candidates:
            status = "ambiguous"

        if status == "resolved" and target is not None:
            if key not in seen_resolved:
                resolved.append(mention.full_label)
                seen_resolved.add(key)
            links.append(
                ScientificObjectLink(
                    source_object_id=mention_object.object_id,
                    target_object_id=target.object_id,
                    relation="references",
                    status="resolved",
                    source_label=mention.full_label,
                    target_label=target.label,
                    source_file=mention.source_file,
                    line_number=mention.line_number,
                    metadata={"category": mention.category},
                )
            )
            continue

        if key not in seen_unresolved:
            unresolved.append(mention.full_label)
            seen_unresolved.add(key)

        link_metadata: dict[str, Any] = {"category": mention.category}
        if candidates:
            link_metadata["candidate_object_ids"] = [candidate.object_id for candidate in candidates]
            link_metadata["candidate_sources"] = [candidate.source_file for candidate in candidates]
        links.append(
            ScientificObjectLink(
                source_object_id=mention_object.object_id,
                target_object_id=target.object_id if target is not None else "",
                relation="references",
                status=status,
                source_label=mention.full_label,
                target_label=target.label if target is not None else mention.full_label,
                source_file=mention.source_file,
                line_number=mention.line_number,
                metadata=link_metadata,
            )
        )

    return CrossRefReport(
        definitions=definitions,
        mentions=mentions,
        resolved=resolved,
        unresolved=unresolved,
        objects=objects,
        links=links,
    )


# ── File / tree operations ───────────────────────────────────────────────────


def analyze_file(file_path: Path) -> CrossRefReport:
    """Analyze cross-references in a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    source_label = stable_source_label(file_path)
    defs = extract_definitions(text, source_file=source_label)
    mentions = extract_mentions(text, source_file=source_label)
    return resolve_references(defs, mentions)


def analyze_tree(input_root: Path) -> CrossRefReport:
    """Analyze cross-references across all Markdown files in a directory."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")

    all_defs: list[RefDefinition] = []
    all_mentions: list[RefMention] = []

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        source_label = stable_source_label(md_path, root=input_root)
        all_defs.extend(extract_definitions(text, source_file=source_label))
        all_mentions.extend(extract_mentions(text, source_file=source_label))

    return resolve_references(all_defs, all_mentions)


# ── Output ───────────────────────────────────────────────────────────────────


def write_report(report: CrossRefReport, output_path: Path) -> Path:
    """Write cross-reference analysis as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    category_counts: dict[str, int] = {}
    for m in report.mentions:
        category_counts[m.category] = category_counts.get(m.category, 0) + 1

    data: dict[str, Any] = {
        "summary": {
            "total_definitions": len(report.definitions),
            "total_mentions": len(report.mentions),
            "resolved": len(report.resolved),
            "unresolved": len(report.unresolved),
            "resolution_rate": round(report.resolution_rate, 3),
            "category_counts": category_counts,
        },
        "definitions": [asdict(d) for d in report.definitions],
        "mentions": [asdict(m) for m in report.mentions],
        "resolved": report.resolved,
        "unresolved": report.unresolved,
        "canonical_ir": {
            "objects": [asdict(obj) for obj in report.objects],
            "links": [asdict(link) for link in report.links],
        },
    }
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and resolve internal cross-references in Markdown.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory")
    parser.add_argument("--output", type=Path, help="Path for cross-ref report (JSON)")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("cross_ref", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "crossref_report.json")).resolve()

    try:
        report = analyze_tree(input_path) if input_path.is_dir() else analyze_file(input_path)
        written = write_report(report, output_path)
        log.info(
            "found %d def(s), %d mention(s), %d unresolved → %s",
            len(report.definitions),
            len(report.mentions),
            len(report.unresolved),
            written,
        )
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
