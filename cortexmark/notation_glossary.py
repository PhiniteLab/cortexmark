"""Mathematical notation glossary builder for scientific Markdown.

Scans converted Markdown to extract and catalogue mathematical symbols
and notations, producing a glossary that maps each symbol to its
definition context. This is invaluable for:
  - RAG pipelines (enriching embeddings with notation meaning)
  - Readers unfamiliar with a paper's specific notation
  - Cross-document notation consistency checking

Detected notation sources:
  - Explicit definitions: "Let $X$ denote the state space"
  - "where" clauses after equations: "$V(s) = …$ where $V$ is …"
  - Notation tables / lists
  - Common mathematical conventions
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import (
    load_config,
    resolve_configured_path,
    resolve_quality_report_path,
    setup_logging,
)
from cortexmark.scientific_ir import OBJECT_NOTATION, ScientificObject, make_object_id, stable_source_label

# ── Detection patterns ───────────────────────────────────────────────────────

# Inline math: $...$
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

# Display math: $$...$$
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

# "Let $X$ denote/be/represent …"
LET_DEFINE_RE = re.compile(
    r"[Ll]et\s+\$([^$]+)\$\s+(?:denote|be|represent|stand\s+for)\s+(.+?)(?:\.|,|;|\n)",
)

# "$X$ is/denotes/represents …"
IS_DEFINE_RE = re.compile(
    r"\$([^$]+)\$\s+(?:is|denotes|represents|stands\s+for|refers\s+to)\s+"
    r"(?:the\s+|a\s+|an\s+)?(.+?)(?:\.|,|;|\n)",
)

# "where $X$ is …" (common after equations)
WHERE_DEFINE_RE = re.compile(
    r"where\s+\$([^$]+)\$\s+(?:is|denotes|represents)\s+"
    r"(?:the\s+|a\s+|an\s+)?(.+?)(?:\.|,|;|\band\b|\n)",
)

# "we define $X$ as …" or "$X$ is defined as …"
DEFINE_AS_RE = re.compile(
    r"(?:we\s+define|define)\s+\$([^$]+)\$\s+(?:as|to\s+be)\s+(.+?)(?:\.|,|;|\n)"
    r"|"
    r"\$([^$]+)\$\s+is\s+defined\s+(?:as|to\s+be)\s+(.+?)(?:\.|,|;|\n)",
    re.IGNORECASE,
)

# Notation list pattern: "- $X$: description" or "| $X$ | description |"
LIST_NOTATION_RE = re.compile(
    r"^[-*]\s+\$([^$]+)\$\s*[:–—]\s*(.+)$",
    re.MULTILINE,
)
TABLE_NOTATION_RE = re.compile(
    r"\|\s*\$([^$]+)\$\s*\|\s*(.+?)\s*\|",
)

# Common mathematical conventions (symbol → typical meaning)
COMMON_NOTATIONS: dict[str, str] = {
    r"\mathbb{R}": "set of real numbers",
    r"\mathbb{Z}": "set of integers",
    r"\mathbb{N}": "set of natural numbers",
    r"\mathbb{E}": "expectation operator",
    r"\mathbb{P}": "probability measure",
    r"\nabla": "gradient operator",
    r"\partial": "partial derivative",
    r"\sum": "summation",
    r"\prod": "product",
    r"\int": "integral",
    r"\infty": "infinity",
    r"\alpha": "learning rate / parameter",
    r"\beta": "parameter / coefficient",
    r"\gamma": "discount factor",
    r"\epsilon": "exploration rate / small value",
    r"\lambda": "regularization / eligibility trace",
    r"\theta": "model parameter vector",
    r"\pi": "policy",
    r"\mu": "mean / measure",
    r"\sigma": "standard deviation",
    r"\tau": "temperature / time constant",
    r"\omega": "angular frequency / weight",
    r"\phi": "feature function / basis",
    r"\psi": "wave function / feature",
    r"\delta": "temporal difference / small change",
    r"\Delta": "change / difference",
    r"\Omega": "sample space",
    r"\forall": "for all",
    r"\exists": "there exists",
    r"\in": "element of",
    r"\subset": "subset of",
    r"\cap": "intersection",
    r"\cup": "union",
    r"\to": "maps to / approaches",
    r"\rightarrow": "maps to / implies",
    r"\Rightarrow": "implies",
    r"\Leftrightarrow": "if and only if",
    r"\approx": "approximately equal",
    r"\sim": "distributed as / similar to",
    r"\propto": "proportional to",
    r"\leq": "less than or equal",
    r"\geq": "greater than or equal",
    r"\neq": "not equal",
    r"\arg\max": "argument that maximises",
    r"\arg\min": "argument that minimises",
    r"\max": "maximum",
    r"\min": "minimum",
    r"\sup": "supremum",
    r"\inf": "infimum",
    r"\log": "logarithm",
    r"\exp": "exponential",
    r"\lim": "limit",
}


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class NotationEntry:
    """A single notation symbol with its definition."""

    symbol: str
    definition: str
    source: str = ""  # "explicit", "where-clause", "list", "table", "convention"
    source_file: str = ""
    line_number: int = 0
    context: str = ""  # surrounding text for disambiguation
    object_id: str = ""


@dataclass
class NotationGlossary:
    """Complete notation glossary for a document or collection."""

    entries: list[NotationEntry] = field(default_factory=list)

    @property
    def unique_symbols(self) -> int:
        return len({e.symbol for e in self.entries})

    def lookup(self, symbol: str) -> list[NotationEntry]:
        """Find all entries for a given symbol."""
        return [e for e in self.entries if e.symbol == symbol]

    def deduplicated(self) -> list[NotationEntry]:
        """Return entries with duplicate symbols merged (most precise definition wins)."""
        priority = {"list": 0, "table": 0, "where-clause": 1, "explicit": 2, "convention": 3}
        seen: dict[str, NotationEntry] = {}
        for entry in self.entries:
            current = seen.get(entry.symbol)
            if current is None:
                seen[entry.symbol] = entry
                continue
            # Secondary key is the NEGATED length so that, among entries with the
            # same source priority, the longer (more descriptive) definition wins
            # -- the shorter one winning would contradict "most precise wins".
            if (priority.get(entry.source, 9), -len(entry.definition)) < (
                priority.get(current.source, 9),
                -len(current.definition),
            ):
                seen[entry.symbol] = entry
        return list(seen.values())


# ── Extraction engine ────────────────────────────────────────────────────────


def extract_explicit_definitions(text: str, source_file: str = "") -> list[NotationEntry]:
    """Extract notation from explicit 'Let X denote…' patterns."""
    entries: list[NotationEntry] = []

    for pattern, source_tag in [
        (LET_DEFINE_RE, "explicit"),
        (IS_DEFINE_RE, "explicit"),
        (WHERE_DEFINE_RE, "where-clause"),
        (DEFINE_AS_RE, "explicit"),
    ]:
        for m in pattern.finditer(text):
            groups = list(m.groups())
            # DEFINE_AS_RE has two alternative groups
            symbol = groups[0] or (groups[2] if len(groups) > 2 else None)
            definition = (groups[1] if len(groups) > 1 else None) or (groups[3] if len(groups) > 3 else None)
            if symbol and definition:
                line_num = text[: m.start()].count("\n") + 1
                entries.append(
                    NotationEntry(
                        symbol=symbol.strip(),
                        definition=definition.strip(),
                        source=source_tag,
                        source_file=source_file,
                        line_number=line_num,
                        context=m.group(0).strip()[:200],
                    )
                )

    return entries


def extract_list_notations(text: str, source_file: str = "") -> list[NotationEntry]:
    """Extract notation from Markdown lists ('- $X$: description')."""
    entries: list[NotationEntry] = []
    for m in LIST_NOTATION_RE.finditer(text):
        line_num = text[: m.start()].count("\n") + 1
        entries.append(
            NotationEntry(
                symbol=m.group(1).strip(),
                definition=m.group(2).strip(),
                source="list",
                source_file=source_file,
                line_number=line_num,
            )
        )
    return entries


def extract_table_notations(text: str, source_file: str = "") -> list[NotationEntry]:
    """Extract notation from Markdown tables ('| $X$ | description |')."""
    entries: list[NotationEntry] = []
    for m in TABLE_NOTATION_RE.finditer(text):
        definition = m.group(2).strip()
        # Skip table header separators
        if re.fullmatch(r"[-:| ]+", definition):
            continue
        line_num = text[: m.start()].count("\n") + 1
        entries.append(
            NotationEntry(
                symbol=m.group(1).strip(),
                definition=definition,
                source="table",
                source_file=source_file,
                line_number=line_num,
            )
        )
    return entries


def detect_common_notations(text: str, source_file: str = "") -> list[NotationEntry]:
    """Detect usage of well-known mathematical symbols in the text."""
    entries: list[NotationEntry] = []
    for symbol, meaning in COMMON_NOTATIONS.items():
        escaped = re.escape(symbol)
        pattern_inline = r"\$" + escaped + r"[\s$}^_,)]"
        # Mirror the inline pattern's trailing boundary (as a lookahead, since the
        # symbol may sit right before the closing "$$") so a custom macro that
        # merely starts with a known symbol name (e.g. \thetaHat vs. \theta) is
        # not misdetected as that symbol.
        pattern_display = r"\$\$[^$]*" + escaped + r"(?![a-zA-Z])"
        if re.search(pattern_inline, text) or re.search(pattern_display, text):
            entries.append(
                NotationEntry(
                    symbol=symbol,
                    definition=meaning,
                    source="convention",
                    source_file=source_file,
                )
            )
    return entries


def extract_all(text: str, source_file: str = "", *, include_conventions: bool = True) -> NotationGlossary:
    """Extract all notation entries from text."""
    entries: list[NotationEntry] = []
    entries.extend(extract_explicit_definitions(text, source_file))
    entries.extend(extract_list_notations(text, source_file))
    entries.extend(extract_table_notations(text, source_file))
    if include_conventions:
        entries.extend(detect_common_notations(text, source_file))
    return NotationGlossary(entries=entries)


# ── File / tree operations ───────────────────────────────────────────────────


def extract_from_file(file_path: Path, *, include_conventions: bool = True) -> NotationGlossary:
    """Extract notation glossary from a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    glossary = extract_all(text, source_file=stable_source_label(file_path), include_conventions=include_conventions)
    return glossary


def extract_from_tree(input_root: Path, *, include_conventions: bool = True) -> NotationGlossary:
    """Extract notation glossary from all Markdown files in a directory."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")

    all_entries: list[NotationEntry] = []
    for md_path in md_files:
        glossary = extract_all(
            md_path.read_text(encoding="utf-8"),
            source_file=stable_source_label(md_path, root=input_root),
            include_conventions=include_conventions,
        )
        all_entries.extend(glossary.entries)

    return NotationGlossary(entries=all_entries)


# ── Output ───────────────────────────────────────────────────────────────────


def build_summary(glossary: NotationGlossary) -> dict[str, Any]:
    """Build summary statistics for a notation glossary."""
    source_counts: dict[str, int] = {}
    for e in glossary.entries:
        source_counts[e.source] = source_counts.get(e.source, 0) + 1
    return {
        "total_entries": len(glossary.entries),
        "unique_symbols": glossary.unique_symbols,
        "source_counts": source_counts,
    }


def glossary_to_scientific_objects(glossary: NotationGlossary) -> list[ScientificObject]:
    """Convert notation entries into stable scholarly objects."""
    objects: list[ScientificObject] = []
    for index, entry in enumerate(glossary.entries, start=1):
        object_id = entry.object_id or make_object_id(
            entry.source_file,
            OBJECT_NOTATION,
            label=entry.symbol,
            name=entry.definition,
            line_number=entry.line_number,
            ordinal=index,
        )
        entry.object_id = object_id
        objects.append(
            ScientificObject(
                object_id=object_id,
                object_type=OBJECT_NOTATION,
                object_kind=OBJECT_NOTATION,
                label=entry.symbol,
                title=entry.symbol,
                source_file=entry.source_file,
                line_number=entry.line_number,
                text=entry.definition,
                evidence_level="explicit" if entry.source != "convention" else "convention",
                metadata={
                    "definition": entry.definition,
                    "source": entry.source,
                    "context": entry.context,
                },
            )
        )
    return objects


def write_report(glossary: NotationGlossary, output_path: Path) -> Path:
    """Write notation glossary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduped = glossary.deduplicated()
    canonical_objects = glossary_to_scientific_objects(glossary)
    data: dict[str, Any] = {
        "summary": build_summary(glossary),
        "glossary": [
            {
                "symbol": e.symbol,
                "definition": e.definition,
                "source": e.source,
                "source_file": e.source_file,
                "line_number": e.line_number,
                "object_id": e.object_id,
            }
            for e in deduped
        ],
        "canonical_ir": {
            "objects": [
                {
                    "object_id": obj.object_id,
                    "object_type": obj.object_type,
                    "label": obj.label,
                    "source_file": obj.source_file,
                    "line_number": obj.line_number,
                    "evidence_level": obj.evidence_level,
                    "metadata": obj.metadata,
                }
                for obj in canonical_objects
            ]
        },
    }
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_markdown_glossary(glossary: NotationGlossary, output_path: Path) -> Path:
    """Write notation glossary as a Markdown table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduped = glossary.deduplicated()

    lines = [
        "# Notation Glossary",
        "",
        "| Symbol | Definition | Source |",
        "|--------|-----------|--------|",
    ]
    for e in deduped:
        symbol_col = f"${e.symbol}$" if not e.symbol.startswith("$") else e.symbol
        lines.append(f"| {symbol_col} | {e.definition} | {e.source} |")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract mathematical notation glossary from Markdown documents.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory")
    parser.add_argument("--output", type=Path, help="Path for glossary report (JSON)")
    parser.add_argument("--markdown", type=Path, help="Path for Markdown glossary output")
    parser.add_argument("--no-conventions", action="store_true", help="Skip common notation detection")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("notation_glossary", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "notation_report.json")).resolve()
    include_conventions = not args.no_conventions

    try:
        if input_path.is_dir():
            glossary = extract_from_tree(input_path, include_conventions=include_conventions)
        else:
            glossary = extract_from_file(input_path, include_conventions=include_conventions)

        written = write_report(glossary, output_path)
        summary = build_summary(glossary)
        log.info(
            "found %d notation(s) (%d unique) → %s",
            summary["total_entries"],
            summary["unique_symbols"],
            written,
        )

        if args.markdown:
            md_path = write_markdown_glossary(glossary, args.markdown)
            log.info("Markdown glossary → %s", md_path)

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
