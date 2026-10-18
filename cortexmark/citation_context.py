"""Citation context extraction and classification for scientific Markdown.

Extends basic citation extraction (citations.py) with:
  - Surrounding sentence extraction for each citation
  - Citation purpose classification (foundational, comparative, methodological, etc.)
  - Purpose phrase detection from linguistic cues
  - Self-citation detection via author overlap with document metadata
  - Co-citation analysis (citations appearing in the same sentence)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cortexmark.common import load_config, resolve_configured_path, resolve_quality_report_path, setup_logging

# ── Constants ─────────────────────────────────────────────────────────────────

# Citation purpose categories
PURPOSE_FOUNDATIONAL = "foundational"
PURPOSE_COMPARATIVE = "comparative"
PURPOSE_METHODOLOGICAL = "methodological"
PURPOSE_EXTENDING = "extending"
PURPOSE_BACKGROUND = "background"
PURPOSE_REFUTING = "refuting"
PURPOSE_UNKNOWN = "unknown"

ALL_PURPOSES: tuple[str, ...] = (
    PURPOSE_FOUNDATIONAL,
    PURPOSE_COMPARATIVE,
    PURPOSE_METHODOLOGICAL,
    PURPOSE_EXTENDING,
    PURPOSE_BACKGROUND,
    PURPOSE_REFUTING,
    PURPOSE_UNKNOWN,
)

# ── Purpose phrase patterns ──────────────────────────────────────────────────
# Each pattern maps to a citation purpose. Matched against the sentence
# containing the citation (case-insensitive).

_PURPOSE_PHRASES: list[tuple[str, str]] = [
    # Foundational — the work is a basis for the current work
    (r"(?:based on|builds? upon|following|grounded in|rooted in|rely(?:ing)? on)", PURPOSE_FOUNDATIONAL),
    (r"(?:pioneered by|introduced by|originally proposed|seminal work)", PURPOSE_FOUNDATIONAL),
    (r"(?:fundamental|foundational) (?:work|result|contribution)", PURPOSE_FOUNDATIONAL),
    # Comparative — comparing with or contrasting against
    (r"(?:in contrast (?:to|with)|unlike|differs? from|compared (?:to|with))", PURPOSE_COMPARATIVE),
    (r"(?:outperforms?|superior to|better than|worse than|comparable to)", PURPOSE_COMPARATIVE),
    (r"(?:as opposed to|on the other hand|alternatively)", PURPOSE_COMPARATIVE),
    # Methodological — using method/approach/tool from the cited work
    (r"(?:using the (?:method|approach|algorithm|framework|technique) (?:of|from|in))", PURPOSE_METHODOLOGICAL),
    (
        r"(?:we (?:use|adopt|employ|apply|follow|implement) (?:the )?(?:method|approach|algorithm|framework))",
        PURPOSE_METHODOLOGICAL,
    ),
    (r"(?:as described (?:in|by)|following the procedure)", PURPOSE_METHODOLOGICAL),
    # Extending — extending or improving upon the cited work
    (r"(?:we extend|we improve|we generali[sz]e|building on|extending)", PURPOSE_EXTENDING),
    (r"(?:our (?:extension|improvement|generali[sz]ation) of)", PURPOSE_EXTENDING),
    (r"(?:an extension of|a generali[sz]ation of|improves? (?:upon|on))", PURPOSE_EXTENDING),
    # Background — providing general context
    (r"(?:has been (?:widely |extensively )?studied|is well[- ]known|see (?:e\.g\.|for example))", PURPOSE_BACKGROUND),
    (r"(?:for (?:a |an )?(?:overview|survey|review|introduction) see)", PURPOSE_BACKGROUND),
    (r"(?:there (?:is|has been) (?:a )?growing|literature on|recent (?:work|studies))", PURPOSE_BACKGROUND),
    # Refuting — disagreeing with or challenging the cited work
    (r"(?:we disagree with|in contradiction|contrary to|challenges? the)", PURPOSE_REFUTING),
    (r"(?:disproves?|refutes?|invalidates?|fails? to)", PURPOSE_REFUTING),
]

PURPOSE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, re.IGNORECASE), purpose) for pat, purpose in _PURPOSE_PHRASES
]

# Citation detection (inline parenthetical and bracketed). The comma before the
# year is optional (",?" not ",") to match citations.py's PAREN_CITE_RE, so
# "(Smith 2020)" (no comma) is detected the same way by both sibling analyzers
# instead of citation_context.py silently under-counting it.
PAREN_CITE_RE = re.compile(
    r"\(([A-Z][a-zA-Z'-]+(?:\s+et\s+al\.?|(?:\s+(?:and|&)\s+[A-Z][a-zA-Z'-]+))*"
    r"(?:\s*,?\s*\d{4}[a-z]?))"
    r"(?:\s*;\s*[A-Z][a-zA-Z'-]+(?:\s+et\s+al\.?|(?:\s+(?:and|&)\s+[A-Z][a-zA-Z'-]+))*"
    r"(?:\s*,?\s*\d{4}[a-z]?))*\)"
)
# (?<!\]) skips reference-style "[text][1]"; (?!\() skips inline Markdown links
# "[1](url)" -- neither is a genuine numeric citation.
BRACKET_CITE_RE = re.compile(r"(?<!\])\[(\d+(?:\s*[,;]\s*\d+)*)\](?!\()")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class CitationContext:
    """A citation with its surrounding context and classified purpose."""

    raw_text: str
    cite_type: str  # "author-year" | "numeric"
    sentence: str
    purpose: str  # one of ALL_PURPOSES
    purpose_confidence: float  # 0.0 - 1.0
    line: int = 0
    source_file: str = ""


@dataclass
class CoCitation:
    """A pair of citations that appear in the same sentence."""

    cite_a: str
    cite_b: str
    sentence: str
    count: int = 1


@dataclass
class SelfCitation:
    """A citation identified as likely self-citation."""

    raw_text: str
    matching_author: str
    line: int = 0


@dataclass
class FileContextReport:
    """Citation context analysis for a single file."""

    file: str
    total_citations: int = 0
    contexts: list[CitationContext] = field(default_factory=list)
    co_citations: list[CoCitation] = field(default_factory=list)
    self_citations: list[SelfCitation] = field(default_factory=list)
    purpose_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class ContextSummary:
    """Aggregate summary across all files."""

    files_scanned: int = 0
    total_citations: int = 0
    total_self_citations: int = 0
    total_co_citation_pairs: int = 0
    purpose_distribution: dict[str, int] = field(default_factory=dict)


# ── Core functions ───────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping citation-bearing context intact."""
    # Collapse newlines within paragraphs
    collapsed = re.sub(r"\n(?!\n)", " ", text)
    sentences = SENTENCE_SPLIT_RE.split(collapsed)
    return [s.strip() for s in sentences if s.strip()]


def classify_purpose(sentence: str) -> tuple[str, float]:
    """Classify the purpose of a citation from its surrounding sentence.

    Returns ``(purpose, confidence)`` where confidence is between 0 and 1.
    """
    for pattern, purpose in PURPOSE_PATTERNS:
        if pattern.search(sentence):
            return purpose, 0.8
    return PURPOSE_UNKNOWN, 0.2


def extract_citation_contexts(text: str, source_file: str = "") -> list[CitationContext]:
    """Extract all citations with their sentence context and purpose."""
    sentences = _split_sentences(text)
    collapsed = re.sub(r"\n(?!\n)", " ", text)
    contexts: list[CitationContext] = []

    # Build a cumulative char offset for line-number computation
    lines = text.split("\n")
    char_offsets: list[int] = []
    offset = 0
    for line in lines:
        char_offsets.append(offset)
        offset += len(line) + 1  # +1 for newline

    def _line_for_pos(pos: int) -> int:
        for i in range(len(char_offsets) - 1, -1, -1):
            if pos >= char_offsets[i]:
                return i + 1
        return 1  # pragma: no cover

    sentence_search_start = 0
    for sentence in sentences:
        cite_matches: list[tuple[str, str, int]] = []

        for m in PAREN_CITE_RE.finditer(sentence):
            cite_matches.append((m.group(0), "author-year", m.start()))
        for m in BRACKET_CITE_RE.finditer(sentence):
            cite_matches.append((m.group(0), "numeric", m.start()))

        if not cite_matches:
            continue

        purpose, confidence = classify_purpose(sentence)
        sent_pos = collapsed.find(sentence, sentence_search_start)
        if sent_pos < 0:
            sent_pos = collapsed.find(sentence)
        if sent_pos >= 0:
            sentence_search_start = sent_pos + len(sentence)

        for raw, cite_type, cite_pos in cite_matches:
            line_no = _line_for_pos(sent_pos + cite_pos) if sent_pos >= 0 else 0
            contexts.append(
                CitationContext(
                    raw_text=raw,
                    cite_type=cite_type,
                    sentence=sentence[:500],  # cap length
                    purpose=purpose,
                    purpose_confidence=confidence,
                    line=line_no,
                    source_file=source_file,
                )
            )

    return contexts


def detect_co_citations(contexts: list[CitationContext]) -> list[CoCitation]:
    """Find citations that appear in the same sentence (co-citation pairs)."""
    sentence_groups: dict[str, list[str]] = {}
    for ctx in contexts:
        key = ctx.sentence[:200]  # normalise key
        sentence_groups.setdefault(key, []).append(ctx.raw_text)

    pairs: Counter[tuple[str, str]] = Counter()
    pair_sentences: dict[tuple[str, str], str] = {}
    for sent_key, cites in sentence_groups.items():
        unique = sorted(set(cites))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair = (unique[i], unique[j])
                pairs[pair] += 1
                pair_sentences.setdefault(pair, sent_key)

    return [
        CoCitation(cite_a=a, cite_b=b, sentence=pair_sentences[(a, b)][:300], count=count)
        for (a, b), count in pairs.most_common()
    ]


def detect_self_citations(
    contexts: list[CitationContext],
    document_authors: list[str] | None = None,
) -> list[SelfCitation]:
    """Detect citations that reference any of the document's own authors.

    If *document_authors* is empty, returns an empty list.
    """
    if not document_authors:
        return []

    # Normalise author surnames to lower
    author_surnames: set[str] = set()
    for name in document_authors:
        parts = name.strip().split()
        if parts:
            author_surnames.add(parts[-1].lower().rstrip(".,"))

    self_cites: list[SelfCitation] = []
    for ctx in contexts:
        if ctx.cite_type != "author-year":
            continue
        # Extract first surname from citation text
        cite_text = ctx.raw_text.strip("()")
        first_name = cite_text.split(",")[0].split(" and ")[0].split(" & ")[0].strip()
        tokens = first_name.split()
        matching_author = tokens[0].rstrip(".,") if tokens else ""
        surname = matching_author.lower()
        if surname in author_surnames:
            self_cites.append(SelfCitation(raw_text=ctx.raw_text, matching_author=matching_author, line=ctx.line))

    return self_cites


# ── File / tree operations ───────────────────────────────────────────────────


def _extract_authors_from_text(text: str) -> list[str]:
    """Heuristic: extract author names from a Markdown document header."""
    # Look for "Author:" or "Authors:" line, or second line after title
    author_line_re = re.compile(r"^(?:\*\*)?Authors?(?:\*\*)?[:\s]+(.+)", re.IGNORECASE | re.MULTILINE)
    m = author_line_re.search(text[:2000])
    if m:
        raw = m.group(1).strip().rstrip("*")
        # Split on comma, "and", "&"
        names = re.split(r"\s*(?:,\s*(?:and\s+)?|(?:\s+and\s+)|&)\s*", raw)
        return [n.strip() for n in names if n.strip()]
    return []


def analyze_file(file_path: Path, document_authors: list[str] | None = None) -> FileContextReport:
    """Analyse citation contexts in a single file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")

    if document_authors is None:
        document_authors = _extract_authors_from_text(text)

    source = str(file_path)
    contexts = extract_citation_contexts(text, source_file=source)
    co_cites = detect_co_citations(contexts)
    self_cites = detect_self_citations(contexts, document_authors)

    purpose_dist: dict[str, int] = {}
    for ctx in contexts:
        purpose_dist[ctx.purpose] = purpose_dist.get(ctx.purpose, 0) + 1

    return FileContextReport(
        file=source,
        total_citations=len(contexts),
        contexts=contexts,
        co_citations=co_cites,
        self_citations=self_cites,
        purpose_distribution=purpose_dist,
    )


def analyze_tree(input_root: Path, document_authors: list[str] | None = None) -> list[FileContextReport]:
    """Analyse citation contexts across all Markdown files in a tree."""
    results: list[FileContextReport] = []
    if not input_root.exists():
        return results
    if input_root.is_file():
        return [analyze_file(input_root, document_authors)]
    for md_file in sorted(input_root.rglob("*.md")):
        results.append(analyze_file(md_file, document_authors))
    return results


def build_summary(reports: list[FileContextReport]) -> ContextSummary:
    """Compute aggregate context statistics."""
    purpose_dist: dict[str, int] = {}
    total_self = 0
    total_co = 0
    total_cites = 0

    for r in reports:
        total_cites += r.total_citations
        total_self += len(r.self_citations)
        total_co += len(r.co_citations)
        for purpose, count in r.purpose_distribution.items():
            purpose_dist[purpose] = purpose_dist.get(purpose, 0) + count

    return ContextSummary(
        files_scanned=len(reports),
        total_citations=total_cites,
        total_self_citations=total_self,
        total_co_citation_pairs=total_co,
        purpose_distribution=purpose_dist,
    )


def write_report(reports: list[FileContextReport], summary: ContextSummary, output_path: Path) -> Path:
    """Write citation context report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "summary": asdict(summary),
        "files": [
            {
                "file": r.file,
                "total_citations": r.total_citations,
                "self_citations": len(r.self_citations),
                "co_citation_pairs": len(r.co_citations),
                "purpose_distribution": r.purpose_distribution,
            }
            for r in reports
        ],
    }
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract citation contexts and classify purposes.")
    p.add_argument("--input", type=Path, help="Markdown file or directory")
    p.add_argument("--output", type=Path, help="Output JSON report path")
    p.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    log = setup_logging("citation_context", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "citation_context.json")).resolve()

    log.info("analysing citation contexts in %s", input_path)
    reports = analyze_tree(input_path)
    summary = build_summary(reports)
    written = write_report(reports, summary, output_path)
    log.info(
        "scanned %d files, %d citations (%d self, %d co-citation pairs) → %s",
        summary.files_scanned,
        summary.total_citations,
        summary.total_self_citations,
        summary.total_co_citation_pairs,
        written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
