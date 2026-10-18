"""Document-type templates: detect document type and apply type-specific processing.

Supported document types:
  - paper:      Academic paper (abstract, keywords, citations, sections)
  - textbook:   Textbook chapter (chapter headings, examples, exercises)
  - syllabus:   Course syllabus (weeks, assignments, grading)
  - slides:     Presentation slides (short sections, bullet-heavy)
  - report:     Technical report (executive summary, numbered sections)
  - generic:    Default fallback

Each type provides:
  - Detection heuristics (regex + structural analysis)
  - Type-specific Markdown template
  - Extracted structural metadata
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import (
    get_path_settings,
    load_config,
    resolve_configured_path,
    resolve_quality_report_path,
    setup_logging,
)

# ── Document types ───────────────────────────────────────────────────────────

PAPER = "paper"
TEXTBOOK = "textbook"
SYLLABUS = "syllabus"
SLIDES = "slides"
REPORT = "report"
GENERIC = "generic"

ALL_TYPES = [PAPER, TEXTBOOK, SYLLABUS, SLIDES, REPORT, GENERIC]

# ── Detection patterns ───────────────────────────────────────────────────────

ABSTRACT_RE = re.compile(r"\b(?:abstract|summary)\b", re.IGNORECASE)
KEYWORDS_RE = re.compile(r"\b(?:keywords?|key\s*words?)\b", re.IGNORECASE)
DOI_RE = re.compile(r"\b10\.\d{4,9}/")
REFERENCES_RE = re.compile(r"^\s*#{1,3}\s+(?:references|bibliography)\b", re.IGNORECASE | re.MULTILINE)
CHAPTER_RE = re.compile(r"^\s*#{1,2}\s+chapter\s+\d+", re.IGNORECASE | re.MULTILINE)
EXAMPLE_RE = re.compile(r"\b(?:example|exercise|problem|definition|theorem|proof)\s+\d+", re.IGNORECASE)
WEEK_RE = re.compile(r"\b(?:week|hafta)\s+\d+", re.IGNORECASE)
GRADING_RE = re.compile(r"\b(?:grading|assessment|evaluation|midterm|final\s+exam)\b", re.IGNORECASE)
SLIDE_MARKER_RE = re.compile(r"(?:^---$|^#{1,2}\s+slide\b)", re.IGNORECASE | re.MULTILINE)
EXEC_SUMMARY_RE = re.compile(r"^\s*#{1,3}\s+(?:executive\s+summary|introduction)\b", re.IGNORECASE | re.MULTILINE)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class DocTypeResult:
    """Result of document type detection."""

    doc_type: str
    confidence: float  # 0.0 - 1.0
    signals: list[str] = field(default_factory=list)
    source_file: str = ""


@dataclass
class DocTemplate:
    """Template structure for a document type."""

    doc_type: str
    sections: list[str] = field(default_factory=list)
    description: str = ""


# ── Detection engine ─────────────────────────────────────────────────────────


def _count_pattern(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def _avg_paragraph_length(text: str) -> float:
    """Average paragraph length (chars) — short = slides-like."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return 0.0
    return sum(len(p) for p in paragraphs) / len(paragraphs)


def _bullet_ratio(text: str) -> float:
    """Fraction of lines that start with a bullet."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return 0.0
    bullet_lines = sum(1 for line in lines if line.startswith(("- ", "* ")) or re.match(r"^\d+\.\s", line))
    return bullet_lines / len(lines)


def detect_paper(text: str) -> tuple[float, list[str]]:
    """Compute paper-likeness score."""
    score = 0.0
    signals: list[str] = []
    if ABSTRACT_RE.search(text):
        score += 0.3
        signals.append("has abstract")
    if KEYWORDS_RE.search(text):
        score += 0.15
        signals.append("has keywords")
    if DOI_RE.search(text):
        score += 0.2
        signals.append("has DOI")
    if REFERENCES_RE.search(text):
        score += 0.25
        signals.append("has references section")
    if _count_pattern(text, re.compile(r"\[\d+\]")) >= 3:
        score += 0.1
        signals.append("has citation markers")
    return min(score, 1.0), signals


def detect_textbook(text: str) -> tuple[float, list[str]]:
    """Compute textbook-likeness score."""
    score = 0.0
    signals: list[str] = []
    chapter_count = _count_pattern(text, CHAPTER_RE)
    if chapter_count >= 1:
        score += 0.3
        signals.append(f"{chapter_count} chapter heading(s)")
    example_count = _count_pattern(text, EXAMPLE_RE)
    if example_count >= 2:
        score += 0.3
        signals.append(f"{example_count} examples/exercises")
    if len(text) > 10000:
        score += 0.2
        signals.append("long document")
    if _count_pattern(text, re.compile(r"^\s*#{1,3}\s+", re.MULTILINE)) >= 10:
        score += 0.2
        signals.append("many headings")
    return min(score, 1.0), signals


def detect_syllabus(text: str) -> tuple[float, list[str]]:
    """Compute syllabus-likeness score."""
    score = 0.0
    signals: list[str] = []
    week_count = _count_pattern(text, WEEK_RE)
    if week_count >= 3:
        score += 0.4
        signals.append(f"{week_count} week references")
    if GRADING_RE.search(text):
        score += 0.3
        signals.append("has grading/assessment")
    if re.search(r"\b(?:instructor|professor|lecturer)\b", text, re.IGNORECASE):
        score += 0.15
        signals.append("mentions instructor")
    if re.search(r"\b(?:course|ders|class)\b", text, re.IGNORECASE):
        score += 0.15
        signals.append("mentions course")
    return min(score, 1.0), signals


def detect_slides(text: str) -> tuple[float, list[str]]:
    """Compute slides-likeness score."""
    score = 0.0
    signals: list[str] = []
    avg_para = _avg_paragraph_length(text)
    if avg_para < 150 and avg_para > 0:
        score += 0.3
        signals.append(f"short paragraphs (avg {avg_para:.0f} chars)")
    bullet_r = _bullet_ratio(text)
    if bullet_r > 0.4:
        score += 0.3
        signals.append(f"high bullet ratio ({bullet_r:.0%})")
    slide_count = _count_pattern(text, SLIDE_MARKER_RE)
    if slide_count >= 3:
        score += 0.3
        signals.append(f"{slide_count} slide markers")
    if len(text) < 5000 and _count_pattern(text, re.compile(r"^#{1,2}\s+", re.MULTILINE)) >= 5:
        score += 0.1
        signals.append("short doc with many headings")
    return min(score, 1.0), signals


def detect_report(text: str) -> tuple[float, list[str]]:
    """Compute report-likeness score."""
    score = 0.0
    signals: list[str] = []
    if EXEC_SUMMARY_RE.search(text):
        score += 0.3
        signals.append("has executive summary/introduction")
    numbered = _count_pattern(text, re.compile(r"^\s*\d+\.\d*\s+", re.MULTILINE))
    if numbered >= 3:
        score += 0.3
        signals.append(f"{numbered} numbered sections")
    if re.search(r"\b(?:conclusion|recommendation|findings)\b", text, re.IGNORECASE):
        score += 0.2
        signals.append("has conclusion/recommendations")
    if 3000 < len(text) < 30000:
        score += 0.2
        signals.append("medium-length document")
    return min(score, 1.0), signals


def detect_type(text: str, source_file: str = "") -> DocTypeResult:
    """Detect the most likely document type."""
    detectors = {
        PAPER: detect_paper,
        TEXTBOOK: detect_textbook,
        SYLLABUS: detect_syllabus,
        SLIDES: detect_slides,
        REPORT: detect_report,
    }

    best_type = GENERIC
    best_score = 0.0
    best_signals: list[str] = []

    for doc_type, detector in detectors.items():
        score, signals = detector(text)
        if score > best_score:
            best_type = doc_type
            best_score = score
            best_signals = signals

    # Require minimum confidence to avoid false positives
    if best_score < 0.3:
        best_type = GENERIC
        best_signals = ["no strong signals detected"]

    return DocTypeResult(
        doc_type=best_type,
        confidence=round(best_score, 2),
        signals=best_signals,
        source_file=source_file,
    )


# ── Template definitions ─────────────────────────────────────────────────────

TEMPLATES: dict[str, DocTemplate] = {
    PAPER: DocTemplate(
        doc_type=PAPER,
        sections=[
            "Title",
            "Authors",
            "Abstract",
            "Keywords",
            "Introduction",
            "Methods",
            "Results",
            "Discussion",
            "Conclusion",
            "References",
        ],
        description="Academic paper with structured sections and citations",
    ),
    TEXTBOOK: DocTemplate(
        doc_type=TEXTBOOK,
        sections=["Chapter Title", "Learning Objectives", "Content Sections", "Examples", "Exercises", "Summary"],
        description="Textbook chapter with examples and exercises",
    ),
    SYLLABUS: DocTemplate(
        doc_type=SYLLABUS,
        sections=["Course Title", "Instructor", "Schedule", "Grading", "Policies", "Weekly Plan"],
        description="Course syllabus with schedule and grading",
    ),
    SLIDES: DocTemplate(
        doc_type=SLIDES,
        sections=["Title Slide", "Outline", "Content Slides", "Summary", "Questions"],
        description="Presentation slides with concise bullet points",
    ),
    REPORT: DocTemplate(
        doc_type=REPORT,
        sections=["Executive Summary", "Introduction", "Methodology", "Findings", "Recommendations", "Conclusion"],
        description="Technical report with numbered sections",
    ),
    GENERIC: DocTemplate(
        doc_type=GENERIC,
        sections=["Title", "Content"],
        description="Generic document without specific structure",
    ),
}


def get_template(doc_type: str) -> DocTemplate:
    """Get the template for a document type."""
    return TEMPLATES.get(doc_type, TEMPLATES[GENERIC])


def render_template_scaffold(template: DocTemplate) -> str:
    """Render a Markdown scaffold from a template."""
    lines = [f"<!-- Document Type: {template.doc_type} -->", f"<!-- {template.description} -->", ""]
    for section in template.sections:
        lines.append(f"## {section}")
        lines.append("")
        lines.append(f"<!-- TODO: Fill {section} content -->")
        lines.append("")
    return "\n".join(lines)


# ── File / tree operations ───────────────────────────────────────────────────


def detect_file(file_path: Path) -> DocTypeResult:
    """Detect document type for a single file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    return detect_type(text, source_file=str(file_path))


def detect_tree(input_root: Path) -> list[DocTypeResult]:
    """Detect document types for all Markdown files under a directory."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")
    return [detect_file(f) for f in md_files]


def write_detection_report(results: list[DocTypeResult], output_path: Path) -> Path:
    """Write detection results as a JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    type_dist: dict[str, int] = {}
    for r in results:
        type_dist[r.doc_type] = type_dist.get(r.doc_type, 0) + 1

    data: dict[str, Any] = {
        "summary": {
            "files_scanned": len(results),
            "type_distribution": type_dist,
        },
        "files": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect document types and apply type-specific templates.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory")
    parser.add_argument("--output", type=Path, help="Path for detection report (JSON)")
    parser.add_argument(
        "--scaffold",
        type=str,
        choices=ALL_TYPES,
        help="Generate a template scaffold for this type",
    )
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("doctype", cfg)

    if args.scaffold:
        template = get_template(args.scaffold)
        scaffold = render_template_scaffold(template)
        out_path = args.output or get_path_settings(cfg).outputs_dir / "templates" / f"{args.scaffold}_scaffold.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(scaffold, encoding="utf-8")
        log.info("wrote scaffold for %s → %s", args.scaffold, out_path)
        return 0

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "doctype_report.json")).resolve()

    try:
        results = detect_tree(input_path) if input_path.is_dir() else [detect_file(input_path)]

        written = write_detection_report(results, output_path)

        # Log type distribution
        type_counts: dict[str, int] = {}
        for r in results:
            type_counts[r.doc_type] = type_counts.get(r.doc_type, 0) + 1
        log.info(
            "scanned %d file(s): %s → %s",
            len(results),
            ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items())),
            written,
        )

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
