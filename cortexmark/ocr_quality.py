"""Assess already-converted Markdown with OCR quality heuristics.

Provides:
  - Confidence scoring for OCR-derived text (character-level heuristics)
  - Garbled-text detection (encoding artefacts, symbol soup)
  - Common function-word frequency for a language sanity check
  - Per-file and per-tree quality reports

The module works on already-converted Markdown (post-convert stage)
and does NOT depend on any external OCR engine at runtime.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import load_config, resolve_configured_path, resolve_quality_report_path, setup_logging

# ── Heuristic patterns ──────────────────────────────────────────────────────

# Characters that commonly appear in garbled OCR output
GARBLE_RE = re.compile(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")
# Sequences of 3+ non-ASCII punctuation / symbols (symbol soup)
SYMBOL_SOUP_RE = re.compile(r"[^\w\s.,;:!?(){}\[\]\"'`\-/\\@#$%&*+=<>]{3,}")
# Repeated character runs (e.g. "aaaaaa") - likely artefacts
REPEAT_RE = re.compile(r"(.)\1{4,}")

# Common English / Turkish function words for language sanity check
COMMON_WORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "and",
        "or",
        "but",
        "not",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "if",
        "it",
        "this",
        "that",
        "these",
        "those",
        "bir",
        "ve",
        "ya",
        "da",
        "ile",
        "de",
        "bu",
        "o",
        "ne",
        "kadar",
        "gibi",
        "için",
        "olan",
        "hem",
        "her",
    ]
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class OCRQualityMetrics:
    """Quality metrics for a single document."""

    char_count: int = 0
    word_count: int = 0
    garble_count: int = 0
    symbol_soup_count: int = 0
    repeat_artefacts: int = 0
    short_noise_lines: int = 0
    common_word_hits: int = 0
    confidence: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class OCRFileReport:
    """Report for a single file."""

    source_file: str
    metrics: OCRQualityMetrics
    grade: str  # A, B, C, D, F


# ── Scoring engine ───────────────────────────────────────────────────────────


def count_garble_chars(text: str) -> int:
    """Count replacement characters and control codes."""
    return len(GARBLE_RE.findall(text))


def count_symbol_soup(text: str) -> int:
    """Count sequences of 3+ unusual symbols."""
    return len(SYMBOL_SOUP_RE.findall(text))


def count_repeat_artefacts(text: str) -> int:
    """Count suspicious character repetitions (5+)."""
    return len(REPEAT_RE.findall(text))


def count_short_noise_lines(text: str) -> int:
    """Count lines that look like OCR noise."""
    return sum(
        1
        for line in text.split("\n")
        if 1 <= len(stripped := line.strip()) <= 5 and not any(char.isalpha() for char in stripped)
    )


def count_common_words(text: str) -> int:
    """Count recognised common function words (language sanity)."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    return sum(1 for w in words if w in COMMON_WORDS)


def assess_quality(text: str, source: str = "") -> OCRQualityMetrics:
    """Compute OCR quality metrics for a text."""
    chars = len(text)
    words = len(text.split())
    garble = count_garble_chars(text)
    soup = count_symbol_soup(text)
    repeats = count_repeat_artefacts(text)
    noise = count_short_noise_lines(text)
    common = count_common_words(text)

    issues: list[str] = []

    # Confidence starts at 1.0, penalties reduce it
    confidence = 1.0

    meaningful_chars = sum(char.isalpha() for char in text)
    if meaningful_chars < 10:
        return OCRQualityMetrics(
            char_count=chars,
            word_count=words,
            garble_count=garble,
            symbol_soup_count=soup,
            repeat_artefacts=repeats,
            short_noise_lines=noise,
            common_word_hits=common,
            confidence=0.0,
            issues=["empty or near-empty document"],
        )

    # Garble penalty
    garble_ratio = garble / chars
    if garble_ratio > 0.01:
        confidence -= min(garble_ratio * 10, 0.4)
        issues.append(f"garble ratio {garble_ratio:.2%}")

    # Symbol soup penalty
    if soup > 0:
        soup_penalty = min(soup * 0.05, 0.3)
        confidence -= soup_penalty
        issues.append(f"{soup} symbol-soup sequences")

    # Repeat artefact penalty
    if repeats > 2:
        confidence -= min(repeats * 0.03, 0.2)
        issues.append(f"{repeats} repeat artefacts")

    # Noise line penalty
    total_lines = max(text.count("\n"), 1)
    noise_ratio = noise / total_lines
    if noise_ratio > 0.1:
        confidence -= min(noise_ratio * 0.5, 0.2)
        issues.append(f"noise line ratio {noise_ratio:.1%}")

    # Common word bonus (indicates real language content)
    if words > 10:
        cw_ratio = common / words
        if cw_ratio < 0.05:
            confidence -= 0.15
            issues.append(f"low common-word ratio {cw_ratio:.1%}")

    confidence = max(round(confidence, 3), 0.0)

    return OCRQualityMetrics(
        char_count=chars,
        word_count=words,
        garble_count=garble,
        symbol_soup_count=soup,
        repeat_artefacts=repeats,
        short_noise_lines=noise,
        common_word_hits=common,
        confidence=confidence,
        issues=issues,
    )


def confidence_to_grade(confidence: float) -> str:
    """Map a 0-1 confidence score to a letter grade."""
    if confidence >= 0.9:
        return "A"
    if confidence >= 0.75:
        return "B"
    if confidence >= 0.6:
        return "C"
    if confidence >= 0.4:
        return "D"
    return "F"


# ── File / tree operations ───────────────────────────────────────────────────


def assess_file(file_path: Path) -> OCRFileReport:
    """Assess a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8", errors="replace")
    metrics = assess_quality(text, source=str(file_path))
    return OCRFileReport(
        source_file=str(file_path),
        metrics=metrics,
        grade=confidence_to_grade(metrics.confidence),
    )


def assess_tree(input_root: Path) -> list[OCRFileReport]:
    """Assess all Markdown files in a directory tree."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")
    return [assess_file(f) for f in md_files]


def write_ocr_report(reports: list[OCRFileReport], output_path: Path) -> Path:
    """Write OCR quality report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grade_dist: dict[str, int] = {}
    total_confidence = 0.0
    for r in reports:
        grade_dist[r.grade] = grade_dist.get(r.grade, 0) + 1
        total_confidence += r.metrics.confidence

    avg_confidence = total_confidence / len(reports) if reports else 0.0

    data: dict[str, Any] = {
        "summary": {
            "files_scanned": len(reports),
            "average_confidence": round(avg_confidence, 3),
            "grade_distribution": grade_dist,
        },
        "files": [asdict(r) for r in reports],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess OCR quality of converted Markdown files.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory")
    parser.add_argument("--output", type=Path, help="Path for quality report (JSON)")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("ocr_quality", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_raw_md", "outputs/raw_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "ocr_report.json")).resolve()

    try:
        reports = assess_tree(input_path) if input_path.is_dir() else [assess_file(input_path)]

        written = write_ocr_report(reports, output_path)

        grade_counts: dict[str, int] = {}
        for r in reports:
            grade_counts[r.grade] = grade_counts.get(r.grade, 0) + 1
        log.info(
            "assessed %d file(s): %s -> %s",
            len(reports),
            ", ".join(f"{g}={c}" for g, c in sorted(grade_counts.items())),
            written,
        )

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
