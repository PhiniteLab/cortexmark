"""Formula fidelity scoring for converted Markdown files.

Scans Markdown for:
  - Recovered equation blockquotes  (> Equation ...)
  - Incomplete formula markers
  - Formula placeholders that were never replaced
  - Algorithm code blocks

Produces a per-file quality report with a fidelity score.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import load_config, resolve_configured_path, resolve_quality_report_path, setup_logging

# ── Detection patterns ───────────────────────────────────────────────────────

RECOVERED_EQUATION_RE = re.compile(
    r"^>\s*Equation\s*\(recovered from PDF\):\s*\n>\s*\n>\s*(.+)$",
    re.MULTILINE,
)
INCOMPLETE_EQUATION_RE = re.compile(
    r"^>\s*Equation present in PDF,\s+but text recovery was incomplete\.",
    re.MULTILINE,
)
PLACEHOLDER_RE = re.compile(r"<!--\s*formula-not-decoded\s*-->")
ALGORITHM_BLOCK_RE = re.compile(r"```text\n(.*?)```", re.DOTALL)

# Heuristic LaTeX validity: must contain at least one operator-like symbol
LATEX_OPERATOR_RE = re.compile(r"[=+\-*/∑∏∫∂∇≤≥≠≈∈∀∃λαβγδεζηθικμνξπρστφχψω]", re.IGNORECASE)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class FormulaDetail:
    """Single detected formula with quality info."""

    kind: str  # "recovered" | "incomplete" | "placeholder" | "algorithm"
    text: str = ""
    valid: bool = True
    issues: list[str] = field(default_factory=list)


@dataclass
class FileReport:
    """Quality report for one Markdown file."""

    file: str
    recovered_count: int = 0
    incomplete_count: int = 0
    placeholder_count: int = 0
    algorithm_count: int = 0
    valid_count: int = 0
    total_count: int = 0
    fidelity_score: float = 0.0
    formulas: list[FormulaDetail] = field(default_factory=list)


# ── Validation helpers ───────────────────────────────────────────────────────


def validate_formula_text(text: str) -> tuple[bool, list[str]]:
    """Run heuristic checks on recovered formula text.

    Returns (is_valid, list_of_issues).
    """
    issues: list[str] = []
    stripped = text.strip()

    if not stripped:
        return False, ["empty formula text"]

    # Must have at least one math-like symbol
    if not LATEX_OPERATOR_RE.search(stripped):
        issues.append("no mathematical operators detected")

    # Check balanced parentheses / brackets
    for open_c, close_c, name in [("(", ")", "parentheses"), ("[", "]", "brackets"), ("{", "}", "braces")]:
        if stripped.count(open_c) != stripped.count(close_c):
            issues.append(f"unbalanced {name}")

    is_valid = len(issues) == 0
    return is_valid, issues


# ── Scoring engine ───────────────────────────────────────────────────────────


def score_markdown(text: str) -> list[FormulaDetail]:
    """Extract and validate all formula-like constructs from Markdown text."""
    formulas: list[FormulaDetail] = []

    # Recovered equations
    for m in RECOVERED_EQUATION_RE.finditer(text):
        formula_text = m.group(1).strip()
        valid, issues = validate_formula_text(formula_text)
        formulas.append(FormulaDetail(kind="recovered", text=formula_text, valid=valid, issues=issues))

    # Incomplete equations
    for _ in INCOMPLETE_EQUATION_RE.finditer(text):
        formulas.append(FormulaDetail(kind="incomplete", text="", valid=False, issues=["recovery failed"]))

    # Unresolved placeholders
    for _ in PLACEHOLDER_RE.finditer(text):
        formulas.append(FormulaDetail(kind="placeholder", text="", valid=False, issues=["never decoded"]))

    # Algorithm blocks
    for m in ALGORITHM_BLOCK_RE.finditer(text):
        algo_text = m.group(1).strip()
        formulas.append(FormulaDetail(kind="algorithm", text=algo_text[:200], valid=True, issues=[]))

    return formulas


def build_file_report(file_path: Path, text: str) -> FileReport:
    """Build a complete quality report for one file."""
    formulas = score_markdown(text)
    recovered = sum(1 for f in formulas if f.kind == "recovered")
    incomplete = sum(1 for f in formulas if f.kind == "incomplete")
    placeholders = sum(1 for f in formulas if f.kind == "placeholder")
    algorithms = sum(1 for f in formulas if f.kind == "algorithm")
    valid = sum(1 for f in formulas if f.valid)
    total = len(formulas)
    fidelity = (valid / total * 100.0) if total > 0 else 100.0

    return FileReport(
        file=str(file_path),
        recovered_count=recovered,
        incomplete_count=incomplete,
        placeholder_count=placeholders,
        algorithm_count=algorithms,
        valid_count=valid,
        total_count=total,
        fidelity_score=round(fidelity, 1),
        formulas=formulas,
    )


def score_file(file_path: Path) -> FileReport:
    """Score a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    return build_file_report(file_path, text)


def score_tree(input_root: Path) -> list[FileReport]:
    """Score all Markdown files under a directory tree."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")
    return [score_file(f) for f in md_files]


def write_report(reports: list[FileReport], output_path: Path) -> Path:
    """Write quality report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "summary": {
            "files_scanned": len(reports),
            "total_formulas": sum(r.total_count for r in reports),
            "total_valid": sum(r.valid_count for r in reports),
            "overall_fidelity": round(
                sum(r.valid_count for r in reports) / max(sum(r.total_count for r in reports), 1) * 100, 1
            ),
        },
        "files": [asdict(r) for r in reports],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score formula fidelity in converted Markdown files.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory to scan")
    parser.add_argument("--output", type=Path, help="Path for JSON quality report")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("formula_score", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "formula_report.json")).resolve()

    try:
        reports = score_tree(input_path) if input_path.is_dir() else [score_file(input_path)]

        written = write_report(reports, output_path)

        total = sum(r.total_count for r in reports)
        valid = sum(r.valid_count for r in reports)
        fidelity = valid / max(total, 1) * 100
        log.info(
            "scanned %d file(s): %d formulas, %d valid (%.1f%% fidelity) → %s",
            len(reports),
            total,
            valid,
            fidelity,
            written,
        )
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
