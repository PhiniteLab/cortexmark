"""Algorithm and pseudocode extraction from scientific Markdown.

Detects, parses, and structures algorithm blocks found in converted
Markdown, whether they appear as:
  - Code-fenced blocks (```text, ```algorithm, ```pseudocode)
  - Keyword-triggered blocks (Initialize, Loop, Input:, …)
  - Explicitly labelled "Algorithm N" sections

Each extracted algorithm receives structured metadata: label, title,
inputs, outputs, and the body lines with indentation-based nesting.
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

# ── Detection patterns ───────────────────────────────────────────────────────

# Code fence openers that signal algorithm blocks
ALGO_FENCE_RE = re.compile(r"^```(?P<language>text|algorithm|pseudocode)?\s*$", re.IGNORECASE)

# Algorithm header: "Algorithm 1: Q-Learning", "**Algorithm 2.1** Value Iteration"
ALGO_HEADER_RE = re.compile(
    r"^(?:#{1,4}\s+)?(?:\*\*)?Algorithm\s+(?P<label>\d+(?:\.\d+)*)"
    r"[.:)]*(?:\*\*)?\s*(?P<title>.*)$",
    re.IGNORECASE,
)

# Pseudocode keywords that help identify algorithmic content
PSEUDOCODE_KEYWORDS: tuple[str, ...] = (
    "Initialize",
    "Input:",
    "Output:",
    "Require:",
    "Ensure:",
    "Algorithm parameter",
    "Loop for each",
    "Loop forever",
    "for each",
    "for all",
    "while",
    "repeat",
    "if ",
    "else",
    "return",
    "until ",
)
STRONG_PSEUDOCODE_KEYWORDS: tuple[str, ...] = PSEUDOCODE_KEYWORDS[:8]

# Input/output extraction patterns
INPUT_RE = re.compile(
    r"^(?:Input|Require|Algorithm\s+parameter)s?\s*:\s*(?P<value>.+)$",
    re.IGNORECASE,
)
OUTPUT_RE = re.compile(
    r"^(?:Output|Ensure)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE,
)

# Structural keywords that denote control flow
CONTROL_FLOW_RE = re.compile(
    r"^(?P<keyword>if|else\s*if|else|for|for\s+each|for\s+all|"
    r"while|repeat|loop|do|until|return|break|continue)\b",
    re.IGNORECASE,
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class AlgoStep:
    """A single line/step within an algorithm."""

    text: str
    indent_level: int = 0
    is_control_flow: bool = False
    keyword: str = ""


@dataclass
class Algorithm:
    """A fully extracted algorithm with structured metadata."""

    label: str = ""
    title: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    steps: list[AlgoStep] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0
    raw_text: str = ""

    @property
    def full_label(self) -> str:
        if self.label and self.title:
            return f"Algorithm {self.label}: {self.title}"
        if self.label:
            return f"Algorithm {self.label}"
        return self.title or "Unnamed Algorithm"

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def max_depth(self) -> int:
        if not self.steps:
            return 0
        return max(s.indent_level for s in self.steps)


# ── Parsing helpers ──────────────────────────────────────────────────────────


def parse_step(line: str, base_indent: int = 0) -> AlgoStep:
    """Parse a single algorithm line into a structured step."""
    stripped = line.strip()
    # Compute indent level from leading whitespace (every 2 spaces = 1 level)
    raw_indent = len(line) - len(line.lstrip())
    indent_level = max(0, (raw_indent - base_indent) // 2)

    cf_match = CONTROL_FLOW_RE.match(stripped)
    return AlgoStep(
        text=stripped,
        indent_level=indent_level,
        is_control_flow=bool(cf_match),
        keyword=cf_match.group("keyword").lower() if cf_match else "",
    )


def parse_algorithm_body(lines: list[str]) -> tuple[list[str], list[str], list[AlgoStep]]:
    """Parse algorithm body lines into inputs, outputs, and steps."""
    inputs: list[str] = []
    outputs: list[str] = []
    steps: list[AlgoStep] = []

    # Determine base indent from first non-empty line
    base_indent = 0
    for line in lines:
        if line.strip():
            base_indent = len(line) - len(line.lstrip())
            break

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        in_match = INPUT_RE.match(stripped)
        if in_match:
            inputs.append(in_match.group("value").strip())
            continue

        out_match = OUTPUT_RE.match(stripped)
        if out_match:
            outputs.append(out_match.group("value").strip())
            continue

        steps.append(parse_step(line, base_indent))

    return inputs, outputs, steps


def is_algorithm_content(text: str) -> bool:
    """Check whether text contains enough pseudocode keywords to be algorithmic."""
    count = sum(1 for kw in PSEUDOCODE_KEYWORDS if kw.lower() in text.lower())
    return count >= 2


# ── Main extraction ──────────────────────────────────────────────────────────


def extract_algorithms(text: str, source_file: str = "") -> list[Algorithm]:
    """Extract all algorithm blocks from Markdown text."""
    algorithms: list[Algorithm] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Case 1: Code-fenced algorithm block
        fence_match = ALGO_FENCE_RE.match(stripped)
        if fence_match:
            block_start = i
            body_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # skip closing fence

            raw = "\n".join(body_lines)
            language = (fence_match.group("language") or "").lower()
            explicitly_algorithmic = language in {"algorithm", "pseudocode"}
            has_strong_signal = any(keyword.lower() in raw.lower() for keyword in STRONG_PSEUDOCODE_KEYWORDS)
            if not (
                explicitly_algorithmic
                or _has_nearby_header(lines, block_start)
                or (has_strong_signal and is_algorithm_content(raw))
            ):
                continue

            # Look backwards for a header
            label, title = _find_header_backward(lines, block_start)
            inputs, outputs, steps = parse_algorithm_body(body_lines)

            algorithms.append(
                Algorithm(
                    label=label,
                    title=title,
                    inputs=inputs,
                    outputs=outputs,
                    steps=steps,
                    source_file=source_file,
                    line_number=block_start + 1,
                    raw_text=raw,
                )
            )
            continue

        # Case 2: Explicit "Algorithm N" header line (not in fence)
        header_match = ALGO_HEADER_RE.match(stripped)
        if header_match:
            header_line = i + 1
            label = header_match.group("label")
            title = header_match.group("title").strip().rstrip(".:)")
            block_lines: list[str] = []
            i += 1
            # Collect subsequent lines until next heading or blank-line gap
            blank_count = 0
            while i < len(lines):
                ls = lines[i].strip()
                if ls.startswith("#") or ALGO_HEADER_RE.match(ls):
                    break
                if not ls:
                    blank_count += 1
                    if blank_count >= 2:
                        break
                    i += 1
                    continue
                blank_count = 0
                # Skip code fences (just collect body)
                if ls.startswith("```"):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith("```"):
                        block_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        i += 1
                    continue
                block_lines.append(lines[i])
                i += 1

            inputs, outputs, steps = parse_algorithm_body(block_lines)
            algorithms.append(
                Algorithm(
                    label=label,
                    title=title,
                    inputs=inputs,
                    outputs=outputs,
                    steps=steps,
                    source_file=source_file,
                    line_number=header_line,
                    raw_text="\n".join(block_lines),
                )
            )
            continue

        i += 1

    return algorithms


def _has_nearby_header(lines: list[str], fence_line: int) -> bool:
    """Check if there's an Algorithm header within 3 lines before the fence."""
    start = max(0, fence_line - 3)
    return any(ALGO_HEADER_RE.match(lines[j].strip()) for j in range(start, fence_line))


def _find_header_backward(lines: list[str], fence_line: int) -> tuple[str, str]:
    """Look backwards from a code fence to find an Algorithm header."""
    start = max(0, fence_line - 3)
    for j in range(fence_line - 1, start - 1, -1):
        m = ALGO_HEADER_RE.match(lines[j].strip())
        if m:
            return m.group("label"), m.group("title").strip().rstrip(".:)")
    return "", ""


# ── File / tree operations ───────────────────────────────────────────────────


def extract_from_file(file_path: Path) -> list[Algorithm]:
    """Extract algorithms from a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    return extract_algorithms(text, source_file=str(file_path))


def extract_from_tree(input_root: Path) -> list[Algorithm]:
    """Extract algorithms from all Markdown files in a directory."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")

    results: list[Algorithm] = []
    for md_path in md_files:
        results.extend(extract_from_file(md_path))
    return results


# ── Output ───────────────────────────────────────────────────────────────────


def build_summary(algorithms: list[Algorithm]) -> dict[str, Any]:
    """Build summary statistics for extracted algorithms."""
    total_steps = sum(a.step_count for a in algorithms)
    max_depth = max((a.max_depth for a in algorithms), default=0)
    labelled = sum(1 for a in algorithms if a.label)
    return {
        "total_algorithms": len(algorithms),
        "total_steps": total_steps,
        "max_nesting_depth": max_depth,
        "labelled": labelled,
        "unlabelled": len(algorithms) - labelled,
    }


def write_report(algorithms: list[Algorithm], output_path: Path) -> Path:
    """Write algorithm extraction report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "summary": build_summary(algorithms),
        "algorithms": [
            {
                "label": a.full_label,
                "inputs": a.inputs,
                "outputs": a.outputs,
                "step_count": a.step_count,
                "max_depth": a.max_depth,
                "source_file": a.source_file,
                "line_number": a.line_number,
                "steps": [asdict(s) for s in a.steps],
            }
            for a in algorithms
        ],
    }
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and structure algorithm/pseudocode blocks from Markdown.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory")
    parser.add_argument("--output", type=Path, help="Path for algorithm report (JSON)")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("algorithm_extract", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "algorithm_report.json")).resolve()

    try:
        algorithms = extract_from_tree(input_path) if input_path.is_dir() else extract_from_file(input_path)

        written = write_report(algorithms, output_path)
        summary = build_summary(algorithms)
        log.info(
            "extracted %d algorithm(s), %d total steps → %s",
            summary["total_algorithms"],
            summary["total_steps"],
            written,
        )
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
