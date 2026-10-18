"""Enhanced LaTeX formula validation for scientific Markdown.

Goes beyond basic regex presence checks (formula_score.py) to perform
structural validation:
  - Balanced delimiter checking (braces, brackets, parentheses)
  - LaTeX environment matching (\\begin{X} / \\end{X})
  - Inline ($...$) vs display ($$...$$) math classification
  - Standard LaTeX command validation
  - Common error detection (unclosed delimiters, nested $ issues)
  - Formula complexity scoring (nesting depth, command count)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import load_config, resolve_configured_path, resolve_quality_report_path, setup_logging

# ── Constants ─────────────────────────────────────────────────────────────────

INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
ENVIRONMENT_BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
ENVIRONMENT_END_RE = re.compile(r"\\end\{([^}]+)\}")
LATEX_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)")

# Standard LaTeX math commands (a representative subset)
KNOWN_COMMANDS: frozenset[str] = frozenset(
    {
        "frac",
        "sqrt",
        "sum",
        "prod",
        "int",
        "lim",
        "inf",
        "sup",
        "sin",
        "cos",
        "tan",
        "log",
        "ln",
        "exp",
        "max",
        "min",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Pi",
        "Sigma",
        "Upsilon",
        "Phi",
        "Psi",
        "Omega",
        "partial",
        "nabla",
        "infty",
        "forall",
        "exists",
        "nexists",
        "in",
        "notin",
        "subset",
        "supset",
        "subseteq",
        "supseteq",
        "cup",
        "cap",
        "setminus",
        "emptyset",
        "varnothing",
        "leq",
        "geq",
        "neq",
        "approx",
        "equiv",
        "sim",
        "simeq",
        "ll",
        "gg",
        "prec",
        "succ",
        "cdot",
        "times",
        "div",
        "pm",
        "mp",
        "oplus",
        "otimes",
        "hat",
        "bar",
        "tilde",
        "vec",
        "dot",
        "ddot",
        "overline",
        "underline",
        "text",
        "mathrm",
        "mathbf",
        "mathbb",
        "mathcal",
        "mathfrak",
        "mathsf",
        "left",
        "right",
        "big",
        "Big",
        "bigg",
        "Bigg",
        "begin",
        "end",
        "tag",
        "label",
        "ref",
        "eqref",
        "quad",
        "qquad",
        "hspace",
        "vspace",
        "binom",
        "tbinom",
        "dbinom",
        "underbrace",
        "overbrace",
        "boldsymbol",
        "operatorname",
    }
)

# Standard LaTeX math environments
KNOWN_ENVIRONMENTS: frozenset[str] = frozenset(
    {
        "equation",
        "equation*",
        "align",
        "align*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "split",
        "aligned",
        "gathered",
        "cases",
        "matrix",
        "pmatrix",
        "bmatrix",
        "vmatrix",
        "Vmatrix",
        "Bmatrix",
        "array",
        "subequations",
    }
)

DELIMITER_PAIRS: dict[str, str] = {"{": "}", "[": "]", "(": ")"}


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class FormulaIssue:
    """A single validation issue found in a formula."""

    kind: str  # "error" | "warning"
    message: str


@dataclass
class Formula:
    """A single extracted and validated formula."""

    text: str
    display: str  # "inline" | "display"
    line: int = 0
    commands: list[str] = field(default_factory=list)
    environments: list[str] = field(default_factory=list)
    issues: list[FormulaIssue] = field(default_factory=list)
    nesting_depth: int = 0
    command_count: int = 0
    valid: bool = True
    complexity: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.kind == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.kind == "warning")


@dataclass
class FileValidation:
    """Validation results for a single file."""

    file: str
    inline_count: int = 0
    display_count: int = 0
    total_count: int = 0
    valid_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    avg_complexity: float = 0.0
    formulas: list[Formula] = field(default_factory=list)


@dataclass
class ValidationSummary:
    """Aggregate summary across all files."""

    files_scanned: int = 0
    total_formulas: int = 0
    total_inline: int = 0
    total_display: int = 0
    total_valid: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    avg_complexity: float = 0.0


# ── Validation helpers ────────────────────────────────────────────────────────


def check_balanced_delimiters(text: str) -> list[FormulaIssue]:
    """Check LaTeX grouping braces and visible delimiter counts."""
    issues: list[FormulaIssue] = []
    brace_stack: list[str] = []
    visible_stack: list[str] = []
    left_right_depth = 0

    i = 0
    while i < len(text):
        ch = text[i]
        if text.startswith(r"\left", i) or text.startswith(r"\right", i):
            command = r"\left" if text.startswith(r"\left", i) else r"\right"
            if command == r"\left":
                left_right_depth += 1
            elif left_right_depth == 0:
                issues.append(FormulaIssue("error", "unmatched \\right delimiter"))
            else:
                left_right_depth -= 1

            i += len(command)
            while i < len(text) and text[i].isspace():
                i += 1
            if i < len(text) and text[i] in ".()[]":
                i += 1
            elif i < len(text) and text[i] == "\\":
                delimiter = re.match(r"\\(?:[a-zA-Z]+|.)", text[i:])
                if delimiter:
                    i += len(delimiter.group(0))
            continue

        # Skip escaped characters
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == "{":
            brace_stack.append(ch)
        elif ch == "}":
            if not brace_stack:
                issues.append(FormulaIssue("error", f"unmatched closing delimiter '{ch}'"))
            else:
                brace_stack.pop()
        elif ch in "([":
            visible_stack.append(ch)
        elif ch in ")]":
            if not visible_stack:
                issues.append(FormulaIssue("error", f"unmatched closing delimiter '{ch}'"))
            else:
                visible_stack.pop()
        i += 1

    for opener in brace_stack:
        issues.append(FormulaIssue("error", f"unclosed delimiter '{opener}'"))
    for opener in visible_stack:
        issues.append(FormulaIssue("error", f"unclosed delimiter '{opener}'"))
    for _ in range(left_right_depth):
        issues.append(FormulaIssue("error", "unclosed \\left delimiter"))

    return issues


def check_environments(text: str) -> list[FormulaIssue]:
    """Check that \\begin{{X}} and \\end{{X}} are properly matched."""
    issues: list[FormulaIssue] = []
    begins = ENVIRONMENT_BEGIN_RE.findall(text)
    ends = ENVIRONMENT_END_RE.findall(text)

    stack: list[str] = []
    # Process in order of occurrence
    events: list[tuple[int, str, str]] = []
    for m in ENVIRONMENT_BEGIN_RE.finditer(text):
        events.append((m.start(), "begin", m.group(1)))
    for m in ENVIRONMENT_END_RE.finditer(text):
        events.append((m.start(), "end", m.group(1)))
    events.sort(key=lambda e: e[0])

    for _pos, action, env_name in events:
        if action == "begin":
            stack.append(env_name)
        elif not stack:
            issues.append(FormulaIssue("error", f"\\end{{{env_name}}} without matching \\begin"))
        elif stack[-1] != env_name:
            issues.append(
                FormulaIssue(
                    "error",
                    f"environment mismatch: \\begin{{{stack[-1]}}} closed by \\end{{{env_name}}}",
                )
            )
            stack.pop()
        else:
            stack.pop()

    for env_name in stack:
        issues.append(FormulaIssue("error", f"unclosed environment \\begin{{{env_name}}}"))

    # Warn on unknown environments
    all_envs = set(begins) | set(ends)
    for env_name in all_envs:
        if env_name not in KNOWN_ENVIRONMENTS:
            issues.append(FormulaIssue("warning", f"unknown environment '{env_name}'"))

    return issues


def check_commands(text: str) -> tuple[list[str], list[FormulaIssue]]:
    """Extract LaTeX commands and flag unknown ones."""
    commands = LATEX_COMMAND_RE.findall(text)
    issues: list[FormulaIssue] = []

    unique = set(commands)
    for cmd in sorted(unique):
        if cmd not in KNOWN_COMMANDS:
            issues.append(FormulaIssue("warning", f"unknown command '\\{cmd}'"))

    return commands, issues


def compute_nesting_depth(text: str) -> int:
    """Compute maximum brace nesting depth."""
    depth = 0
    max_depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
        i += 1
    return max_depth


def compute_complexity(formula: Formula) -> float:
    """Score formula complexity based on nesting, command count, and length.

    Returns a value in [0, 100] where higher = more complex.
    """
    length_score = min(len(formula.text) / 200.0, 1.0) * 30
    depth_score = min(formula.nesting_depth / 5.0, 1.0) * 30
    cmd_score = min(formula.command_count / 15.0, 1.0) * 25
    env_score = min(len(formula.environments) / 3.0, 1.0) * 15
    return round(length_score + depth_score + cmd_score + env_score, 1)


# ── Core validation ──────────────────────────────────────────────────────────


def validate_formula(text: str, display: str = "inline", line: int = 0) -> Formula:
    """Validate a single formula string and return a Formula with issues."""
    issues: list[FormulaIssue] = []

    if not text.strip():
        issues.append(FormulaIssue("error", "empty formula"))
        return Formula(
            text=text,
            display=display,
            line=line,
            issues=issues,
            valid=False,
        )

    # Check delimiters
    issues.extend(check_balanced_delimiters(text))

    # Check environments
    issues.extend(check_environments(text))

    # Check commands
    commands, cmd_issues = check_commands(text)
    issues.extend(cmd_issues)

    environments = ENVIRONMENT_BEGIN_RE.findall(text)
    nesting = compute_nesting_depth(text)
    cmd_count = len(commands)

    has_errors = any(i.kind == "error" for i in issues)

    formula = Formula(
        text=text,
        display=display,
        line=line,
        commands=sorted(set(commands)),
        environments=environments,
        issues=issues,
        nesting_depth=nesting,
        command_count=cmd_count,
        valid=not has_errors,
    )
    formula.complexity = compute_complexity(formula)
    return formula


def extract_and_validate(text: str) -> list[Formula]:
    """Extract all formulas from Markdown text and validate each one."""
    formulas: list[Formula] = []

    # Track positions to avoid double-counting display math as inline
    display_spans: list[tuple[int, int]] = []
    for m in DISPLAY_MATH_RE.finditer(text):
        display_spans.append((m.start(), m.end()))
        content = m.group(1)
        line_no = text[: m.start()].count("\n") + 1
        formulas.append(validate_formula(content, display="display", line=line_no))

    for m in INLINE_MATH_RE.finditer(text):
        # Skip if this match falls inside a display math span
        if any(s <= m.start() < e for s, e in display_spans):
            continue
        content = m.group(1)
        line_no = text[: m.start()].count("\n") + 1
        formulas.append(validate_formula(content, display="inline", line=line_no))

    return formulas


# ── File / tree operations ───────────────────────────────────────────────────


def validate_file(file_path: Path) -> FileValidation:
    """Validate all formulas in a single Markdown file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    formulas = extract_and_validate(text)

    inline_count = sum(1 for f in formulas if f.display == "inline")
    display_count = sum(1 for f in formulas if f.display == "display")
    valid_count = sum(1 for f in formulas if f.valid)
    err_count = sum(f.error_count for f in formulas)
    warn_count = sum(f.warning_count for f in formulas)
    avg_cx = (sum(f.complexity for f in formulas) / len(formulas)) if formulas else 0.0

    return FileValidation(
        file=str(file_path),
        inline_count=inline_count,
        display_count=display_count,
        total_count=len(formulas),
        valid_count=valid_count,
        error_count=err_count,
        warning_count=warn_count,
        avg_complexity=round(avg_cx, 1),
        formulas=formulas,
    )


def validate_tree(input_root: Path) -> list[FileValidation]:
    """Validate all Markdown files under *input_root*."""
    results: list[FileValidation] = []
    if not input_root.exists():
        return results
    if input_root.is_file():
        return [validate_file(input_root)]
    for md_file in sorted(input_root.rglob("*.md")):
        results.append(validate_file(md_file))
    return results


def build_summary(validations: list[FileValidation]) -> ValidationSummary:
    """Compute aggregate statistics from file validations."""
    total_formulas = sum(v.total_count for v in validations)
    total_cx = sum(v.avg_complexity * v.total_count for v in validations)
    return ValidationSummary(
        files_scanned=len(validations),
        total_formulas=total_formulas,
        total_inline=sum(v.inline_count for v in validations),
        total_display=sum(v.display_count for v in validations),
        total_valid=sum(v.valid_count for v in validations),
        total_errors=sum(v.error_count for v in validations),
        total_warnings=sum(v.warning_count for v in validations),
        avg_complexity=round(total_cx / total_formulas, 1) if total_formulas else 0.0,
    )


def write_report(validations: list[FileValidation], summary: ValidationSummary, output_path: Path) -> Path:
    """Write formula validation report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialise(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return str(obj)  # pragma: no cover

    data = {
        "summary": asdict(summary),
        "files": [
            {
                "file": v.file,
                "inline_count": v.inline_count,
                "display_count": v.display_count,
                "total_count": v.total_count,
                "valid_count": v.valid_count,
                "error_count": v.error_count,
                "warning_count": v.warning_count,
                "avg_complexity": v.avg_complexity,
            }
            for v in validations
        ],
    }
    output_path.write_text(json.dumps(data, indent=2, default=_serialise) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate LaTeX formulas in Markdown files.")
    p.add_argument("--input", type=Path, help="Markdown file or directory to validate")
    p.add_argument("--output", type=Path, help="Output JSON report path")
    p.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    log = setup_logging("formula_validate", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "formula_validation.json")).resolve()

    log.info("validating formulas in %s", input_path)
    validations = validate_tree(input_path)
    summary = build_summary(validations)
    written = write_report(validations, summary, output_path)
    log.info(
        "scanned %d files, %d formulas (%d valid, %d errors) → %s",
        summary.files_scanned,
        summary.total_formulas,
        summary.total_valid,
        summary.total_errors,
        written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
