"""Scientific document quality assurance checks.

Extends the general QA pipeline (qa_pipeline.py) with checks specific to
scientific and mathematical documents:
  - Theorem-proof pairing (every theorem should have a proof or acknowledgement)
  - Definition-before-use (terms should be defined before they are referenced)
  - Notation consistency (same symbol should not have conflicting definitions)
  - Cross-reference completeness (all internal references should resolve)
  - Algorithm validation (inputs/outputs declared, steps non-empty)
  - Formula quality gate (flag files with low formula fidelity)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cortexmark.common import load_config, resolve_configured_path, resolve_quality_report_path, setup_logging

# ── Constants ─────────────────────────────────────────────────────────────────

CHECK_THEOREM_PROOF = "theorem_proof_pairing"
CHECK_DEF_BEFORE_USE = "definition_before_use"
CHECK_NOTATION_CONSISTENCY = "notation_consistency"
CHECK_CROSSREF_COMPLETENESS = "crossref_completeness"
CHECK_ALGORITHM_VALIDITY = "algorithm_validity"
CHECK_FORMULA_QUALITY = "formula_quality"

ALL_CHECKS: tuple[str, ...] = (
    CHECK_THEOREM_PROOF,
    CHECK_DEF_BEFORE_USE,
    CHECK_NOTATION_CONSISTENCY,
    CHECK_CROSSREF_COMPLETENESS,
    CHECK_ALGORITHM_VALIDITY,
    CHECK_FORMULA_QUALITY,
)

# ── Patterns ─────────────────────────────────────────────────────────────────

# Theorem-like statements (with label numbers)
THEOREM_HEADING_RE = re.compile(
    r"(?:\*\*)?(?P<kind>Theorem|Lemma|Proposition|Corollary|Conjecture)"
    r"\s+(?P<label>[\d]+(?:\.[\d]+)*)"
    r"(?:\s*\((?P<name>[^)]+)\))?"
    r"[.:)]*(?:\*\*)?",
    re.IGNORECASE,
)


# Line-anchored (mirrors DEFINITION_SITE_RE below) and word-boundary-guarded so that
# a bare mention of the word "proof" in running prose ("we defer the proof to the
# appendix") or inside a longer word ("bulletproof") is never mistaken for a proof
# heading. "of <kind> N" is captured with the kind so a labeled proof only resolves
# the statement of the same kind and label (not any statement sharing the numeral).
PROOF_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\b(?:Proof|PROOF)\b"
    r"(?:\s+of\s+(?P<proof_kind>Theorem|Lemma|Proposition|Corollary|Conjecture)"
    r"\s+(?P<label>[\d]+(?:\.[\d]+)*))?"
    r"[.:)]*(?:\*\*)?",
    re.IGNORECASE | re.MULTILINE,
)

PROOF_OMITTED_RE = re.compile(
    r"proof\s+(?:is\s+)?(?:omitted|left\s+(?:to|as)\s+(?:the\s+)?(?:reader|exercise|appendix))",
    re.IGNORECASE,
)

# Definition sites
DEFINITION_RE = re.compile(
    r"(?:\*\*)?(?:Definition)\s+(?P<label>[\d]+(?:\.[\d]+)*)"
    r"(?:\s*\((?P<name>[^)]+)\))?"
    r"[.:)]*(?:\*\*)?",
    re.IGNORECASE,
)
DEFINITION_SITE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?Definition\s+(?P<label>\d+(?:\.\d+)*)",
    re.IGNORECASE | re.MULTILINE,
)

# References to definitions/theorems
REFERENCE_RE = re.compile(
    r"(?:Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Conjecture"
    r"|Figure|Fig\.|Table|Equation|Eq\.|Section|Chapter|Algorithm)"
    r"\s+(?P<label>[\d]+(?:\.[\d]+)*)",
    re.IGNORECASE,
)

# Assumption definition sites (not paired with proofs, unlike THEOREM_HEADING_RE).
# Line-anchored (like DEFINITION_SITE_RE) rather than matching anywhere, so an
# in-text mention ("Under Assumption 7 ...") isn't itself mistaken for a
# definition site — used only to validate "Assumption N" cross-references, see
# check_crossref_completeness.
ASSUMPTION_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:Assumption)\s+(?P<label>[\d]+(?:\.[\d]+)*)"
    r"(?:\s*\((?P<name>[^)]+)\))?"
    r"[.:)]*(?:\*\*)?",
    re.IGNORECASE | re.MULTILINE,
)

# Figure definition sites: docling picture generation is disabled (see convert.py),
# so figures surface as plain "Figure N: caption" text, never "![Figure N" alt-text.
FIGURE_CAPTION_RE = re.compile(
    r"^\s{0,3}(?:Figure|Fig\.?)\s+(?P<label>\d+(?:\.\d+)*)\s*[:.]",
    re.IGNORECASE | re.MULTILINE,
)

# Equation definition sites: numbered equations recovered from PDF are emitted as
# "> Equation (recovered from PDF):\n>\n> ... (N)" blockquotes (see
# convert.py:format_formula_block); legacy "$$...$$ (N)" / "\tag{N}" forms (for
# hand-authored Markdown) are also recognised.
EQUATION_LABEL_RE = re.compile(
    r"(?:\$\$.*?\$\$\s*\((?P<label>\d+(?:\.\d+)*)\))"
    r"|(?:\\tag\{(?P<tag>[^}]+)\})"
    r"|(?:Equation \(recovered from PDF\):\s*\n>\s*\n>[^\n]*\((?P<recovered>\d+(?:\.\d+)*)\)[ \t]*(?=\n|$))",
    re.DOTALL,
)

# Notation definition: "Let $X$ ..." or "where $X$ is/denotes/represents"
NOTATION_DEF_RE = re.compile(
    r"(?:(?:Let|let|define|Define|where|Where|denote|set)\s+)"
    r"(?:\$(?P<symbol>[^$]+)\$)"
    r"\s+(?:be|denote|represent|is|=)\s+(?P<meaning>[^.\n;]+)",
    re.IGNORECASE,
)

# Algorithm block detection
ALGORITHM_HEADING_RE = re.compile(
    r"(?:\*\*)?Algorithm\s+(?P<label>[\d]+(?:\.[\d]+)*)"
    r"(?:[:\s]+(?P<name>[^\n*]+))?"
    r"(?:\*\*)?",
    re.IGNORECASE,
)

ALGORITHM_INPUT_RE = re.compile(r"^\s*(?:\*\*)?(?:Input|Require|Given)[s:]?\s*(?:\*\*)?", re.IGNORECASE | re.MULTILINE)
ALGORITHM_OUTPUT_RE = re.compile(
    r"^\s*(?:\*\*)?(?:Output|Ensure|Return)[s:]?\s*(?:\*\*)?", re.IGNORECASE | re.MULTILINE
)

FORMULA_BLOCK_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
INLINE_FORMULA_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class SciQAIssue:
    """A single scientific QA issue."""

    check: str  # one of ALL_CHECKS
    severity: str  # "error" | "warning" | "info"
    message: str
    line: int = 0


@dataclass
class FileSciQAReport:
    """Scientific QA results for a single file."""

    file: str
    issues: list[SciQAIssue] = field(default_factory=list)
    theorems_found: int = 0
    proofs_found: int = 0
    definitions_found: int = 0
    algorithms_found: int = 0
    formulas_found: int = 0
    unresolved_refs: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def badge(self) -> str:
        if self.error_count > 0:
            return "fail"
        if self.warning_count > 2:
            return "bronze"
        if self.warning_count > 0:
            return "silver"
        return "gold"


@dataclass
class SciQASummary:
    """Aggregate scientific QA summary."""

    files_scanned: int = 0
    total_issues: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    badge_distribution: dict[str, int] = field(default_factory=dict)
    total_theorems: int = 0
    total_proofs: int = 0
    total_definitions: int = 0
    total_algorithms: int = 0
    total_unresolved: int = 0


# ── Check functions ──────────────────────────────────────────────────────────


def check_theorem_proof_pairing(text: str) -> list[SciQAIssue]:
    """Check that theorems/lemmas have corresponding proofs."""
    issues: list[SciQAIssue] = []

    # Theorem-like statements keyed by (kind, label), in document order, so a
    # "Theorem 1" and a "Lemma 1" sharing a numeral are tracked independently
    # instead of the later one silently overwriting the earlier.
    theorem_sites: list[tuple[str, str, int, int]] = []  # (kind, label, line, start_pos)
    seen_keys: set[tuple[str, str]] = set()
    for m in THEOREM_HEADING_RE.finditer(text):
        kind = m.group("kind").capitalize()
        label = m.group("label")
        key = (kind, label)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        line = text[: m.start()].count("\n") + 1
        theorem_sites.append((kind, label, line, m.start()))

    # Explicit "Proof of <Kind> N" markers resolve only the matching (kind, label)
    # pair; unlabeled ("generic") proof markers are associated with whichever
    # theorem's window (its heading up to the next theorem heading) they fall in,
    # rather than suppressing every missing-proof warning in the whole document.
    proof_keys: set[tuple[str, str]] = set()
    generic_proof_positions: list[int] = []
    for m in PROOF_RE.finditer(text):
        label = m.group("label")
        proof_kind = m.group("proof_kind")
        if label and proof_kind:
            proof_keys.add((proof_kind.capitalize(), label))
        elif not label:
            generic_proof_positions.append(m.start())

    omitted_positions = [m.start() for m in PROOF_OMITTED_RE.finditer(text)]

    for index, (kind, label, line, start_pos) in enumerate(theorem_sites):
        window_end = theorem_sites[index + 1][3] if index + 1 < len(theorem_sites) else len(text)

        has_labeled_proof = (kind, label) in proof_keys
        has_generic_proof = any(start_pos <= pos < window_end for pos in generic_proof_positions)
        has_omitted = any(start_pos <= pos < window_end for pos in omitted_positions)

        if not (has_labeled_proof or has_generic_proof or has_omitted):
            issues.append(
                SciQAIssue(
                    CHECK_THEOREM_PROOF,
                    "warning",
                    f"{kind} {label} has no corresponding proof",
                    line=line,
                )
            )

    return issues


def check_definition_before_use(text: str) -> list[SciQAIssue]:
    """Check that defined terms appear before their first reference."""
    issues: list[SciQAIssue] = []

    # Collect definition sites with their positions
    def_positions: dict[str, int] = {}
    for m in DEFINITION_SITE_RE.finditer(text):
        label = m.group("label")
        if label not in def_positions:
            def_positions[label] = m.start()

    # Collect all references
    for m in REFERENCE_RE.finditer(text):
        label = m.group("label")
        ref_text = m.group(0)
        # Only check "Definition X" references
        if not ref_text.lower().startswith("definition"):
            continue
        if label in def_positions and m.start() < def_positions[label]:
            line = text[: m.start()].count("\n") + 1
            issues.append(
                SciQAIssue(
                    CHECK_DEF_BEFORE_USE,
                    "warning",
                    f"Definition {label} referenced before it is defined",
                    line=line,
                )
            )

    return issues


def check_notation_consistency(text: str) -> list[SciQAIssue]:
    """Check that notation symbols are not defined with conflicting meanings."""
    issues: list[SciQAIssue] = []

    # Collect all notation definitions: symbol → list of normalized meanings
    definitions: dict[str, list[tuple[str, int]]] = {}  # symbol → [(meaning, line)]
    for m in NOTATION_DEF_RE.finditer(text):
        symbol = m.group("symbol").strip()
        line = text[: m.start()].count("\n") + 1
        meaning = re.sub(r"\s+", " ", m.group("meaning")).strip(" ,:").casefold()
        definitions.setdefault(symbol, []).append((meaning, line))

    for symbol, defs in definitions.items():
        if len({meaning for meaning, _line in defs}) > 1:
            lines = [d[1] for d in defs]
            issues.append(
                SciQAIssue(
                    CHECK_NOTATION_CONSISTENCY,
                    "warning",
                    f"Symbol ${symbol}$ defined with conflicting meanings (lines {', '.join(str(n) for n in lines)})",
                    line=lines[0],
                )
            )

    return issues


def check_crossref_completeness(text: str) -> list[SciQAIssue]:
    """Check that internal references point to existing definitions."""
    issues: list[SciQAIssue] = []
    kind_map: dict[str, str] = {
        "theorem": "theorem",
        "lemma": "theorem",
        "proposition": "theorem",
        "corollary": "theorem",
        "conjecture": "theorem",
        "definition": "definition",
        "assumption": "assumption",
        "figure": "figure",
        "fig.": "figure",
        "table": "table",
        "equation": "equation",
        "eq.": "equation",
        "algorithm": "algorithm",
        "section": "section",
        "chapter": "section",
    }

    # Collect all definition sites
    defined_labels: dict[str, set[str]] = {}  # kind → set of labels
    for m in THEOREM_HEADING_RE.finditer(text):
        kind = m.group("kind").lower()
        defined_labels.setdefault(kind_map.get(kind, kind), set()).add(m.group("label"))
    for m in DEFINITION_RE.finditer(text):
        defined_labels.setdefault("definition", set()).add(m.group("label"))
    for m in ASSUMPTION_HEADING_RE.finditer(text):
        defined_labels.setdefault("assumption", set()).add(m.group("label"))
    for m in ALGORITHM_HEADING_RE.finditer(text):
        defined_labels.setdefault("algorithm", set()).add(m.group("label"))

    # Figure/Equation definition sites (patterns aligned to real pipeline output —
    # see FIGURE_CAPTION_RE / EQUATION_LABEL_RE docstrings above).
    for m in FIGURE_CAPTION_RE.finditer(text):
        defined_labels.setdefault("figure", set()).add(m.group("label"))
    for m in EQUATION_LABEL_RE.finditer(text):
        label = m.group("label") or m.group("tag") or m.group("recovered")
        if label:
            defined_labels.setdefault("equation", set()).add(label)

    # Check all references
    for m in REFERENCE_RE.finditer(text):
        ref_kind_raw = m.group(0).split()[0].lower().rstrip(".")
        label = m.group("label")
        kind_key = kind_map.get(ref_kind_raw)

        # Skip section/chapter references (hard to validate in Markdown)
        if kind_key == "section":
            continue

        if kind_key and kind_key in defined_labels and label not in defined_labels[kind_key]:
            line = text[: m.start()].count("\n") + 1
            issues.append(
                SciQAIssue(
                    CHECK_CROSSREF_COMPLETENESS,
                    "warning",
                    f"Reference to {m.group(0).split()[0]} {label} but no definition found",
                    line=line,
                )
            )

    return issues


def check_algorithm_validity(text: str) -> list[SciQAIssue]:
    """Validate algorithm blocks have proper structure."""
    issues: list[SciQAIssue] = []

    for m in ALGORITHM_HEADING_RE.finditer(text):
        label = m.group("label")
        line = text[: m.start()].count("\n") + 1

        # Extract algorithm body (text until next heading or end)
        body_start = m.end()
        next_heading = re.search(r"\n#{1,3}\s", text[body_start:])
        next_algo = ALGORITHM_HEADING_RE.search(text[body_start:])
        body_end = len(text)
        if next_heading:
            body_end = min(body_end, body_start + next_heading.start())
        if next_algo:
            body_end = min(body_end, body_start + next_algo.start())
        body = text[body_start:body_end]

        # Check for input/output declarations
        has_input = bool(ALGORITHM_INPUT_RE.search(body))
        has_output = bool(ALGORITHM_OUTPUT_RE.search(body))

        if not has_input:
            issues.append(
                SciQAIssue(
                    CHECK_ALGORITHM_VALIDITY,
                    "info",
                    f"Algorithm {label} has no declared inputs",
                    line=line,
                )
            )
        if not has_output:
            issues.append(
                SciQAIssue(
                    CHECK_ALGORITHM_VALIDITY,
                    "info",
                    f"Algorithm {label} has no declared outputs",
                    line=line,
                )
            )

        # Check body is non-trivial (at least some text beyond whitespace)
        body_stripped = body.strip()
        if len(body_stripped) < 20:
            issues.append(
                SciQAIssue(
                    CHECK_ALGORITHM_VALIDITY,
                    "warning",
                    f"Algorithm {label} body is very short ({len(body_stripped)} chars)",
                    line=line,
                )
            )

    return issues


def check_formula_quality(text: str, *, min_fidelity: float = 50.0) -> list[SciQAIssue]:
    """Flag documents where formula fidelity appears low.

    Counts display-math and inline-math blocks and checks for common issues:
    incomplete placeholders, empty math blocks, and very short formulas.
    """
    issues: list[SciQAIssue] = []

    display_blocks = FORMULA_BLOCK_RE.findall(text)
    inline_blocks = INLINE_FORMULA_RE.findall(text)
    decoded_count = len(display_blocks) + len(inline_blocks)

    # Check for placeholder markers
    placeholder_count = text.count("<!-- formula-not-decoded -->")
    incomplete_count = len(re.findall(r"Equation present in PDF.*?incomplete", text, re.IGNORECASE))

    problem_count = placeholder_count + incomplete_count
    total = decoded_count + problem_count
    if total == 0:
        return issues

    fidelity = decoded_count / total

    if fidelity * 100 < min_fidelity:
        issues.append(
            SciQAIssue(
                CHECK_FORMULA_QUALITY,
                "error",
                f"Low formula fidelity: {fidelity:.0%} ({problem_count} of {total} formulas have issues)",
            )
        )

    # Check for empty formula blocks
    empty_display = sum(1 for b in display_blocks if not b.strip("$ \t\n\r"))
    empty_inline = sum(1 for b in inline_blocks if not b.strip())
    empty_total = empty_display + empty_inline
    if empty_total > 0:
        issues.append(
            SciQAIssue(
                CHECK_FORMULA_QUALITY,
                "warning",
                f"{empty_total} empty formula block(s) found",
            )
        )

    return issues


# ── Composite check ──────────────────────────────────────────────────────────


def run_all_checks(text: str) -> list[SciQAIssue]:
    """Run all scientific QA checks on a text."""
    issues: list[SciQAIssue] = []
    issues.extend(check_theorem_proof_pairing(text))
    issues.extend(check_definition_before_use(text))
    issues.extend(check_notation_consistency(text))
    issues.extend(check_crossref_completeness(text))
    issues.extend(check_algorithm_validity(text))
    issues.extend(check_formula_quality(text))
    return issues


# ── File / tree operations ───────────────────────────────────────────────────


def analyze_file(file_path: Path) -> FileSciQAReport:
    """Run scientific QA checks on a single file."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    issues = run_all_checks(text)

    theorems = len(THEOREM_HEADING_RE.findall(text))
    proofs = len(PROOF_RE.findall(text))
    definitions = len(DEFINITION_RE.findall(text))
    algorithms = len(ALGORITHM_HEADING_RE.findall(text))
    formulas = len(FORMULA_BLOCK_RE.findall(text)) + len(INLINE_FORMULA_RE.findall(text))
    unresolved = sum(1 for i in issues if i.check == CHECK_CROSSREF_COMPLETENESS)

    return FileSciQAReport(
        file=str(file_path),
        issues=issues,
        theorems_found=theorems,
        proofs_found=proofs,
        definitions_found=definitions,
        algorithms_found=algorithms,
        formulas_found=formulas,
        unresolved_refs=unresolved,
    )


def analyze_tree(input_root: Path) -> list[FileSciQAReport]:
    """Run scientific QA checks on all Markdown files under *input_root*."""
    results: list[FileSciQAReport] = []
    if not input_root.exists():
        return results
    if input_root.is_file():
        return [analyze_file(input_root)]
    for md_file in sorted(input_root.rglob("*.md")):
        results.append(analyze_file(md_file))
    return results


def build_summary(reports: list[FileSciQAReport]) -> SciQASummary:
    """Compute aggregate scientific QA statistics."""
    badge_dist: dict[str, int] = {}
    for r in reports:
        badge_dist[r.badge] = badge_dist.get(r.badge, 0) + 1

    return SciQASummary(
        files_scanned=len(reports),
        total_issues=sum(len(r.issues) for r in reports),
        total_errors=sum(r.error_count for r in reports),
        total_warnings=sum(r.warning_count for r in reports),
        badge_distribution=badge_dist,
        total_theorems=sum(r.theorems_found for r in reports),
        total_proofs=sum(r.proofs_found for r in reports),
        total_definitions=sum(r.definitions_found for r in reports),
        total_algorithms=sum(r.algorithms_found for r in reports),
        total_unresolved=sum(r.unresolved_refs for r in reports),
    )


def write_report(reports: list[FileSciQAReport], summary: SciQASummary, output_path: Path) -> Path:
    """Write scientific QA report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "summary": asdict(summary),
        "files": [
            {
                "file": r.file,
                "badge": r.badge,
                "error_count": r.error_count,
                "warning_count": r.warning_count,
                "theorems": r.theorems_found,
                "proofs": r.proofs_found,
                "definitions": r.definitions_found,
                "algorithms": r.algorithms_found,
                "formulas": r.formulas_found,
                "unresolved_refs": r.unresolved_refs,
                "issues": [asdict(i) for i in r.issues],
            }
            for r in reports
        ],
    }
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scientific document quality checks.")
    p.add_argument("--input", type=Path, help="Markdown file or directory to check")
    p.add_argument("--output", type=Path, help="Output JSON report path")
    p.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    log = setup_logging("scientific_qa", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "scientific_qa.json")).resolve()

    log.info("running scientific QA on %s", input_path)
    reports = analyze_tree(input_path)
    summary = build_summary(reports)
    written = write_report(reports, summary, output_path)
    log.info(
        "scanned %d files, %d issues (%d errors, %d warnings), badge: %s → %s",
        summary.files_scanned,
        summary.total_issues,
        summary.total_errors,
        summary.total_warnings,
        summary.badge_distribution,
        written,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
