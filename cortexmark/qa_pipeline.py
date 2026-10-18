"""Quality-assurance pipeline for converted Markdown documents.

Checks performed:
  - Missing text detection (suspiciously short files)
  - Encoding errors (replacement chars, mojibake)
  - Broken internal references / links
  - Empty chunks
  - Orphan headings (heading with no body)
  - Table integrity (malformed tables)

Produces a per-file + summary quality report with badge-style ratings.
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

REPLACEMENT_CHAR_RE = re.compile(r"[\ufffd]")
_CP1252_CONTINUATION_CHARS = "\u0080-\u00bf€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ"
MOJIBAKE_RE = re.compile(
    rf"(?:[ÃÂ][{_CP1252_CONTINUATION_CHARS}]|Â[ \u00a0]|"
    rf"â[{_CP1252_CONTINUATION_CHARS}][{_CP1252_CONTINUATION_CHARS}])"
)
BROKEN_LINK_RE = re.compile(r"\[([^\]]*)\]\((?!\s*http)([^)]*)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
TABLE_SEP_RE = re.compile(r"^\|\s*[-:]+[-|\s:]*\|$")

# ── Quality thresholds ───────────────────────────────────────────────────────

MIN_FILE_CHARS = 50  # Files shorter than this are suspicious
MIN_CHUNK_CHARS = 10  # Chunks shorter than this are "empty"

# ── Badge levels ─────────────────────────────────────────────────────────────

BADGE_GOLD = "gold"
BADGE_SILVER = "silver"
BADGE_BRONZE = "bronze"
BADGE_FAIL = "fail"


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class QAIssue:
    """A single QA issue detected."""

    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    line: int = 0


@dataclass
class FileQAReport:
    """QA report for a single file."""

    file: str
    char_count: int = 0
    line_count: int = 0
    issues: list[QAIssue] = field(default_factory=list)
    badge: str = ""

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


@dataclass
class QASummary:
    """Overall QA summary."""

    files_scanned: int = 0
    total_issues: int = 0
    total_errors: int = 0
    total_warnings: int = 0
    badge_distribution: dict[str, int] = field(default_factory=dict)
    overall_badge: str = ""


# ── Check functions ──────────────────────────────────────────────────────────


def check_encoding_errors(text: str) -> list[QAIssue]:
    """Detect replacement characters and mojibake patterns."""
    issues: list[QAIssue] = []
    for i, line in enumerate(text.split("\n"), 1):
        if REPLACEMENT_CHAR_RE.search(line):
            issues.append(
                QAIssue(
                    check="encoding",
                    severity="error",
                    message="Unicode replacement character (U+FFFD) found",
                    line=i,
                )
            )
        if MOJIBAKE_RE.search(line):
            issues.append(
                QAIssue(
                    check="encoding",
                    severity="warning",
                    message="Possible mojibake detected",
                    line=i,
                )
            )
    return issues


def check_missing_text(text: str, *, min_chars: int = MIN_FILE_CHARS) -> list[QAIssue]:
    """Flag files that are suspiciously short."""
    char_count = len(text.strip())
    if char_count < min_chars:
        return [
            QAIssue(
                check="missing_text",
                severity="warning",
                message=f"File has only {char_count} characters (threshold: {min_chars})",
            )
        ]
    return []


def check_broken_links(text: str) -> list[QAIssue]:
    """Detect Markdown links that point to non-URL targets (likely broken)."""
    issues: list[QAIssue] = []
    for i, line in enumerate(text.split("\n"), 1):
        for m in BROKEN_LINK_RE.finditer(line):
            target = m.group(2).strip()
            # Skip anchors (#section) and common patterns
            if target.startswith(("#", "mailto:")):
                continue
            # Flag if target looks incomplete or suspicious
            if not target or target.startswith(".."):
                issues.append(
                    QAIssue(
                        check="broken_link",
                        severity="warning",
                        message=f"Possibly broken link: [{m.group(1)}]({target})",
                        line=i,
                    )
                )
    return issues


def check_orphan_headings(text: str) -> list[QAIssue]:
    """Detect headings immediately followed by another heading (no body)."""
    issues: list[QAIssue] = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if HEADING_RE.match(line.strip()):
            # Scan forward for body text
            has_body = False
            for next_line in lines[i + 1 :]:
                stripped = next_line.strip()
                if not stripped:
                    continue
                if HEADING_RE.match(stripped):
                    break
                has_body = True
                break
            if not has_body:
                issues.append(
                    QAIssue(
                        check="orphan_heading",
                        severity="info",
                        message=f"Heading with no body text: {line.strip()[:80]}",
                        line=i + 1,
                    )
                )
    return issues


def check_table_integrity(text: str) -> list[QAIssue]:
    """Check for malformed tables (mismatched columns, missing separators)."""
    issues: list[QAIssue] = []
    lines = text.split("\n")
    table_start = -1
    table_rows: list[tuple[int, str]] = []

    def check_table_block() -> None:
        if len(table_rows) < 2:
            return
        col_counts = [row.count("|") - 1 for _, row in table_rows]
        # Check separator exists
        has_sep = any(TABLE_SEP_RE.match(row.strip()) for _, row in table_rows)
        if not has_sep:
            issues.append(
                QAIssue(
                    check="table_integrity",
                    severity="warning",
                    message="Table missing separator row",
                    line=table_rows[0][0],
                )
            )
        # Check consistent column count
        if len(set(col_counts)) > 1:
            issues.append(
                QAIssue(
                    check="table_integrity",
                    severity="warning",
                    message=f"Table has inconsistent column counts: {set(col_counts)}",
                    line=table_rows[0][0],
                )
            )

        # Flag rows whose cell count is wildly out of line with the table's
        # median (e.g. one row having 3x the median cells). This catches severe
        # extraction corruption -- such as several source rows getting merged
        # into one during PDF conversion -- that the plain "not all equal"
        # check above also reports, but without pinpointing which row is at
        # fault or distinguishing it from a harmless one-cell typo.
        sorted_counts = sorted(col_counts)
        median_count = sorted_counts[len(sorted_counts) // 2]
        if median_count > 0:
            for line_no, count in zip((ln for ln, _ in table_rows), col_counts, strict=True):
                if count >= median_count * 3:
                    issues.append(
                        QAIssue(
                            check="table_integrity",
                            severity="warning",
                            message=(
                                f"Table row has {count} cells vs. median {median_count} "
                                "-- possible corrupted/merged row"
                            ),
                            line=line_no,
                        )
                    )

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if table_start < 0:
                table_start = i
            table_rows.append((i, stripped))
        elif table_rows:
            check_table_block()
            table_rows = []
            table_start = -1

    if table_rows:
        check_table_block()

    return issues


def check_empty_chunks(text: str, *, min_chars: int = MIN_CHUNK_CHARS) -> list[QAIssue]:
    """Check if the chunk has meaningful content beyond headings."""
    lines = text.split("\n")
    body_chars = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or HEADING_RE.match(stripped):
            continue
        body_chars += len(stripped)

    if body_chars < min_chars:
        return [
            QAIssue(
                check="empty_chunk",
                severity="warning",
                message=f"Chunk body only {body_chars} chars (threshold: {min_chars})",
            )
        ]
    return []


# ── Badge computation ────────────────────────────────────────────────────────


def compute_badge(report: FileQAReport) -> str:
    """Assign a quality badge based on issue severity counts."""
    if report.error_count > 0:
        return BADGE_FAIL
    if report.warning_count > 2:
        return BADGE_BRONZE
    if report.warning_count > 0:
        return BADGE_SILVER
    return BADGE_GOLD


def compute_overall_badge(reports: list[FileQAReport]) -> str:
    """Compute overall badge across all files."""
    if not reports:
        return BADGE_GOLD
    badges = [r.badge for r in reports]
    if any(b == BADGE_FAIL for b in badges):
        return BADGE_FAIL
    if sum(1 for b in badges if b == BADGE_BRONZE) > len(badges) * 0.3:
        return BADGE_BRONZE
    if any(b == BADGE_BRONZE for b in badges):
        return BADGE_SILVER
    if any(b == BADGE_SILVER for b in badges):
        return BADGE_SILVER
    return BADGE_GOLD


# ── Scoring engine ───────────────────────────────────────────────────────────


def qa_check_text(text: str, *, is_chunk: bool = False) -> list[QAIssue]:
    """Run all QA checks on a text block."""
    issues: list[QAIssue] = []
    issues.extend(check_encoding_errors(text))
    issues.extend(check_missing_text(text))
    issues.extend(check_broken_links(text))
    issues.extend(check_orphan_headings(text))
    issues.extend(check_table_integrity(text))
    if is_chunk:
        issues.extend(check_empty_chunks(text))
    return issues


def build_file_report(file_path: Path, text: str, *, is_chunk: bool = False) -> FileQAReport:
    """Build a complete QA report for one file."""
    issues = qa_check_text(text, is_chunk=is_chunk)
    report = FileQAReport(
        file=str(file_path),
        char_count=len(text),
        line_count=text.count("\n") + 1,
        issues=issues,
    )
    report.badge = compute_badge(report)
    return report


def qa_file(file_path: Path, *, is_chunk: bool = False) -> FileQAReport:
    """QA-check a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return build_file_report(file_path, text, is_chunk=is_chunk)


def qa_tree(input_root: Path, *, is_chunk: bool = False) -> list[FileQAReport]:
    """QA-check all Markdown files under a directory."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")
    return [qa_file(f, is_chunk=is_chunk) for f in md_files]


def build_summary(reports: list[FileQAReport]) -> QASummary:
    """Aggregate file reports into a summary."""
    badge_dist: dict[str, int] = {}
    for r in reports:
        badge_dist[r.badge] = badge_dist.get(r.badge, 0) + 1

    return QASummary(
        files_scanned=len(reports),
        total_issues=sum(len(r.issues) for r in reports),
        total_errors=sum(r.error_count for r in reports),
        total_warnings=sum(r.warning_count for r in reports),
        badge_distribution=badge_dist,
        overall_badge=compute_overall_badge(reports),
    )


def write_report(reports: list[FileQAReport], summary: QASummary, output_path: Path) -> Path:
    """Write QA report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "summary": asdict(summary),
        "files": [asdict(r) for r in reports],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def write_markdown_report(reports: list[FileQAReport], summary: QASummary, output_path: Path) -> Path:
    """Write a human-readable Markdown QA report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    badge_emoji = {
        BADGE_GOLD: "🥇",
        BADGE_SILVER: "🥈",
        BADGE_BRONZE: "🥉",
        BADGE_FAIL: "❌",
    }

    lines.append("# Quality Assurance Report")
    lines.append("")
    lines.append(f"**Overall Badge:** {badge_emoji.get(summary.overall_badge, '')} {summary.overall_badge.upper()}")
    lines.append(f"**Files Scanned:** {summary.files_scanned}")
    lines.append(
        f"**Total Issues:** {summary.total_issues} ({summary.total_errors} errors, {summary.total_warnings} warnings)"
    )
    lines.append("")
    lines.append("## Badge Distribution")
    lines.append("")
    lines.append("| Badge | Count |")
    lines.append("| --- | --- |")
    for badge, count in sorted(summary.badge_distribution.items()):
        lines.append(f"| {badge_emoji.get(badge, '')} {badge} | {count} |")
    lines.append("")

    # Per-file details (only files with issues)
    files_with_issues = [r for r in reports if r.issues]
    if files_with_issues:
        lines.append("## Files with Issues")
        lines.append("")
        for report in files_with_issues:
            lines.append(f"### {report.file}")
            lines.append(f"Badge: {badge_emoji.get(report.badge, '')} {report.badge}")
            lines.append("")
            lines.append("| Check | Severity | Line | Message |")
            lines.append("| --- | --- | --- | --- |")
            for issue in report.issues:
                lines.append(f"| {issue.check} | {issue.severity} | {issue.line or '-'} | {issue.message} |")
            lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run quality assurance checks on converted Markdown files.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory to check")
    parser.add_argument("--output", type=Path, help="Path for QA report (JSON)")
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument("--chunks", action="store_true", help="Enable chunk-specific checks")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("qa", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "qa_report.json")).resolve()

    try:
        if input_path.is_dir():
            reports = qa_tree(input_path, is_chunk=args.chunks)
        else:
            reports = [qa_file(input_path, is_chunk=args.chunks)]

        summary = build_summary(reports)

        if args.format in ("json", "both"):
            written = write_report(reports, summary, output_path)
            log.info("wrote JSON QA report → %s", written)

        if args.format in ("markdown", "both"):
            md_path = output_path.with_suffix(".md")
            written_md = write_markdown_report(reports, summary, md_path)
            log.info("wrote Markdown QA report → %s", written_md)

        log.info(
            "QA complete: %d file(s), %d issue(s), badge=%s",
            summary.files_scanned,
            summary.total_issues,
            summary.overall_badge,
        )

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
