"""Figure / image extraction and cataloguing from Markdown documents.

Scans converted Markdown for image references (![alt](path)) and:
  - Catalogues all figures with metadata (alt text, path, context)
  - Validates that referenced image files exist on disk
  - Generates a figure index (JSON manifest)
  - Produces a Markdown gallery page

Works on the post-convert Markdown outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import load_config, resolve_configured_path, resolve_quality_report_path, setup_logging

# ── Patterns ─────────────────────────────────────────────────────────────────

# Standard Markdown image: ![alt](path "optional title")
IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)")
# HTML img tag: <img src="..." alt="..."> (any attribute order)
HTML_IMG_TAG_RE = re.compile(r"<img\s[^>]*?>", re.IGNORECASE)
HTML_IMG_SRC_RE = re.compile(r'<img\s[^>]*?src=["\']([^"\']+)["\']', re.IGNORECASE)
HTML_IMG_ALT_RE = re.compile(r'<img\s[^>]*?alt=["\']([^"\']*)["\']', re.IGNORECASE)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class FigureEntry:
    """A single figure/image reference in a document."""

    alt_text: str
    image_path: str
    title: str = ""
    source_file: str = ""
    line_number: int = 0
    exists: bool = False
    context: str = ""


@dataclass
class FigureReport:
    """Report for all figures in a file or tree."""

    total_figures: int = 0
    missing_files: int = 0
    figures: list[FigureEntry] = field(default_factory=list)


# ── Extraction engine ────────────────────────────────────────────────────────


def extract_figures_from_text(
    text: str,
    source_file: str = "",
    base_dir: Path | None = None,
) -> list[FigureEntry]:
    """Extract all image references from Markdown text."""
    entries: list[FigureEntry] = []
    lines = text.split("\n")

    for line_num, line in enumerate(lines, start=1):
        # Markdown images
        for match in IMG_RE.finditer(line):
            alt, path, title = match.group(1), match.group(2), match.group(3) or ""
            # Context: surrounding lines
            start = max(0, line_num - 2)
            end = min(len(lines), line_num + 1)
            ctx = "\n".join(lines[start:end]).strip()

            exists = False
            if "://" in path:
                exists = True
            elif base_dir:
                candidate = base_dir / path
                exists = candidate.exists()

            entries.append(
                FigureEntry(
                    alt_text=alt,
                    image_path=path,
                    title=title,
                    source_file=source_file,
                    line_number=line_num,
                    exists=exists,
                    context=ctx,
                )
            )

        # HTML images
        for match in HTML_IMG_TAG_RE.finditer(line):
            tag = match.group(0)
            src_match = HTML_IMG_SRC_RE.search(tag)
            if not src_match:
                continue
            src = src_match.group(1)
            alt_match = HTML_IMG_ALT_RE.search(tag)
            alt = alt_match.group(1) if alt_match else ""
            # Skip if already captured by Markdown pattern
            if any(e.image_path == src and e.line_number == line_num for e in entries):
                continue

            exists = False
            if "://" in src:
                exists = True
            elif base_dir:
                candidate = base_dir / src
                exists = candidate.exists()

            entries.append(
                FigureEntry(
                    alt_text=alt,
                    image_path=src,
                    source_file=source_file,
                    line_number=line_num,
                    exists=exists,
                )
            )

    return entries


def build_figure_report(entries: list[FigureEntry]) -> FigureReport:
    """Build a report from a list of figure entries."""
    missing = sum(1 for e in entries if not e.exists)
    return FigureReport(
        total_figures=len(entries),
        missing_files=missing,
        figures=entries,
    )


# ── File / tree operations ───────────────────────────────────────────────────


def extract_from_file(file_path: Path) -> FigureReport:
    """Extract figures from a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = file_path.read_text(encoding="utf-8")
    entries = extract_figures_from_text(
        text,
        source_file=str(file_path),
        base_dir=file_path.parent,
    )
    return build_figure_report(entries)


def extract_from_tree(input_root: Path) -> FigureReport:
    """Extract figures from all Markdown files in a directory tree."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")

    all_entries: list[FigureEntry] = []
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        entries = extract_figures_from_text(
            text,
            source_file=str(md_path),
            base_dir=md_path.parent,
        )
        all_entries.extend(entries)

    return build_figure_report(all_entries)


# ── Output writers ───────────────────────────────────────────────────────────


def write_figure_manifest(report: FigureReport, output_path: Path) -> Path:
    """Write figure manifest as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "summary": {
            "total_figures": report.total_figures,
            "missing_files": report.missing_files,
        },
        "figures": [asdict(e) for e in report.figures],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def write_gallery_page(report: FigureReport, output_path: Path) -> Path:
    """Write a Markdown gallery page listing all figures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Figure Gallery", "", f"Total figures: {report.total_figures}", ""]

    if report.missing_files > 0:
        lines.append(f"**Missing files: {report.missing_files}**")
        lines.append("")

    for i, fig in enumerate(report.figures, start=1):
        status = "✓" if fig.exists else "✗ MISSING"
        lines.append(f"## Figure {i}")
        lines.append("")
        lines.append(f"- **Path:** `{fig.image_path}`")
        lines.append(f"- **Alt:** {fig.alt_text or '(none)'}")
        lines.append(f"- **Source:** {fig.source_file}:{fig.line_number}")
        lines.append(f"- **Status:** {status}")
        if fig.exists:
            lines.append("")
            if "://" in fig.image_path:
                embed = fig.image_path
            else:
                absolute_image = (Path(fig.source_file).parent / fig.image_path).resolve()
                embed = Path(os.path.relpath(absolute_image, output_path.parent)).as_posix()
            lines.append(f"![{fig.alt_text}]({embed})")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and catalogue figures from Markdown files.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory")
    parser.add_argument("--output", type=Path, help="Path for figure manifest (JSON)")
    parser.add_argument(
        "--gallery",
        type=Path,
        help="Path for gallery page (Markdown)",
    )
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("figures", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_raw_md", "outputs/raw_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "figures.json")).resolve()

    try:
        report = extract_from_tree(input_path) if input_path.is_dir() else extract_from_file(input_path)

        written = write_figure_manifest(report, output_path)
        log.info(
            "found %d figure(s), %d missing -> %s",
            report.total_figures,
            report.missing_files,
            written,
        )

        if args.gallery:
            gallery = write_gallery_page(report, args.gallery)
            log.info("gallery page -> %s", gallery)

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
