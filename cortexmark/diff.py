"""Standalone diff and change detection for saved Markdown outputs.

Compares two versions of converted/cleaned Markdown outputs:
  - Line-level diffs (unified format)
  - Section-level change summaries
  - Added/removed/modified file detection for trees
  - JSON report with change statistics

Run explicitly to compare saved outputs; pipeline runners do not invoke it.
"""

from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.common import load_config, resolve_quality_report_path, setup_logging

# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class FileDiff:
    """Diff result for a single file pair."""

    file_path: str
    status: str  # "added", "removed", "modified", "unchanged"
    lines_added: int = 0
    lines_removed: int = 0
    diff_text: str = ""


@dataclass
class TreeDiff:
    """Diff result for a directory tree comparison."""

    files_added: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_unchanged: list[str] = field(default_factory=list)
    file_diffs: list[FileDiff] = field(default_factory=list)
    total_lines_added: int = 0
    total_lines_removed: int = 0


# ── Diff engine ──────────────────────────────────────────────────────────────


def diff_texts(old_text: str, new_text: str, label: str = "") -> FileDiff:
    """Compute unified diff between two text strings."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"old/{label}",
            tofile=f"new/{label}",
            lineterm="\n",
        )
    )

    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    status = "unchanged" if not diff else "modified"

    return FileDiff(
        file_path=label,
        status=status,
        lines_added=added,
        lines_removed=removed,
        diff_text="".join(diff),
    )


def diff_files(old_path: Path, new_path: Path) -> FileDiff:
    """Compute diff between two files."""
    label = old_path.name
    old_text = old_path.read_text(encoding="utf-8") if old_path.exists() else ""
    new_text = new_path.read_text(encoding="utf-8") if new_path.exists() else ""

    if not old_path.exists():
        lines = new_text.count("\n") + (1 if new_text and not new_text.endswith("\n") else 0)
        return FileDiff(file_path=label, status="added", lines_added=lines)
    if not new_path.exists():
        lines = old_text.count("\n") + (1 if old_text and not old_text.endswith("\n") else 0)
        return FileDiff(file_path=label, status="removed", lines_removed=lines)

    return diff_texts(old_text, new_text, label=label)


def diff_trees(old_root: Path, new_root: Path) -> TreeDiff:
    """Compare two directory trees of Markdown files."""
    old_files = {p.relative_to(old_root): p for p in old_root.rglob("*.md") if p.is_file()}
    new_files = {p.relative_to(new_root): p for p in new_root.rglob("*.md") if p.is_file()}

    all_keys = sorted(set(old_files) | set(new_files))
    result = TreeDiff()
    total_added = 0
    total_removed = 0

    for rel in all_keys:
        old_p = old_files.get(rel)
        new_p = new_files.get(rel)
        name = str(rel)

        if old_p is None and new_p is not None:
            text = new_p.read_text(encoding="utf-8")
            lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
            result.files_added.append(name)
            result.file_diffs.append(FileDiff(file_path=name, status="added", lines_added=lines))
            total_added += lines

        elif old_p is not None and new_p is None:
            text = old_p.read_text(encoding="utf-8")
            lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
            result.files_removed.append(name)
            result.file_diffs.append(FileDiff(file_path=name, status="removed", lines_removed=lines))
            total_removed += lines

        else:
            assert old_p is not None and new_p is not None
            fd = diff_texts(
                old_p.read_text(encoding="utf-8"),
                new_p.read_text(encoding="utf-8"),
                label=name,
            )
            result.file_diffs.append(fd)
            if fd.status == "modified":
                result.files_modified.append(name)
            else:
                result.files_unchanged.append(name)
            total_added += fd.lines_added
            total_removed += fd.lines_removed

    result.total_lines_added = total_added
    result.total_lines_removed = total_removed
    return result


# ── Report writer ────────────────────────────────────────────────────────────


def write_diff_report(tree_diff: TreeDiff, output_path: Path) -> Path:
    """Write diff report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "summary": {
            "files_added": len(tree_diff.files_added),
            "files_removed": len(tree_diff.files_removed),
            "files_modified": len(tree_diff.files_modified),
            "files_unchanged": len(tree_diff.files_unchanged),
            "total_lines_added": tree_diff.total_lines_added,
            "total_lines_removed": tree_diff.total_lines_removed,
        },
        "added": tree_diff.files_added,
        "removed": tree_diff.files_removed,
        "modified": tree_diff.files_modified,
        "file_diffs": [asdict(fd) for fd in tree_diff.file_diffs if fd.status != "unchanged"],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def write_unified_diff(tree_diff: TreeDiff, output_path: Path) -> Path:
    """Write concatenated unified diffs as a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for fd in tree_diff.file_diffs:
        if fd.diff_text:
            parts.append(fd.diff_text)
    output_path.write_text("\n".join(parts) + "\n" if parts else "", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two versions of pipeline outputs.")
    parser.add_argument("--old", type=Path, required=True, help="Old output directory or file")
    parser.add_argument("--new", type=Path, required=True, help="New output directory or file")
    parser.add_argument("--output", type=Path, help="Path for diff report (JSON)")
    parser.add_argument("--unified", type=Path, help="Path for unified diff text output")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("diff", cfg)

    old_path = args.old.resolve()
    new_path = args.new.resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "diff_report.json")).resolve()

    try:
        if old_path.is_dir() and new_path.is_dir():
            td = diff_trees(old_path, new_path)
        elif old_path.is_file() and new_path.is_file():
            fd = diff_files(old_path, new_path)
            td = TreeDiff(file_diffs=[fd])
            if fd.status == "modified":
                td.files_modified = [fd.file_path]
            td.total_lines_added = fd.lines_added
            td.total_lines_removed = fd.lines_removed
        else:
            log.error("both --old and --new must be files or both directories")
            return 1

        written = write_diff_report(td, output_path)
        log.info(
            "diff: +%d/-%d lines, %d added, %d removed, %d modified -> %s",
            td.total_lines_added,
            td.total_lines_removed,
            len(td.files_added),
            len(td.files_removed),
            len(td.files_modified),
            written,
        )

        if args.unified:
            write_unified_diff(td, args.unified)
            log.info("unified diff -> %s", args.unified)

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
