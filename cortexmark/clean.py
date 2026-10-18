from __future__ import annotations

import argparse
import html
import re
from collections import Counter
from pathlib import Path
from typing import Any

from cortexmark.common import (
    Manifest,
    config_fingerprint,
    get_source_id,
    load_config,
    mirror_directory_tree,
    resolve_configured_path,
    resolve_manifest_path,
    setup_logging,
)

PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^\s*chapter\b[:\s-]*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*section\b[:\s-]*(.*)$", re.IGNORECASE)
COMBINED_HEADING_RE = re.compile(
    r"^\s*chapter\b[:\s-]*(.*?)\s+section\b[:\s-]*(.*)$",
    re.IGNORECASE,
)
TABLE_CELL_STRIP_RE = re.compile(r"\s{2,}")
INLINE_CODE_SPAN_RE = re.compile(r"(`[^`]*`)")


def _fence_mask(lines: list[str]) -> list[bool]:
    """Flag lines that fall strictly inside a fenced (```) code block.

    Fence delimiter lines themselves are never flagged, so they stay eligible
    for header/footer/page-number/heading processing; only the raw content
    between a pair of delimiters is protected from stripping or deletion.
    """
    mask: list[bool] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            mask.append(False)
            in_fence = not in_fence
        else:
            mask.append(in_fence)
    return mask


def strip_markdown_heading(line: str) -> str:
    return re.sub(r"^\s*#{1,6}\s*", "", line).strip()


def normalize_heading_lines(line: str) -> list[str]:
    original = line.strip()
    stripped = strip_markdown_heading(line)
    if not stripped:
        return [""]

    combined_match = COMBINED_HEADING_RE.match(stripped)
    if combined_match:
        chapter_title = combined_match.group(1).strip() or "Chapter"
        section_title = combined_match.group(2).strip() or "Section"
        return [f"# {chapter_title}", f"## {section_title}"]

    chapter_match = CHAPTER_RE.match(stripped)
    if chapter_match:
        title = chapter_match.group(1).strip() or stripped
        return [f"# {title}" if title != stripped else f"# {stripped}"]

    section_match = SECTION_RE.match(stripped)
    if section_match:
        title = section_match.group(1).strip() or stripped
        return [f"## {title}" if title != stripped else f"## {stripped}"]

    if original.startswith("#"):
        return [original]

    return [stripped]


def remove_page_numbers(lines: list[str]) -> list[str]:
    mask = _fence_mask(lines)
    return [
        line for line, protected in zip(lines, mask, strict=True) if protected or not PAGE_NUMBER_RE.match(line.strip())
    ]


def remove_repeated_headers_footers(lines: list[str], *, min_count: int = 3, max_length: int = 80) -> list[str]:
    mask = _fence_mask(lines)
    normalized = [line.strip() for line, protected in zip(lines, mask, strict=True) if line.strip() and not protected]
    counts = Counter(normalized)
    repeated = {
        line
        for line, count in counts.items()
        if count >= min_count
        and len(line) <= max_length
        and not line.startswith("#")
        and not line.startswith("##")
        and not line.startswith("```")
        and not line.startswith(">")
        # A pipe-table row repeated across pages is usually a legitimate table
        # header/separator that recurs once per page, not running boilerplate.
        and not (line.startswith("|") and line.endswith("|"))
    }
    return [line for line, protected in zip(lines, mask, strict=True) if protected or line.strip() not in repeated]


def fix_wrapped_lines(lines: list[str]) -> list[str]:
    rebuilt: list[str] = []
    paragraph: list[str] = []
    in_code_block = False

    def flush_paragraph() -> None:
        if not paragraph:
            return
        merged = paragraph[0]
        for fragment in paragraph[1:]:
            if merged.endswith("-") and fragment and fragment[0].islower():
                merged = merged[:-1] + fragment.lstrip()
            else:
                merged = merged.rstrip() + " " + fragment.lstrip()
        rebuilt.append(merged.strip())
        paragraph.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
            rebuilt.append(stripped)
            in_code_block = not in_code_block
            continue
        if in_code_block:
            rebuilt.append(line)
            continue
        if not stripped:
            flush_paragraph()
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
            rebuilt.append(stripped)
            continue
        if stripped.startswith("#"):
            flush_paragraph()
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
            rebuilt.append(stripped)
            rebuilt.append("")
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            rebuilt.append(stripped)
            continue
        if re.match(r"^(?:[-*+]|\d+[.)])\s+", stripped):
            flush_paragraph()
            rebuilt.append(stripped)
            continue
        paragraph.append(stripped)

    flush_paragraph()

    while rebuilt and rebuilt[-1] == "":
        rebuilt.pop()

    compact: list[str] = []
    for item in rebuilt:
        if item == "" and compact and compact[-1] == "":
            continue
        compact.append(item)
    return compact


def clean_markdown(text: str, *, min_repeated: int = 3, max_header_len: int = 80) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = remove_page_numbers(lines)
    lines = remove_repeated_headers_footers(lines, min_count=min_repeated, max_length=max_header_len)
    mask = _fence_mask(lines)
    normalized_lines: list[str] = []
    for line, protected in zip(lines, mask, strict=True):
        if protected:
            # Inside a fence, indentation IS the content (algorithm/pseudocode
            # nesting, table columns); never route it through heading stripping.
            normalized_lines.append(line)
        else:
            normalized_lines.extend(normalize_heading_lines(line))
    cleaned_lines = fix_wrapped_lines(normalized_lines)
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = normalize_table_blocks(cleaned_text)
    cleaned_text = decode_html_entities(cleaned_text)
    return cleaned_text.strip() + "\n"


def decode_html_entities(text: str) -> str:
    """Decode HTML entities (&amp; &lt; &gt; ...) outside fenced code and inline code spans.

    Converter output (docling) HTML-escapes prose it serializes, leaving
    literal '&amp;'/'&gt;'/'&lt;' in citations and inequalities. Code content
    is left untouched since an entity there may be meaningfully escaped.
    """
    lines = text.split("\n")
    mask = _fence_mask(lines)
    decoded_lines = [
        line if protected else _decode_entities_outside_inline_code(line)
        for line, protected in zip(lines, mask, strict=True)
    ]
    return "\n".join(decoded_lines)


def _decode_entities_outside_inline_code(line: str) -> str:
    parts = INLINE_CODE_SPAN_RE.split(line)
    return "".join(part if index % 2 else html.unescape(part) for index, part in enumerate(parts))


def normalize_table_cell(cell: str) -> str:
    """Collapse internal whitespace in a table cell."""
    return TABLE_CELL_STRIP_RE.sub(" ", cell.strip())


def normalize_table_blocks(text: str) -> str:
    """Normalize Markdown table blocks: trim cells, ensure separator alignment, remove empty tables."""
    result_lines: list[str] = []
    table_lines: list[str] = []
    in_table = False

    def flush_table() -> None:
        if not table_lines:
            return
        # Parse rows into cells
        rows: list[list[str]] = []
        for tl in table_lines:
            cells = [normalize_table_cell(c) for c in tl.strip().strip("|").split("|")]
            rows.append(cells)

        if not rows:
            return

        # Find separator row (all cells match ---+ pattern)
        sep_idx = None
        for i, row in enumerate(rows):
            if all(re.match(r"^:?-{1,}:?$", c.strip()) or c.strip() == "" for c in row):
                sep_idx = i
                break

        # If no separator found, look for header + data pattern and insert one
        if sep_idx is None and len(rows) >= 2:
            sep_idx = 1
            rows.insert(1, ["---"] * len(rows[0]))

        # Normalise column count
        col_count = max(len(r) for r in rows)
        for i, row in enumerate(rows):
            while len(row) < col_count:
                row.append("---" if (sep_idx is not None and i == sep_idx) else "")

        # Drop only tables with no content anywhere: a header-plus-separator table
        # still carries data (docling renders single-row Q&A pairs in that shape).
        non_sep_rows = [r for idx, r in enumerate(rows) if idx != sep_idx]
        has_content = any(c.strip() and not re.match(r"^:?-{1,}:?$", c.strip()) for r in non_sep_rows for c in r)
        if not has_content:
            return

        # Rebuild
        for row in rows:
            result_lines.append("| " + " | ".join(row) + " |")

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(stripped)
        else:
            if in_table:
                flush_table()
                table_lines = []
                in_table = False
            result_lines.append(line)

    if in_table:
        flush_table()

    return "\n".join(result_lines)


def clean_file(input_path: Path, output_path: Path, *, cfg: dict[str, Any] | None = None) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input markdown not found: {input_path}")

    clean_cfg = (cfg or {}).get("clean", {})
    cleaned = clean_markdown(
        input_path.read_text(encoding="utf-8"),
        min_repeated=clean_cfg.get("min_repeated_header_count", 3),
        max_header_len=clean_cfg.get("max_repeated_header_length", 80),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    return output_path


def _clean_fingerprint(cfg: dict[str, Any] | None) -> str:
    return config_fingerprint((cfg or {}).get("clean", {}))


def clean_tree(
    input_root: Path,
    output_root: Path,
    *,
    cfg: dict[str, Any] | None = None,
    manifest: Manifest | None = None,
    include_input_root_name: bool = True,
) -> list[Path]:
    markdown_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not markdown_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")

    mirror_directory_tree(input_root, output_root)
    fingerprint = _clean_fingerprint(cfg)
    written: list[Path] = []
    for markdown_path in markdown_files:
        relative = markdown_path.relative_to(input_root)
        base_output = output_root / input_root.name if include_input_root_name else output_root
        output_path = base_output / relative
        if manifest and not manifest.needs_update(markdown_path, fingerprint=fingerprint, outputs=(output_path,)):
            continue
        written.append(clean_file(markdown_path, output_path, cfg=cfg))
        if manifest:
            manifest.record(markdown_path, fingerprint=fingerprint, outputs=(output_path,))
    return written


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean raw Markdown extracted from PDF files.")
    parser.add_argument("--input", type=Path, help="Raw Markdown file or directory")
    parser.add_argument("--output-dir", type=Path, help="Root directory for cleaned Markdown output")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    parser.add_argument("--no-manifest", action="store_true", help="Disable idempotency manifest")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("clean", cfg)
    fingerprint = _clean_fingerprint(cfg)

    source_id = get_source_id(cfg)
    input_path = (args.input or resolve_configured_path(cfg, "output_raw_md", "outputs/raw_md") / source_id).resolve()
    output_dir = (args.output_dir or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()

    manifest = None
    idem_cfg = cfg.get("idempotency", {})
    if idem_cfg.get("enabled", True) and not args.no_manifest:
        manifest = Manifest(resolve_manifest_path(cfg))

    try:
        if input_path.is_dir():
            written = clean_tree(input_path, output_dir, cfg=cfg, manifest=manifest)
            if manifest:
                manifest.save()
            log.info("cleaned %d markdown file(s) under %s", len(written), input_path)
            return 0

        output_path = output_dir / input_path.name
        written_path = clean_file(input_path, output_path, cfg=cfg)
        if manifest:
            manifest.record(input_path, fingerprint=fingerprint, outputs=(written_path,))
            manifest.save()
        log.info("wrote cleaned markdown to %s", written_path)
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1
    except ValueError as exc:
        log.error("invalid input: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
