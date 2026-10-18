from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

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

# Default split levels match original behaviour (#, ##)
DEFAULT_SPLIT_LEVELS: list[int] = [1, 2]
MAX_CHUNK_CHARS = 8000
TARGET_CHUNK_CHARS = 4000
CHUNK_FORMAT_VERSION = 2


def _chunk_fingerprint(split_levels: list[int] | None) -> str:
    return config_fingerprint(
        {
            "split_levels": split_levels or DEFAULT_SPLIT_LEVELS,
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "format_version": CHUNK_FORMAT_VERSION,
        }
    )


def build_heading_re(split_levels: list[int] | None = None) -> re.Pattern[str]:
    """Build a heading regex that matches the given heading levels."""
    levels = split_levels or DEFAULT_SPLIT_LEVELS
    alts = "|".join(f"#{{{n}}}" for n in sorted(levels))
    return re.compile(rf"^({alts})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    chapter: str | None = None
    section: str | None = None
    body: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.section or self.chapter or "untitled"

    def render(self) -> str:
        parts: list[str] = []
        if self.chapter:
            parts.append(f"# {self.chapter}")
        if self.section:
            parts.append(f"## {self.section}")
        if parts:
            parts.append("")
        parts.extend(self.body)
        return "\n".join(parts).strip() + "\n"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80]
    return slug or "untitled"


def _body_blocks(lines: list[str]) -> list[list[str]]:
    """Group body lines into paragraphs or complete fenced-code blocks."""
    blocks: list[list[str]] = []
    current: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence and current:
                blocks.append(current)
                current = []
            current.append(line)
            in_fence = not in_fence
            if not in_fence:
                blocks.append(current)
                current = []
            continue
        if in_fence or stripped:
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _split_lines(lines: list[str], target_chars: int) -> list[list[str]]:
    """Split a block on line boundaries without exceeding the target when possible."""
    parts: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        candidate = [*current, line]
        if current and len("\n".join(candidate)) > target_chars:
            parts.append(current)
            current = [line]
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _split_large_block(block: list[str], target_chars: int, max_chars: int) -> list[list[str]]:
    """Split an indivisible paragraph or fence only when it exceeds the hard limit."""
    if len("\n".join(block)) <= max_chars:
        return [block]

    is_complete_fence = len(block) >= 2 and block[0].strip().startswith("```") and block[-1].strip().startswith("```")
    if not is_complete_fence:
        return _split_lines(block, target_chars)

    opening, *content, closing = block
    fence_overhead = len(opening) + len(closing) + 2
    content_target = max(1, target_chars - fence_overhead)
    return [[opening, *part, closing] for part in _split_lines(content, content_target)]


def _split_oversized_chunk(chunk: Chunk, raw_body: list[str]) -> list[Chunk]:
    """Sub-split a heading chunk at paragraph boundaries when it is too large."""
    header_chars = len(Chunk(chapter=chunk.chapter, section=chunk.section).render())
    max_body_chars = max(1, MAX_CHUNK_CHARS - header_chars)
    target_body_chars = min(TARGET_CHUNK_CHARS, max_body_chars)
    blocks = [
        part
        for block in _body_blocks(raw_body)
        for part in _split_large_block(block, target_body_chars, max_body_chars)
    ]

    bodies: list[list[str]] = []
    current: list[str] = []
    for block in blocks:
        candidate = [*current, "", *block] if current else block
        candidate_chunk = Chunk(chapter=chunk.chapter, section=chunk.section, body=candidate)
        if current and len(candidate_chunk.render()) > TARGET_CHUNK_CHARS:
            bodies.append(current)
            current = block
        else:
            current = candidate
    if current:
        bodies.append(current)
    return [Chunk(chapter=chunk.chapter, section=chunk.section, body=body) for body in bodies]


def parse_chunks(text: str, *, split_levels: list[int] | None = None) -> list[Chunk]:
    heading_re = build_heading_re(split_levels)
    levels = sorted(split_levels or DEFAULT_SPLIT_LEVELS)
    min_level = levels[0] if levels else 1

    chapter: str | None = None
    section: str | None = None
    body: list[str] = []
    chunks: list[Chunk] = []
    in_fence = False

    def flush() -> None:
        nonlocal body
        # Regroup into paragraph/fence blocks (fence-aware, so blank lines
        # inside a ``` block survive) and rejoin with a single blank line
        # between blocks, restoring paragraph boundaries that a flat
        # "drop every blank line" filter used to collapse.
        blocks = _body_blocks(body)
        if blocks:
            content: list[str] = []
            for block in blocks:
                if content:
                    content.append("")
                content.extend(block)
            chunk = Chunk(chapter=chapter, section=section, body=content)
            if len(chunk.render()) > MAX_CHUNK_CHARS:
                chunks.extend(_split_oversized_chunk(chunk, body))
            else:
                chunks.append(chunk)
        body = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            body.append(raw_line.rstrip())
            continue
        if in_fence:
            body.append(raw_line.rstrip())
            continue

        match = heading_re.match(stripped)
        if not match:
            body.append(raw_line.rstrip())
            continue

        level_str, title = match.groups()
        level = len(level_str)
        if level == min_level:
            flush()
            chapter = title.strip()
            section = None
            continue
        # Any deeper split level becomes a section
        flush()
        section = title.strip()

    flush()
    return [chunk for chunk in chunks if chunk.body or chunk.section or chunk.chapter]


def chunk_file(input_path: Path, output_dir: Path, *, split_levels: list[int] | None = None) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned markdown not found: {input_path}")

    text = input_path.read_text(encoding="utf-8-sig")
    chunks = parse_chunks(text, split_levels=split_levels)

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("chunk_*.md"):
        stale_path.unlink()

    written: list[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        path = output_dir / f"chunk_{index:03d}_{slugify(chunk.title)}.md"
        path.write_text(chunk.render(), encoding="utf-8")
        written.append(path)
    return written


def chunk_tree(
    input_root: Path,
    output_root: Path,
    *,
    manifest: Manifest | None = None,
    split_levels: list[int] | None = None,
    include_input_root_name: bool = True,
) -> list[Path]:
    markdown_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not markdown_files:
        raise FileNotFoundError(f"No cleaned markdown files found under: {input_root}")

    mirror_directory_tree(input_root, output_root)
    fingerprint = _chunk_fingerprint(split_levels)
    written: list[Path] = []
    for markdown_path in markdown_files:
        relative = markdown_path.relative_to(input_root)
        base_output = output_root / input_root.name if include_input_root_name else output_root
        target_dir = base_output / relative.parent / markdown_path.stem
        existing_outputs = sorted(target_dir.glob("chunk_*.md"))
        if manifest and not manifest.needs_update(
            markdown_path,
            fingerprint=fingerprint,
            outputs=existing_outputs,
        ):
            continue
        chunk_paths = chunk_file(markdown_path, target_dir, split_levels=split_levels)
        written.extend(chunk_paths)
        if manifest:
            manifest.record(markdown_path, fingerprint=fingerprint, outputs=chunk_paths)
    return written


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split cleaned Markdown into logical chapter/section chunks.")
    parser.add_argument("--input", type=Path, help="Cleaned Markdown file or directory")
    parser.add_argument("--output-dir", type=Path, help="Chunk output root directory")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    parser.add_argument("--no-manifest", action="store_true", help="Disable idempotency manifest")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("chunk", cfg)

    chunk_cfg = cfg.get("chunk", {})
    split_levels = chunk_cfg.get("split_levels", DEFAULT_SPLIT_LEVELS)
    fingerprint = _chunk_fingerprint(split_levels)

    source_id = get_source_id(cfg)
    input_path = (
        args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md") / source_id
    ).resolve()
    output_dir = (args.output_dir or resolve_configured_path(cfg, "output_chunks", "outputs/chunks")).resolve()

    manifest = None
    idem_cfg = cfg.get("idempotency", {})
    if idem_cfg.get("enabled", True) and not args.no_manifest:
        manifest = Manifest(resolve_manifest_path(cfg))

    try:
        if input_path.is_dir():
            chunk_paths = chunk_tree(input_path, output_dir, manifest=manifest, split_levels=split_levels)
            if manifest:
                manifest.save()
            log.info("wrote %d chunk(s) for markdown files under %s", len(chunk_paths), input_path)
            return 0

        chunk_paths = chunk_file(input_path, output_dir / input_path.stem, split_levels=split_levels)
        if manifest:
            manifest.record(input_path, fingerprint=fingerprint, outputs=chunk_paths)
            manifest.save()
        log.info("wrote %d chunk(s) to %s", len(chunk_paths), output_dir / input_path.stem)
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1
    except ValueError as exc:
        log.error("invalid input: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
