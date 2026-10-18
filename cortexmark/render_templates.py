from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from cortexmark.common import (
    get_source_id,
    load_config,
    resolve_configured_path,
    setup_logging,
)

SECTION_HEADING_RE = re.compile(
    r"^##\s+(?:Section|Module|Chapter|Unit|Topic|Week)\s+(\d+)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
LINE_VALUE_RE_TEMPLATE = r"(?m)^(?:##\s+)?{label}:\s*(.+?)\s*$"
ACRONYMS = {"rl": "RL", "mdp": "MDP", "td": "TD", "hjb": "HJB", "pg": "PG"}
DEFAULT_OUTLINE_CANDIDATES: tuple[str, ...] = (
    "00_meta/outline.md",
    "00_meta/source_outline.md",
    "00_meta/syllabus.md",
    "00_meta/course_syllabus.md",
)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def resolve_outline_path(
    raw_root: Path,
    *,
    cfg: dict[str, Any] | None = None,
    override: Path | None = None,
) -> Path | None:
    """Resolve outline path from explicit arg, config, known candidates, then discovery."""
    if override:
        candidate = override if override.is_absolute() else raw_root / override
        if not candidate.exists():
            raise FileNotFoundError(f"Outline file not found: {candidate}")
        return candidate.resolve()

    rt_cfg = (cfg or {}).get("render_templates", {})
    seen: set[str] = set()
    candidates: list[Path] = []
    configured = rt_cfg.get("outline_file")
    if configured:
        candidates.append(Path(str(configured)))
    for rel in DEFAULT_OUTLINE_CANDIDATES:
        candidates.append(Path(rel))

    for rel_path in candidates:
        key = str(rel_path)
        if key in seen:
            continue
        seen.add(key)
        candidate = rel_path if rel_path.is_absolute() else raw_root / rel_path
        if candidate.exists():
            return candidate.resolve()

    meta_root = raw_root / "00_meta"
    if meta_root.exists():
        for pattern in ("*outline*.md", "*outline*.txt", "*syllabus*.md", "*syllabus*.txt"):
            for candidate in sorted(meta_root.glob(pattern)):
                if candidate.is_file():
                    return candidate.resolve()
    return None


def extract_line_value(text: str, label: str, fallback: str) -> str:
    match = re.search(LINE_VALUE_RE_TEMPLATE.format(label=re.escape(label)), text)
    return match.group(1).strip() if match else fallback


def extract_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"(?ms)^##\s+{re.escape(heading)}:?\s*$\n(.*?)(?=^##\s+|\Z)",
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def clean_inline(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def bullet_lines_from_section(section: str) -> list[str]:
    lines: list[str] = []
    for raw_line in section.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("- "):
            lines.append(stripped[2:].strip())
    return lines


def headings_from_markdown(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = re.sub(r"^\s*#{1,6}\s*", "", stripped).strip()
            if title:
                headings.append(title)
    return headings


def paragraphs_from_markdown(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(clean_inline(" ".join(current)))
                current = []
            continue
        if stripped.startswith("#"):
            if current:
                paragraphs.append(clean_inline(" ".join(current)))
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(clean_inline(" ".join(current)))
    return paragraphs


def humanize_topic(slug: str) -> str:
    parts = slug.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    rendered: list[str] = []
    for part in parts:
        lowered = part.lower()
        rendered.append(ACRONYMS.get(lowered, lowered.capitalize()))
    return " ".join(rendered) if rendered else slug


def parse_section_entries(outline_text: str) -> dict[int, dict[str, list[str] | str]]:
    lines = outline_text.splitlines()
    entries: dict[int, dict[str, list[str] | str]] = {}
    current_section: int | None = None
    current_title = ""
    current_bullets: list[str] = []

    def flush() -> None:
        nonlocal current_section, current_title, current_bullets
        if current_section is not None:
            entries[current_section] = {
                "title": clean_inline(current_title),
                "bullets": [clean_inline(item) for item in current_bullets if item.strip()],
            }
        current_section = None
        current_title = ""
        current_bullets = []

    for line in lines:
        heading_match = SECTION_HEADING_RE.match(line.strip())
        if heading_match:
            flush()
            current_section = int(heading_match.group(1))
            current_title = heading_match.group(2).strip()
            continue
        if current_section is None:
            continue
        stripped = line.strip()
        if stripped.startswith("## "):
            flush()
            continue
        if stripped.startswith("- "):
            current_bullets.append(stripped[2:].strip())

    flush()
    return entries


def extract_programs(section: str) -> list[str]:
    bullets = bullet_lines_from_section(section)
    if bullets:
        parts: list[str] = []
        for bullet in bullets:
            parts.extend(item.strip() for item in bullet.split(",") if item.strip())
        return parts
    if section:
        return [item.strip() for item in section.split(",") if item.strip()]
    return []


def first_items(values: list[str], limit: int) -> list[str]:
    unique: list[str] = []
    for value in values:
        stripped = clean_inline(value)
        if stripped and stripped not in unique:
            unique.append(stripped)
        if len(unique) >= limit:
            break
    return unique


def summarize_text(text: str, max_chars: int = 240) -> str:
    cleaned = clean_inline(text)
    if len(cleaned) <= max_chars:
        return cleaned
    sentence_match = re.match(rf"^(.{{0,{max_chars}}}?[.!?])(?:\s|$)", cleaned)
    if sentence_match:
        return sentence_match.group(1).strip()
    return cleaned[: max_chars - 1].rstrip() + "…"


def build_source_profile_text(
    source_name: str,
    source_cycle: str,
    maintainer: str,
    main_topics: list[str],
    programs: list[str],
    notes: list[str],
) -> str:
    lines = [
        "# Source Profile",
        "",
        "## Source Name",
        source_name,
        "",
        "## Source Cycle",
        source_cycle,
        "",
        "## Maintainer",
        maintainer,
        "",
        "## Primary Topics",
    ]
    lines.extend(f"- {topic}" for topic in main_topics)
    lines.extend(["", "## Recommended Tools"])
    lines.extend(f"- {program}" for program in programs)
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines).strip() + "\n"


def build_global_rules_text(existing_rules: list[str], ai_rules: list[str], admin_rules: list[str]) -> str:
    lines = ["# Global Rules", ""]
    for rule in first_items(existing_rules + ai_rules + admin_rules, limit=12):
        lines.append(f"- {rule}")
    return "\n".join(lines).strip() + "\n"


def build_section_rules_text(
    section_number: int,
    title: str,
    scope_items: list[str],
    exclude_items: list[str],
    output_items: list[str],
    *,
    max_scope: int = 6,
) -> str:
    lines = [
        f"# Section {section_number:02d} Rules",
        "",
        "## Scope",
    ]
    lines.extend(f"- {item}" for item in first_items(scope_items, limit=max_scope))
    lines.extend(["", "## Exclude"])
    lines.extend(f"- {item}" for item in first_items(exclude_items, limit=5))
    lines.extend(["", "## Output"])
    lines.extend(f"- {item}" for item in first_items(output_items, limit=5))
    return "\n".join(lines).strip() + "\n"


def build_assignment_text(
    section_number: int,
    objective: str,
    tasks: list[str],
    submission: list[str],
    *,
    max_tasks: int = 5,
) -> str:
    lines = [
        f"# Section {section_number:02d} Tasks",
        "",
        "## Objective",
        objective,
        "",
        "## Tasks",
    ]
    lines.extend(f"{index + 1}. {task}" for index, task in enumerate(first_items(tasks, limit=max_tasks)))
    lines.extend(["", "## Deliverables"])
    lines.extend(f"- {item}" for item in first_items(submission, limit=4))
    return "\n".join(lines).strip() + "\n"


def render_meta_templates(
    source_root: Path,
    outline_text: str,
    section_entries: dict[int, dict[str, list[str] | str]],
) -> list[Path]:
    source_name = extract_line_value(
        outline_text,
        "Source Name",
        fallback=source_root.name,
    )
    maintainer = extract_line_value(
        outline_text,
        "Maintainer",
        fallback="Not specified",
    )
    source_code = extract_line_value(
        outline_text,
        "Source Code",
        fallback="Not specified",
    )
    programs = extract_programs(extract_section(outline_text, "Recommended Tools"))
    assessment = bullet_lines_from_section(extract_section(outline_text, "Evaluation"))
    if not assessment:
        raw_eval = extract_section(outline_text, "Evaluation")
        assessment = [clean_inline(line) for line in raw_eval.splitlines() if line.strip()]

    main_topics = []
    for entry in section_entries.values():
        title = str(entry["title"])
        main_topics.append(title)
    main_topics = first_items(main_topics, limit=8)

    ai_section = extract_section(outline_text, "Usage Notes")
    other_rules = extract_section(outline_text, "Global Rules")
    ai_rules = bullet_lines_from_section(ai_section)
    admin_rules = bullet_lines_from_section(other_rules)

    notes = [
        f"Source Code: {source_code}",
        "Evaluation: " + "; ".join(assessment) if assessment else "Evaluation: not provided.",
        "This profile and section templates are generated deterministically from available source material.",
    ]

    source_profile_path = source_root / "00_meta" / "source_profile.md"
    source_profile_path.parent.mkdir(parents=True, exist_ok=True)
    source_profile_path.write_text(
        build_source_profile_text(
            source_name=source_name,
            source_cycle="Not specified",
            maintainer=maintainer,
            main_topics=main_topics,
            programs=programs or ["Not specified"],
            notes=notes,
        ),
        encoding="utf-8",
    )

    global_rules_path = source_root / "00_meta" / "global_rules.md"
    existing_rules = [
        "Use only evidence available in the source files.",
        "Do not mix content across unrelated sections or sources.",
        "Keep outputs concise, structured, and technically accurate.",
        "Flag assumptions when source evidence is missing.",
        "Avoid speculative expansion outside the provided material.",
    ]
    global_rules_path.write_text(
        build_global_rules_text(existing_rules=existing_rules, ai_rules=ai_rules, admin_rules=admin_rules),
        encoding="utf-8",
    )
    return [source_profile_path, global_rules_path]


def render_section_templates(
    source_root: Path,
    raw_root: Path,
    cleaned_root: Path,
    section_entries: dict[int, dict[str, list[str] | str]],
    *,
    cfg: dict[str, Any] | None = None,
) -> list[Path]:
    rt_cfg = (cfg or {}).get("render_templates", {})
    max_summary = rt_cfg.get("max_summary_chars", 240)
    max_scope = rt_cfg.get("max_scope_items", 6)
    max_tasks = rt_cfg.get("max_tasks", 5)

    written: list[Path] = []
    section_dirs = [p for p in sorted(source_root.iterdir()) if p.is_dir() and p.name != "00_meta"]
    for index, section_dir in enumerate(section_dirs, start=1):
        section_prefix = section_dir.name.split("_", 1)[0]
        section_number = int(section_prefix) if section_prefix.isdigit() else index
        section_entry = section_entries.get(section_number, {})
        section_title = str(section_entry.get("title", humanize_topic(section_dir.name)))
        section_bullets = [str(item) for item in section_entry.get("bullets", [])]

        raw_content_path = raw_root / section_dir.name / "content.md"
        cleaned_content_path = cleaned_root / section_dir.name / "content.md"
        content_text = read_text(raw_content_path) if raw_content_path.exists() else ""
        if not content_text and cleaned_content_path.exists():
            content_text = read_text(cleaned_content_path)
        content_headings = [
            heading
            for heading in headings_from_markdown(content_text)
            if not re.fullmatch(r"Chapter\s+\d+", heading, flags=re.IGNORECASE)
        ]
        content_paragraphs = paragraphs_from_markdown(content_text)

        scope_items = [
            f"Primary focus: {section_title}",
            *section_bullets,
            *content_headings[:3],
        ]
        exclude_items = [
            "Concepts from unrelated sections or source fragments",
            "Claims not grounded in the source material",
            "Off-topic implementation details",
            "Unnecessary framework-specific detours",
        ]
        output_items = [
            "Produce concise, instructional Markdown output",
            "Include equations, algorithms, or step lists when relevant",
            "Connect key concepts in a structured summary",
            "Use explicit assumptions only when source context is missing",
        ]

        objective_source = content_paragraphs[0] if content_paragraphs else ""
        if objective_source:
            objective = (
                f"Study {section_title} and use this source-grounded framing: "
                f"{summarize_text(objective_source, max_chars=max_summary)}"
            )
        else:
            objective = f"Explain {section_title} using only the available outline/content context."

        task_seed = section_bullets or content_headings or [section_title]
        tasks = [
            f"Explain '{item}' in a concise but technically accurate way." for item in first_items(task_seed, limit=3)
        ]
        tasks.append("Add a short worked example or scenario that reinforces the main idea.")
        tasks.append("Deliver the output using a few clear section headings.")

        submission = [
            "Submit a single, well-structured Markdown file.",
            "Keep terminology scoped to the active section.",
            "Include concise summary + core concepts + example where useful.",
            "Document open assumptions explicitly.",
        ]

        rules_path = section_dir / "rules.md"
        rules_path.write_text(
            build_section_rules_text(
                section_number=section_number,
                title=section_title,
                scope_items=scope_items,
                exclude_items=exclude_items,
                output_items=output_items,
                max_scope=max_scope,
            ),
            encoding="utf-8",
        )
        written.append(rules_path)

        tasks_dir = section_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        task_list_path = tasks_dir / "task_list.md"
        task_list_path.write_text(
            build_assignment_text(
                section_number=section_number,
                objective=objective,
                tasks=tasks,
                submission=submission,
                max_tasks=max_tasks,
            ),
            encoding="utf-8",
        )
        written.append(task_list_path)
    return written


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Populate source templates deterministically from Markdown content.")
    parser.add_argument("--source-root", type=Path, help="Source directory under data/raw")
    parser.add_argument("--raw-root", type=Path, help="Raw markdown root for the same source")
    parser.add_argument("--cleaned-root", type=Path, help="Cleaned markdown root for the same source")
    parser.add_argument(
        "--outline-file",
        type=Path,
        help="Optional outline path (absolute or relative to --raw-root).",
    )
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("render_templates", cfg)

    source_id = get_source_id(cfg)
    source_root = (args.source_root or resolve_configured_path(cfg, "data_raw", "data/raw") / source_id).resolve()
    raw_root = (args.raw_root or resolve_configured_path(cfg, "output_raw_md", "outputs/raw_md") / source_id).resolve()
    cleaned_root = (
        args.cleaned_root or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md") / source_id
    ).resolve()

    try:
        outline_text = ""
        section_entries: dict[int, dict[str, list[str] | str]] = {}
        outline_path = resolve_outline_path(raw_root, cfg=cfg, override=args.outline_file)
        if outline_path:
            outline_text = read_text(outline_path)
            section_entries = parse_section_entries(outline_text)
        else:
            log.warning(
                "No outline file detected under %s; generating templates from folder/content structure only.", raw_root
            )
        written: list[Path] = []
        written.extend(render_meta_templates(source_root, outline_text, section_entries))
        written.extend(render_section_templates(source_root, raw_root, cleaned_root, section_entries, cfg=cfg))
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1
    except ValueError as exc:
        log.error("invalid input: %s", exc)
        return 1

    log.info("populated %d template file(s) under %s", len(written), source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
