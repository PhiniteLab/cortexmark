"""Scholarly metadata extraction from converted Markdown files.

Extracts:
  - Title, authors, abstract, keywords
  - DOI, journal, volume/issue, date
  - Funding acknowledgement

Produces YAML front-matter and optional BibTeX/APA citation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cortexmark.common import load_config, resolve_configured_path, resolve_quality_report_path, setup_logging

# ── Extraction patterns ──────────────────────────────────────────────────────

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;\"')\]]+)")
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
KEYWORDS_HEADER_RE = re.compile(r"^\*{0,2}Keywords?\*{0,2}\s*[:.]?\s*", re.IGNORECASE)
ABSTRACT_HEADER_RE = re.compile(r"^\*{0,2}Abstract\*{0,2}\s*[:.]?\s*", re.IGNORECASE)
FUNDING_RE = re.compile(
    r"(?:fund(?:ed|ing)|grant|support(?:ed)?|acknowledg(?:e|ement))\b.{0,200}",
    re.IGNORECASE,
)
VOLUME_ISSUE_RE = re.compile(
    r"(?:vol(?:ume)?\.?\s*(\d+))[,\s]*(?:(?:no|issue|number)\.?\s*(\d+))?",
    re.IGNORECASE,
)
JOURNAL_LINE_RE = re.compile(
    r"(?:journal|proceedings|transactions|letters|review|annals)\s+(?:of\s+)?[\w\s&-]+",
    re.IGNORECASE,
)
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?(?:references|bibliography|works\s+cited)(?:\*{1,2})?\s*:?\s*$"
    # Non-anchored ALL-CAPS fallback for reference sections flattened onto a single running line by
    # the converter; restricted to a token immediately followed by a numbered-entry marker (e.g.
    # "REFERENCES [1] ...") so prose that merely mentions REFERENCES/BIBLIOGRAPHY is not truncated.
    r"|(?-i:\b(?:REFERENCES|BIBLIOGRAPHY)\b)\s*(?=\[\d)",
    re.IGNORECASE | re.MULTILINE,
)
INSTITUTION_RE = re.compile(
    r"\b(?:universit\w*|institut\w*|departments?|dept|laborator\w*|lab|schools?|colleges?|facult\w*|"
    r"divisions?|centres?|centers?|hospitals?|ministr\w*|academ\w*|foundations?|corporations?|compan(?:y|ies)|"
    r"inc|ltd|gmbh)\b",
    re.IGNORECASE,
)
ADDRESS_RE = re.compile(r"\d{4,}|\b[A-Z]{2}\s+\d{4,5}\b")
NAME_PARTICLE_RE = re.compile(r"^(?:al|da|de|del|der|di|dos|du|la|le|van|von)$", re.IGNORECASE)
# Any Markdown heading level (docling commonly emits the title as H2/H3, not H1).
TITLE_HEADING_RE = re.compile(r"^#{1,6}\s+")
# Glued footnote/affiliation markers a converter can leave stuck to a surname (e.g. "Vaswani1", "Doe2,3").
AUTHOR_FOOTNOTE_SUFFIX_RE = re.compile(r"[0-9,;:†‡*]+$")


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class ScholarlyMetadata:
    """Structured metadata extracted from a document."""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    doi: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    year: str = ""
    emails: list[str] = field(default_factory=list)
    funding: str = ""
    source_file: str = ""
    provenance: dict[str, str] = field(default_factory=dict)


# ── Extraction engine ────────────────────────────────────────────────────────


def extract_title(lines: list[str]) -> str:
    """Extract the document title from the first heading (any level) or first non-empty line."""
    for line in lines[:30]:
        stripped = line.strip()
        if TITLE_HEADING_RE.match(stripped):
            return TITLE_HEADING_RE.sub("", stripped).strip()
    # Fallback: first substantial non-empty line
    for line in lines[:10]:
        stripped = line.strip()
        if stripped and not stripped.startswith(("---", "```", "|", ">")):
            return TITLE_HEADING_RE.sub("", stripped).strip()
    return ""


def _strip_footnote_markers(value: str) -> str:
    """Strip glued footnote/affiliation markers (superscript digits, daggers, asterisks) from each token."""
    tokens = (AUTHOR_FOOTNOTE_SUFFIX_RE.sub("", token) for token in value.split())
    return " ".join(token for token in tokens if token)


def _looks_like_author_name(value: str) -> bool:
    """Return whether a short string has the shape of a person's name."""
    tokens = value.split()
    if not 2 <= len(tokens) <= 6:
        return False
    return all(
        NAME_PARTICLE_RE.fullmatch(token)
        or (token[0].isupper() and all(char.isalpha() or char in "-'’." for char in token))
        for token in tokens
    )


def extract_authors(lines: list[str]) -> list[str]:
    """Heuristic: look for author-like lines near the title (lines 1-20).

    Authors are typically comma/and-separated names after the title.
    """
    authors: list[str] = []
    title_seen = False
    for line in lines[:25]:
        stripped = line.strip()
        if TITLE_HEADING_RE.match(stripped):
            title_seen = True
            continue
        if not title_seen:
            continue
        if not stripped or stripped.startswith(("#", "---", "```", "|", ">")):
            if authors:
                break
            continue
        # Skip lines that look like abstract/keyword headers
        if ABSTRACT_HEADER_RE.match(stripped) or KEYWORDS_HEADER_RE.match(stripped):
            break
        if INSTITUTION_RE.search(stripped) or ADDRESS_RE.search(stripped) or EMAIL_RE.search(stripped):
            continue
        parts = re.split(r",\s*|\s+and\s+", stripped, flags=re.IGNORECASE)
        for part in parts:
            name = _strip_footnote_markers(part.strip().rstrip("."))
            if (
                name
                and not INSTITUTION_RE.search(name)
                and not ADDRESS_RE.search(name)
                and _looks_like_author_name(name)
            ):
                authors.append(name)
        if authors:
            break
    return authors


def extract_abstract(text: str) -> str:
    """Extract abstract section from the document text."""
    lines = text.split("\n")
    collecting = False
    abstract_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if ABSTRACT_HEADER_RE.match(stripped):
            # Start collecting; strip the header prefix
            remainder = ABSTRACT_HEADER_RE.sub("", stripped).strip()
            if remainder:
                abstract_lines.append(remainder)
            collecting = True
            continue
        if collecting:
            # Stop at next heading, keywords, or empty after content
            if stripped.startswith("#") or KEYWORDS_HEADER_RE.match(stripped):
                break
            if not stripped and abstract_lines:
                break
            if stripped:
                abstract_lines.append(stripped)

    return " ".join(abstract_lines).strip()


def extract_keywords(text: str) -> list[str]:
    """Extract keywords from a Keywords: line."""
    for line in text.split("\n"):
        stripped = line.strip()
        if KEYWORDS_HEADER_RE.match(stripped):
            remainder = KEYWORDS_HEADER_RE.sub("", stripped).strip()
            if remainder:
                # Split by comma or semicolon
                parts = re.split(r"[;,]\s*", remainder)
                return [kw.strip().rstrip(".") for kw in parts if kw.strip()]
    return []


def _text_before_references(text: str) -> str:
    """Return document text excluding its bibliography or references section."""
    heading = REFERENCE_HEADING_RE.search(text)
    return text[: heading.start()] if heading else text


def extract_doi(text: str) -> str:
    """Extract the first DOI before the reference list."""
    m = DOI_RE.search(_text_before_references(text))
    return m.group(1).rstrip(".") if m else ""


def extract_year(text: str) -> str:
    """Extract the most likely publication year from the first 50 lines."""
    lines_text = "\n".join(_text_before_references(text).split("\n")[:50])
    years = YEAR_RE.findall(lines_text)
    if not years:
        return ""
    # Prefer 4-digit years in a reasonable range
    valid = [y for y in years if 1950 <= int(y) <= 2030]
    return valid[0] if valid else ""


def extract_emails(text: str) -> list[str]:
    """Extract email addresses from the document."""
    return list(dict.fromkeys(EMAIL_RE.findall(text[:3000])))


def extract_journal(text: str) -> str:
    """Heuristic: find a journal name in front matter before the reference list."""
    lines_text = "\n".join(_text_before_references(text).split("\n")[:30])
    m = JOURNAL_LINE_RE.search(lines_text)
    return m.group(0).strip() if m else ""


def extract_volume_issue(text: str) -> tuple[str, str]:
    """Extract volume and issue numbers."""
    lines_text = "\n".join(text.split("\n")[:30])
    m = VOLUME_ISSUE_RE.search(lines_text)
    if m:
        return m.group(1) or "", m.group(2) or ""
    return "", ""


def extract_funding(text: str) -> str:
    """Extract funding/acknowledgement snippet."""
    m = FUNDING_RE.search(text)
    if m:
        return m.group(0).strip()[:300]
    return ""


def extract_metadata(text: str, source_file: str = "") -> ScholarlyMetadata:
    """Run all extractors on a document and return structured metadata."""
    lines = text.split("\n")
    title = extract_title(lines)
    authors = extract_authors(lines)
    journal = extract_journal(text)
    funding = extract_funding(text)
    volume, issue = extract_volume_issue(text)
    provenance: dict[str, str] = {}
    if title:
        has_title_heading = any(line.strip().startswith("# ") for line in lines[:30])
        provenance["title"] = "heading" if has_title_heading else "low-confidence-heuristic"
    if authors:
        provenance["authors"] = "heuristic-byline"
    if journal:
        provenance["journal"] = "low-confidence-heuristic"
    if funding:
        provenance["funding"] = "low-confidence-heuristic"

    return ScholarlyMetadata(
        title=title,
        authors=authors,
        abstract=extract_abstract(text),
        keywords=extract_keywords(text),
        doi=extract_doi(text),
        journal=journal,
        volume=volume,
        issue=issue,
        year=extract_year(text),
        emails=extract_emails(text),
        funding=funding,
        source_file=source_file,
        provenance=provenance,
    )


# ── Output formatters ────────────────────────────────────────────────────────


def to_yaml_frontmatter(meta: ScholarlyMetadata) -> str:
    """Render metadata as YAML front-matter block."""

    def quoted(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    parts = ["---"]
    if meta.title:
        parts.append(f"title: {quoted(meta.title)}")
    if meta.authors:
        parts.append("authors:")
        for a in meta.authors:
            parts.append(f"  - {quoted(a)}")
    if meta.doi:
        parts.append(f"doi: {quoted(meta.doi)}")
    if meta.journal:
        parts.append(f"journal: {quoted(meta.journal)}")
    if meta.volume:
        parts.append(f"volume: {quoted(meta.volume)}")
    if meta.issue:
        parts.append(f"issue: {quoted(meta.issue)}")
    if meta.year:
        parts.append(f"year: {quoted(meta.year)}")
    if meta.keywords:
        parts.append("keywords:")
        for kw in meta.keywords:
            parts.append(f"  - {quoted(kw)}")
    if meta.abstract:
        # Truncate for front-matter
        abstract_short = meta.abstract[:500]
        parts.append(f"abstract: {quoted(abstract_short)}")
    if meta.funding:
        parts.append(f"funding: {quoted(meta.funding)}")
    if meta.provenance:
        parts.append("provenance:")
        for field_name, evidence in meta.provenance.items():
            parts.append(f"  {field_name}: {quoted(evidence)}")
    parts.append("---")
    return "\n".join(parts) + "\n"


def to_bibtex(meta: ScholarlyMetadata) -> str:
    """Render metadata as a BibTeX entry."""
    # Generate a cite key from first author surname + year
    cite_key = "unknown"
    if meta.authors:
        surname = meta.authors[0].split()[-1].lower() if meta.authors[0] else "unknown"
        cite_key = f"{surname}{meta.year}" if meta.year else surname
    lines = [f"@article{{{cite_key},"]
    if meta.title:
        lines.append(f"  title = {{{meta.title}}},")
    if meta.authors:
        lines.append(f"  author = {{{' and '.join(meta.authors)}}},")
    if meta.journal:
        lines.append(f"  journal = {{{meta.journal}}},")
    if meta.volume:
        lines.append(f"  volume = {{{meta.volume}}},")
    if meta.issue:
        lines.append(f"  number = {{{meta.issue}}},")
    if meta.year:
        lines.append(f"  year = {{{meta.year}}},")
    if meta.doi:
        lines.append(f"  doi = {{{meta.doi}}},")
    lines.append("}")
    return "\n".join(lines) + "\n"


def to_apa7(meta: ScholarlyMetadata) -> str:
    """Render metadata as an APA 7 citation string."""
    parts: list[str] = []
    # Authors
    if meta.authors:
        author_str = ", ".join(meta.authors)
        parts.append(author_str)
    # Year
    if meta.year:
        parts.append(f"({meta.year}).")
    else:
        parts.append("(n.d.).")
    # Title
    if meta.title:
        parts.append(f"{meta.title}.")
    # Journal (italic)
    if meta.journal:
        journal_part = f"*{meta.journal}*"
        if meta.volume:
            journal_part += f", *{meta.volume}*"
        if meta.issue:
            journal_part += f"({meta.issue})"
        journal_part += "."
        parts.append(journal_part)
    # DOI
    if meta.doi:
        parts.append(f"https://doi.org/{meta.doi}")
    return " ".join(parts)


# ── File/tree operations ─────────────────────────────────────────────────────


def extract_file(file_path: Path) -> ScholarlyMetadata:
    """Extract metadata from a single Markdown file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    # Tolerant read: a non-UTF-8 input degrades to U+FFFD replacement chars instead of a raw
    # UnicodeDecodeError traceback (matches the errors="replace" convention already used by the
    # QA-focused module readers such as formula_validate/ocr_quality).
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return extract_metadata(text, source_file=str(file_path))


def extract_tree(input_root: Path) -> list[ScholarlyMetadata]:
    """Extract metadata from all Markdown files under a directory."""
    md_files = sorted(p for p in input_root.rglob("*.md") if p.is_file())
    if not md_files:
        raise FileNotFoundError(f"No markdown files found under: {input_root}")
    return [extract_file(f) for f in md_files]


def write_metadata_report(metadatas: list[ScholarlyMetadata], output_path: Path) -> Path:
    """Write metadata as a JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "documents": len(metadatas),
        "entries": [asdict(m) for m in metadatas],
    }
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract scholarly metadata from converted Markdown files.")
    parser.add_argument("--input", type=Path, help="Markdown file or directory to scan")
    parser.add_argument("--output", type=Path, help="Path for JSON metadata report")
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "bibtex", "apa"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("metadata", cfg)

    input_path = (args.input or resolve_configured_path(cfg, "output_cleaned_md", "outputs/cleaned_md")).resolve()
    output_path = (args.output or resolve_quality_report_path(cfg, "metadata_report.json")).resolve()

    try:
        metadatas = extract_tree(input_path) if input_path.is_dir() else [extract_file(input_path)]

        if args.format == "json":
            written = write_metadata_report(metadatas, output_path)
            log.info("wrote metadata report for %d file(s) → %s", len(metadatas), written)
        elif args.format == "yaml":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            parts = [to_yaml_frontmatter(m) for m in metadatas]
            output_path.write_text("\n".join(parts), encoding="utf-8")
            log.info("wrote YAML front-matter for %d file(s) → %s", len(metadatas), output_path)
        elif args.format == "bibtex":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            parts = [to_bibtex(m) for m in metadatas]
            output_path.write_text("\n".join(parts), encoding="utf-8")
            log.info("wrote BibTeX for %d file(s) → %s", len(metadatas), output_path)
        elif args.format == "apa":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            parts = [to_apa7(m) for m in metadatas]
            output_path.write_text("\n".join(parts), encoding="utf-8")
            log.info("wrote APA 7 for %d file(s) → %s", len(metadatas), output_path)

    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
