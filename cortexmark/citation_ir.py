"""Canonical citation/reference intermediate representation helpers.

This module introduces a reusable citation IR that can be shared across
multiple scholarly-analysis modules without forcing an immediate rewrite of
their public outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_YEAR_SUFFIX_RE = re.compile(r"(?P<year>\d{4}[a-z]?)$", re.IGNORECASE)


@dataclass
class CitationMention:
    """A single inline citation occurrence or inline citation cluster."""

    raw_text: str
    surface_text: str = ""
    source_file: str = ""
    line_number: int = 0
    cite_type: str = ""  # "author-year" | "numeric"
    mention_id: str = ""
    target_hints: list[str] = field(default_factory=list)


@dataclass
class Reference:
    """A canonical bibliography/reference entry."""

    key: str
    raw_text: str
    authors: str = ""
    year: str = ""
    title: str = ""
    doi: str = ""
    source_file: str = ""
    line_number: int = 0
    ref_type: str = ""  # "numeric" | "author-year"
    reference_id: str = ""
    aliases: list[str] = field(default_factory=list)
    normalized_title: str = ""


@dataclass
class CitationLink:
    """Link from a citation mention to a reference target."""

    source_doc: str
    target_ref: str
    mention_id: str = ""
    reference_id: str = ""
    candidate_reference_ids: list[str] = field(default_factory=list)
    status: str = "resolved"  # "resolved" | "missing" | "ambiguous"
    confidence: float = 1.0


@dataclass
class DuplicateReferenceCluster:
    """References that appear to describe the same work."""

    reason: str
    signature: str
    reference_ids: list[str] = field(default_factory=list)


@dataclass
class CitationAudit:
    """Audit summary for citation/reference integrity."""

    missing_references: list[str] = field(default_factory=list)
    phantom_references: list[str] = field(default_factory=list)
    ambiguous_references: list[str] = field(default_factory=list)
    duplicate_references: list[DuplicateReferenceCluster] = field(default_factory=list)


def normalize_author_token(authors: str) -> str:
    """Return a conservative first-author token for matching."""
    cleaned = authors.strip()
    if not cleaned:
        return ""

    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("et al.", "").replace("et al", "")
    cleaned = cleaned.split(" and ", 1)[0].strip()
    cleaned = cleaned.split(",", 1)[0].strip()
    tokens = [tok for tok in re.split(r"\s+", cleaned) if tok]
    if not tokens:
        return ""
    surname = _NON_ALNUM_RE.sub("", tokens[0].lower())
    return surname


def normalize_title(title: str) -> str:
    """Normalize a title for duplicate detection."""
    lowered = title.lower()
    return _NON_ALNUM_RE.sub(" ", lowered).strip()


def build_author_year_key(authors: str, year: str) -> str:
    """Build a stable normalized author-year key."""
    author_token = normalize_author_token(authors)
    year_token = year.strip().lower()
    if author_token and year_token:
        return f"{author_token}{year_token}"
    return author_token or year_token


def parse_author_year_targets(raw_text: str) -> list[str]:
    """Extract normalized author-year target hints from a citation cluster."""
    hints: list[str] = []
    for segment in raw_text.split(";"):
        piece = segment.strip()
        if not piece:
            continue
        match = re.search(r"(?P<authors>.+?)(?:,)?\s*(?P<year>\d{4}[a-z]?)$", piece, re.IGNORECASE)
        if not match:
            continue
        hint = build_author_year_key(match.group("authors"), match.group("year"))
        if hint and hint not in hints:
            hints.append(hint)
    return hints


def parse_numeric_targets(raw_text: str) -> list[str]:
    """Extract numeric citation targets from a citation cluster."""
    return list(dict.fromkeys(re.findall(r"\d+", raw_text)))


def slugify_identifier(value: str, *, default: str = "item") -> str:
    """Create a filesystem/JSON-safe identifier token."""
    lowered = value.lower()
    slug = _NON_ALNUM_RE.sub("-", lowered).strip("-")
    return slug or default


def year_suffix(value: str) -> str:
    """Return a normalized year suffix when present."""
    match = _YEAR_SUFFIX_RE.search(value.strip())
    return match.group("year").lower() if match else ""
