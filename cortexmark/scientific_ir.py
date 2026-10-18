"""Shared scientific object model for scholarly structure and retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cortexmark.citation_ir import slugify_identifier

OBJECT_THEOREM = "theorem"
OBJECT_PROOF = "proof"
OBJECT_DEFINITION = "definition"
OBJECT_EQUATION = "equation"
OBJECT_NOTATION = "notation"
OBJECT_ALGORITHM = "algorithm"
OBJECT_EXAMPLE = "example"
OBJECT_REMARK = "remark"
OBJECT_NARRATIVE = "narrative"


@dataclass
class ScientificObject:
    """A stable scholarly object that can be linked across modules."""

    object_id: str
    object_type: str
    object_kind: str = ""
    label: str = ""
    name: str = ""
    title: str = ""
    source_file: str = ""
    line_number: int = 0
    chapter: str = ""
    section: str = ""
    text: str = ""
    evidence_level: str = "explicit"
    parent_object_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    page_no: int = 0
    bbox: tuple[float, float, float, float] | None = None
    engine: str = ""


@dataclass
class ScientificObjectLink:
    """A typed edge between scientific objects or object references."""

    source_object_id: str
    target_object_id: str = ""
    relation: str = ""
    status: str = "resolved"
    source_label: str = ""
    target_label: str = ""
    source_file: str = ""
    line_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def stable_source_label(file_path: Path, *, root: Path | None = None) -> str:
    """Return a cwd-independent source label, relative to root when provided."""
    if root is not None:
        try:
            return file_path.relative_to(root).as_posix()
        except ValueError:
            pass
    return file_path.name


def make_object_id(
    source_file: str,
    object_type: str,
    *,
    label: str = "",
    name: str = "",
    line_number: int = 0,
    ordinal: int = 0,
) -> str:
    """Build a deterministic, path-stable object ID."""
    source_part = slugify_identifier(source_file or "document", default="document")
    semantic_part = slugify_identifier(label or name or object_type, default=object_type)
    raw = f"{source_file}|{object_type}|{label}|{name}|{line_number}|{ordinal}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:10]
    return f"{source_part}:{object_type}:{semantic_part}:{digest}"
