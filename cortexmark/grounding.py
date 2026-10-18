"""Ground generated document fields against extracted source text."""

from __future__ import annotations

import re
from typing import Any

MAX_TITLE_CHARS = 250
MAX_TITLE_SENTENCES = 2


def _normalize(value: str) -> str:
    """Lowercase text and collapse non-alphanumeric characters to spaces."""
    characters = (character.lower() if character.isalnum() else " " for character in value)
    return " ".join("".join(characters).split())


def _tokens(value: str) -> set[str]:
    return set(_normalize(value).split())


def is_grounded(value: str, source_text: str, *, threshold: float = 0.7) -> tuple[bool, float]:
    """Return whether a generated value is supported by source text and its token coverage."""
    normalized_value = _normalize(value)
    normalized_source = _normalize(source_text)
    value_tokens = _tokens(value)
    source_tokens = _tokens(source_text)
    coverage = len(value_tokens & source_tokens) / max(1, len(value_tokens))
    grounded = bool(normalized_value) and (normalized_value in normalized_source or coverage >= threshold)
    return grounded, coverage


def _plausibility(field_name: str, value: str) -> tuple[bool, str]:
    """Return whether a generated value has a plausible shape for its field."""
    if field_name != "title":
        return True, "plausible"
    if len(value.strip()) > MAX_TITLE_CHARS:
        return False, f"title exceeds {MAX_TITLE_CHARS} characters"
    sentence_count = len(re.findall(r"[.!?](?=\s|$)", value.strip()))
    if sentence_count > MAX_TITLE_SENTENCES:
        return False, f"title spans more than {MAX_TITLE_SENTENCES} sentences"
    return True, "plausible"


def _ground_field(field_name: str, value: str, source_text: str) -> dict[str, Any]:
    present_in_source, coverage = is_grounded(value, source_text)
    plausible, plausibility_reason = _plausibility(field_name, value)
    grounded = present_in_source and plausible
    if not plausible:
        reason = plausibility_reason
    elif present_in_source:
        reason = "supported by source"
    else:
        reason = "not sufficiently supported by source"
    return {
        "value": value,
        "grounded": grounded,
        "coverage": coverage,
        "plausible": plausible,
        "reason": reason,
    }


def ground_fields(
    fields: dict[str, str | list[str]],
    source_text: str,
) -> dict[str, dict[str, Any]]:
    """Ground each non-empty scalar or list field against source text."""
    grounded_fields: dict[str, dict[str, Any]] = {}
    for field_name, raw_value in fields.items():
        if isinstance(raw_value, list):
            values = [value for value in raw_value if value.strip()]
            if not values:
                continue
            elements = [_ground_field(field_name, value, source_text) for value in values]
            grounded_count = sum(bool(element["grounded"]) for element in elements)
            plausible = all(bool(element["plausible"]) for element in elements)
            grounded = grounded_count > len(elements) / 2 and plausible
            grounded_fields[field_name] = {
                "value": values,
                "grounded": grounded,
                "coverage": sum(float(element["coverage"]) for element in elements) / len(elements),
                "plausible": plausible,
                "reason": "supported by source" if grounded else "one or more values are not grounded",
                "elements": elements,
            }
            continue

        if not raw_value.strip():
            continue
        grounded_fields[field_name] = _ground_field(field_name, raw_value, source_text)
    return grounded_fields


def build_grounding_report(
    source_text: str,
    *,
    metadata: dict[str, Any] | None,
    topics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build an explicit source-versus-generated grounding report."""
    metadata = metadata or {}
    topics = topics or {}
    fields: dict[str, str | list[str]] = {
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", []),
        "journal": metadata.get("journal", ""),
        "doi": metadata.get("doi", ""),
        "keywords": metadata.get("keywords", []),
        "primary_topic": topics.get("primary_topic", ""),
    }
    generated_fields = ground_fields(fields, source_text)

    atomic_results: list[tuple[str, dict[str, Any]]] = []
    for field_name, result in generated_fields.items():
        elements = result.get("elements")
        if elements is None:
            atomic_results.append((field_name, result))
        else:
            atomic_results.extend((field_name, element) for element in elements)

    ungrounded = [
        {
            "field": field_name,
            "value": str(result["value"]),
            "coverage": float(result["coverage"]),
            "plausible": bool(result["plausible"]),
            "reason": str(result["reason"]),
        }
        for field_name, result in atomic_results
        if not result["grounded"]
    ]
    n_grounded = sum(bool(result["grounded"]) for _, result in atomic_results)
    return {
        "source_channel": "pdf-extracted-text (raw/cleaned/ir explicit+ocr blocks)",
        "generated_fields": generated_fields,
        "n_generated": len(atomic_results),
        "n_grounded": n_grounded,
        "n_ungrounded": len(ungrounded),
        "ungrounded": ungrounded,
    }


__all__ = ["build_grounding_report", "ground_fields", "is_grounded"]
