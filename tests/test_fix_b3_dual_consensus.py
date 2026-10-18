"""Regression tests for B3 (dual-engine merge + engine-aware consensus).

Covers findings C24/C25 (dual-engine merge dedup + supplement provenance),
C26 (markitdown multi-column garble warning) and C62 (consensus honours engine).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cortexmark import convert
from cortexmark.consensus import document_consensus
from cortexmark.convert import (
    convert_pdf,
    looks_column_garbled,
    merge_docling_markitdown,
)


def _born_digital_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 14 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode())
        data.extend(obj + b"\nendobj\n")
    xref_offset = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode())
    return bytes(data)


# ── C24: 2-column garble is deduplicated, not appended as a near-doubling supplement ──


def _column_scramble(text: str) -> str:
    """Approximate markitdown's 2-column failure: strip inter-word space, keep characters."""
    return text.replace(" ", "")


def test_c24_whitespace_stripped_duplicate_is_deduped_not_doubled() -> None:
    body = (
        "Instruction tuning fine-tunes a large language model on pairs of instructions and "
        "desired responses so the model learns to generalize to unseen tasks. "
        "Whereas large instruction-tuning datasets of increasing quality have been proposed in "
        "recent years, the effect of instruction diversity on generalization remains open. "
        "We study this question through symbolic experiments on Markov string-rewriting tasks."
    )
    docling = f"## Introduction\n\n{body}\n"
    # markitdown re-emits the same content whitespace-stripped (column interleave signature).
    markitdown = _column_scramble(body)

    merged = merge_docling_markitdown(docling, markitdown)

    assert merged == docling  # fully deduplicated: no headingless supplement, no doubling
    assert "<!-- markitdown-supplement -->" not in merged
    assert len(merged) < 1.3 * len(docling)


def test_c24_supplement_is_capped_for_substantial_documents(caplog: pytest.LogCaptureFixture) -> None:
    import random

    docling = "## Backbone\n\n" + "Structured docling paragraph number zero about control theory. " * 200
    assert len(docling) >= 2000
    # Genuinely novel, mutually distinct markitdown paragraphs (random tokens) that would,
    # unchecked, more than double the document; each survives dedup against docling and the
    # others, so the 30% cap is the only thing that bounds the supplement.
    rng = random.Random(1234)

    def _para() -> str:
        return " ".join("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(10)) for _ in range(8))

    novel = "\n\n".join(_para() for _ in range(300))
    with caplog.at_level("WARNING", logger="cortexmark.convert"):
        merged = merge_docling_markitdown(docling, novel)

    assert len(merged) <= len(docling) * 1.35 + 100  # capped near 30% of the backbone
    assert "exceeded" in caplog.text and "backbone" in caplog.text


# ── C25: an appended supplement carries its own heading (no misattribution) ──


def test_c25_supplement_carries_scoped_heading() -> None:
    docling = "## Method\n\nDocling backbone paragraph describing the experimental protocol in detail.\n"
    markitdown = "Genuinely new markitdown paragraph that docling missed entirely and must be preserved verbatim."
    merged = merge_docling_markitdown(docling, markitdown)

    assert "<!-- markitdown-supplement -->" in merged
    assert "## Supplementary Text (markitdown)" in merged
    # The supplement heading is a real level-2 heading so the chunker scopes it separately
    # instead of attributing the tail to the last docling section ("## Method").
    from cortexmark.chunk import parse_chunks

    chunks = parse_chunks(merged, split_levels=[1, 2])
    tail_sections = {chunk.section for chunk in chunks if "Genuinely new markitdown" in "\n".join(chunk.body)}
    assert tail_sections and "Method" not in tail_sections


# ── C26: markitdown-alone multi-column garble is detected and warned ──


def test_c26_looks_column_garbled_detects_run_together_words() -> None:
    garbled = " ".join(["Instructiontuningfinetuningalargelanguagemodelonpairs"] * 6)
    clean = "Instruction tuning fine tunes a large language model on pairs of instructions."
    assert looks_column_garbled(garbled) is True
    assert looks_column_garbled(clean) is False


def test_c26_convert_pdf_warns_for_garbled_markitdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(_born_digital_pdf("placeholder"))
    out_path = tmp_path / "out.md"
    garbled = " ".join(["Whereaslargeinstructiontuningdatasetsofincreasingquality"] * 8)
    monkeypatch.setattr(convert, "convert_with_markitdown", lambda _path: garbled)

    with caplog.at_level("WARNING", logger="cortexmark.convert"):
        convert_pdf(pdf_path, out_path, engine="markitdown", cfg={"convert": {"ocr": "off"}})

    assert "no layout model" in caplog.text and "multi-column" in caplog.text


# ── C62: consensus honours the chosen engine instead of a fixed markitdown baseline ──


def test_c62_confidence_varies_with_engine_reference(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    source = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    pdf_path.write_bytes(_born_digital_pdf(source * 6))

    # engine_markdown overrides the reference: an unrelated reference lowers coverage,
    # a matching one raises it -> the metric is no longer constant across engines.
    matching = document_consensus(pdf_path, engine_markdown=source * 6)
    unrelated = document_consensus(pdf_path, engine_markdown="zulu yankee xray whiskey victor uniform tango")

    assert matching["doc_consensus_confidence"] > unrelated["doc_consensus_confidence"]
    assert matching["doc_consensus_confidence"] != unrelated["doc_consensus_confidence"]


def test_c62_engine_markdown_skips_reconversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(_born_digital_pdf("alpha bravo charlie delta echo foxtrot " * 6))

    def _boom(_path: object) -> str:
        raise AssertionError("convert_with_markitdown must not run when engine_markdown is provided")

    monkeypatch.setattr(convert, "convert_with_markitdown", _boom)

    report = document_consensus(pdf_path, engine_markdown="alpha bravo charlie delta echo foxtrot")
    assert report["engine"] == "markitdown"  # default engine label when cfg omits it


def test_c62_report_exposes_engine_and_defaults_to_markitdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(_born_digital_pdf("alpha bravo charlie delta echo foxtrot golf hotel " * 4))
    seen: dict[str, bool] = {"markitdown_called": False}

    def _fake(_path: object) -> str:
        seen["markitdown_called"] = True
        return "alpha bravo charlie delta echo foxtrot golf hotel"

    monkeypatch.setattr(convert, "convert_with_markitdown", _fake)

    report = document_consensus(pdf_path)  # no cfg -> markitdown baseline preserved
    assert seen["markitdown_called"] is True
    assert report["engine"] == "markitdown"
    assert "coverage_by_engine" in report["per_page"][0]
