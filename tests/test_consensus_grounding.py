from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from cortexmark import cli, convert
from cortexmark.consensus import document_consensus
from cortexmark.grounding import build_grounding_report, is_grounded


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


def _write_scanned_pdf(path: Path) -> None:
    image = Image.new("RGB", (1400, 500), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=54)
    draw.text((70, 100), "Theorem 3.2 Bellman Optimality", font=font, fill="black")
    draw.text((70, 210), "Value function for a finite MDP", font=font, fill="black")
    image.save(path, "PDF", resolution=150)


def test_born_digital_pdf_has_high_consensus_without_low_agreement(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    source = (
        "CortexMark extracts reliable embedded document tokens for transparent scientific evidence and "
        "reproducible source comparison. "
    ) * 3
    pdf_path.write_bytes(_born_digital_pdf(source))

    report = document_consensus(pdf_path)

    assert report["doc_consensus_confidence"] >= 0.8
    assert report["per_page"]
    assert not any(page["low_agreement"] for page in report["per_page"])


def test_consensus_confidence_uses_document_coverage_not_symmetric_jaccard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    source_tokens = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    pdf_path.write_bytes(_born_digital_pdf(f"{source_tokens} " * 8))
    independent_text = (
        "alpha bravo charlie delta echo foxtrot golf hotel "
        "kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu"
    )
    monkeypatch.setattr(convert, "convert_with_markitdown", lambda _path: independent_text)

    report = document_consensus(pdf_path)

    assert report["doc_consensus_confidence"] == pytest.approx(0.8)
    assert report["token_jaccard"] < 0.4
    assert report["doc_consensus_confidence"] > report["token_jaccard"]


def test_scanned_page_is_not_flagged_as_low_agreement(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    _write_scanned_pdf(pdf_path)

    report = document_consensus(pdf_path)

    assert report["n_pages"] == 1
    assert report["per_page"][0]["low_agreement"] is False


def test_is_grounded_distinguishes_source_value_from_fabrication() -> None:
    source = "The paper introduces CortexMark consensus confidence for extracted PDF evidence."

    assert is_grounded("CortexMark consensus confidence", source) == (True, 1.0)
    fabricated_grounded, fabricated_coverage = is_grounded("quantum banana synthesis", source)
    assert fabricated_grounded is False
    assert fabricated_coverage == 0.0


def test_grounding_report_marks_fabricated_metadata_title_ungrounded() -> None:
    report = build_grounding_report(
        "A factual paper about source evidence.",
        metadata={"title": "Fabricated Quantum Banana Study"},
        topics=None,
    )

    assert report["generated_fields"]["title"]["grounded"] is False
    assert report["n_generated"] == 1
    assert report["n_ungrounded"] == 1
    assert report["ungrounded"][0]["field"] == "title"


def test_grounding_report_rejects_overlong_title_even_when_present_in_source() -> None:
    title = "This is page content presented as metadata rather than a real title. " * 8

    report = build_grounding_report(title, metadata={"title": title}, topics=None)

    title_record = report["generated_fields"]["title"]
    assert title_record["coverage"] == 1.0
    assert title_record["plausible"] is False
    assert title_record["grounded"] is False  # was: True
    assert title_record["reason"] == "title exceeds 250 characters"
    assert report["ungrounded"][0]["reason"] == "title exceeds 250 characters"


def test_run_writes_consensus_grounding_and_summary_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(_born_digital_pdf("CortexMark machine learning source evidence and metadata. " * 4))
    output_root = tmp_path / "output"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    assert cli.main(["run", str(pdf_path), "--out", str(output_root), "--json"]) == 0

    bundle = output_root / "paper"
    consensus = json.loads((bundle / "consensus.json").read_text(encoding="utf-8"))
    grounding = json.loads((bundle / "grounding.json").read_text(encoding="utf-8"))
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    assert summary["consensus_confidence"] == consensus["doc_consensus_confidence"]
    assert summary["low_agreement_pages"] == sum(page["low_agreement"] for page in consensus["per_page"])
    assert summary["n_generated"] == grounding["n_generated"]
    assert summary["n_ungrounded"] == grounding["n_ungrounded"]
