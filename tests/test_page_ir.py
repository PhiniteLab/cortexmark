from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from cortexmark import cli, convert
from cortexmark.page_ir import build_document_ir, classify_pages
from cortexmark.scientific_ir import ScientificObject


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
    draw.text((70, 100), "Page-aware RapidOCR text", font=font, fill="black")
    image.save(path, "PDF", resolution=150)


def _assert_normalized_bbox(bbox: list[float]) -> None:
    assert len(bbox) == 4
    assert all(0.0 <= coordinate <= 1.0 for coordinate in bbox)


def test_scientific_object_page_provenance_defaults() -> None:
    obj = ScientificObject(object_id="object-1", object_type="theorem")

    assert obj.page_no == 0
    assert obj.bbox is None
    assert obj.engine == ""


def test_classify_pages_distinguishes_born_digital_and_scanned(tmp_path: Path) -> None:
    born_path = tmp_path / "born.pdf"
    scanned_path = tmp_path / "scan.pdf"
    born_path.write_bytes(_born_digital_pdf("Embedded text for page classification. " * 4))
    _write_scanned_pdf(scanned_path)

    assert classify_pages(born_path)[0]["page_type"] == "born-digital"
    assert classify_pages(scanned_path)[0]["page_type"] == "scanned"


def test_build_document_ir_extracts_normalized_born_digital_blocks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "born.pdf"
    expected_text = "Recovered born-digital page text"
    pdf_path.write_bytes(_born_digital_pdf(f"{expected_text}. " * 4))

    ir = build_document_ir(pdf_path)

    assert ir["coord_space"] == "normalized-topleft-0-1"
    assert ir["blocks"]
    assert all(block["engine"] == "pypdfium2" for block in ir["blocks"])
    assert expected_text in " ".join(block["text"] for block in ir["blocks"])
    for block in ir["blocks"]:
        _assert_normalized_bbox(block["bbox"])


def test_build_document_ir_extracts_normalized_rapidocr_blocks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    _write_scanned_pdf(pdf_path)

    ir = build_document_ir(pdf_path, ocr_mode="on")

    assert ir["blocks"]
    assert all(block["engine"] == "rapidocr" for block in ir["blocks"])
    assert all(block["page_no"] == 1 for block in ir["blocks"])
    for block in ir["blocks"]:
        _assert_normalized_bbox(block["bbox"])


def test_run_writes_page_ir_and_summary_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(_born_digital_pdf("CortexMark page IR bundle text. " * 5))
    output_root = tmp_path / "output"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    assert cli.main(["run", str(pdf_path), "--out", str(output_root), "--json"]) == 0

    bundle = output_root / "paper"
    ir = json.loads((bundle / "ir.json").read_text(encoding="utf-8"))
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    assert ir["blocks"]
    assert summary["page_types"] == {"born-digital": 1}
    assert summary["n_ir_blocks"] == len(ir["blocks"])


def test_build_document_ir_routes_each_page_of_a_mixed_document(tmp_path: Path) -> None:
    # Page-level routing: a mixed doc must extract page 1 (born-digital) via pypdfium2 and
    # page 2 (scanned image) via RapidOCR, each block carrying its own page_no + engine.
    import pypdfium2 as pdfium

    born = tmp_path / "born.pdf"
    born.write_bytes(_born_digital_pdf("Section 1. Introduction with a real text layer. " * 3))
    scan = tmp_path / "scan.pdf"
    _write_scanned_pdf(scan)
    mixed = tmp_path / "mixed.pdf"
    dst = pdfium.PdfDocument.new()
    dst.import_pages(pdfium.PdfDocument(born))
    dst.import_pages(pdfium.PdfDocument(scan))
    dst.save(str(mixed))

    ir = build_document_ir(mixed, ocr_mode="auto")

    page_one = [block for block in ir["blocks"] if block["page_no"] == 1]
    page_two = [block for block in ir["blocks"] if block["page_no"] == 2]
    assert page_one and all(block["engine"] == "pypdfium2" for block in page_one)
    assert page_two and all(block["engine"] == "rapidocr" for block in page_two)
    assert {page["page_type"] for page in ir["pages"]} == {"born-digital", "scanned"}
    for block in ir["blocks"]:
        _assert_normalized_bbox(block["bbox"])
