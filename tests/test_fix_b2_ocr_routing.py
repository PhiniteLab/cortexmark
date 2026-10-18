"""Regression tests for the B2 OCR-routing fixes (C01, C04, C07, C17, C18, C20, C33, C34, C58).

These lock in that scanned/image-only pages are recovered into the primary Markdown
deliverable for every engine (including the default docling path), that ``--ocr on`` never
duplicates born-digital pages, that the OCR provenance marker (hence summary.ocr_used) is
truthful, that docling's diacritic and pseudocode-heading artifacts are normalized, and that
the scanned-page IR blocks are emitted in reading order using shared constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image, ImageDraw, ImageFont

from cortexmark import convert, ocr, page_ir

# ── Shared PDF builders ──────────────────────────────────────────────────────


def _born_digital_pdf(text: str, pages: int = 1) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = f"BT /F1 14 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    stream = b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream"
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(pages))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
    ]
    for i in range(pages):
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 " + f"{3 + 2 * pages} 0 R".encode() + b" >> >> "
            b"/Contents " + f"{4 + 2 * i} 0 R".encode() + b" >>"
        )
        objects.append(stream)
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

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


# ── C34: shared constants, no drift between the two OCR paths ─────────────────


def test_scanned_threshold_and_dpi_are_shared_constants() -> None:
    assert ocr.SCANNED_CHAR_THRESHOLD == 100
    assert ocr.OCR_RENDER_DPI == 200
    # page_ir must consume the same constants, not private copies.
    import inspect

    assert inspect.signature(ocr.page_is_scanned).parameters["threshold"].default is ocr.SCANNED_CHAR_THRESHOLD
    assert inspect.signature(page_ir.classify_pages).parameters["threshold"].default is ocr.SCANNED_CHAR_THRESHOLD
    assert inspect.signature(page_ir.build_document_ir).parameters["threshold"].default is ocr.SCANNED_CHAR_THRESHOLD
    assert inspect.signature(ocr.ocr_pdf_pages).parameters["dpi"].default is ocr.OCR_RENDER_DPI


# ── C18: diacritic-with-inserted-space normalization ─────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Moˇ ckus", "Močkus"),  # caron -> c-with-caron
        ("Hern´ andez", "Hernández"),  # acute -> a-with-acute
        ("d´ emarrage", "démarrage"),  # acute -> e (French)
    ],
)
def test_normalize_diacritic_spacing_recomposes(raw: str, expected: str) -> None:
    assert convert.normalize_diacritic_spacing(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "a ^ b",  # circumflex is not a targeted accent; legit math untouched
        "´ evolutive",  # accent at a word start (no letter before) is left alone
        "cost × time",  # multiplication sign untouched
        "already composed é",  # composed text untouched
    ],
)
def test_normalize_diacritic_spacing_leaves_legitimate_text(text: str) -> None:
    assert convert.normalize_diacritic_spacing(text) == text


# ── C17: pseudocode heading demotion ─────────────────────────────────────────


@pytest.mark.parametrize(
    "heading",
    ["## end for", "## end while", "## end if", "## while n N do", "## for each item do", "## x := y + 1"],
)
def test_demote_pseudocode_headings_demotes_control_flow(heading: str) -> None:
    demoted = convert.demote_pseudocode_headings(heading)
    assert not demoted.lstrip().startswith("#")
    assert demoted == heading.split(maxsplit=1)[1]


@pytest.mark.parametrize(
    "heading",
    [
        "## For Loop Optimization",
        "## While You Wait",
        "## Method Details",
        "## Results and Discussion",
        "# Introduction",
    ],
)
def test_demote_pseudocode_headings_keeps_real_headings(heading: str) -> None:
    assert convert.demote_pseudocode_headings(heading) == heading


def test_demote_pseudocode_headings_prevents_duplicate_end_for_sections() -> None:
    markdown = "## end for\n\nEstimate ...\n\n## end for\n\nReturn ...\n"
    demoted = convert.demote_pseudocode_headings(markdown)
    assert "## end for" not in demoted
    assert "Estimate ..." in demoted and "Return ..." in demoted


# ── C07: --ocr on must not duplicate born-digital pages ──────────────────────


def test_ocr_on_does_not_duplicate_born_digital_pages(tmp_path: Path) -> None:
    if not ocr.ocr_available():
        pytest.skip("RapidOCR is not installed")
    pdf_path = tmp_path / "born.pdf"
    out_path = tmp_path / "born.md"
    pdf_path.write_bytes(_born_digital_pdf("SENTINELTOKEN born digital body text " * 4, pages=2))

    convert.convert_pdf(pdf_path, out_path, engine="markitdown", cfg={"convert": {"ocr": "on"}})
    markdown = out_path.read_text(encoding="utf-8")

    # No scanned pages -> no OCR append at all, so the sentinel is not re-emitted.
    assert "OCR recovered page" not in markdown
    assert not convert.markdown_uses_ocr(markdown)


# ── C01 / C04 / C20: default docling path recovers scanned text + truthful flag ─


def test_docling_path_recovers_scanned_text_and_marks_ocr_used(tmp_path: Path) -> None:
    if not ocr.ocr_available():
        pytest.skip("RapidOCR is not installed")
    pdf_path = tmp_path / "scan.pdf"
    out_path = tmp_path / "scan.md"
    _write_scanned_pdf(pdf_path)

    # Force the docling branch of convert_pdf without importing docling: docling only runs
    # when engine is docling/dual, so we exercise the OCR append that the docling path now
    # shares. build_converter is bypassed because we pass a pre-built no-op converter that
    # returns empty docling markdown, mimicking a scanned page docling cannot read.
    class _EmptyDoc:
        def export_to_markdown(self) -> str:
            return ""

        def iterate_items(self):
            return iter(())

    class _Result:
        document = _EmptyDoc()

    class _Converter:
        def convert(self, _path: str) -> _Result:
            return _Result()

    for mode in ("auto", "on"):
        convert.convert_pdf(
            pdf_path,
            out_path,
            converter=cast("Any", _Converter()),
            engine="docling",
            cfg={"convert": {"ocr": mode}},
        )
        markdown = out_path.read_text(encoding="utf-8")
        assert convert.markdown_uses_ocr(markdown), f"ocr provenance marker missing for --ocr {mode}"
        assert "Bellman" in markdown, f"scanned text not recovered for --ocr {mode}"


def test_build_converter_never_enables_docling_internal_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    captured: dict[str, object] = {}

    class _PipelineOptions:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_modules = {
        name: types.ModuleType(name)
        for name in (
            "docling",
            "docling.datamodel",
            "docling.datamodel.accelerator_options",
            "docling.datamodel.base_models",
            "docling.datamodel.pipeline_options",
            "docling.document_converter",
        )
    }
    fake_attrs: list[tuple[str, str, object]] = [
        ("docling.datamodel.accelerator_options", "AcceleratorOptions", lambda **kw: None),
        ("docling.datamodel.base_models", "InputFormat", types.SimpleNamespace(PDF="pdf")),
        ("docling.datamodel.pipeline_options", "PdfPipelineOptions", _PipelineOptions),
        ("docling.document_converter", "DocumentConverter", lambda **kw: None),
        ("docling.document_converter", "PdfFormatOption", lambda **kw: None),
    ]
    for module_name, attr_name, attr_value in fake_attrs:
        setattr(fake_modules[module_name], attr_name, attr_value)
    for module_name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)

    # Even when the config explicitly requests docling OCR, it must stay disabled.
    convert.build_converter({"convert": {"docling": {"do_ocr": True}}})
    assert captured["do_ocr"] is False


# ── C33: scanned-page IR blocks are emitted in reading order ─────────────────


def test_scanned_blocks_are_sorted_into_reading_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # A RapidOCR result whose raw emission order is NOT top-to-bottom; _scanned_blocks must
    # reorder by (min_y, min_x) to match ocr.ocr_image and the born-digital block order.
    def _box(x: float, y: float) -> list[list[float]]:
        return [[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]]

    fake_result = (
        [
            (_box(10, 300), "third", 0.9),
            (_box(10, 10), "first", 0.9),
            (_box(200, 10), "second", 0.9),
        ],
    )

    class _Bitmap:
        def to_pil(self):
            img = Image.new("RGB", (400, 400), "white")
            return img

        def close(self) -> None:
            pass

    class _Page:
        def render(self, *, scale: float) -> _Bitmap:
            return _Bitmap()

    monkeypatch.setattr(ocr, "_get_rapidocr", lambda: lambda _image: fake_result)

    page_info = {"page_no": 1, "page_type": "scanned"}
    blocks = page_ir._scanned_blocks(_Page(), page_info)
    assert [block["text"] for block in blocks] == ["first", "second", "third"]
