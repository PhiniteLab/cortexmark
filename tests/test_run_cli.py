from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from cortexmark import cli, convert, ocr


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


def test_ocr_available_with_installed_rapidocr() -> None:
    assert ocr.ocr_available() is True


@pytest.mark.parametrize(("ocr_mode", "expected_do_ocr"), [("auto", False), ("on", True), ("off", False)])
def test_default_config_enables_docling_quality_options(
    tmp_path: Path,
    ocr_mode: str,
    expected_do_ocr: bool,
) -> None:
    cfg = cli._default_config(engine="docling", ocr_mode=ocr_mode, project_root=tmp_path)

    assert cfg["convert"]["docling"] == {
        "do_table_structure": True,
        "table_structure_mode": "accurate",
        # was: True/True — formula/code enrichment are ~8x slower on CPU; kept opt-in (config only).
        "do_formula_enrichment": False,
        "do_code_enrichment": False,
        "do_ocr": expected_do_ocr,
    }


# was: (True, "dual") — auto now prefers pure docling for quality (no markitdown garbage merge)
@pytest.mark.parametrize(("docling_present", "expected"), [(False, "markitdown"), (True, "docling")])
def test_auto_engine_resolution(
    monkeypatch: pytest.MonkeyPatch,
    docling_present: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(convert, "docling_available", lambda: docling_present)
    assert convert._resolve_engine("auto") == expected


def test_run_born_digital_pdf_writes_bundle_without_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    sample = "Born-digital CortexMark text with enough embedded characters for OCR detection. " * 3
    pdf_path.write_bytes(_born_digital_pdf(sample))
    output_root = tmp_path / "output"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    assert cli.main(["run", str(pdf_path), "--out", str(output_root), "--json"]) == 0

    bundle = output_root / "paper"
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    assert summary["engine"] == "markitdown"
    assert summary["ocr_used"] is False
    assert summary["pages"] == 1
    assert summary["n_chunks"] == 1
    assert summary["char_count"] > 100
    assert (bundle / "paper.md").is_file()
    assert list((bundle / "chunks").glob("chunk_*.md"))


def test_unknown_first_argument_delegates_to_legacy_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    from cortexmark import run_pipeline

    received: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        received.append(argv)
        return 17

    monkeypatch.setattr(run_pipeline, "main", fake_main)

    assert cli.main(["--config", "pipeline.yaml", "--stages", "clean"]) == 17
    assert received == [["--config", "pipeline.yaml", "--stages", "clean"]]


def test_scanned_pdf_ocr_has_provenance_marker(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    output_path = tmp_path / "scan.md"
    _write_scanned_pdf(pdf_path)

    convert.convert_pdf(
        pdf_path,
        output_path,
        engine="markitdown",
        cfg={"convert": {"ocr": "on"}},
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert "<!-- ocr-recovered: torch-free RapidOCR, pages {1} -->" in markdown
    assert "Bellman" in markdown


def test_run_batch_skips_corrupt_pdf_and_converts_the_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # was: a single corrupt PDF in a directory aborted the whole `run` batch (exit 1, 0 documents).
    batch = tmp_path / "batch"
    batch.mkdir()
    sample = "Born-digital CortexMark text with enough embedded characters for OCR detection. " * 3
    (batch / "good.pdf").write_bytes(_born_digital_pdf(sample))
    (batch / "broken.pdf").write_bytes(b"%PDF-1.4 not-a-real-pdf")
    output_root = tmp_path / "output"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    # The corrupt PDF must be skipped, the valid one converted, and the batch must still succeed.
    assert cli.main(["run", str(batch), "--out", str(output_root), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [Path(doc["source"]).name for doc in report["documents"]] == ["good.pdf"]
    assert (output_root / "good" / "good.md").is_file()
    assert (output_root / "good" / "summary.json").is_file()
    # the corrupt document is not reported and never gets a completed bundle summary
    assert not (output_root / "broken" / "summary.json").is_file()
