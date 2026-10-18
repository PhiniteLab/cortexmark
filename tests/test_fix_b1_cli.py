"""Regression tests for the B1-cli findings (C06, C14, C15, C28, C31, C59)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortexmark import cli, consensus, convert, ocr, page_ir, semantic_chunk


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


def _make_pdf(tmp_path: Path, name: str = "paper.pdf") -> Path:
    pdf_path = tmp_path / name
    pdf_path.write_bytes(_born_digital_pdf("CortexMark source evidence text with symbols x and y. " * 5))
    return pdf_path


# --- C14: analyze stage must isolate a semantic-chunk failure and still finish the bundle ---


def test_analyze_semantic_chunk_failure_does_not_drop_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = _make_pdf(tmp_path)
    output_root = tmp_path / "out"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise ValueError("No semantic chunks could be produced")

    monkeypatch.setattr(semantic_chunk, "chunk_file", _boom)

    exit_code = cli.main(
        ["run", str(pdf_path), "--out", str(output_root), "--stages", "convert", "clean", "chunk", "analyze"]
    )
    assert exit_code == 0
    bundle = output_root / "paper"
    # The document survives with a complete summary despite the semantic-chunk failure.
    assert (bundle / "summary.json").is_file()
    # The sibling cross-reference artifact is isolated independently and still produced.
    assert (bundle / "crossref_report.json").is_file()


# --- C15: page count is guarded and no longer unconditionally re-parses the PDF ---


def test_pages_reuses_page_ir_count_without_extra_pdfium_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = _make_pdf(tmp_path)
    output_root = tmp_path / "out"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("pdfium page-count must not be called when page IR has the count")

    monkeypatch.setattr(ocr, "pdf_embedded_text_chars", _explode)

    assert cli.main(["run", str(pdf_path), "--out", str(output_root), "--json"]) == 0
    summary = json.loads((output_root / "paper" / "summary.json").read_text(encoding="utf-8"))
    assert summary["pages"] == 1


def test_pages_survive_page_ir_failure_via_open_pdf_without_full_text_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = _make_pdf(tmp_path)
    output_root = tmp_path / "out"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("page IR degraded")

    def _text_extraction_broken(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("get_text_range must not be used to count pages")

    # page IR degrades AND full-text extraction is broken, yet the PDF is still openable:
    # the page count comes from len(open_pdf(pdf)) and the completed bundle is NOT discarded.
    monkeypatch.setattr(page_ir, "build_document_ir", _explode)
    monkeypatch.setattr(ocr, "pdf_embedded_text_chars", _text_extraction_broken)

    assert cli.main(["run", str(pdf_path), "--out", str(output_root), "--json"]) == 0
    summary = json.loads((output_root / "paper" / "summary.json").read_text(encoding="utf-8"))
    assert summary["pages"] == 1


# --- C28: partial batch failures are surfaced in --json (exit stays 0 by contract) ---


def test_json_reports_failed_documents_on_partial_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "good.pdf").write_bytes(_born_digital_pdf("Born-digital CortexMark text. " * 4))
    (batch / "broken.pdf").write_bytes(b"%PDF-1.4 not-a-real-pdf")
    output_root = tmp_path / "out"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    assert cli.main(["run", str(batch), "--out", str(output_root), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [Path(doc["source"]).name for doc in report["documents"]] == ["good.pdf"]
    assert report["failures"] == 1
    assert len(report["failed"]) == 1
    assert Path(report["failed"][0]["source"]).name == "broken.pdf"
    assert report["failed"][0]["error"]  # non-empty class: message string


# --- C31: enrichment failure warnings carry the exception class name ---


def test_enrichment_failure_warning_includes_exception_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pdf_path = _make_pdf(tmp_path)
    output_root = tmp_path / "out"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("consensus subsystem regressed")

    monkeypatch.setattr(consensus, "document_consensus", _boom)

    with caplog.at_level("WARNING", logger="cortexmark.run"):
        assert cli.main(["run", str(pdf_path), "--out", str(output_root), "--json"]) == 0
    assert "consensus generation failed" in caplog.text
    assert "RuntimeError" in caplog.text  # was a silent zero; now the class is visible


# --- C06: consensus receives the RESOLVED engine through cfg ---


def test_consensus_receives_resolved_engine_via_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = _make_pdf(tmp_path)
    output_root = tmp_path / "out"
    monkeypatch.setattr(convert, "docling_available", lambda: False)  # auto -> markitdown

    seen: dict[str, object] = {}

    def _capture(path: object, *, cfg: dict | None = None) -> dict:
        seen["engine"] = (cfg or {}).get("convert", {}).get("engine")
        return {"doc_consensus_confidence": 0.0, "per_page": []}

    monkeypatch.setattr(consensus, "document_consensus", _capture)

    assert cli.main(["run", str(pdf_path), "--out", str(output_root), "--engine", "auto", "--json"]) == 0
    assert seen["engine"] == "markitdown"


# --- C59: run's analyze/validate stages emit the rich analysis artifacts ---


def test_run_analyze_validate_emit_rich_analysis_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_path = _make_pdf(tmp_path)
    output_root = tmp_path / "out"
    monkeypatch.setattr(convert, "docling_available", lambda: False)

    exit_code = cli.main(
        [
            "run",
            str(pdf_path),
            "--out",
            str(output_root),
            "--stages",
            "convert",
            "clean",
            "chunk",
            "analyze",
            "validate",
        ]
    )
    assert exit_code == 0
    bundle = output_root / "paper"
    for artifact in (
        "algorithm_report.json",
        "notation_report.json",
        "formula_validation.json",
        "scientific_qa.json",
        "citation_context.json",
        "crossref_report.json",
        "qa.json",
    ):
        assert (bundle / artifact).is_file(), f"missing {artifact}"
    assert (bundle / "semantic_chunks").is_dir()
