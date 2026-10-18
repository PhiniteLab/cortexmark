"""Torch-free local OCR helpers backed by RapidOCR and pypdfium2."""

from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from cortexmark.pdfutil import open_pdf

log = logging.getLogger(__name__)

# Shared OCR routing constants. Both the Markdown OCR path (this module) and the IR
# provenance path (page_ir.py) import these so the scanned/born-digital cutoff and the
# render resolution cannot silently drift apart between the two artifacts.
SCANNED_CHAR_THRESHOLD = 100
OCR_RENDER_DPI = 200


def ocr_available() -> bool:
    """Return whether either supported RapidOCR package is importable."""
    return (
        importlib.util.find_spec("rapidocr_onnxruntime") is not None or importlib.util.find_spec("rapidocr") is not None
    )


@lru_cache(maxsize=1)
def _get_rapidocr() -> Any:
    """Create and cache the first available torch-free RapidOCR engine."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        try:
            from rapidocr import RapidOCR  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                'RapidOCR is required for local OCR. Install it with: pip install "cortexmark[ocr]"'
            ) from None
    return RapidOCR()


def pdf_embedded_text_chars(path: str | Path) -> list[int]:
    """Return the number of embedded-text characters on each PDF page."""
    counts: list[int] = []
    with open_pdf(path) as pdf:
        for page in pdf:
            try:
                text_page = page.get_textpage()
                try:
                    counts.append(len(text_page.get_text_range(index=0, count=text_page.count_chars())))
                finally:
                    text_page.close()
            finally:
                page.close()
    return counts


def pdf_embedded_text(path: str | Path) -> str:
    """Return the concatenated embedded text layer of every page."""
    texts: list[str] = []
    with open_pdf(path) as pdf:
        for page in pdf:
            try:
                text_page = page.get_textpage()
                try:
                    texts.append(text_page.get_text_range(index=0, count=text_page.count_chars()))
                finally:
                    text_page.close()
            finally:
                page.close()
    return "\n".join(texts)


def page_is_scanned(char_count: int, threshold: int = SCANNED_CHAR_THRESHOLD) -> bool:
    """Return whether a page has too little embedded text to be considered born-digital."""
    return char_count < threshold


def _line_sort_key(box: Any) -> tuple[float, float]:
    try:
        points = list(box)
        return min(float(point[1]) for point in points), min(float(point[0]) for point in points)
    except (IndexError, TypeError, ValueError):
        return 0.0, 0.0


def _recognized_lines(result: Any) -> list[tuple[Any, str]]:
    payload = result[0] if isinstance(result, tuple) else result
    if payload is None:
        return []

    if isinstance(payload, dict):
        texts = payload.get("txts")
        if texts is None:
            texts = payload.get("texts")
        if texts is None:
            texts = []
        boxes = payload.get("boxes")
        if boxes is None:
            boxes = [None] * len(texts)
        return [(box, str(text)) for box, text in zip(boxes, texts, strict=False) if text]

    texts = getattr(payload, "txts", None)
    if texts is not None:
        boxes = getattr(payload, "boxes", None)
        if boxes is None:
            boxes = [None] * len(texts)
        return [(box, str(text)) for box, text in zip(boxes, texts, strict=False) if text]

    lines: list[tuple[Any, str]] = []
    for item in payload:
        if isinstance(item, list | tuple) and len(item) >= 2 and item[1]:
            lines.append((item[0], str(item[1])))
    return lines


def ocr_image(image_or_path: Any) -> str:
    """Run RapidOCR on an image and join recognized lines in reading order."""
    result = _get_rapidocr()(image_or_path)
    lines = sorted(_recognized_lines(result), key=lambda line: _line_sort_key(line[0]))
    return " ".join(text.strip() for _box, text in lines if text.strip())


def ocr_pdf_pages(
    path: str | Path,
    *,
    dpi: int = OCR_RENDER_DPI,
    only_scanned: bool = True,
    threshold: int = SCANNED_CHAR_THRESHOLD,
) -> dict[int, str]:
    """OCR selected PDF pages and return recovered text keyed by one-based page number."""
    text_counts = pdf_embedded_text_chars(path) if only_scanned else []
    recovered: dict[int, str] = {}
    with open_pdf(path) as pdf:
        for page_index in range(len(pdf)):
            page_number = page_index + 1
            if only_scanned and not page_is_scanned(text_counts[page_index], threshold=threshold):
                continue
            try:
                page = pdf[page_index]
                try:
                    bitmap = page.render(scale=dpi / 72)  # type: ignore[arg-type]
                    try:
                        image = bitmap.to_pil()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
                text = ocr_image(image).strip()
                if text:
                    recovered[page_number] = text
            except Exception as exc:
                log.warning("OCR failed for page %d of %s: %s", page_number, path, exc)
    return recovered


__all__ = [
    "OCR_RENDER_DPI",
    "SCANNED_CHAR_THRESHOLD",
    "ocr_available",
    "ocr_image",
    "ocr_pdf_pages",
    "page_is_scanned",
    "pdf_embedded_text_chars",
]
