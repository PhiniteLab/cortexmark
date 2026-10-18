"""Page-aware, torch-free intermediate representation for PDF documents."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from cortexmark import ocr
from cortexmark.pdfutil import open_pdf

log = logging.getLogger(__name__)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _pdf_bbox(
    rect: tuple[float, float, float, float],
    *,
    width: float,
    height: float,
) -> list[float]:
    left, bottom, right, top = rect
    return [
        _clamp(left / width),
        _clamp((height - top) / height),
        _clamp(right / width),
        _clamp((height - bottom) / height),
    ]


def _ocr_bbox(box: Any, *, width: int, height: int) -> list[float] | None:
    try:
        points = list(box)
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (IndexError, TypeError, ValueError):
        return None
    if not xs or not ys or width <= 0 or height <= 0:
        return None
    return [
        _clamp(min(xs) / width),
        _clamp(min(ys) / height),
        _clamp(max(xs) / width),
        _clamp(max(ys) / height),
    ]


def _ocr_lines(result: Any) -> list[tuple[Any, str, float]]:
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
        scores = payload.get("scores")
        if scores is None:
            scores = payload.get("confidences")
        if scores is None:
            scores = [0.0] * len(texts)
        return [(box, str(text), float(score)) for box, text, score in zip(boxes, texts, scores, strict=False) if text]

    texts = getattr(payload, "txts", None)
    if texts is not None:
        boxes = getattr(payload, "boxes", None)
        if boxes is None:
            boxes = [None] * len(texts)
        scores = getattr(payload, "scores", None)
        if scores is None:
            scores = [0.0] * len(texts)
        return [(box, str(text), float(score)) for box, text, score in zip(boxes, texts, scores, strict=False) if text]

    lines: list[tuple[Any, str, float]] = []
    for item in payload:
        if isinstance(item, list | tuple) and len(item) >= 2 and item[1]:
            confidence = float(item[2]) if len(item) >= 3 else 0.0
            lines.append((item[0], str(item[1]), confidence))
    return lines


def classify_pages(path: str | Path, *, threshold: int = ocr.SCANNED_CHAR_THRESHOLD) -> list[dict[str, Any]]:
    """Classify PDF pages from embedded text and renderable page objects."""
    pages: list[dict[str, Any]] = []
    with open_pdf(path) as pdf:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            try:
                width, height = page.get_size()
                text_page = page.get_textpage()
                try:
                    embedded_chars = len(text_page.get_text_range(index=0, count=text_page.count_chars()))
                finally:
                    text_page.close()
                if embedded_chars >= threshold:
                    page_type = "born-digital"
                elif any(page.get_objects()):
                    page_type = "scanned"
                else:
                    page_type = "empty"
                pages.append(
                    {
                        "page_no": page_index + 1,
                        "page_type": page_type,
                        "embedded_chars": embedded_chars,
                        "width": float(width),
                        "height": float(height),
                    }
                )
            finally:
                page.close()
    return pages


def _born_digital_blocks(page: Any, page_info: dict[str, Any]) -> list[dict[str, Any]]:
    width = float(page_info["width"])
    height = float(page_info["height"])
    if width <= 0 or height <= 0:
        return []

    blocks: list[dict[str, Any]] = []
    text_page = page.get_textpage()
    try:
        for rect_index in range(text_page.count_rects()):
            rect = text_page.get_rect(rect_index)
            text = text_page.get_text_bounded(*rect).strip()
            if not text:
                continue
            blocks.append(
                {
                    "page_no": page_info["page_no"],
                    "bbox": _pdf_bbox(rect, width=width, height=height),
                    "text": text,
                    "engine": "pypdfium2",
                    "page_type": page_info["page_type"],
                    "confidence": 1.0,
                    "evidence_level": "explicit",
                }
            )
    finally:
        text_page.close()
    return blocks


def _scanned_blocks(page: Any, page_info: dict[str, Any]) -> list[dict[str, Any]]:
    bitmap = page.render(scale=ocr.OCR_RENDER_DPI / 72)
    try:
        image = bitmap.to_pil()
        try:
            result = ocr._get_rapidocr()(image)
            image_width, image_height = image.size
        finally:
            image.close()
    finally:
        bitmap.close()

    blocks: list[dict[str, Any]] = []
    # Sort into reading order (top-to-bottom, then left-to-right) to match the Markdown OCR
    # path (ocr.ocr_image) and the born-digital block order; RapidOCR's raw emission order is
    # not reliably (min_y, min_x).
    ordered_lines = sorted(_ocr_lines(result), key=lambda line: ocr._line_sort_key(line[0]))
    for box, text, confidence in ordered_lines:
        normalized_bbox = _ocr_bbox(box, width=image_width, height=image_height)
        stripped_text = text.strip()
        if normalized_bbox is None or not stripped_text:
            continue
        blocks.append(
            {
                "page_no": page_info["page_no"],
                "bbox": normalized_bbox,
                "text": stripped_text,
                "engine": "rapidocr",
                "page_type": page_info["page_type"],
                "confidence": confidence,
                "evidence_level": "ocr",
            }
        )
    return blocks


def build_document_ir(
    path: str | Path,
    *,
    ocr_mode: str = "auto",
    threshold: int = ocr.SCANNED_CHAR_THRESHOLD,
) -> dict[str, Any]:
    """Build a serializable page-aware IR for a PDF document."""
    pages = classify_pages(path, threshold=threshold)
    blocks: list[dict[str, Any]] = []
    with open_pdf(path) as pdf:
        for page_index, page_info in enumerate(pages):
            page = pdf[page_index]
            try:
                if page_info["page_type"] == "born-digital":
                    blocks.extend(_born_digital_blocks(page, page_info))
                elif page_info["page_type"] == "scanned" and ocr_mode in {"auto", "on"}:
                    if not ocr.ocr_available():
                        log.warning("RapidOCR is unavailable; skipping page %d of %s", page_index + 1, path)
                        continue
                    try:
                        blocks.extend(_scanned_blocks(page, page_info))
                    except Exception as exc:
                        log.warning("OCR IR extraction failed for page %d of %s: %s", page_index + 1, path, exc)
            finally:
                page.close()

    return {
        "source": str(path),
        "n_pages": len(pages),
        "pages": pages,
        "coord_space": "normalized-topleft-0-1",
        "blocks": blocks,
    }


def write_ir(ir: dict[str, Any], path: Path) -> None:
    """Write a page-aware IR as stable, human-readable JSON."""
    path.write_text(json.dumps(ir, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = ["build_document_ir", "classify_pages", "write_ir"]
