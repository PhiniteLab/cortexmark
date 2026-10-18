"""Torch-free agreement checks between PDF text sources."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cortexmark.pdfutil import open_pdf


def _tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric word tokens with at least three characters."""
    return {token for token in re.findall(r"[^\W_]+", text.lower()) if len(token) >= 3}


def _reference_markdown(
    path: str | Path,
    *,
    engine: str,
    cfg: dict[str, Any] | None,
    engine_markdown: str | None,
) -> str:
    """Return the Markdown the chosen engine actually delivered, to corroborate the PDF text.

    When the caller already produced the engine's Markdown it is passed in directly and no
    conversion is re-run. Otherwise the chosen engine is re-run so the confidence tracks the
    engine actually selected instead of a fixed MarkItDown baseline; docling/dual fall back
    to MarkItDown when docling is unavailable so the metric stays computable.
    """
    if engine_markdown is not None:
        return engine_markdown
    from cortexmark.convert import convert_with_markitdown, docling_available

    if engine in ("docling", "dual") and docling_available():
        from cortexmark.convert import build_converter, merge_docling_markitdown

        docling_md = build_converter(cfg).convert(str(path)).document.export_to_markdown()
        if engine == "dual":
            return merge_docling_markitdown(docling_md, convert_with_markitdown(Path(path)))
        return docling_md
    return convert_with_markitdown(Path(path))


def document_consensus(
    path: str | Path,
    *,
    cfg: dict[str, Any] | None = None,
    engine_markdown: str | None = None,
) -> dict[str, Any]:
    """Measure how much non-scanned PDF token mass is corroborated by the chosen engine.

    The document confidence is the PDFium-token-weighted mean of per-page coverage by the
    engine actually selected (``cfg["convert"]["engine"]``, MarkItDown by default), so the
    metric varies with the engine instead of being a constant PDFium-vs-MarkItDown baseline.
    Pass ``engine_markdown`` to reuse an already-produced conversion and skip the re-run.
    Symmetric token Jaccard is reported separately for diagnostic use.
    """
    from cortexmark.page_ir import classify_pages

    engine = str((cfg or {}).get("convert", {}).get("engine", "markitdown")).strip().lower()
    reference_tokens = _tokens(_reference_markdown(path, engine=engine, cfg=cfg, engine_markdown=engine_markdown))
    classified_pages = classify_pages(path)
    per_page: list[dict[str, Any]] = []
    pdfium_tokens: set[str] = set()
    corroborated_token_mass = 0.0
    document_token_mass = 0

    with open_pdf(path) as pdf:
        for page_index, page_info in enumerate(classified_pages):
            page = pdf[page_index]
            try:
                text_page = page.get_textpage()
                try:
                    page_tokens = _tokens(text_page.get_text_range(index=0, count=text_page.count_chars()))
                finally:
                    text_page.close()
            finally:
                page.close()

            pdfium_tokens.update(page_tokens)
            coverage = len(page_tokens & reference_tokens) / max(1, len(page_tokens))
            low_agreement = page_info["page_type"] != "scanned" and len(page_tokens) >= 20 and coverage < 0.6
            if page_info["page_type"] != "scanned":
                corroborated_token_mass += coverage * len(page_tokens)
                document_token_mass += len(page_tokens)
            per_page.append(
                {
                    "page_no": int(page_info["page_no"]),
                    "pdfium_token_count": len(page_tokens),
                    "coverage_by_engine": coverage,
                    "low_agreement": low_agreement,
                }
            )

    combined_tokens = reference_tokens | pdfium_tokens
    token_jaccard = len(reference_tokens & pdfium_tokens) / max(1, len(combined_tokens))
    confidence = corroborated_token_mass / max(1, document_token_mass)
    return {
        "engine": engine,
        "n_pages": len(classified_pages),
        "per_page": per_page,
        "doc_consensus_confidence": confidence,
        "token_jaccard": token_jaccard,
        "engine_only_sample": sorted(reference_tokens - pdfium_tokens)[:20],
        "pdfium_only_sample": sorted(pdfium_tokens - reference_tokens)[:20],
    }


__all__ = ["document_consensus"]
