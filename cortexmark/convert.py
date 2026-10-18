from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cortexmark.common import (
    Manifest,
    config_fingerprint,
    detect_device,
    get_source_id,
    load_config,
    mirror_directory_tree,
    resolve_configured_path,
    resolve_manifest_path,
    runtime_env_value,
    setup_logging,
)

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

FORMULA_PLACEHOLDER = "<!-- formula-not-decoded -->"
ALGORITHM_HINTS = (
    "Algorithm parameter",
    "Algorithm parameters",
    "Input:",
    "Initialize",
    "Loop for each",
    "Loop forever",
    "Take action",
    "Observe",
    "Choose ",
    "until ",
)
VALID_ENGINES = ("auto", "docling", "markitdown", "dual")
VALID_OCR_MODES = ("auto", "on", "off")
OCR_PROVENANCE_PREFIX = "<!-- ocr-recovered: torch-free RapidOCR, pages "

log = logging.getLogger(__name__)


# ── Docling helpers ──────────────────────────────────────────────────────────


def docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


def _docling_import_error() -> ImportError:
    return ImportError(
        "The 'docling' package is required for the docling/dual conversion engine.\n"
        'Install it with:  pip install "cortexmark[docling]"\n'
        'Or for GPU support: pip install "cortexmark[gpu]"'
    )


def _resolve_engine(engine: str) -> str:
    if engine not in VALID_ENGINES:
        raise ValueError(f"Unknown conversion engine: {engine!r}. Expected one of {VALID_ENGINES}.")
    if engine == "auto":
        # Prefer pure docling for quality (layout, reading order, tables, headings). Avoid "dual",
        # whose markitdown supplement re-injects garbled 2-column text on scientific PDFs.
        return "docling" if docling_available() else "markitdown"
    if engine == "docling" and not docling_available():
        raise _docling_import_error()
    if engine == "dual" and not docling_available():
        log.warning("docling is unavailable; degrading conversion engine from 'dual' to 'markitdown'")
        return "markitdown"
    return engine


def _ocr_mode(cfg: dict[str, Any] | None) -> str:
    mode = str((cfg or {}).get("convert", {}).get("ocr", "auto")).strip().lower()
    if mode not in VALID_OCR_MODES:
        raise ValueError(f"Unknown OCR mode: {mode!r}. Expected one of {VALID_OCR_MODES}.")
    return mode


def _append_rapidocr_text(markdown: str, input_path: Path, cfg: dict[str, Any] | None) -> str:
    from cortexmark import ocr

    mode = _ocr_mode(cfg)
    if mode == "off":
        return markdown
    if not ocr.ocr_available():
        if mode == "on":
            log.warning('OCR was requested but RapidOCR is unavailable; install "cortexmark[ocr]"')
        return markdown

    try:
        if mode == "auto":
            text_counts = ocr.pdf_embedded_text_chars(input_path)
            if not any(ocr.page_is_scanned(count) for count in text_counts):
                return markdown
        # Both "auto" and "on" recover text for scanned/image-only pages only. "on" never
        # OCRs born-digital pages, which would otherwise re-emit their already-extracted body
        # a second time beneath the primary text (a size/quality regression, not "force OCR").
        recovered = ocr.ocr_pdf_pages(input_path, only_scanned=True)
    except Exception as exc:
        log.warning("OCR recovery failed for %s: %s", input_path, exc)
        return markdown

    if not recovered:
        return markdown

    page_numbers = ", ".join(str(page) for page in recovered)
    recovered_sections = "\n\n".join(
        f"### OCR recovered page {page_number}\n\n{text}" for page_number, text in recovered.items()
    )
    return markdown.rstrip() + f"\n\n{OCR_PROVENANCE_PREFIX}{{{page_numbers}}} -->\n\n" + recovered_sections + "\n"


def markdown_uses_ocr(markdown: str) -> bool:
    """Return whether Markdown contains a CortexMark OCR provenance block."""
    return OCR_PROVENANCE_PREFIX in markdown


def build_converter(cfg: dict[str, Any] | None = None) -> DocumentConverter:
    dc = (cfg or {}).get("convert", {}).get("docling", {})
    table_mode_name = str(dc.get("table_structure_mode", "accurate")).strip().lower()
    if table_mode_name not in {"accurate", "fast"}:
        raise ValueError(f"convert.docling.table_structure_mode must be 'accurate' or 'fast', got {table_mode_name!r}")

    try:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        raise _docling_import_error() from None

    do_table = dc.get("do_table_structure", False)

    table_kwargs: dict[str, Any] = {}
    if do_table:
        from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions

        table_mode = TableFormerMode.ACCURATE if table_mode_name == "accurate" else TableFormerMode.FAST
        table_kwargs["table_structure_options"] = TableStructureOptions(mode=table_mode)

    raw_device = dc.get("device", "auto")
    device = detect_device() if raw_device == "auto" else raw_device

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=device,
            num_threads=dc.get("num_threads", 1),
        ),
        enable_remote_services=False,
        allow_external_plugins=False,
        # CortexMark performs scanned-page OCR itself, torch-free, via the RapidOCR append in
        # convert_pdf(). Docling's own OCR is left off so the two paths cannot double-OCR a
        # page and so the recovered text, its reading order, ir.json, and summary.ocr_used all
        # stay consistent. The requested OCR mode is honored by _append_rapidocr_text().
        do_ocr=False,
        do_table_structure=do_table,
        force_backend_text=True,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        generate_parsed_pages=False,
        **table_kwargs,
    )

    for option_name, enabled in (
        ("do_formula_enrichment", dc.get("do_formula_enrichment", False)),
        ("do_code_enrichment", dc.get("do_code_enrichment", False)),
    ):
        if hasattr(pipeline_options, option_name):
            setattr(pipeline_options, option_name, enabled)

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )


def normalize_markdown(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized + "\n"


def normalize_recovered_text(text: str) -> str:
    normalized = " ".join(text.replace("\n", " ").split())
    normalized = normalized.replace(" . =", " =").replace(" . =", " =")
    normalized = normalized.replace(" ,", ",").replace(" .", ".")
    normalized = re.sub(r"\s+([)\]])", r"\1", normalized)
    normalized = re.sub(r"([([=+\-/*])\s+", r"\1 ", normalized)
    return normalized.strip()


def is_algorithmic_text(text: str) -> bool:
    return any(hint in text for hint in ALGORITHM_HINTS)


def format_algorithm_block(text: str) -> str:
    formatted = text
    split_markers = [
        "Input:",
        "Algorithm parameter",
        "Algorithm parameters",
        "Initialize",
        "Loop for each episode:",
        "Loop for each step of episode:",
        "Loop forever:",
        "Loop for t =",
        "Choose ",
        "Take action",
        "Observe ",
        "If ",
        "Else ",
        "until ",
        "Until ",
    ]
    for marker in split_markers:
        formatted = formatted.replace(marker, f"\n{marker}")
    lines = [line.strip() for line in formatted.splitlines() if line.strip()]
    return "```text\n" + "\n".join(lines) + "\n```"


def format_formula_block(text: str) -> str:
    return "\n".join(
        [
            "> Equation (recovered from PDF):",
            ">",
            f"> {text}",
        ]
    )


def format_incomplete_formula_block() -> str:
    return "> Equation present in PDF, but text recovery was incomplete."


# Spacing-modifier accents docling emits as standalone glyphs, mapped to their combining
# equivalents so an accent + inserted-space + base letter can be recomposed to one NFC glyph.
_SPACING_TO_COMBINING = {
    "´": "́",  # acute
    "`": "̀",  # grave
    "ˇ": "̌",  # caron
    "˘": "̆",  # breve
    "¨": "̈",  # diaeresis
    "˙": "̇",  # dot above
    "¸": "̧",  # cedilla
    "˚": "̊",  # ring above
    "˝": "̋",  # double acute
    "¯": "̄",  # macron
    "˜": "̃",  # small tilde
}
_DIACRITIC_SPACING = re.compile(
    r"(?<=[^\W\d_])([" + "".join(_SPACING_TO_COMBINING) + r"]) ([a-z])",
)
_PSEUDOCODE_HEADING = re.compile(
    r"(?im)^#{1,6}[ \t]+("
    r"end[ \t]+(?:while|for|if|loop|function|procedure)"  # end while / end for / end if …
    r"|(?:while|for|for[ \t]+each|repeat|if)\b.*\b(?:do|then)"  # while … do / if … then
    r"|.*:=.*"  # contains a pseudocode assignment
    r")[ \t]*$",
)


def normalize_diacritic_spacing(text: str) -> str:
    """Repair docling's spacing-modifier accent glyphs split from their base letter.

    Docling's layout reconstruction renders a standalone accent (caron, acute, grave, ...)
    followed by a spurious space before the letter it modifies, so a caron + space + "c"
    that should read as a single composed letter arrives as three characters. Only a
    letter-anchored accent + single space + lowercase letter that recomposes to a single NFC
    glyph is collapsed, so legitimately spaced text (maths, quotations) is left untouched.
    """

    def _compose(match: re.Match[str]) -> str:
        base = match.group(2)
        composed = unicodedata.normalize("NFC", base + _SPACING_TO_COMBINING[match.group(1)])
        # Keep the change only when the accent actually merges into one character; otherwise
        # fall back to removing just the spurious space (accent kept before the base letter).
        return composed if len(composed) == 1 else match.group(1) + base

    return _DIACRITIC_SPACING.sub(_compose, text)


_WORD_TOKEN = re.compile(r"\w+")


def repair_intraword_splits(markdown: str, reference_text: str) -> str:
    """Merge glyph-run word splits docling introduces ("spl it" -> "split"; letter-spaced runs like "B O L U M" -> "BOLUM").

    Docling's layout reconstruction occasionally splits one word into spaced
    fragments at glyph-run boundaries; the accent normalizer above cannot repair
    these because Turkish letters are full code points, not spacing accents. A
    merge is applied only when the joined word occurs in the PDF's own embedded
    text layer while the spaced fragment sequence does not, so text that is
    genuinely spaced in the source is never collapsed.
    """
    if not reference_text:
        return markdown
    vocab = {w.casefold() for w in _WORD_TOKEN.findall(reference_text) if len(w) >= 4}
    if not vocab:
        return markdown
    ref_flat = " ".join(reference_text.split()).casefold()
    out: list[str] = []
    in_fence = False
    for line in markdown.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
        elif in_fence:
            out.append(line)
        else:
            out.append(_merge_split_tokens(line, vocab, ref_flat))
    return "\n".join(out)


def _segment_by_vocab(collapsed: str, vocab: set[str]) -> list[str] | None:
    """Greedy longest-prefix segmentation of a collapsed letter run into vocab words."""
    segments: list[str] = []
    rest = collapsed
    while rest:
        for end in range(len(rest), 2, -1):
            candidate = rest[:end]
            if candidate.casefold() in vocab:
                segments.append(candidate)
                rest = rest[end:]
                break
        else:
            return None
    return segments


def _merge_split_tokens(line: str, vocab: set[str], ref_flat: str) -> str:
    parts = line.split(" ")
    merged_any = False
    result: list[str] = []
    i = 0
    while i < len(parts):
        # Letter-spaced runs (>=4 single-letter tokens, e.g. stylized headings) are
        # collapsed and re-segmented against the PDF vocabulary, all-or-nothing.
        run_end = i
        while run_end < len(parts) and len(parts[run_end]) == 1 and parts[run_end].isalpha():
            run_end += 1
        if run_end - i >= 4:
            window = parts[i:run_end]
            if " ".join(window).casefold() not in ref_flat:
                segments = _segment_by_vocab("".join(window), vocab)
                if segments:
                    result.extend(segments)
                    merged_any = True
                    i = run_end
                    continue
        merged = None
        for width in (6, 5, 4, 3, 2):
            window = parts[i : i + width]
            if len(window) < width or not all(p.isalpha() for p in window):
                continue
            if min(len(p) for p in window) > 2:
                continue
            joined = "".join(window)
            if len(joined) < 4 or joined.casefold() not in vocab:
                continue
            if " ".join(window).casefold() in ref_flat:
                continue
            merged = joined
            i += width
            break
        if merged is None:
            result.append(parts[i])
            i += 1
        else:
            result.append(merged)
            merged_any = True
    return " ".join(result) if merged_any else line


def demote_pseudocode_headings(markdown: str) -> str:
    """Demote algorithm control-flow lines docling promoted to Markdown headings.

    docling exports pseudocode loop/branch lines (``end for``, ``while n < N do``) as ``##``
    headings, which then split the chunker into orphaned, indistinguishable sections. Only
    lines that are unmistakably pseudocode -- ``end while/for/if/loop``, a ``while/for/if``
    line ending in ``do``/``then``, or an assignment (``:=``) -- are demoted to body text, so
    real section headings are never touched.
    """
    return _PSEUDOCODE_HEADING.sub(lambda match: match.group(1), markdown)


def reformat_algorithm_sections(markdown: str) -> str:
    pattern = re.compile(
        r"(?m)^(##\s+.+?)\s+(Algorithm parameter[s]?:.+)$",
    )

    def replacer(match: re.Match[str]) -> str:
        heading = match.group(1).strip()
        algorithm_text = normalize_recovered_text(match.group(2))
        return f"{heading}\n\n{format_algorithm_block(algorithm_text)}"

    return pattern.sub(replacer, markdown)


def render_formula_item(item: Any) -> str:
    recovered = normalize_recovered_text(item.text or item.orig)
    if not recovered:
        return FORMULA_PLACEHOLDER
    if len(re.findall(r"[\w\dα-ωΑ-Ω]", recovered)) < 3:
        return format_incomplete_formula_block()
    if is_algorithmic_text(recovered):
        return format_algorithm_block(recovered)
    return format_formula_block(recovered)


def recover_formula_placeholders(markdown: str, formula_items: list[Any]) -> str:
    recovered_markdown = markdown
    for item in formula_items:
        replacement = render_formula_item(item)
        recovered_markdown = recovered_markdown.replace(FORMULA_PLACEHOLDER, replacement, 1)
    return recovered_markdown


# ── markitdown helpers ───────────────────────────────────────────────────────


def convert_with_markitdown(input_path: Path) -> str:
    """Return raw markdown text produced by markitdown."""
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(input_path))
    return result.text_content or ""


# A run of 26+ letters with no separator does not occur in normal prose (the longest common
# English words are ~20 letters); several such runs signal markitdown/pdfminer collapsing a
# multi-column layout into interleaved, whitespace-stripped text.
_GARBLED_RUN = re.compile(r"[^\W\d_]{26,}")


def looks_column_garbled(markdown: str) -> bool:
    """Heuristically detect markitdown's multi-column failure mode from its own output."""
    return len(_GARBLED_RUN.findall(markdown)) >= 5


# ── Merge logic ──────────────────────────────────────────────────────────────

_SHINGLE_SIZE = 8
# A markitdown paragraph is treated as already present in docling when this fraction of its
# character shingles is found in the docling backbone.
_DUPLICATE_CONTAINMENT = 0.65
# The supplement may not exceed this fraction of the backbone; a larger one means dedup
# missed near-duplicate content (the 2-column failure) and would nearly double the document.
_SUPPLEMENT_CAP_FRACTION = 0.3
# Only cap substantial documents; unit-test-sized snippets legitimately carry a supplement
# comparable in size to a tiny backbone.
_SUPPLEMENT_CAP_MIN_BASE = 2000
_SUPPLEMENT_HEADING = "## Supplementary Text (markitdown)"


def _dedup_key(text: str) -> str:
    """Collapse text to lowercase alphanumerics only (drop whitespace and punctuation)."""
    return re.sub(r"[\W_]+", "", text.lower())


def _shingles(text: str, size: int = _SHINGLE_SIZE) -> set[str]:
    """Character n-gram set of the whitespace/punctuation-stripped, lowercased text.

    Comparing character shingles rather than a reflowed word prefix lets duplicate
    detection survive markitdown's 2-column failure mode, where pdfminer interleaves the
    columns and strips inter-word whitespace: the reflowed text bears no resemblance to
    docling's, yet the underlying character runs still overlap heavily.
    """
    key = _dedup_key(text)
    if len(key) <= size:
        return {key} if key else set()
    return {key[i : i + size] for i in range(len(key) - size + 1)}


def merge_docling_markitdown(docling_md: str, markitdown_md: str) -> str:
    """Merge docling (structure-rich) and markitdown (text-rich) outputs.

    docling stays the structural backbone; only markitdown paragraphs whose content is
    genuinely absent from docling are appended. Column-interleaved, whitespace-stripped
    markitdown text shares almost no reflowed prefix with docling but still overlaps at the
    character level, so duplicate detection uses character-shingle containment rather than
    an exact prefix match. The supplement is additionally capped at a fraction of the
    backbone so a dedup miss cannot silently double a real document, and it is placed under
    its own heading so the chunker cannot misattribute it to the last docling section.
    """
    docling_shingles = _shingles(docling_md)
    extra_paragraphs: list[str] = []

    base_len = len(docling_md)
    cap = int(base_len * _SUPPLEMENT_CAP_FRACTION) if base_len >= _SUPPLEMENT_CAP_MIN_BASE else None
    supplement_len = 0
    capped = False

    for block in re.split(r"\n{2,}", markitdown_md):
        stripped = block.strip()
        if not stripped or stripped.startswith(("#", ">", "```")):
            continue
        # Skip markitdown table artifacts (page layout garbage)
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        normalised = " ".join(stripped.lower().split())
        if len(normalised) <= 40:
            continue
        para_shingles = _shingles(stripped)
        if not para_shingles:
            continue
        containment = len(para_shingles & docling_shingles) / len(para_shingles)
        if containment >= _DUPLICATE_CONTAINMENT:
            continue
        if cap is not None and supplement_len + len(stripped) > cap:
            capped = True
            break
        extra_paragraphs.append(stripped)
        supplement_len += len(stripped) + 2
        # Also dedup within markitdown so repeated garbled blocks are appended at most once.
        docling_shingles |= para_shingles

    if capped:
        log.warning(
            "markitdown supplement exceeded %.0f%% of the docling backbone; truncating "
            "likely-duplicate 2-column text (append %d paragraph(s))",
            _SUPPLEMENT_CAP_FRACTION * 100,
            len(extra_paragraphs),
        )

    if not extra_paragraphs:
        return docling_md

    supplement = "\n\n".join(extra_paragraphs)
    return docling_md.rstrip() + f"\n\n<!-- markitdown-supplement -->\n\n{_SUPPLEMENT_HEADING}\n\n" + supplement + "\n"


# ── Single-file conversion ──────────────────────────────────────────────────


def convert_pdf(
    input_path: Path,
    output_path: Path,
    *,
    converter: DocumentConverter | None = None,
    engine: str = "dual",
    cfg: dict[str, Any] | None = None,
) -> Path:
    engine = _resolve_engine(engine)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")
    if input_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {input_path.name}")

    docling_md = ""
    markitdown_md = ""

    if engine in ("docling", "dual"):
        active_converter = converter or build_converter(cfg)
        result = active_converter.convert(str(input_path))
        document = result.document
        docling_md = document.export_to_markdown()
        try:
            from docling_core.types.doc.document import FormulaItem  # pyright: ignore[reportPrivateImportUsage]
        except ImportError:
            raise ImportError(
                "The 'docling' package is required for formula recovery.\n"
                'Install it with:  pip install "cortexmark[docling]"'
            ) from None

        undecoded_formulas = [
            item
            for item, _level in document.iterate_items()
            if isinstance(item, FormulaItem) and not item.text and item.orig
        ]
        docling_md = recover_formula_placeholders(docling_md, undecoded_formulas)
        docling_md = reformat_algorithm_sections(docling_md)
        docling_md = normalize_diacritic_spacing(docling_md)
        from cortexmark import ocr as _ocr

        try:
            _reference_text = _ocr.pdf_embedded_text(input_path)
        except Exception:  # the repair pass must never break conversion
            _reference_text = ""
        docling_md = repair_intraword_splits(docling_md, _reference_text)
        docling_md = demote_pseudocode_headings(docling_md)

    if engine in ("markitdown", "dual"):
        markitdown_md = convert_with_markitdown(input_path)

    if engine == "dual":
        markdown = merge_docling_markitdown(docling_md, markitdown_md)
    elif engine == "markitdown":
        markdown = markitdown_md
        if looks_column_garbled(markitdown_md):
            log.warning(
                "engine=markitdown has no layout model and this PDF looks multi-column: the "
                "extracted text is likely garbled (interleaved columns, run-together words). "
                "Re-run with --engine docling for layout-aware extraction (%s).",
                input_path.name,
            )
    else:
        markdown = docling_md

    # Every engine recovers scanned/image-only pages through the same torch-free RapidOCR
    # append (docling's own OCR is disabled in build_converter). This keeps the Markdown
    # deliverable consistent with ir.json and lets summary.ocr_used reflect real OCR use.
    markdown = _append_rapidocr_text(markdown, input_path, cfg)

    markdown = normalize_markdown(markdown)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown)
    return output_path


# ── Tree conversion ─────────────────────────────────────────────────────────


def derive_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    *,
    include_input_root_name: bool = True,
) -> Path:
    relative = input_path.relative_to(input_root)
    base = output_root / input_root.name if include_input_root_name else output_root
    return base / relative.with_suffix(".md")


def _convert_fingerprint(engine: str, cfg: dict[str, Any] | None) -> str:
    convert_cfg = (cfg or {}).get("convert", {})
    return config_fingerprint(
        {
            "docling": convert_cfg.get("docling", {}),
            "engine": engine,
            "markitdown": convert_cfg.get("markitdown", {}),
            "ocr": convert_cfg.get("ocr", "auto"),
        }
    )


def convert_tree(
    input_root: Path,
    output_root: Path,
    *,
    engine: str = "dual",
    cfg: dict[str, Any] | None = None,
    manifest: Manifest | None = None,
    include_input_root_name: bool = True,
    only_files: tuple[Path, ...] | None = None,
) -> list[Path]:
    engine = _resolve_engine(engine)
    pdf_files = (
        sorted(p for p in input_root.rglob("*.pdf") if p.is_file())
        if only_files is None
        else sorted(p for p in only_files if p.is_file() and p.suffix.lower() == ".pdf")
    )
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under: {input_root}")

    if only_files is None:
        mirror_directory_tree(input_root, output_root)
    converter = None
    fingerprint = _convert_fingerprint(engine, cfg)
    written: list[Path] = []
    for pdf_path in pdf_files:
        output_path = derive_output_path(
            pdf_path,
            input_root,
            output_root,
            include_input_root_name=include_input_root_name,
        )
        if manifest and not manifest.needs_update(pdf_path, fingerprint=fingerprint, outputs=(output_path,)):
            continue
        if converter is None and engine in ("docling", "dual"):
            converter = build_converter(cfg)
        try:
            written_path = convert_pdf(pdf_path, output_path, converter=converter, engine=engine, cfg=cfg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            log.warning("skipping unconvertible PDF %s: %s", pdf_path, exc)
            continue
        written.append(written_path)
        if manifest:
            manifest.record(pdf_path, fingerprint=fingerprint, outputs=(output_path,))
    return written


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown (docling + markitdown).")
    parser.add_argument("--input", type=Path, help="Source PDF file or source directory")
    parser.add_argument("--output", type=Path, help="Target Markdown path for single-file conversion")
    parser.add_argument("--output-dir", type=Path, help="Output root directory for directory mode")
    parser.add_argument("--engine", choices=VALID_ENGINES, help="Conversion engine override")
    parser.add_argument("--config", type=Path, help="Path to pipeline.yaml")
    parser.add_argument("--no-manifest", action="store_true", help="Disable idempotency manifest")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("convert", cfg)

    source_id = get_source_id(cfg)
    input_path = (args.input or resolve_configured_path(cfg, "data_raw", "data/raw") / source_id).resolve()
    output_dir = (args.output_dir or resolve_configured_path(cfg, "output_raw_md", "outputs/raw_md")).resolve()
    engine = (
        args.engine
        or runtime_env_value("PIPELINE_ENGINE", "CORTEXMARK_ENGINE", cfg=cfg)
        or cfg.get("convert", {}).get("engine", "dual")
    )

    manifest = None
    idem_cfg = cfg.get("idempotency", {})
    if idem_cfg.get("enabled", True) and not args.no_manifest:
        manifest = Manifest(resolve_manifest_path(cfg))

    try:
        engine = _resolve_engine(engine)
        fingerprint = _convert_fingerprint(engine, cfg)
        if input_path.is_dir():
            if args.output is not None:
                raise ValueError("--output can only be used when --input points to a single PDF file")
            written = convert_tree(input_path, output_dir, engine=engine, cfg=cfg, manifest=manifest)
            if manifest:
                manifest.save()
            log.info("converted %d pdf file(s) under %s [engine=%s]", len(written), input_path, engine)
            return 0

        if args.output is not None:
            output_path = args.output.resolve()
        else:
            output_path = output_dir / input_path.with_suffix(".md").name

        if manifest and not manifest.needs_update(input_path, fingerprint=fingerprint, outputs=(output_path,)):
            log.info("skipped unchanged PDF %s [engine=%s]", input_path, engine)
            return 0

        written_path = convert_pdf(input_path, output_path, engine=engine, cfg=cfg)
        if manifest:
            manifest.record(input_path, fingerprint=fingerprint, outputs=(written_path,))
            manifest.save()
        log.info("wrote markdown to %s [engine=%s]", written_path, engine)
    except FileNotFoundError as exc:
        log.error("file not found: %s", exc)
        return 1
    except ValueError as exc:
        log.error("invalid input: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
