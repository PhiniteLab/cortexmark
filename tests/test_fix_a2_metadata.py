"""Regression tests for A2-metadata audit fixes (C02, C12, C19, C40, C41, C49).

Covers cortexmark/metadata.py:
  - C02: extract_authors() must not silently return [] when the title is an H2/H3 heading.
  - C12/C19: extract_title() must strip ALL leading Markdown hashes, at any heading level.
  - C40: REFERENCE_HEADING_RE must not truncate on an inline uppercase REFERENCES/BIBLIOGRAPHY
    token that is not a real reference-list marker, while still truncating flattened reference
    sections that begin with a numbered entry marker.
  - C41: the author-name tokenizer must strip glued footnote/affiliation digits instead of
    rejecting the whole name.
  - C49: extract_file() must tolerate non-UTF-8 bytes instead of raising UnicodeDecodeError.
"""

from __future__ import annotations

from pathlib import Path

from cortexmark.metadata import (
    _looks_like_author_name,
    _strip_footnote_markers,
    extract_authors,
    extract_doi,
    extract_file,
    extract_journal,
    extract_metadata,
    extract_title,
    extract_year,
)


class TestTitleAnyHeadingLevel:
    """C12 / C19: extract_title() strips hashes at every heading level."""

    def test_h2_title_is_stripped(self) -> None:
        lines = ["## A Contextual-Bandit Approach to Personalized News", "", "Lihong Li"]
        assert extract_title(lines) == "A Contextual-Bandit Approach to Personalized News"

    def test_h3_title_is_stripped(self) -> None:
        lines = ["### Some H3 Title", ""]
        assert extract_title(lines) == "Some H3 Title"

    def test_h1_title_still_works(self) -> None:
        lines = ["# Proper H1 Title", ""]
        assert extract_title(lines) == "Proper H1 Title"

    def test_fallback_line_never_leaks_hashes(self) -> None:
        # No heading at all in the first 30 lines: fallback path must still strip any
        # stray leading hashes rather than returning them verbatim.
        lines = ["## Stray heading-like first line", "", "Body text."]
        # This is actually caught by the primary loop (any heading level matches), but
        # guard the fallback branch directly too.
        assert extract_title(lines) == "Stray heading-like first line"

    def test_no_leading_hash_in_bibtex(self) -> None:
        text = "## A Tutorial on Bayesian Optimization\n\nPeter I. Frazier\n\nAbstract: ...\n"
        meta = extract_metadata(text)
        assert not meta.title.startswith("#")
        assert meta.title == "A Tutorial on Bayesian Optimization"


class TestAuthorsWithNonH1Title:
    """C02: extract_authors() must find the byline when the title is H2/H3."""

    def test_h2_title_authors_are_found(self) -> None:
        lines = [
            "## A Contextual-Bandit Approach to Personalized News Article Recommendation",
            "",
            "Lihong Li, Wei Chu, John Langford, Robert E. Schapire",
            "",
        ]
        authors = extract_authors(lines)
        assert authors == ["Lihong Li", "Wei Chu", "John Langford", "Robert E. Schapire"]

    def test_h3_title_authors_are_found(self) -> None:
        lines = ["### A Title", "Alice Wang and Bob Chen", ""]
        assert extract_authors(lines) == ["Alice Wang", "Bob Chen"]

    def test_h1_title_still_works(self) -> None:
        lines = ["# A Title", "Alice Wang and Bob Chen", ""]
        assert extract_authors(lines) == ["Alice Wang", "Bob Chen"]

    def test_provenance_records_authors_from_h2_title(self) -> None:
        text = "## A Title\n\nAlice Wang, Bob Chen\n\nAbstract: text.\n"
        meta = extract_metadata(text)
        assert meta.authors == ["Alice Wang", "Bob Chen"]
        assert meta.provenance["authors"] == "heuristic-byline"


class TestReferenceHeadingAnchoring:
    """C40: inline uppercase REFERENCES/BIBLIOGRAPHY must not truncate front matter."""

    def test_inline_uppercase_references_in_prose_does_not_truncate(self) -> None:
        text = (
            "As noted in the REFERENCES cited above, DOI 10.1234/abcd was assigned; "
            "the Journal of Testing published it in 2021.\n\n"
            "## References\n\n1. Some real reference.\n"
        )
        assert extract_doi(text) == "10.1234/abcd"
        assert extract_year(text) == "2021"
        assert "Journal of Testing" in extract_journal(text)

    def test_flattened_numbered_reference_section_still_truncates(self) -> None:
        # Regression guard for tests.test_pipeline_structure.TestExtractDoi
        # .test_reference_doi_is_not_attributed_to_document: a converter that flattens the
        # reference list onto a single running line, starting with a numbered marker, must
        # still be recognized and truncated.
        text = (
            "# Current Paper\n\nJane Doe\n\nThe document body ends here. REFERENCES "
            "[37] Other Author (2017). Proceedings of Example Research. 10.18653/v1/S17-2001\n"
        )
        meta = extract_metadata(text)
        assert meta.doi == ""
        assert meta.journal == ""
        assert meta.year == ""


class TestAuthorFootnoteMarkers:
    """C41: glued footnote/affiliation digits are stripped, not treated as a rejection."""

    def test_strip_footnote_markers_helper(self) -> None:
        assert _strip_footnote_markers("Vaswani1") == "Vaswani"
        assert _strip_footnote_markers("Doe2,3") == "Doe"
        assert _strip_footnote_markers("Jane Doe") == "Jane Doe"

    def test_looks_like_author_name_still_rejects_raw_digits(self) -> None:
        # _looks_like_author_name itself is unchanged; stripping happens before it is called.
        assert _looks_like_author_name("Ashish Vaswani1") is False
        assert _looks_like_author_name("Ashish Vaswani") is True

    def test_extract_authors_strips_glued_footnote_digits(self) -> None:
        lines = ["# Attention Is All You Need", "", "Ashish Vaswani1, Noam Shazeer2", ""]
        assert extract_authors(lines) == ["Ashish Vaswani", "Noam Shazeer"]

    def test_extract_authors_strips_comma_footnote_cluster(self) -> None:
        lines = ["# A Title", "", "John Doe2,3", ""]
        assert extract_authors(lines) == ["John Doe"]


class TestTolerantFileRead:
    """C49: extract_file() degrades non-UTF-8 input instead of raising UnicodeDecodeError."""

    def test_non_utf8_file_does_not_raise(self, tmp_path: Path) -> None:
        bad = tmp_path / "latin.md"
        bad.write_bytes(b"# T\xffle\n\nJane Doe\n")
        meta = extract_file(bad)
        assert "T" in meta.title
        assert "�" in meta.title

    def test_valid_utf8_file_unaffected(self, tmp_path: Path) -> None:
        good = tmp_path / "good.md"
        good.write_text("# A Title\n\nJane Doe\n", encoding="utf-8")
        meta = extract_file(good)
        assert meta.title == "A Title"
