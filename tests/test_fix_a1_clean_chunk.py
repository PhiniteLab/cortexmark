"""Regression tests for A1-clean-chunk audit fixes (C08, C09, C10, C11, C61).

Covers cortexmark/clean.py and cortexmark/chunk.py:
  - C08: clean_markdown() must not flatten leading indentation inside fenced code
    blocks (algorithm/pseudocode nesting, indented list structure).
  - C09: remove_repeated_headers_footers() must not delete legitimate repeated
    pipe-table header/separator rows (e.g. a table header repeated once per page)
    or repeated lines inside fenced code blocks.
  - C10: remove_page_numbers() must not delete bare-integer lines that live inside
    a fenced code block or numeric data block.
  - C11: chunk.py's parse_chunks()/flush() must preserve blank-line paragraph
    boundaries in prose and blank lines inside fenced code blocks, instead of
    stripping every blank line from the chunk body.
  - C61: clean_markdown() must decode HTML entities (&amp; &lt; &gt; ...) in prose
    while leaving fenced code and inline code spans untouched.
"""

from __future__ import annotations

from cortexmark.chunk import parse_chunks
from cortexmark.clean import clean_markdown


class TestFenceIndentationPreserved:
    """C08: indentation inside fenced code blocks must survive clean_markdown()."""

    def test_nested_pseudocode_indentation_preserved(self) -> None:
        text = (
            "```text\n"
            "for i in range(n):\n"
            "    for j in range(n):\n"
            "        if a[i] < a[j]:\n"
            "            swap(a, i, j)\n"
            "```\n"
        )
        out = clean_markdown(text)
        assert "    for j in range(n):" in out
        assert "        if a[i] < a[j]:" in out
        assert "            swap(a, i, j)" in out

    def test_non_fence_heading_still_stripped(self) -> None:
        # Outside a fence, normalize_heading_lines() should still strip leading
        # markdown-heading hashes/whitespace as before -- only fence interiors
        # are protected.
        text = "  ## A Heading  \n\nbody\n"
        out = clean_markdown(text)
        assert "## A Heading" in out
        assert "  ## A Heading  " not in out


class TestNoDeletionInsideFences:
    """C09 + C10: repeated-line and page-number removal must skip fence interiors."""

    def test_bare_integer_inside_fence_survives(self) -> None:
        text = "```python\ndef f():\n    x = 42\n    print(x)\n42\n```\n"
        out = clean_markdown(text)
        lines = out.splitlines()
        assert "42" in lines
        assert "    x = 42" in out

    def test_bare_page_number_outside_fence_still_removed(self) -> None:
        # Sanity: the original page-number removal behaviour for real page
        # numbers (outside any fence) must be unchanged.
        text = "some text\n\n42\n\nmore text\n"
        out = clean_markdown(text)
        assert "\n42\n" not in out

    def test_repeated_short_line_inside_fence_survives(self) -> None:
        block = "```python\ndef a():\n    return None\n```\n"
        text = block * 3
        out = clean_markdown(text)
        assert out.count("return None") == 3


class TestRepeatedTableHeaderSpared:
    """C09: a legitimate multi-page table header/separator must not be deleted."""

    def test_table_header_repeated_across_pages_kept(self) -> None:
        text = (
            "# Results\n\n"
            "| Method | Accuracy | F1 |\n"
            "| --- | --- | --- |\n"
            "| A | 0.9 | 0.8 |\n\n"
            "Some page-1 text.\n\n"
            "| Method | Accuracy | F1 |\n"
            "| --- | --- | --- |\n"
            "| B | 0.7 | 0.6 |\n\n"
            "Some page-2 text.\n\n"
            "| Method | Accuracy | F1 |\n"
            "| --- | --- | --- |\n"
            "| C | 0.5 | 0.4 |\n"
        )
        out = clean_markdown(text)
        assert out.count("Method") == 3
        assert out.count("Accuracy") == 3
        assert "| A | 0.9 | 0.8 |" in out
        assert "| B | 0.7 | 0.6 |" in out
        assert "| C | 0.5 | 0.4 |" in out

    def test_genuine_repeated_boilerplate_line_still_removed(self) -> None:
        # Non-table running headers/footers repeated >= min_count times should
        # still be treated as boilerplate and removed -- only pipe-table rows
        # and fence interiors are protected.
        text = "\n\n".join(["Confidential Draft - Do Not Distribute"] * 3 + ["real content here"])
        out = clean_markdown(text)
        assert out.count("Confidential Draft - Do Not Distribute") == 0
        assert "real content here" in out


class TestChunkBlankLinesPreserved:
    """C11: parse_chunks() must preserve paragraph and fence-internal blank lines."""

    def test_paragraph_boundary_preserved(self) -> None:
        text = (
            "# Introduction\n\n"
            "This is the first paragraph with some content.\n\n"
            "This is a second, logically distinct paragraph.\n"
        )
        chunks = parse_chunks(text)
        assert len(chunks) == 1
        rendered = chunks[0].render()
        assert (
            "This is the first paragraph with some content.\n\n"
            "This is a second, logically distinct paragraph." in rendered
        )

    def test_blank_line_inside_fence_preserved(self) -> None:
        text = "# Code\n\n```python\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n```\n"
        chunks = parse_chunks(text)
        assert len(chunks) == 1
        rendered = chunks[0].render()
        assert "    return 1\n\ndef bar():" in rendered


class TestHtmlEntitiesDecoded:
    """C61: HTML entities are decoded in prose but left alone in code."""

    def test_ampersand_and_angle_brackets_decoded_in_prose(self) -> None:
        text = "Citations: Chen &amp; He (2020), lambda &gt; 1, mu &lt; 1.\n"
        out = clean_markdown(text)
        assert "Chen & He (2020)" in out
        assert "lambda > 1" in out
        assert "mu < 1" in out
        assert "&amp;" not in out

    def test_entities_inside_fence_left_untouched(self) -> None:
        text = "prose &amp; more.\n\n```text\nx &amp; y\n```\n"
        out = clean_markdown(text)
        assert "prose & more." in out
        assert "x &amp; y" in out

    def test_entities_inside_inline_code_span_left_untouched(self) -> None:
        text = "See `inline &amp; code` outside prose &amp; here.\n"
        out = clean_markdown(text)
        assert "`inline &amp; code`" in out
        assert "outside prose & here" in out
