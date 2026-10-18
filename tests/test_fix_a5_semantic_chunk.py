"""Regression tests for A5-semantic-chunk findings C21, C36, C37 (+ the
semantic_chunk.chunk_file root-contract fix consumed by C14).

Covers:
  - C21: extract_formulas() must recognize docling's dollar-free
    "Equation (recovered from PDF):" blockquote convention.
  - C14 (root fix): chunk_file() must not raise ValueError on empty or
    whitespace-only cleaned text; it should emit a minimal single chunk.
  - C36: chunk_file() must remove stale chunk_*.md files before writing a
    new chunk set.
  - C37: extract_cross_refs() must not leave a dangling unbalanced
    "Theorem (3.2" artifact for parenthesized references.
"""

from __future__ import annotations

from pathlib import Path

from cortexmark.semantic_chunk import chunk_file, extract_cross_refs, extract_formulas


class TestExtractFormulasRecoveredEquation:
    """C21: docling emits no '$' delimiters -- recovered blockquotes must
    still be picked up as formulas."""

    def test_single_recovered_equation_blockquote(self) -> None:
        text = "> Equation (recovered from PDF):\n>\n> v ( Z) = 1 d d ∑ j =1 max(0, γ -S ( z j, ϵ)), (1)\n"
        formulas = extract_formulas(text)
        assert len(formulas) == 1
        assert "v ( Z)" in formulas[0]

    def test_multiple_recovered_equations_no_dollar_signs(self) -> None:
        text = (
            "> Equation (recovered from PDF):\n"
            ">\n"
            "> S(x,ϵ)=√Var(x)+ϵ (2)\n"
            "\n"
            "Some narrative text in between.\n"
            "\n"
            "> Equation (recovered from PDF):\n"
            ">\n"
            "> ℓ(Z,Z′)=λs(...)+µ[v(Z)+v(Z′)]+ν[c(Z)+c(Z′)] (6)\n"
        )
        assert "$" not in text
        formulas = extract_formulas(text)
        assert len(formulas) == 2

    def test_dollar_delimited_still_works(self) -> None:
        # Non-regression: the original LaTeX-delimited path is untouched.
        text = "We have $$V(s)=E[R]$$ as the value."
        assert extract_formulas(text) == ["V(s)=E[R]"]

    def test_no_formulas_plain_text(self) -> None:
        assert extract_formulas("No math here.") == []


class TestChunkFileEmptyText:
    """Root-contract fix behind C14: empty/whitespace cleaned text must not
    raise; it must produce a minimal single-chunk result."""

    def test_empty_string_returns_minimal_chunk(self, tmp_path: Path) -> None:
        src = tmp_path / "empty.md"
        src.write_text("", encoding="utf-8")
        written = chunk_file(src, tmp_path / "out")
        assert len(written) == 1
        assert written[0].exists()

    def test_whitespace_only_returns_minimal_chunk(self, tmp_path: Path) -> None:
        src = tmp_path / "whitespace.md"
        src.write_text("   \n\n  \n", encoding="utf-8")
        written = chunk_file(src, tmp_path / "out")
        assert len(written) == 1
        assert written[0].exists()

    def test_structured_text_unaffected(self, tmp_path: Path) -> None:
        src = tmp_path / "structured.md"
        src.write_text("# Chapter\n\n**Theorem 1.** Statement.\n", encoding="utf-8")
        written = chunk_file(src, tmp_path / "out")
        assert len(written) == 1
        assert "Theorem" in written[0].read_text(encoding="utf-8")


class TestChunkFileStaleCleanup:
    """C36: a shrinking re-run must not leave orphaned chunk_*.md files."""

    def test_stale_chunks_removed_on_rerun(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "chunk_050_stale.md").write_text("stale content", encoding="utf-8")
        (out / "chunk_051_stale.md").write_text("more stale content", encoding="utf-8")

        src = tmp_path / "only_heading.md"
        src.write_text("# Only Heading\n\nSome body text.\n", encoding="utf-8")
        written = chunk_file(src, out)

        remaining = sorted(p.name for p in out.glob("chunk_*.md"))
        assert remaining == sorted(p.name for p in written)
        assert "chunk_050_stale.md" not in remaining
        assert "chunk_051_stale.md" not in remaining

    def test_shrinking_rerun_leaves_no_orphans(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        src = tmp_path / "doc.md"

        src.write_text(
            "# Alpha\n\nBody alpha.\n\n# Beta\n\nBody beta.\n\n# Gamma\n\nBody gamma.\n",
            encoding="utf-8",
        )
        first_written = chunk_file(src, out)
        assert len(first_written) == 3

        src.write_text("# Alpha\n\nBody alpha only.\n", encoding="utf-8")
        second_written = chunk_file(src, out)
        assert len(second_written) == 1

        remaining = sorted(p.name for p in out.glob("chunk_*.md"))
        assert remaining == [p.name for p in second_written]


class TestExtractCrossRefsBalancedParens:
    """C37: parenthesized references must not leave a dangling '(' artifact."""

    def test_theorem_paren_reference_is_balanced(self) -> None:
        refs = extract_cross_refs("see Theorem (3.2) above")
        assert refs == ["Theorem 3.2"]
        assert not any("(" in r or ")" in r for r in refs)

    def test_equation_paren_reference_is_balanced(self) -> None:
        refs = extract_cross_refs("as in Equation (4.1)")
        assert refs == ["Equation 4.1"]

    def test_figure_paren_reference_is_balanced(self) -> None:
        refs = extract_cross_refs("refer to Figure (2)")
        assert refs == ["Figure 2"]

    def test_bare_and_parenthesized_forms_unify(self) -> None:
        # Same underlying reference spelled two ways should collapse to one
        # normalized token so a cross-reference index doesn't double-count.
        bare = extract_cross_refs("per Theorem 3.2 above")
        parenthesized = extract_cross_refs("see Theorem (3.2) above")
        assert bare == parenthesized == ["Theorem 3.2"]

    def test_mixed_bare_and_paren_refs_in_one_string(self) -> None:
        refs = extract_cross_refs("Section 4.1 and Eq. (5.3)")
        assert refs == ["Section 4.1", "Eq. 5.3"]

    def test_no_refs(self) -> None:
        assert extract_cross_refs("Plain text.") == []
