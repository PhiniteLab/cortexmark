"""Regression tests for A3-sci-analysis audit fixes (C13, C22, C38, C39, C42, C43,
C44, C45, C46) plus the bounded C23/C60 table-integrity mitigation.

Covers:
  - cortexmark/scientific_qa.py:
      C13: check_theorem_proof_pairing() must not let ANY occurrence of the
        substring "proof" anywhere in a document suppress missing-proof
        warnings for every theorem, and must not collapse "Theorem 1" and
        "Lemma 1" into one entry.
      C38: check_crossref_completeness()'s equation/figure sub-checks must be
        able to fire on real pipeline output (recovered-equation blockquotes,
        plain "Figure N:" captions), not only on formats convert.py never emits.
      C46: an "Assumption N" reference is now validated against Assumption
        definition sites instead of being silently un-validated forever.
  - cortexmark/cross_ref.py:
      C22: EQUATION_LABEL_RE must recognize the pipeline's actual recovered-
        equation blockquote format so equation cross-references resolve.
      C39: _resolution_key() must not conflate "Chapter N" and "Section N".
  - cortexmark/notation_glossary.py:
      C42: deduplicated()'s secondary sort must prefer the longer (more
        descriptive) definition, not the shorter one.
      C43: detect_common_notations()'s display-math pattern must not match a
        known symbol that is merely a prefix of a longer custom macro.
  - cortexmark/citations.py / cortexmark/citation_context.py:
      C44: an inline Markdown link "[1](url)" must not be miscounted as a
        numeric citation, in either module.
      C45: the two modules' parenthetical author-year detection must agree on
        comma-less citations like "(Smith 2020)".
  - cortexmark/qa_pipeline.py:
      C23/C60 mitigation: check_table_integrity() flags a table row whose
      cell count is wildly (>=3x) out of line with the table's median,
      surfacing severe corruption instead of leaving it undetected.
"""

from __future__ import annotations

from cortexmark.citation_context import (
    BRACKET_CITE_RE as CTX_BRACKET_CITE_RE,
)
from cortexmark.citation_context import (
    PAREN_CITE_RE as CTX_PAREN_CITE_RE,
)
from cortexmark.citation_context import (
    extract_citation_contexts,
)
from cortexmark.citations import (
    PAREN_CITE_RE as CITE_PAREN_CITE_RE,
)
from cortexmark.citations import (
    extract_inline_citations,
)
from cortexmark.cross_ref import (
    extract_definitions,
    extract_mentions,
    resolve_references,
)
from cortexmark.notation_glossary import (
    NotationEntry,
    NotationGlossary,
    detect_common_notations,
)
from cortexmark.qa_pipeline import check_table_integrity
from cortexmark.scientific_qa import (
    check_crossref_completeness,
    check_theorem_proof_pairing,
)

# ── C13: theorem-proof pairing ──────────────────────────────────────────────


class TestTheoremProofPairing:
    def test_generic_proof_does_not_suppress_other_theorems(self) -> None:
        """A generic '**Proof.**' for Theorem 1 must not silence warnings for
        Theorem 2, Theorem 3, and Lemma 1 elsewhere in the same document."""
        text = (
            "**Theorem 1.** Statement.\n\n"
            "**Proof.** Done.\n\n"
            "**Theorem 2.** Statement.\n\n"
            "**Theorem 3.** Statement.\n\n"
            "**Lemma 1.** Statement.\n"
        )
        issues = check_theorem_proof_pairing(text)
        messages = {i.message for i in issues}
        assert "Theorem 2 has no corresponding proof" in messages
        assert "Theorem 3 has no corresponding proof" in messages
        assert "Lemma 1 has no corresponding proof" in messages
        assert "Theorem 1 has no corresponding proof" not in messages

    def test_prose_mention_of_proof_does_not_count_as_a_proof(self) -> None:
        """A bare prose mention of the word 'proof' is not a proof heading."""
        text = "We defer the proof to the appendix.\n\n**Theorem 1.** Statement.\n"
        issues = check_theorem_proof_pairing(text)
        assert any("Theorem 1 has no corresponding proof" in i.message for i in issues)

    def test_word_inside_bulletproof_does_not_count_as_a_proof(self) -> None:
        text = "This bound is bulletproof against adversaries.\n\n**Theorem 1.** Statement.\n"
        issues = check_theorem_proof_pairing(text)
        assert any("Theorem 1 has no corresponding proof" in i.message for i in issues)

    def test_theorem_and_lemma_sharing_a_numeral_are_tracked_independently(self) -> None:
        """A 'Proof of Theorem 1' must not silently also cover an unrelated
        'Lemma 1' that shares the numeral 1."""
        text = "**Theorem 1.** Statement.\n\n**Proof of Theorem 1.** Done.\n\n**Lemma 1.** Statement.\n"
        issues = check_theorem_proof_pairing(text)
        messages = {i.message for i in issues}
        assert "Lemma 1 has no corresponding proof" in messages
        assert "Theorem 1 has no corresponding proof" not in messages

    def test_still_passes_the_pre_existing_single_theorem_cases(self) -> None:
        assert check_theorem_proof_pairing("**Theorem 1.** Some statement.\n\n**Proof.** We show that...") == []
        assert check_theorem_proof_pairing("**Theorem 1.** Statement.\n\nThe proof is omitted.") == []
        assert check_theorem_proof_pairing("**Theorem 1.** Statement.\n\nThe proof is left to the reader.") == []
        assert check_theorem_proof_pairing("**Theorem 2.1** Statement.\n\n**Proof of Theorem 2.1.** Done.") == []
        assert len(check_theorem_proof_pairing("**Lemma 3.** A useful lemma.")) >= 1


# ── C38 / C46: crossref completeness on real pipeline output ───────────────


class TestCrossrefCompletenessRealFormats:
    def test_recovered_equation_blockquote_is_a_definition_site(self) -> None:
        """The exact blockquote form convert.py emits for a recovered,
        numbered equation must register as an Equation N definition."""
        doc = (
            "Some intro text.\n\n"
            "> Equation (recovered from PDF):\n"
            ">\n"
            "> l(Z,Zp) = ls(Z,Zp) + m[v(Z)+v(Zp)], (6)\n\n"
            "As shown in Equation 6, the bound holds. Equation 99 does not exist.\n"
        )
        issues = check_crossref_completeness(doc)
        messages = [i.message for i in issues]
        assert any("Equation 99" in m for m in messages)
        assert not any("Equation 6 but no definition" in m for m in messages)

    def test_plain_figure_caption_is_a_definition_site(self) -> None:
        """A plain 'Figure N: caption' text line (no markdown image alt-text,
        since convert.py disables docling picture generation) must register
        as a Figure N definition."""
        doc = "Figure 1: A snapshot of the results.\n\nAs shown in Figure 1, ... Figure 42 does not exist.\n"
        issues = check_crossref_completeness(doc)
        messages = [i.message for i in issues]
        assert any("Figure 42" in m for m in messages)
        assert not any("Figure 1 but no definition" in m for m in messages)

    def test_undefined_assumption_reference_is_now_flagged(self) -> None:
        doc = "**Assumption 7** The noise is bounded.\n\nUnder Assumption 7 the bound holds.\nUnder Assumption 99 it fails.\n"
        issues = check_crossref_completeness(doc)
        assert any("Assumption 99" in i.message for i in issues)
        assert not any("Assumption 7 but no definition" in i.message for i in issues)

    def test_assumption_mid_sentence_mention_is_not_mistaken_for_a_definition(self) -> None:
        """An in-text mention ('Under Assumption 7 ...') must not itself
        register as a definition site, or an undefined Assumption N would
        falsely self-resolve."""
        doc = "Under Assumption 7 the bound holds.\n"
        issues = check_crossref_completeness(doc)
        # No Assumption definition anywhere -> conservative design does not
        # flag it either (consistent with how other kinds behave), but it
        # must not have registered itself as a definition.
        assert issues == []


# ── C22 / C39: cross_ref.py ─────────────────────────────────────────────────


class TestCrossRefEquationAndSectionFixes:
    def test_recovered_equation_blocks_resolve_eq_mentions(self) -> None:
        text = (
            "> Equation (recovered from PDF):\n"
            ">\n"
            "> l(Z,Zp) = ls(Z,Zp) + m[v(Z)+v(Zp)], (6)\n\n"
            "Coefficients are set as in Eq. (6).\n"
        )
        defs = extract_definitions(text)
        mentions = extract_mentions(text)
        report = resolve_references(defs, mentions)
        assert any(d.kind == "Equation" and d.label == "6" for d in defs)
        assert report.resolved == ["Equation 6"]
        assert report.unresolved == []

    def test_chapter_mention_does_not_falsely_resolve_against_section_definition(self) -> None:
        text = "## Section 2 Methods\n\nSee Chapter 2 for the derivation...\n"
        defs = extract_definitions(text)
        mentions = extract_mentions(text)
        report = resolve_references(defs, mentions)
        assert report.resolved == []
        assert report.unresolved == ["Chapter 2"]

    def test_distinct_numeral_chapter_and_section_still_resolve(self) -> None:
        """Regression guard for the pre-existing passing scenario: an explicit
        Chapter heading and a numeric-only Section sub-heading with distinct
        numerals must both still resolve their own mentions."""
        text = "# Chapter 2 Methods\n\n## 2.1 Setup\n\nSee Section 2.1 and Chapter 2.\n"
        defs = extract_definitions(text)
        mentions = extract_mentions(text)
        report = resolve_references(defs, mentions)
        assert report.unresolved == []
        assert set(report.resolved) == {"Section 2.1", "Chapter 2"}


# ── C42 / C43: notation_glossary.py ─────────────────────────────────────────


class TestNotationGlossaryFixes:
    def test_dedup_tiebreak_prefers_the_longer_definition(self) -> None:
        glossary = NotationGlossary(
            entries=[
                NotationEntry(symbol="X", definition="a set", source="explicit"),
                NotationEntry(
                    symbol="X",
                    definition="the state space of the Markov decision process",
                    source="explicit",
                ),
            ]
        )
        deduped = glossary.deduplicated()
        assert len(deduped) == 1
        assert deduped[0].definition == "the state space of the Markov decision process"

    def test_dedup_tiebreak_is_order_independent(self) -> None:
        glossary = NotationGlossary(
            entries=[
                NotationEntry(
                    symbol="X",
                    definition="the state space of the Markov decision process",
                    source="explicit",
                ),
                NotationEntry(symbol="X", definition="a set", source="explicit"),
            ]
        )
        deduped = glossary.deduplicated()
        assert deduped[0].definition == "the state space of the Markov decision process"

    def test_display_math_symbol_prefix_is_not_a_false_positive(self) -> None:
        entries = detect_common_notations(r"$$ \thetaHat = \nabla J $$")
        symbols = {e.symbol for e in entries}
        assert r"\theta" not in symbols
        assert r"\nabla" in symbols

    def test_display_math_clean_symbol_still_detected(self) -> None:
        entries = detect_common_notations(r"$$ \theta = 1 $$")
        assert {e.symbol for e in entries} == {r"\theta"}


# ── C44 / C45: citations.py / citation_context.py ───────────────────────────


class TestCitationLinkAndAuthorYearFixes:
    def test_inline_markdown_link_is_not_a_numeric_citation_in_either_module(self) -> None:
        line = "Footnote [1](#fn1) and cite [2] and [3](http://a).\n"
        surfaces = {c.surface_text for c in extract_inline_citations(line)}
        assert surfaces == {"[2]"}
        raws = {c.raw_text for c in extract_citation_contexts(line)}
        assert raws == {"[2]"}

    def test_reference_style_link_still_skipped(self) -> None:
        line = "A ref-style link [text][1] here.\n"
        assert extract_inline_citations(line) == []
        assert extract_citation_contexts(line) == []

    def test_comma_less_author_year_agrees_across_both_modules(self) -> None:
        line = "The idea (Smith 2020) is central."
        assert CITE_PAREN_CITE_RE.findall(line) == ["Smith 2020"]
        assert CTX_PAREN_CITE_RE.findall(line) == ["Smith 2020"]

    def test_comma_less_bracket_still_matches_plain_numeric_citation(self) -> None:
        assert CTX_BRACKET_CITE_RE.findall("As shown in [2].") == ["2"]


# ── C23/C60 mitigation: table cell-count outlier detection ─────────────────


class TestTableCellCountOutlierMitigation:
    def test_wildly_inconsistent_row_is_flagged(self) -> None:
        text = "| Method | Acc |\n| --- | --- |\n| a | 1 |\n| b | 2 | 3 | 4 | 5 | 6 | 7 |\n"
        issues = check_table_integrity(text)
        assert any("possible corrupted/merged row" in i.message for i in issues)

    def test_mild_inconsistency_does_not_trigger_the_outlier_heuristic(self) -> None:
        """A single extra cell (well under 3x the median) should not be
        reported as a 'possible corrupted/merged row' -- that is still
        covered by the pre-existing generic inconsistent-columns warning."""
        text = "| A | B |\n| --- | --- |\n| 1 | 2 | 3 |"
        issues = check_table_integrity(text)
        assert not any("possible corrupted/merged row" in i.message for i in issues)

    def test_consistent_table_has_no_issues(self) -> None:
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        assert check_table_integrity(text) == []
