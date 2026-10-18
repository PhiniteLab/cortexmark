"""Regression tests for the Turkish-corpus verification findings (2026-07-24).

Covers two confirmed defects: normalize_table_blocks() dropping header-only
tables (Casebook Q&A data loss) and docling glyph-run word splits that the
accent normalizer cannot repair (a spaced fragment run like "t a k..." that
should be one Turkish word).
"""

from cortexmark.clean import normalize_table_blocks
from cortexmark.convert import repair_intraword_splits


class TestHeaderOnlyTablesKept:
    def test_header_only_table_with_content_survives(self) -> None:
        table = "| Pozisyon 3.1.1 soru metni | Karar: sayı tekrarlanır |\n| --- | --- |"
        result = normalize_table_blocks(table)
        assert "Pozisyon 3.1.1 soru metni" in result
        assert "Karar: sayı tekrarlanır" in result

    def test_fully_empty_table_still_dropped(self) -> None:
        table = "|  |  |\n| --- | --- |\n|  |  |"
        result = normalize_table_blocks(table)
        assert "|" not in result

    def test_normal_table_unchanged_semantics(self) -> None:
        table = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        result = normalize_table_blocks(table)
        assert "| 1 | 2 |" in result


class TestIntrawordSplitRepair:
    REF = (
        "Görüntü talep eden takımın ilk talebi sonucunda hakemlerin kararında "
        "bir değişiklik olmazsa ev de kalan oyuncular sahada birlikte bekler"
    )

    def test_confirmed_turkish_split_is_merged(self) -> None:
        md = "Görüntü talep eden t a kımın ilk talebi sonucunda"
        fixed = repair_intraword_splits(md, self.REF)
        assert "takımın" in fixed
        assert "t a kımın" not in fixed

    def test_legitimate_spaced_words_never_merged(self) -> None:
        # "ev de" is genuinely spaced in the reference, so it must survive even
        # though "evde" would be a plausible Turkish word.
        md = "oyuncular ev de kalan"
        fixed = repair_intraword_splits(md, self.REF + " evde oturmak")
        assert "ev de" in fixed

    def test_merge_refused_when_spaced_form_exists_in_reference(self) -> None:
        ref = "başlık t a kımın diye ayrık yazılmış ve takımın kelimesi de var"
        md = "metin t a kımın metin"
        assert repair_intraword_splits(md, ref) == md

    def test_merge_refused_when_word_not_in_reference(self) -> None:
        md = "bu k elime sözlükte yok"
        assert repair_intraword_splits(md, self.REF) == md

    def test_code_fences_untouched(self) -> None:
        md = "```\nt a kımın\n```"
        assert repair_intraword_splits(md, self.REF) == md

    def test_empty_reference_is_a_noop(self) -> None:
        md = "t a kımın"
        assert repair_intraword_splits(md, "") == md

    def test_unmodified_lines_keep_original_spacing(self) -> None:
        md = "çift  boşluklu  satır değişmemeli"
        assert repair_intraword_splits(md, self.REF) == md


class TestLetterSpacedRunRepair:
    REF = "BÖLÜM BAYRAK İŞARETLERİ İÇİNDEKİLER bölüm sonu ve devamı"

    def test_letter_spaced_heading_collapsed(self) -> None:
        fixed = repair_intraword_splits("## B Ö L Ü M 1", self.REF)
        assert "BÖLÜM 1" in fixed

    def test_two_word_letter_spaced_run_segmented(self) -> None:
        fixed = repair_intraword_splits("B A Y R A K İ Ş A R E T L E R İ", self.REF)
        assert fixed == "BAYRAK İŞARETLERİ"

    def test_run_kept_when_source_faithful(self) -> None:
        ref = self.REF + " B Ö L Ü M başlığı harf aralıklı dizilmiş"
        md = "B Ö L Ü M"
        assert repair_intraword_splits(md, ref) == md

    def test_unsegmentable_run_untouched(self) -> None:
        md = "X Q W Z Y"
        assert repair_intraword_splits(md, self.REF) == md
