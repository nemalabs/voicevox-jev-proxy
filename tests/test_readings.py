import gzip
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from voicevox_jev_proxy.accents import Word
from voicevox_jev_proxy.cli import plan_text
from voicevox_jev_proxy.readings import (
    AccentDictionary,
    Entry,
    Gloss,
    GlossDictionary,
    ReadingCandidate,
    ReadingLexicon,
    ReadingOption,
    accent_targets,
    accent_types,
    build_reading_questions,
    choose_options,
    clause_around,
    common_only,
    group_options,
    load_jmdict,
    mora_count,
    narrow,
    option_context,
    pos_label,
    reading_edits,
    single_sense,
    to_hiragana,
    voicevox_keys,
    with_voicevox_reading,
)
from voicevox_jev_proxy.typesafe import ChoiceAnswer
from voicevox_jev_proxy.voicevox import AccentPhrase, AudioQuery, Mora

DICT_PATH = Path(os.environ.get("SUDACHI_DICT_PATH", ""))
needs_sudachi = pytest.mark.skipif(not DICT_PATH.is_file(), reason="SUDACHI_DICT_PATH not set to a dictionary file")

COMMON = "名詞-普通名詞-一般"
MIYAKO_ENTRIES = [
    Entry("ミヤコ", "都", COMMON, (0,)),
    Entry("ミヤコ", "ミヤコ", "名詞-固有名詞-人名-名", (1,)),
    Entry("ミヤコ", "ミヤコ", "名詞-固有名詞-地名-一般", (1,)),
    Entry("ト", "都", COMMON, (1,)),
]


def choice(value: str, confidence: float = 0.9) -> ChoiceAnswer:
    return ChoiceAnswer(type="choice", choice=value, confidence=confidence, probabilities={value: confidence})


def option(reading: str, accents: tuple[int, ...], lemma: str = "都") -> ReadingOption:
    return ReadingOption(reading, accents, (lemma,), (COMMON,), (lemma,), lemma_readings=((lemma, reading),))


def miyako(attached: int = 1) -> ReadingCandidate:
    options = {
        "ミヤコ(都)": option("ミヤコ", (0,)),
        "ミヤコ(人名)": option("ミヤコ", (1,), "ミヤコ"),
        "ト": option("ト", (1,)),
    }
    return ReadingCandidate(6, 7, "都", "名詞", options, attached)


def test_to_hiragana_keeps_prolonged_sound_mark():
    assert to_hiragana("ミヤコ") == "みやこ"
    assert to_hiragana("コーヒー") == "こーひー"


def test_mora_count_skips_small_kana_only():
    assert mora_count("ミヤコ") == 3
    assert mora_count("トウキョウ") == 4
    assert mora_count("キャッチャー") == 4


def test_accent_types_keep_every_value_in_dictionary_order():
    assert accent_types("0") == (0,)
    assert accent_types("2,1") == (2, 1)
    assert accent_types("*") == ()


def test_pos_label_keeps_the_kind_of_proper_noun():
    assert pos_label(("名詞", "普通名詞", "一般", "*", "*", "*")) == COMMON
    assert pos_label(("名詞", "固有名詞", "人名", "姓", "*", "*")) == "名詞-固有名詞-人名-姓"


def test_group_options_splits_a_reading_by_accent_and_names_the_kind():
    options = group_options(MIYAKO_ENTRIES, by_accent=True)
    assert list(options) == ["ミヤコ(都)", "ミヤコ(人名・地名)", "ト"]
    assert [o.accents for o in options.values()] == [(0,), (1,), (1,)]
    assert options["ミヤコ(人名・地名)"].pos == ("名詞-固有名詞-人名-名", "名詞-固有名詞-地名-一般")


def test_group_options_merges_accents_of_inflected_words():
    entries = [Entry("フル", "降る", "動詞-一般", (1,)), Entry("フル", "降る", "動詞-一般", (0,))]
    assert list(group_options(entries, by_accent=False)) == ["フル"]


def test_build_reading_questions_asks_reading_for_kanji_and_word_for_kana():
    text = "今日、そして都の大路を歩いた。"
    kana = ReadingCandidate(3, 6, "そして", "接続詞", {"ソシテ": option("ソシテ", (0,), "然して")})
    questions = build_reading_questions(text, [miyako(), kana])
    assert list(questions) == ["r6_reading", "r3_reading"]
    assert list(questions["r6_reading"]["criteria"]) == ["ミヤコ(都)", "ミヤコ(人名)", "ト", "none"]
    assert questions["r6_reading"]["criteria"]["ト"] == {
        "reading": "ト",
        "lemma": ["都"],
        "pos": [COMMON],
        "context": ["そしてとの大路を歩いた"],
    }
    assert questions["r3_reading"]["criteria"]["ソシテ"]["context"] == ["然して都の大路を歩いた"]
    assert "どう読むか" in questions["r6_reading"]["instructions"]["question"]
    assert "ひらがな" in questions["r6_reading"]["instructions"]["note"]
    assert "どの語か" in questions["r3_reading"]["instructions"]["question"]
    assert "表記" in questions["r3_reading"]["instructions"]["note"]


@pytest.mark.parametrize(
    ("text", "start", "end", "expected"),
    [
        ("人気のない夜道を、一人で歩いた。", 0, 2, ("", "のない夜道を")),
        ("人気のない夜道を、一人で歩いた。", 9, 11, ("", "で歩いた")),
        ("庭のかきが赤い", 2, 4, ("庭の", "が赤い")),
    ],
)
def test_clause_around_stops_at_punctuation(text: str, start: int, end: int, expected: tuple[str, str]):
    assert clause_around(text, start, end) == expected


def test_option_context_writes_kana_words_with_each_lemma():
    kana = ReadingCandidate(2, 4, "かき", "名詞", {"カキ": option("カキ", (1,))})
    both = ReadingOption("カキ", (1,), ("花卉", "牡蠣"), (COMMON,), ("花卉", "牡蠣"))
    assert option_context("庭のかきが赤い", kana, both) == ["庭の花卉が赤い", "庭の牡蠣が赤い"]


def test_choose_options_ignores_none_unknown_and_low_confidence():
    candidates = [miyako()]
    assert choose_options(candidates, {"r6_reading": choice("none")}, 0.6) == []
    assert choose_options(candidates, {"r6_reading": choice("ミヤコ(都)", 0.3)}, 0.6) == []
    assert choose_options(candidates, {"r6_reading": choice("キョウ")}, 0.6) == []
    (chosen,) = choose_options(candidates, {"r6_reading": choice("ミヤコ(都)")}, 0.6)
    assert chosen.option.accents == (0,)


def test_reading_edits_replace_kanji_but_leave_kana_words():
    kana = ReadingCandidate(
        0, 2, "あめ", "名詞", {"アメ(飴)": option("アメ", (0,), "飴"), "アメ(雨)": option("アメ", (1,))}
    )
    answers = {"r6_reading": choice("ミヤコ(都)"), "r0_reading": choice("アメ(飴)")}
    edits = reading_edits(choose_options([miyako(), kana], answers, 0.6))
    assert [(e.start, e.end, e.replacement, e.change.field) for e in edits] == [(6, 7, "みやこ", "reading")]


def test_accent_targets_carry_the_chosen_accent_and_skip_unknown_ones():
    unknown = ReadingCandidate(0, 2, "推し", "名詞", {"オシ": option("オシ", ()), "スイシ": option("スイシ", ())})
    verb = ReadingCandidate(3, 5, "降る", "動詞", {"フル": option("フル", ()), "クダル": option("クダル", ())})
    answers = {"r6_reading": choice("ミヤコ(都)"), "r0_reading": choice("オシ"), "r3_reading": choice("フル")}
    targets, skipped = accent_targets(choose_options([miyako(attached=1), unknown, verb], answers, 0.6))
    assert [(t.id, t.start, t.end, t.accents, t.moras, t.attached_moras) for t in targets] == [("r6", 6, 7, (0,), 3, 1)]
    assert [(c.phrase_id, c.field, c.reason) for c in skipped] == [("r0", "accent_skipped", "no dictionary accent")]


@pytest.mark.parametrize(
    ("pos", "followed_by", "cut"),
    [
        ("名詞-普通名詞-副詞可能", "名詞", True),
        ("名詞-普通名詞-副詞可能", "接尾辞", False),
        ("名詞-普通名詞-副詞可能", "", False),
        (COMMON, "名詞", False),
    ],
)
def test_accent_targets_let_only_an_adverbial_noun_before_an_independent_word_cut_its_phrase(
    pos: str, followed_by: str, cut
):
    time = ReadingOption("トキ", (2,), ("時",), (pos,), ("時",))
    candidate = ReadingCandidate(2, 4, "とき", "名詞", {"トキ(時)": time}, 0, followed_by)
    (target,), _ = accent_targets(choose_options([candidate], {"r2_reading": choice("トキ(時)")}, 0.6))
    assert target.cut is cut


def test_accent_dictionary_lists_unidic_entries_of_the_same_part_of_speech():
    entries = AccentDictionary().entries("都", "名詞")
    assert Entry("ミヤコ", "都", COMMON, (0,), "ミヤコ") in entries
    assert Entry("ト", "都", COMMON, (1,), "ト") in entries
    assert all(entry.pos.startswith("名詞") for entry in entries)


@pytest.mark.parametrize(
    ("surface", "kept", "changed"),
    [
        ("人", ["ヒト"], {"ビト"}),
        ("近く", ["チカク"], {"ジカク"}),
        ("三", ["サン", "サン", "ミ"], {"ミイ", "ミッ"}),
    ],
)
def test_accent_dictionary_leaves_out_sound_changed_forms(surface: str, kept: list[str], changed: set[str]):
    dictionary = AccentDictionary()
    assert sorted(entry.reading for entry in dictionary.entries(surface, "名詞")) == kept
    assert dictionary.changed_readings(surface, "名詞") == changed


def test_single_sense_holds_only_when_every_reading_has_the_same_gloss():
    def glossed(reading: str, meaning: str) -> ReadingOption:
        return replace(option(reading, (0,), "明日"), meanings=(meaning,) if meaning else ())

    tomorrow = {"アシタ": glossed("アシタ", "tomorrow"), "アス": glossed("アス", "tomorrow")}
    assert single_sense(tomorrow)
    assert not single_sense(tomorrow | {"ミョウニチ": glossed("ミョウニチ", "")})
    assert not single_sense(tomorrow | {"アス#2": glossed("アス", "the next day")})
    name = replace(ReadingOption("アス", (1,), ("アス",), ("名詞-固有名詞-人名-名",), ("人名",)), meanings=())
    assert not single_sense(tomorrow | {"アス(人名)": name})
    assert single_sense(tomorrow | {"アス": replace(tomorrow["アス"], later_meanings=("the near future",))})


def test_single_sense_takes_nested_glosses_as_one_sense_only_for_inflected_words():
    def glossed(reading: str, meaning: str) -> ReadingOption:
        return replace(option(reading, (), "入る"), meanings=(meaning,))

    enter = {
        "ハイル": glossed("ハイル", "to enter; to come in; to arrive"),
        "イル": glossed("イル", "to enter; to come in"),
    }
    assert single_sense(enter, inflected=True)
    assert not single_sense(enter)
    assert not single_sense(enter | {"イル#2": glossed("イル", "to enter; to be needed")}, inflected=True)


def test_common_only_drops_rare_readings_but_keeps_names():
    glass = {
        "ガラス": replace(option("ガラス", (0,), "硝子"), common=True),
        "ショウシ": option("ショウシ", (0,), "硝子"),
        "ショウシ(人名)": ReadingOption("ショウシ", (1,), ("ショウシ",), ("名詞-固有名詞-人名-姓",), ("人名",)),
    }
    assert list(common_only(glass)) == ["ガラス", "ショウシ(人名)"]
    rare = {key: replace(value, common=False) for key, value in glass.items()}
    assert list(common_only(rare)) == list(glass)


@needs_sudachi
def test_lexicon_offers_the_common_and_proper_readings_of_miyako():
    lexicon = ReadingLexicon(DICT_PATH, AccentDictionary())
    (candidate,) = lexicon.candidates("それから都の大路をぶらぶら歩いた。")
    assert (candidate.surface, candidate.pos, candidate.attached_moras) == ("都", "名詞", 1)
    assert candidate.options["ミヤコ(都)"].accents == (0,)
    assert candidate.options["ミヤコ(人名・地名)"].accents == (1,)


@needs_sudachi
def test_lexicon_offers_kana_homographs_by_lemma():
    (candidate,) = ReadingLexicon(DICT_PATH, AccentDictionary()).candidates("はしをわたった")
    assert {key: o.accents for key, o in candidate.options.items()} == {
        "ハシ(箸)": (1,),
        "ハシ(端)": (0,),
        "ハシ(橋)": (2,),
    }


@needs_sudachi
def test_lexicon_leaves_numerals_counters_and_compound_only_forms_to_voicevox():
    lexicon = ReadingLexicon(DICT_PATH, AccentDictionary())
    assert lexicon.candidates("午後三時から第二会議室で") == []
    assert lexicon.candidates("近くにいた人に傘を渡した") == []


@needs_sudachi
def test_lexicon_keeps_verbs_with_several_readings_and_drops_single_readings():
    lexicon = ReadingLexicon(DICT_PATH, AccentDictionary())
    assert lexicon.candidates("ぶらぶら歩いた") == []
    candidates = lexicon.candidates("雨が降る")
    assert [(c.surface, sorted(o.reading for o in c.options.values())) for c in candidates] == [
        ("雨", ["アマ", "アメ"]),
        ("降る", ["クダル", "フル"]),
    ]


JMDICT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE JMdict [
<!ENTITY n "noun (common) (futsuumeishi)">
<!ENTITY uk "word usually written using kana alone">
<!ENTITY dated "dated term">
]>
<JMdict>
<entry><k_ele><keb>人気</keb></k_ele><r_ele><reb>にんき</reb></r_ele>
<sense><pos>&n;</pos><gloss>ninki gloss 1</gloss><gloss>ninki gloss 2</gloss></sense>
<sense><gloss>ninki later gloss</gloss></sense></entry>
<entry><k_ele><keb>人気</keb></k_ele><k_ele><keb>人け</keb></k_ele>
<r_ele><reb>ひとけ</reb></r_ele><r_ele><reb>ひとげ</reb><re_restr>人気</re_restr></r_ele>
<sense><gloss>hitoke gloss</gloss></sense><sense><misc>&dated;</misc><gloss>hitoke later gloss</gloss></sense></entry>
<entry><k_ele><keb>父</keb></k_ele><r_ele><reb>てて</reb></r_ele>
<sense><misc>&uk;</misc><misc>&dated;</misc><gloss>tete gloss</gloss></sense></entry>
<entry><k_ele><keb>最中</keb></k_ele><r_ele><reb>もなか</reb></r_ele>
<sense><stagr>さなか</stagr><gloss>sanaka gloss</gloss></sense><sense><gloss>monaka gloss</gloss></sense></entry>
<entry><k_ele><keb>橋</keb></k_ele><r_ele><reb>はし</reb></r_ele><sense><gloss>hashi gloss</gloss></sense></entry>
<entry><k_ele><keb>降る</keb></k_ele><r_ele><reb>ふる</reb></r_ele><sense><gloss>furu gloss</gloss></sense></entry>
<entry><k_ele><keb>降り</keb></k_ele><r_ele><reb>ふり</reb></r_ele><sense><gloss>furi gloss</gloss></sense></entry>
<entry><k_ele><keb>硝子</keb></k_ele><r_ele><reb>ガラス</reb><re_pri>ichi1</re_pri></r_ele>
<sense><gloss>garasu gloss 1</gloss><gloss>garasu gloss 2</gloss></sense></entry>
</JMdict>
"""


@pytest.fixture
def jmdict() -> Iterator[Path]:
    path = Path(".build/tests/JMdict_e.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(JMDICT_XML.encode()))
    yield path
    path.unlink()


def test_load_jmdict_keeps_every_fitting_sense_and_register_tags(jmdict: Path):
    index = load_jmdict(jmdict)
    assert index[("人気", "にんき")] == [Gloss("ninki gloss 1; ninki gloss 2", (), later=("ninki later gloss",))]
    assert index[("人け", "ひとけ")] == [Gloss("hitoke gloss", (), later=("hitoke later gloss (dated term)",))]
    assert ("人気", "ひとげ") in index
    assert ("人け", "ひとげ") not in index
    assert index[("父", "てて")] == [Gloss("tete gloss", ("dated term",))]
    assert index[("最中", "もなか")] == [Gloss("monaka gloss", ())]
    assert index[("硝子", "がらす")] == [Gloss("garasu gloss 1; garasu gloss 2", (), common=True)]


def test_gloss_dictionary_looks_up_kanji_by_surface_and_kana_by_lemma(jmdict: Path):
    glosses = GlossDictionary(jmdict)
    kanji = glosses.gloss(
        {"ニンキ": option("ニンキ", (0,), "人気"), "ジンキ": option("ジンキ", (0,), "人気")},
        "人気",
        "名詞",
    )
    assert [o.meanings for o in kanji.values()] == [("ninki gloss 1; ninki gloss 2",), ()]
    assert [o.later_meanings for o in kanji.values()] == [("ninki later gloss",), ()]
    variant = glosses.gloss({"ヒトケ": option("ヒトケ", (0,), "人気")}, "人け", "名詞")["ヒトケ"]
    assert (variant.meanings, variant.later_meanings) == (("hitoke gloss",), ("hitoke later gloss (dated term)",))
    kana = glosses.gloss({"ハシ(橋)": option("ハシ", (2,), "橋")}, "はし", "名詞")
    assert kana["ハシ(橋)"].meanings == ("hashi gloss",)
    name = ReadingOption("ハシ", (1,), ("ハシ",), ("名詞-固有名詞-人名-姓",), ("人名",))
    assert glosses.gloss({"ハシ(人名)": name}, "橋", "名詞")["ハシ(人名)"].meanings == ()
    father = glosses.gloss({"テテ": option("テテ", (1,), "父")}, "父", "名詞")
    assert (father["テテ"].meanings, father["テテ"].usage) == (("tete gloss",), ("dated term",))
    glass = glosses.gloss({"ガラス": option("ガラス", (0,), "硝子")}, "硝子", "名詞")
    assert (glass["ガラス"].meanings, glass["ガラス"].common) == (("garasu gloss 1; garasu gloss 2",), True)


def test_gloss_dictionary_looks_up_inflected_words_by_lemma_only(jmdict: Path):
    fall = ReadingOption("フリ", (), ("降る",), ("動詞-一般",), ("降る",), lemma_readings=(("降る", "フル"),))
    glossed = GlossDictionary(jmdict).gloss({"フリ": fall}, "降り", "動詞")
    assert glossed["フリ"].meanings == ("furu gloss",)


def test_build_reading_questions_switch_to_english_once_meanings_are_in():
    text = "人気のない夜道"
    options = {
        "ヒトケ": replace(
            option("ヒトケ", (0,), "人気"),
            meanings=("sign of life",),
            later_meanings=("presence of people (dated term)",),
        ),
        "テテ": replace(option("テテ", (1,), "人気"), meanings=(), usage=("dated term",)),
    }
    questions = build_reading_questions(text, [ReadingCandidate(0, 2, "人気", "名詞", options)])
    question = questions["r0_reading"]
    assert question["instructions"]["question"] == "In the context of `sentence`, which reading of `word` is correct?"
    assert "hiragana" in question["instructions"]["note"]
    assert "`other_meaning`" in question["instructions"]["note"]
    assert question["criteria"]["ヒトケ"]["meaning"] == ["sign of life"]
    assert question["criteria"]["ヒトケ"]["other_meaning"] == ["presence of people (dated term)"]
    assert "other_meaning" not in question["criteria"]["テテ"]
    assert "usage" not in question["criteria"]["ヒトケ"]
    assert question["criteria"]["テテ"]["usage"] == ["dated term"]
    assert question["criteria"]["none"] == "none of the candidates fits"


def test_build_reading_questions_leave_out_meaning_without_a_gloss_dictionary():
    question = build_reading_questions("それから都の", [miyako()])["r6_reading"]
    assert "meaning" not in question["criteria"]["ト"]
    assert question["criteria"]["none"] == "どの候補も当てはまらない"


def spoken_query(reading: str, accent: int = 1) -> AudioQuery:
    moras = [Mora(text=ch, vowel="a", vowel_length=0.1, pitch=5.5) for ch in reading]
    return AudioQuery(
        accent_phrases=[AccentPhrase(moras=moras, accent=accent)] if moras else [],
        speedScale=1.0,
        pitchScale=0.0,
        intonationScale=1.0,
        volumeScale=1.0,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        outputSamplingRate=24000,
        outputStereo=False,
    )


def spoken_reader(readings: dict[str, str]):
    def read(text: str) -> AudioQuery:
        return spoken_query(readings[text])

    return read


def test_voicevox_keys_find_the_option_voicevox_reads_as_it_sounds():
    options = {"ハイロウ": option("ハイロウ", (), "入る"), "イロウ": option("イロウ", (), "入る")}
    candidate = ReadingCandidate(0, 3, "入ろう", "動詞", options)
    read = spoken_reader({"入ろうじゃないか": "ハイロオジャナイカ", "": ""})
    assert voicevox_keys(candidate, "入ろうじゃないか", read("入ろうじゃないか"), read) == {"ハイロウ"}


def test_voicevox_keys_prefer_the_longest_reading_that_fits():
    candidate = ReadingCandidate(0, 1, "入", "動詞", {"イ": option("イ", ()), "イリ": option("イリ", ())})
    read = spoken_reader({"入り口": "イリグチ", "": ""})
    assert voicevox_keys(candidate, "入り口", read("入り口"), read) == {"イリ"}


def toki() -> ReadingCandidate:
    options = {
        "トキ(鴇)": replace(option("トキ", (1,), "鴇"), meanings=("crested ibis",)),
        "トキ(時)": replace(option("トキ", (2,), "時"), meanings=("time",), common=True),
    }
    return ReadingCandidate(0, 2, "とき", "名詞", options, 1)


@pytest.mark.parametrize(("accent", "expected"), [(1, {"トキ(鴇)"}), (2, {"トキ(時)"}), (3, set())])
def test_voicevox_keys_tell_the_lemmas_of_a_kana_word_apart_by_accent(accent: int, expected: set[str]):
    read = spoken_reader({"": ""})
    assert voicevox_keys(toki(), "ときは", spoken_query("トキワ", accent), read) == expected


def test_voicevox_keys_keep_every_lemma_whose_reading_fits_when_the_word_starts_no_phrase():
    read = spoken_reader({"その": "ソノ"})
    candidate = replace(toki(), start=2, end=4)
    assert voicevox_keys(candidate, "そのときは", spoken_query("ソノトキワ"), read) == {"トキ(鴇)", "トキ(時)"}


def test_narrow_asks_about_a_kana_word_voicevox_accents_as_a_rare_lemma():
    narrowed = narrow(toki(), frozenset({"トキ(鴇)"}))
    assert narrowed is not None
    assert list(narrowed.options) == ["トキ(鴇)", "トキ(時)"]
    assert narrow(toki(), frozenset({"トキ(時)"})) is None


def belly() -> ReadingCandidate:
    options = {
        "ハラ": replace(option("ハラ", (2,), "腹"), meanings=("belly",), common=True),
        "フク": replace(option("フク", (2,), "腹"), meanings=("abdomen",)),
        "オナカ": replace(option("オナカ", (0,), "お腹"), meanings=("stomach",), common=True),
    }
    return ReadingCandidate(0, 1, "腹", "名詞", options)


def test_narrow_drops_rare_readings_when_voicevox_reads_a_common_one():
    narrowed = narrow(belly(), frozenset({"ハラ"}))
    assert narrowed is not None
    assert list(narrowed.options) == ["ハラ", "オナカ"]


def test_narrow_keeps_every_reading_when_voicevox_reads_a_rare_one_or_none_of_them():
    assert list(narrow(belly(), frozenset({"フク"})).options) == ["ハラ", "フク", "オナカ"]
    assert list(narrow(belly(), frozenset()).options) == ["ハラ", "フク", "オナカ"]


def test_narrow_counts_a_kana_word_read_as_a_common_lemma():
    options = {
        "トコロ(野老)": replace(option("トコロ", (0,), "野老"), meanings=("yam",)),
        "トコロ(所)": replace(option("トコロ", (0,), "所"), meanings=("place",), common=True),
        "トコ(床)": replace(option("トコ", (0,), "床"), meanings=("bed",), common=True),
    }
    kana = ReadingCandidate(0, 3, "ところ", "名詞", options)
    narrowed = narrow(kana, frozenset({"トコロ(野老)", "トコロ(所)"}))
    assert narrowed is not None
    assert list(narrowed.options) == ["トコロ(所)", "トコ(床)"]


def test_narrow_drops_the_question_when_one_reading_is_left():
    two = replace(belly(), options={key: belly().options[key] for key in ("ハラ", "フク")})
    assert narrow(two, frozenset({"ハラ"})) is None


def test_narrow_keeps_the_dropped_readings_for_a_second_question():
    narrowed = narrow(belly(), frozenset({"ハラ"}))
    assert narrowed is not None
    assert narrowed.all_options is not None
    assert list(narrowed.all_options) == ["ハラ", "フク", "オナカ"]
    assert narrow(belly(), frozenset({"フク"})).all_options is None


def test_build_reading_questions_offer_every_reading_beside_the_common_ones():
    questions = build_reading_questions("腹が減った", [narrow(belly(), frozenset({"ハラ"}))])
    assert list(questions) == ["r0_reading", "r0_reading_all"]
    assert list(questions["r0_reading"]["criteria"]) == ["ハラ", "オナカ", "none"]
    assert list(questions["r0_reading_all"]["criteria"]) == ["ハラ", "フク", "オナカ", "none"]


@pytest.mark.parametrize(
    ("common", "every", "expected"),
    [
        (choice("none"), choice("フク"), ["フク"]),
        (choice("none", 0.3), choice("フク"), ["フク"]),
        (choice("ハラ"), choice("フク"), ["ハラ"]),
        (choice("none"), choice("オナカ"), []),
        (choice("none"), choice("フク", 0.3), []),
        (choice("none"), None, []),
    ],
)
def test_choose_options_takes_a_dropped_reading_only_after_none_of_the_common_ones_fits(
    common: ChoiceAnswer, every: ChoiceAnswer | None, expected: list[str]
):
    answers = {"r0_reading": common} | ({} if every is None else {"r0_reading_all": every})
    chosen = choose_options([narrow(belly(), frozenset({"ハラ"}))], answers, 0.6)
    assert [picked.key for picked in chosen] == expected


def test_with_voicevox_reading_offers_what_voicevox_says_beside_the_dictionary():
    text = "こんなことを云いながら"
    read = spoken_reader(
        {text: "コンナコトオユイナガラ", "こんなことを": "コンナコトオ", "こんなことを云い": "コンナコトオユイ"}
    )
    say = ReadingCandidate(6, 8, "云い", "動詞", {"イイ": option("イイ", (), "云う")})
    widened = with_voicevox_reading(say, text, read(text), read)
    assert widened is not None
    assert list(widened.options) == ["イイ", "ユイ"]
    assert widened.options["ユイ"].kinds == ("VOICEVOX",)
    assert widened.options["ユイ"].meanings == ()


def test_with_voicevox_reading_leaves_kana_words_alone():
    text = "あめが"
    read = spoken_reader({text: "アメガ", "": "", "あめ": "アメ"})
    kana = ReadingCandidate(0, 2, "あめ", "名詞", {"アメ(飴)": option("アメ", (0,), "飴")})
    assert with_voicevox_reading(kana, text, read(text), read) == kana


def test_with_voicevox_reading_leaves_the_voiced_start_of_a_compound_to_voicevox():
    text = "西洋造りの家"
    read = spoken_reader({text: "セイヨオズクリノイエ", "西洋": "セイヨオ", "西洋造り": "セイヨオズクリ"})
    build = ReadingCandidate(2, 4, "造り", "名詞", {"ツクリ": option("ツクリ", (0,), "造り")})
    assert with_voicevox_reading(build, text, read(text), read) is None
    after_particle = "親分の書きよう"
    read = spoken_reader({after_particle: "オヤブンノガキヨオ", "親分の": "オヤブンノ", "親分の書き": "オヤブンノガキ"})
    write = ReadingCandidate(3, 5, "書き", "動詞", {"カキ": option("カキ", (), "書く")})
    widened = with_voicevox_reading(write, after_particle, read(after_particle), read)
    assert widened is not None
    assert list(widened.options) == ["カキ", "ガキ"]


class CompoundLexicon:
    def candidates(self, text: str, *, minimum: int = 2) -> list[ReadingCandidate]:
        mouth = {"クチ": option("クチ", (0,), "口"), "ク": option("ク", (1,), "口")}
        return [ReadingCandidate(2, 3, "口", "名詞", mouth)]

    def words(self, text: str) -> list[Word]:
        return []


def test_plan_text_asks_nothing_about_a_word_voicevox_voices_inside_a_compound():
    text = "裏戸口から"
    read = spoken_reader({text: "ウラトグチカラ", "裏戸": "ウラト", "裏戸口": "ウラトグチ"})
    assert plan_text(text, CompoundLexicon(), read(text), read).candidates == []
