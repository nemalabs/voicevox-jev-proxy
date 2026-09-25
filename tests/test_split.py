import pytest

from voicevox_jev_proxy.accents import Word
from voicevox_jev_proxy.edits import Edit, Rephrase, apply_edits, effective_edits, reading_of, shifted, spoken
from voicevox_jev_proxy.prosody import Change
from voicevox_jev_proxy.split import (
    Span,
    Stretch,
    build_boundary_questions,
    build_split_questions,
    choose_boundaries,
    choose_splits,
    find_spans,
    find_stretches,
    join_moved_boundaries,
    strip_inserted_pauses,
)
from voicevox_jev_proxy.typesafe import ChoiceAnswer
from voicevox_jev_proxy.voicevox import AccentPhrase, AudioQuery, Mora


def choice(value: str, confidence: float = 0.9) -> ChoiceAnswer:
    return ChoiceAnswer(type="choice", choice=value, confidence=confidence, probabilities={value: confidence})


def phrase(reading: str, *, pause: bool = False) -> AccentPhrase:
    moras = [Mora(text=ch, vowel="a", vowel_length=0.1, pitch=5.5) for ch in reading]
    pause_mora = Mora(text="、", vowel="pau", vowel_length=0.3, pitch=0.0) if pause else None
    return AccentPhrase(moras=moras, accent=1, pause_mora=pause_mora)


def query(*phrases: AccentPhrase) -> AudioQuery:
    return AudioQuery(
        accent_phrases=list(phrases),
        speedScale=1.0,
        pitchScale=0.0,
        intonationScale=1.0,
        volumeScale=1.0,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        outputSamplingRate=24000,
        outputStereo=False,
    )


TEXT = "野を越え山越え、十里はなれた此のシラクスの市にやって来た。"


def test_find_spans_requires_two_kana_after_kanji():
    spans = find_spans(TEXT)
    assert [TEXT[s.start : s.end] for s in spans] == ["十里はなれた", "市にやって"]
    assert find_spans("村を出た") == []
    assert find_spans("十里はなれた") == [Span(0, 6)]


def test_build_split_questions_lists_every_boundary_and_none():
    questions = build_split_questions("十里はなれた", [Span(0, 6)])
    criteria = questions["s0_split"]["criteria"]
    assert criteria["whole"] == "十里はなれた"
    assert criteria["split_2"] == "十里 | はなれた"
    assert list(criteria) == ["whole", "split_1", "split_2", "split_3", "split_4", "split_5", "none"]


def test_choose_splits_and_insert_separators_track_positions():
    spans = find_spans(TEXT)
    answers = {"s8_split": choice("split_2"), "s21_split": choice("split_2")}
    edits = choose_splits(TEXT, spans, answers, 0.6)
    assert [(e.start, e.end, e.replacement, e.change.phrase_id, e.change.before, e.change.after) for e in edits] == [
        (10, 10, "、", "s8", "十里はなれた", "十里、はなれた"),
        (23, 23, "、", "s21", "市にやって", "市に、やって"),
    ]
    text, inserted = apply_edits(TEXT, edits)
    assert text == "野を越え山越え、十里、はなれた此のシラクスの市に、やって来た。"
    assert inserted == [10, 24]
    assert [text[i] for i in inserted] == ["、", "、"]


@pytest.mark.parametrize("value", ["whole", "none"])
def test_choose_splits_ignores_whole_none_and_low_confidence(value: str):
    spans = find_spans(TEXT)
    answers = {"s8_split": choice(value), "s21_split": choice("split_2", confidence=0.3)}
    assert choose_splits(TEXT, spans, answers, 0.6) == []


@pytest.mark.parametrize("value", ["split_0", "split_-1", "split_6", "split_99", "split_x", "cut_2"])
def test_choose_splits_ignores_offsets_that_were_not_offered(value: str):
    text = "十里はなれた"
    assert choose_splits(text, find_spans(text), {"s0_split": choice(value)}, 0.6) == []


def test_effective_splits_keeps_only_reading_changes():
    text = "十里はなれた市にやって"
    edits = choose_splits(text, find_spans(text), {"s0_split": choice("split_2"), "s6_split": choice("split_2")}, 0.6)
    original = query(phrase("ジュウリワナレタシニヤッテ"))

    def read(candidate: str) -> AudioQuery:
        if candidate.startswith("十里、"):
            return query(phrase("ジュウリ", pause=True), phrase("ハナレタシニヤッテ"))
        return query(phrase("ジュウリワナレタシニ", pause=True), phrase("ヤッテ"))

    kept, skipped = effective_edits(text, edits, original, read)
    assert [e.start for e in kept] == [2]
    assert [(c.phrase_id, c.field, c.reason) for c in skipped] == [("s6", "split_skipped", "reading unchanged")]
    assert reading_of(original) == "ジュウリワナレタシニヤッテ"


def reader(mora_counts: dict[str, int]):
    def read(text: str) -> AudioQuery:
        return query(phrase("ア" * mora_counts[text]))

    return read


def test_strip_inserted_pauses_removes_only_inserted_ones():
    text = "あ！！野を越え山越え、十里、はなれた"
    q = query(
        phrase("ア", pause=True),
        phrase("ノオコエ"),
        phrase("ヤマゴエ", pause=True),
        phrase("ジュウリ", pause=True),
        phrase("ハナレタ"),
    )
    updated, unmatched = strip_inserted_pauses(q, text, [13], reader({"あ！！野を越え山越え、十里": 13}))
    assert [p.pause_mora is not None for p in updated.accent_phrases] == [True, False, True, False, False]
    assert [p.pause_mora is not None for p in q.accent_phrases] == [True, False, True, True, False]
    assert unmatched == []


def test_strip_inserted_pauses_reports_separators_it_cannot_place():
    q = query(phrase("ジュウリ"), phrase("ハナレタ"))
    updated, unmatched = strip_inserted_pauses(q, "十里、はなれた", [2], reader({"十里": 3}))
    assert [p.pause_mora for p in updated.accent_phrases] == [None, None]
    assert unmatched == [2]


def reading_edit(start: int, end: int, replacement: str, reading: str) -> Edit:
    return Edit(start, end, replacement, Change(f"r{start}", "reading", "", replacement, "test"), reading)


def reading_reader(readings: dict[str, str]):
    def read(text: str) -> AudioQuery:
        return query(phrase(readings[text]))

    return read


def test_spoken_spells_long_vowels_as_they_sound():
    assert [spoken(r) for r in ("トウ", "センセイ", "ラーメン", "キョウ", "ハナヂ", "ハラ")] == [
        "トオ",
        "センセエ",
        "ラアメン",
        "キョオ",
        "ハナジ",
        "ハラ",
    ]


def test_effective_edits_skip_a_word_voicevox_already_reads_the_chosen_way():
    text = "どうも腹が空いた"
    read = reading_reader(
        {
            text: "ドオモハラガアイタ",
            "どうもはらが空いた": "ドオモワラガアイタ",
            "どうもハラが空いた": "ドオモハラガアイタ",
        }
    )
    kept, skipped = effective_edits(text, [reading_edit(3, 4, "はら", "ハラ")], read(text), read)
    assert kept == []
    assert [(c.field, c.reason) for c in skipped] == [("reading_skipped", "reading unchanged")]


def test_effective_edits_retry_in_katakana_when_the_kana_fuse_with_their_neighbours():
    text = "なったし腹は"
    read = reading_reader(
        {text: "ナッタシフクワ", "なったしはらは": "ナッタシワラワ", "なったしハラは": "ナッタシハラワ"}
    )
    kept, skipped = effective_edits(text, [reading_edit(4, 5, "はら", "ハラ")], read(text), read)
    assert [(e.replacement, e.change.after) for e in kept] == [("ハラ", "ハラ")]
    assert skipped == []


def test_effective_edits_compare_readings_as_they_sound():
    read = reading_reader({"硝子の": "ガラスノ", "しょうしの": "ショオシノ", "当軒": "トオノキ", "とう軒": "トウノキ"})
    kept, _ = effective_edits("硝子の", [reading_edit(0, 2, "しょうし", "ショウシ")], read("硝子の"), read)
    assert [e.replacement for e in kept] == ["しょうし"]
    kept, skipped = effective_edits("当軒", [reading_edit(0, 1, "とう", "トウ")], read("当軒"), read)
    assert kept == []
    assert [c.reason for c in skipped] == ["reading unchanged"]


def test_effective_edits_skip_a_replacement_voicevox_misreads_either_way():
    text = "どうも腹が"
    read = reading_reader({text: "ドオモフクガ", "どうもはらが": "ドオモワラガ", "どうもハラが": "ドオモワラガ"})
    kept, skipped = effective_edits(text, [reading_edit(3, 4, "はら", "ハラ")], read(text), read)
    assert kept == []
    assert [c.reason for c in skipped] == ["VOICEVOX does not read the replacement as ハラ in its own phrasing"]


def test_effective_edits_do_not_take_a_word_boundary_for_a_long_vowel():
    text = "なかへ入り"
    read = reading_reader({text: "ナカエハイリ", "なかへいり": "ナカエイリ"})
    kept, skipped = effective_edits(text, [reading_edit(3, 5, "いり", "イリ")], read(text), read)
    assert [e.replacement for e in kept] == ["いり"]
    assert skipped == []


def phrased(*readings: str) -> AudioQuery:
    return query(*(phrase(reading) for reading in readings))


def test_effective_edits_retry_in_katakana_when_the_kana_join_the_phrase_before():
    text = "村の茂平と"
    readings = {
        text: phrased("ムラノ", "モビラト"),
        "村のもへいと": phrased("ムラノモ", "ヘイト"),
        "村のモヘイと": phrased("ムラノ", "モヘイト"),
    }

    def read(candidate: str) -> AudioQuery:
        return readings[candidate]

    kept, skipped = effective_edits(text, [reading_edit(2, 4, "もへい", "モヘイ")], read(text), read)
    assert [e.replacement for e in kept] == ["モヘイ"]
    assert skipped == []


def test_effective_edits_skip_a_replacement_that_breaks_the_phrasing_either_way():
    text = "村の茂平と"
    readings = {
        text: phrased("ムラノ", "モビラト"),
        "村のもへいと": phrased("ムラノモ", "ヘイト"),
        "村のモヘイと": phrased("ムラノモ", "ヘイト"),
    }

    def read(candidate: str) -> AudioQuery:
        return readings[candidate]

    kept, skipped = effective_edits(text, [reading_edit(2, 4, "もへい", "モヘイ")], read(text), read)
    assert kept == []
    assert [c.reason for c in skipped] == ["VOICEVOX does not read the replacement as モヘイ in its own phrasing"]


def test_effective_edits_accept_a_word_that_starts_a_phrase_of_its_own():
    text = "その間、"
    readings = {text: phrased("ソノカン"), "そのあいだ、": phrased("ソノ", "アイダ")}

    def read(candidate: str) -> AudioQuery:
        return readings[candidate]

    kept, skipped = effective_edits(text, [reading_edit(2, 3, "あいだ", "アイダ")], read(text), read)
    assert [e.replacement for e in kept] == ["あいだ"]
    assert skipped == []


def word(start: int, surface: str, *pos: str) -> Word:
    return Word(start, surface, (*pos, "*"))


def counter(mora_counts: dict[str, int]):
    def read(text: str) -> AudioQuery:
        return query(phrase("ア" * mora_counts[text])) if mora_counts[text] else query()

    return read


HI_MO = [word(0, "日", "名詞"), word(1, "も", "助詞"), word(2, "ごん", "名詞"), word(4, "は", "助詞")]
HI_MO_COUNTS = {"": 0, "日": 1, "日も": 2, "日もご": 3, "日もごん": 4}
SONO_TOKI = [
    word(0, "その", "連体詞"),
    word(2, "とき", "名詞", "普通名詞"),
    word(4, "兵十", "名詞", "固有名詞"),
    word(6, "は", "助詞"),
]
SONO_TOKI_COUNTS = {"": 0, "その": 2, "そのとき": 4, "そのとき兵十": 8}


def test_find_stretches_finds_a_phrase_that_starts_at_a_particle_and_runs_into_the_next_word():
    q = phrased("ヒ", "モゴンワ")
    assert find_stretches("日もごんは", q, HI_MO, counter(HI_MO_COUNTS)) == [Stretch(0, 5, 1, (2,))]


def test_find_stretches_finds_a_phrase_that_runs_one_noun_into_the_next():
    q = phrased("ソノ", "トキヘエジュウワ")
    assert find_stretches("そのとき兵十は", q, SONO_TOKI, counter(SONO_TOKI_COUNTS)) == [Stretch(2, 7, None, (2,))]


@pytest.mark.parametrize(
    ("text", "words", "counts", "readings"),
    [
        (
            "じれったくなって",
            [word(0, "じれったく", "形容詞"), word(5, "なっ", "動詞", "非自立可能"), word(7, "て", "助詞")],
            {"": 0, "じれったく": 5, "じれったくなっ": 7},
            ("ジレッタクナッテ",),
        ),
        (
            "日もお城みたいに",
            [
                *HI_MO[:2],
                word(2, "お", "接頭辞"),
                word(3, "城", "名詞"),
                word(4, "みたい", "形状詞", "助動詞語幹"),
                word(7, "に", "助動詞"),
            ],
            {"": 0, "日": 1, "日も": 2, "日もお": 3, "日もお城": 5, "日もお城みたい": 8},
            ("ヒモオ", "シロミタイニ"),
        ),
        ("日もごんは", HI_MO, HI_MO_COUNTS, ("ヒモ", "ゴンワ")),
        (
            "日の新衣装",
            [word(0, "日", "名詞"), word(1, "の", "助詞"), word(2, "新", "接頭辞"), word(3, "衣装", "名詞")],
            {"": 0, "日": 1, "日の": 2, "日の新": 4},
            ("ヒノ", "シンイショオ"),
        ),
        (
            "二千四百円の",
            [word(0, "二千四百", "名詞", "数詞"), word(4, "円", "名詞", "普通名詞"), word(5, "の", "助詞")],
            {"": 0, "二千四百": 7, "二千四百円": 9},
            ("ニセン", "ヨンヒャクエンノ"),
        ),
        (
            "書いてあるじゃないか",
            [
                word(0, "書い", "動詞", "一般"),
                word(2, "て", "助詞", "接続助詞"),
                word(3, "ある", "動詞", "非自立可能"),
                word(5, "じゃ", "助動詞"),
                word(7, "ない", "形容詞", "非自立可能"),
                word(9, "か", "助詞"),
            ],
            {"": 0, "書い": 2, "書いて": 3, "書いてある": 5, "書いてあるじゃ": 6, "書いてあるじゃない": 8},
            ("カイテ", "アルジャナイカ"),
        ),
        (
            "雨、もごんは",
            [
                word(0, "雨", "名詞"),
                word(1, "、", "補助記号"),
                word(2, "も", "助詞"),
                word(3, "ごん", "名詞"),
                word(5, "は", "助詞"),
            ],
            {"": 0, "雨": 2, "雨、": 2, "雨、も": 3, "雨、もごん": 5},
            ("アメ", "モゴンワ"),
        ),
    ],
)
def test_find_stretches_leaves_phrases_that_join_other_words_or_follow_the_words(
    text: str, words: list[Word], counts: dict[str, int], readings: tuple[str, ...]
):
    assert find_stretches(text, phrased(*readings), words, counter(counts)) == []


def test_build_boundary_questions_offers_voicevox_phrasing_first():
    stretches = [Stretch(0, 5, 1, (2,)), Stretch(6, 11, None, (2,))]
    questions = build_boundary_questions("日もごんは、とき兵十は", stretches)
    assert questions["b0_boundary"]["criteria"] == {
        "cut_1": "日 | もごんは",
        "cut_2": "日も | ごんは",
        "none": "どの区切り方も正しくない",
    }
    assert list(questions["b6_boundary"]["criteria"].items())[:2] == [
        ("whole", "とき兵十は"),
        ("cut_2", "とき | 兵十は"),
    ]


def test_choose_boundaries_inserts_a_phrasing_separator_at_the_chosen_cut():
    stretches = [Stretch(1, 6, 1, (2,)), Stretch(7, 12, None, (2,))]
    answers = {"b1_boundary": choice("cut_2", 0.8), "b7_boundary": choice("cut_2", 0.8)}
    edits = choose_boundaries("、日もごんは、とき兵十は", stretches, answers, 0.6)
    assert [(e.start, e.end, e.replacement, e.rephrase) for e in edits] == [
        (3, 3, "、", Rephrase(1, 6, 2)),
        (9, 9, "、", Rephrase(7, 12, None)),
    ]
    assert (edits[0].change.field, edits[0].change.before, edits[0].change.after) == (
        "boundary",
        "日 | もごんは",
        "日も | ごんは",
    )


@pytest.mark.parametrize(("value", "confidence"), [("cut_1", 0.9), ("whole", 0.9), ("none", 0.9), ("cut_2", 0.5)])
def test_choose_boundaries_keeps_voicevox_phrasing_otherwise(value: str, confidence: float):
    stretches = [Stretch(0, 5, 1, (2,))]
    assert choose_boundaries("日もごんは", stretches, {"b0_boundary": choice(value, confidence)}, 0.6) == []


HI_MO_AME = "日もごんは、雨"


def boundary_edit(position: int) -> Edit:
    change = Change(f"b{position}", "boundary", "", "", "test")
    return Edit(position, position, "、", change, rephrase=Rephrase(0, 5))


def boundary_reads(separated: tuple[str, ...], head: tuple[str, ...] = ("ヒモ",)):
    readings = {
        HI_MO_AME: phrased("ヒ", "モゴンワ", "アメ"),
        "日も、ごんは、雨": phrased(*separated),
        "": phrased(),
        "日もごんは": phrased("ヒモゴンワ"),
        "日も": phrased(*head),
    }
    return readings.__getitem__


@pytest.mark.parametrize("separated", [("ヒモ", "ゴンワ", "アメ"), ("ヒモ", "ゴンハ", "アメ")])
def test_effective_edits_keep_a_separator_that_moves_the_boundary_rereading_only_the_stretch(
    separated: tuple[str, ...],
):
    read = boundary_reads(separated)
    kept, skipped = effective_edits(HI_MO_AME, [boundary_edit(2)], read(HI_MO_AME), read)
    assert [e.start for e in kept] == [2]
    assert skipped == []


@pytest.mark.parametrize(
    ("separated", "head", "reason"),
    [
        (
            ("ヒモ", "ゴンワ", "アマ"),
            ("ヒモ",),
            "VOICEVOX reads the text outside the stretch differently with the separator",
        ),
        (
            ("ヒモ", "ゴンワ", "アメ"),
            ("ニチモ",),
            "VOICEVOX reads the text before the separator differently on its own",
        ),
        (("ヒ", "モゴンワ", "アメ"), ("ヒモ",), "VOICEVOX does not start a phrase at the separator"),
    ],
)
def test_effective_edits_skip_a_separator_that_does_not_move_the_boundary_alone(
    separated: tuple[str, ...], head: tuple[str, ...], reason: str
):
    read = boundary_reads(separated, head)
    kept, skipped = effective_edits(HI_MO_AME, [boundary_edit(2)], read(HI_MO_AME), read)
    assert kept == []
    assert [(c.field, c.reason) for c in skipped] == [("boundary_skipped", reason)]


def moved_edit(position: int, moves_from: int) -> Edit:
    change = Change("b3", "boundary", "", "", "test")
    return Edit(position, position, "、", change, rephrase=Rephrase(3, 12, moves_from))


SONOMAMA = "ごんはそのまま、横っとびに"
SONOMAMA_COUNTS = {"ごんはその": 5}


def test_join_moved_boundaries_joins_the_break_voicevox_kept():
    q = phrased("ゴンワ", "ソノ", "ママ", "ヨコットビニ")
    updated, changes = join_moved_boundaries(q, SONOMAMA, [moved_edit(7, 5)], counter(SONOMAMA_COUNTS))
    assert [p.reading for p in updated.accent_phrases] == ["ゴンワ", "ソノママ", "ヨコットビニ"]
    assert [(c.phrase_id, c.field, c.before, c.after) for c in changes] == [
        ("b3", "merge", "ソノ(1)+ママ(1)", "ソノママ(1)")
    ]
    assert len(q.accent_phrases) == 4


@pytest.mark.parametrize(
    ("readings", "pause"),
    [(("ゴンワ", "ソノママ", "ヨコットビニ"), False), (("ゴンワ", "ソノ", "ママ", "ヨコットビニ"), True)],
)
def test_join_moved_boundaries_leaves_a_break_voicevox_dropped_or_pauses_at(readings: tuple[str, ...], pause):
    q = phrased(*readings)
    if pause:
        q.accent_phrases[1].pause_mora = phrase("", pause=True).pause_mora
    updated, changes = join_moved_boundaries(q, SONOMAMA, [moved_edit(7, 5)], counter(SONOMAMA_COUNTS))
    assert [p.reading for p in updated.accent_phrases] == list(readings)
    assert changes == []


def test_shifted_follows_edits_before_the_position_and_rejects_one_around_it():
    edits = [
        Edit(1, 1, "、", Change("s1", "split", "", "", "")),
        Edit(3, 5, "みやこ", Change("r3", "reading", "", "", "")),
    ]
    assert shifted(2, edits) == 3
    assert shifted(5, edits) == 7
    assert shifted(4, edits) is None
