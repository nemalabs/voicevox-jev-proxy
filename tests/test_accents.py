from dataclasses import replace

import pytest

from voicevox_jev_proxy.accents import AccentTarget, Word, apply_accents, attachable_phrases, moved, phrase_accent
from voicevox_jev_proxy.edits import Edit
from voicevox_jev_proxy.prosody import Change
from voicevox_jev_proxy.voicevox import AccentPhrase, AudioQuery, Mora

PAUSE = Mora(text="、", vowel="pau", vowel_length=0.3, pitch=0.0)


def phrase(texts: list[str], accent: int) -> AccentPhrase:
    return AccentPhrase(moras=[Mora(text=t, vowel="a", vowel_length=0.1, pitch=5.5) for t in texts], accent=accent)


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


def reader(mora_counts: dict[str, int]):
    def read(text: str) -> AudioQuery:
        count = mora_counts[text]
        return query(phrase(["ア"] * count, count)) if count else query()

    return read


def target(start: int, end: int, accent: int | tuple[int, ...], moras: int, attached: int = 1) -> AccentTarget:
    accents = accent if isinstance(accent, tuple) else (accent,)
    return AccentTarget("r0", start, end, accents, moras, attached, "test")


def edit(start: int, end: int, replacement: str) -> Edit:
    return Edit(start, end, replacement, Change("e", "x", "", replacement, ""))


def test_moved_follows_earlier_edits_and_the_word_replacement():
    word = target(6, 7, 0, 3)
    placed = moved(word, [edit(3, 3, "、"), edit(6, 7, "みやこ"), edit(9, 9, "、")])
    assert placed is not None
    assert (placed.start, placed.end) == (7, 10)


def test_moved_rejects_an_edit_inside_the_word():
    assert moved(target(0, 2, 0, 3), [edit(1, 1, "、")]) is None


def test_moved_leaves_the_word_when_an_insertion_follows_it():
    placed = moved(target(0, 2, 0, 3), [edit(2, 2, "、")])
    assert placed is not None
    assert (placed.start, placed.end) == (0, 2)


@pytest.mark.parametrize(
    ("texts", "voicevox_accent", "word_moras", "word_accents", "expected"),
    [
        (["ミ", "ヤ", "コ", "ノ"], 1, 3, (0,), 4),
        (["ア", "メ", "ガ"], 3, 2, (1,), 1),
        (["ハ", "シ", "ガ"], 1, 2, (2,), 2),
        (["ミ", "ヤ", "コ", "デ", "ス"], 1, 3, (0,), 5),
        (["ミ", "ヤ", "コ", "デ", "ス"], 4, 3, (0,), 4),
        (["チ", "チ", "ノ"], 1, 2, (2, 1), 1),
        (["チ", "チ", "ノ"], 3, 2, (2, 1), 2),
        (["シ", "キ", "シ", "オ"], 4, 3, (0, 2), 4),
        (["シ", "キ", "シ", "オ"], 1, 3, (0, 2), 4),
    ],
)
def test_phrase_accent_keeps_voicevox_when_any_dictionary_type_agrees(
    texts: list[str], voicevox_accent: int, word_moras: int, word_accents: tuple[int, ...], expected: int
):
    assert phrase_accent(phrase(texts, voicevox_accent), word_moras, word_accents) == expected


def test_apply_accents_flattens_the_phrase_the_word_starts():
    text = "それからみやこの大路を"
    q = query(
        phrase(["ソ", "レ", "カ", "ラ"], 4), phrase(["ミ", "ヤ", "コ", "ノ"], 1), phrase(["オ", "オ", "ジ", "オ"], 1)
    )
    read = reader({"それから": 4, "それからみやこ": 7})
    updated, changes, handled = apply_accents(q, text, [target(4, 7, 0, 3)], read)
    assert [p.accent for p in updated.accent_phrases] == [4, 4, 1]
    assert q.accent_phrases[1].accent == 1
    assert [(c.field, c.before, c.after) for c in changes] == [("accent", "ミヤコノ(1)", "ミヤコノ(4)")]
    assert handled == frozenset({1})


def test_apply_accents_reports_a_matching_phrase_as_handled_without_change():
    q = query(phrase(["ア", "メ", "ガ"], 1))
    updated, changes, handled = apply_accents(q, "あめが", [target(0, 2, 1, 2)], reader({"": 0, "あめ": 2}))
    assert updated.accent_phrases[0].accent == 1
    assert changes == []
    assert handled == frozenset({0})


def test_apply_accents_joins_a_word_voicevox_split_across_phrases():
    q = query(phrase(["ナ", "マ"], 1), phrase(["モ", "ノ", "ワ"], 2), phrase(["ハ", "ヤ", "メ", "ニ"], 3))
    updated, changes, handled = apply_accents(q, "生物は早めに", [target(0, 2, 2, 4)], reader({"": 0, "生物": 4}))
    assert [(p.reading, p.accent) for p in updated.accent_phrases] == [("ナマモノワ", 2), ("ハヤメニ", 3)]
    assert [(c.field, c.before, c.after) for c in changes] == [("accent", "ナマ(1)+モノワ(2)", "ナマモノワ(2)")]
    assert handled == frozenset({0})
    assert len(q.accent_phrases) == 3


def test_apply_accents_places_later_words_after_a_join():
    q = query(phrase(["ナ", "マ"], 1), phrase(["モ", "ノ", "ワ"], 2), phrase(["ハ", "シ", "ガ"], 1))
    read = reader({"": 0, "AB": 4, "ABC": 5, "ABCDE": 7})
    words = [target(3, 5, 2, 2), target(0, 2, 2, 4)]
    updated, changes, handled = apply_accents(q, "ABCDEF", words, read)
    assert [(p.reading, p.accent) for p in updated.accent_phrases] == [("ナマモノワ", 2), ("ハシガ", 2)]
    assert [c.after for c in changes] == ["ナマモノワ(2)", "ハシガ(2)"]
    assert handled == frozenset({0, 1})


def test_apply_accents_leaves_a_word_voicevox_pauses_inside():
    left = phrase(["ナ", "マ"], 1)
    left.pause_mora = PAUSE
    q = query(left, phrase(["モ", "ノ", "ワ"], 2))
    updated, changes, handled = apply_accents(q, "生物は", [target(0, 2, 2, 4)], reader({"": 0, "生物": 4}))
    assert len(updated.accent_phrases) == 2
    assert [(c.field, c.reason) for c in changes] == [("accent_skipped", "VOICEVOX pauses inside the word")]
    assert handled == frozenset()


def sono_toki() -> AudioQuery:
    long = phrase(["ト", "キ", "ヘ", "エ", "ジュ", "ウ", "ワ"], 4)
    long.pause_mora = PAUSE
    return query(phrase(["ソ", "ノ"], 2), long, phrase(["フ", "ト"], 2))


def test_apply_accents_cuts_a_phrase_that_runs_on_into_the_next_word():
    q = sono_toki()
    read = reader({"その": 2, "そのとき": 4})
    toki = replace(target(2, 4, 2, 2, attached=0), cut=True)
    updated, changes, handled = apply_accents(q, "そのとき兵十は", [toki], read)
    assert [(p.reading, p.accent) for p in updated.accent_phrases] == [
        ("ソノ", 2),
        ("トキ", 2),
        ("ヘエジュウワ", 2),
        ("フト", 2),
    ]
    assert [p.pause_mora is not None for p in updated.accent_phrases] == [False, False, True, False]
    assert [(c.field, c.before, c.after) for c in changes] == [
        ("accent", "トキヘエジュウワ(4)", "トキ(2)+ヘエジュウワ(2)")
    ]
    assert handled == frozenset({1})
    assert len(q.accent_phrases) == 3


@pytest.mark.parametrize(
    ("cut", "nucleus", "reason"),
    [
        (False, 4, "the accent phrase goes on past the particles after the word"),
        (True, 1, "the accent phrase goes on past the particles after the word, with its nucleus before them"),
    ],
)
def test_apply_accents_leaves_a_run_on_phrase_it_may_not_cut(cut, nucleus: int, reason: str):
    q = sono_toki()
    q.accent_phrases[1].accent = nucleus
    read = reader({"その": 2, "そのとき": 4})
    toki = replace(target(2, 4, 2, 2, attached=0), cut=cut)
    updated, changes, handled = apply_accents(q, "そのとき兵十は", [toki], read)
    assert [p.reading for p in updated.accent_phrases] == ["ソノ", "トキヘエジュウワ", "フト"]
    assert [(c.field, c.reason) for c in changes] == [("accent_skipped", reason)]
    assert handled == frozenset()


AMEGA = ["ア", "メ", "ガ"]


@pytest.mark.parametrize(
    ("texts", "text", "counts", "word", "reason"),
    [
        (AMEGA, "あめが", {"": 0, "あめ": 3}, target(0, 2, 1, 2), "VOICEVOX reads 3 moras here"),
        (["オ", *AMEGA], "おあめが", {"お": 1, "おあめ": 3}, target(1, 3, 1, 2), "does not start an accent phrase"),
        (AMEGA, "あめが", {"": 0, "あめ": 2}, target(0, 2, 1, 2, attached=0), "goes on past the particles"),
        (AMEGA, "あめが", {"": 0, "あめ": 2}, target(0, 2, (1, 3), 2), "do not fit 2 moras"),
        (["ア"], "あめが", {"": 0, "あめ": 2}, target(0, 2, 1, 2), "runs past the last accent phrase"),
    ],
)
def test_apply_accents_skips_words_it_cannot_place(
    texts: list[str], text: str, counts: dict[str, int], word: AccentTarget, reason: str
):
    q = query(phrase(texts, 1))
    updated, changes, handled = apply_accents(q, text, [word], reader(counts))
    assert updated.accent_phrases[0].accent == 1
    assert [c.field for c in changes] == ["accent_skipped"]
    assert reason in changes[0].reason
    assert handled == frozenset()


def word(start: int, surface: str, *pos: str) -> Word:
    return Word(start, surface, pos)


def test_attachable_phrases_takes_particles_and_helpers_after_te_but_not_main_verbs():
    text = "習っていて、連絡をくれた"
    words = [
        word(0, "習っ", "動詞", "一般"),
        word(2, "て", "助詞", "接続助詞"),
        word(3, "い", "動詞", "非自立可能"),
        word(4, "て", "助詞", "接続助詞"),
        word(5, "、", "補助記号", "読点"),
        word(6, "連絡", "名詞", "普通名詞"),
        word(8, "を", "助詞", "格助詞"),
        word(9, "くれ", "動詞", "非自立可能"),
        word(11, "た", "助動詞", "*"),
    ]
    q = query(
        phrase(["ナ", "ラ", "ッ", "テ"], 2),
        phrase(["イ", "テ"], 2),
        phrase(["レ", "ン", "ラ", "ク", "オ"], 5),
        phrase(["ク", "レ", "タ"], 1),
    )
    counts = {"習っ": 3, "習って": 4, "習ってい": 5, "習っていて": 6, "習っていて、連絡": 10, "習っていて、連絡を": 11}
    counts |= {"習っていて、連絡をくれ": 13}
    assert attachable_phrases(q, text, words, reader(counts)) == frozenset({1})


def test_attachable_phrases_ignores_words_inside_a_phrase():
    words = [word(0, "雨", "名詞", "普通名詞"), word(1, "が", "助詞", "格助詞")]
    q = query(phrase(["ア", "メ", "ガ"], 1))
    assert attachable_phrases(q, "雨が", words, reader({"雨": 2})) == frozenset()
