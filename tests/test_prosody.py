import pytest

from voicevox_jev_proxy.prosody import (
    Counterpart,
    PhraseRoles,
    Policy,
    apply_structure,
    build_questions,
    build_state,
    joined_accent,
    match_phrases,
    needs_repitch,
)
from voicevox_jev_proxy.typesafe import ChoiceAnswer
from voicevox_jev_proxy.voicevox import AccentPhrase, AudioQuery, Mora

POLICY = Policy(threshold=0.6)


def mora(text: str, vowel: str = "a", pitch: float = 5.5) -> Mora:
    return Mora(text=text, vowel=vowel, vowel_length=0.1, pitch=pitch)


def phrase(texts: list[str], accent: int) -> AccentPhrase:
    return AccentPhrase(moras=[mora(t) for t in texts], accent=accent)


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


def choice(value: str, confidence: float = 0.9) -> ChoiceAnswer:
    return ChoiceAnswer(type="choice", choice=value, confidence=confidence, probabilities={value: confidence})


def test_build_state_lists_phrase_ids_and_readings():
    state = build_state("雨が降った", query(phrase(["ア", "メ", "ガ"], 1), phrase(["フ", "ッ", "タ"], 1)))
    assert state == {
        "sentence": "雨が降った",
        "phrases": [{"id": "p1", "reading": "アメガ"}, {"id": "p2", "reading": "フッタ"}],
    }


def test_match_phrases_follows_the_words_through_the_edits():
    before = query(
        phrase(["ワ", "ラ"], 1),
        phrase(["ア", "メ", "ガ"], 1),
        phrase(["ガ", "ン", "ブ", "タ", "ヲ"], 1),
        phrase(["マ", "ガ", "ッ", "タ"], 1),
        phrase(["ム", "ラ", "ノ", "モ"], 3),
        phrase(["ヘ", "イ", "ガ"], 1),
    )
    after = query(
        phrase(["ア", "メ", "ガ"], 1),
        phrase(["マ", "ブ", "タ", "ヲ"], 1),
        phrase(["マ", "ガ", "ッ", "タ"], 1),
        phrase(["ム", "ラ", "ノ"], 3),
        phrase(["モ", "ヘ", "イ", "ガ"], 1),
    )
    assert match_phrases(before, after) == {
        1: Counterpart(0, same_reading=True),
        2: Counterpart(1, same_reading=False),
        3: Counterpart(2, same_reading=True),
    }


def test_build_questions_ask_only_about_standard_intonation():
    q = query(phrase(["ア", "メ", "ガ"], 1), phrase(["フ", "ッ", "タ"], 1), phrase(["ヨ"], 1))
    assert set(build_questions(q)) == {"sentence_type", "p2_unit", "p3_unit"}


@pytest.mark.parametrize("kind", ["question", "confirmation"])
def test_apply_structure_rising_types_mark_last_phrase(kind: str):
    q = query(phrase(["ア", "メ", "ガ"], 1), phrase(["フ", "ッ", "タ"], 1))
    updated, changes = apply_structure(q, {"sentence_type": choice(kind)}, POLICY)
    assert updated.accent_phrases[0].is_interrogative is False
    assert updated.accent_phrases[1].is_interrogative is True
    assert [c.field for c in changes] == ["is_interrogative"]
    assert needs_repitch(changes) is False


def test_apply_structure_statement_or_low_confidence_leaves_flag():
    q = query(phrase(["フ", "ッ", "タ"], 1))
    for answer in (choice("statement"), choice("question", 0.3)):
        updated, changes = apply_structure(q, {"sentence_type": answer}, POLICY)
        assert updated.accent_phrases[0].is_interrogative is False
        assert changes == []


def test_build_questions_asks_unit_from_second_phrase():
    q = query(phrase(["ヤ", "ル"], 2), phrase(["コ", "ト", "ガ"], 2), phrase(["ア", "ル"], 1))
    questions = build_questions(q)
    assert "p1_unit" not in questions
    assert questions["p2_unit"]["type"] == "choice"
    assert questions["p3_unit"]["type"] == "choice"


def test_joined_accent_keeps_first_real_nucleus():
    assert joined_accent(phrase(["ソ", "オ"], 1), phrase(["カ", "モ"], 2)) == 1
    assert joined_accent(phrase(["ヤ", "ル"], 2), phrase(["コ", "ト", "ガ"], 2)) == 4
    assert joined_accent(phrase(["ヤ", "ル"], 2), phrase(["コ", "ト"], 2)) == 4


def test_apply_structure_merges_attached_phrases():
    q = query(phrase(["ヤ", "ル"], 2), phrase(["コ", "ト", "ガ"], 2), phrase(["ア", "ル"], 1))
    answers = {"p2_unit": choice("attached"), "p3_unit": choice("independent")}
    updated, changes = apply_structure(q, answers, POLICY, PhraseRoles(attachable=frozenset({1, 2})))
    assert [p.reading for p in updated.accent_phrases] == ["ヤルコトガ", "アル"]
    assert updated.accent_phrases[0].accent == 4
    assert needs_repitch(changes) is True
    assert changes[0].field == "merge"
    assert changes[0].phrase_id == "p1"


def test_apply_structure_merge_stops_at_pause_and_below_threshold():
    left = phrase(["ヤ", "ル"], 2)
    left.pause_mora = Mora(text="、", vowel="pau", vowel_length=0.2, pitch=0.0)
    q = query(left, phrase(["コ", "ト", "ガ"], 2), phrase(["ア", "ル"], 1))
    answers = {"p2_unit": choice("attached"), "p3_unit": choice("attached", confidence=0.5)}
    updated, changes = apply_structure(q, answers, POLICY, PhraseRoles(attachable=frozenset({1, 2})))
    assert len(updated.accent_phrases) == 3
    assert changes == []


def test_apply_structure_merges_only_phrases_the_morphology_allows():
    q = query(phrase(["レ", "ン", "ラ", "ク", "オ"], 5), phrase(["ク", "レ", "タ"], 1))
    updated, changes = apply_structure(q, {"p2_unit": choice("attached")}, POLICY, PhraseRoles())
    assert [p.reading for p in updated.accent_phrases] == ["レンラクオ", "クレタ"]
    assert changes == []


def test_apply_structure_rise_skips_trailing_laugh_phrases():
    q = query(phrase(["ム", "リ"], 1), phrase(["デ", "ショ"], 1), phrase(["ワ", "ラ"], 1))
    roles = PhraseRoles(laughs=frozenset({2}))
    updated, changes = apply_structure(q, {"sentence_type": choice("confirmation")}, POLICY, roles)
    assert [p.is_interrogative for p in updated.accent_phrases] == [False, True, False]
    assert [(c.phrase_id, c.field) for c in changes] == [("p2", "is_interrogative")]


def test_apply_structure_laugh_phrases_join_nothing():
    q = query(phrase(["ヤ", "ル"], 2), phrase(["ワ", "ラ"], 1), phrase(["ヨ"], 1), phrase(["ネ"], 1))
    answers = {f"p{index}_unit": choice("attached") for index in (2, 3, 4)} | {"sentence_type": choice("question")}
    roles = PhraseRoles(attachable=frozenset({1, 2, 3}), laughs=frozenset({1}))
    updated, changes = apply_structure(q, answers, POLICY, roles)
    assert [p.reading for p in updated.accent_phrases] == ["ヤル", "ワラ", "ヨネ"]
    assert updated.accent_phrases[2].is_interrogative is True
    assert [(c.phrase_id, c.field) for c in changes] == [("p3", "merge"), ("p3", "is_interrogative")]


def test_apply_structure_merge_carries_rise_to_merged_last_phrase():
    q = query(phrase(["ソ", "オ"], 1), phrase(["カ", "ナ"], 2))
    answers = {"p2_unit": choice("fixed"), "sentence_type": choice("question")}
    updated, changes = apply_structure(q, answers, POLICY, PhraseRoles(attachable=frozenset({1})))
    assert len(updated.accent_phrases) == 1
    assert updated.accent_phrases[0].is_interrogative is True
    assert [c.field for c in changes] == ["merge", "is_interrogative"]
