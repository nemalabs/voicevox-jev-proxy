from vovovo.pilot import (
    ATAMADAKA,
    HEIBAN_OR_ODAKA,
    NAKADAKA,
    accent_from_label,
    build_questions,
    cases_from_query,
    category_for,
    pattern_criteria,
    pattern_label,
    render_summary,
    score_cases,
)
from vovovo.typesafe import ChoiceAnswer
from vovovo.voicevox import AccentPhrase, AudioQuery, Mora


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


def test_category_for_maps_accent_positions():
    assert category_for(1, 3) == ATAMADAKA
    assert category_for(2, 3) == NAKADAKA
    assert category_for(3, 3) == HEIBAN_OR_ODAKA


def test_pattern_label_and_parse_round_trip_with_small_kana():
    moras = ("キョ", "ウ", "ワ")
    labels = [pattern_label(moras, k) for k in (1, 2, 3)]
    assert labels == ["キョ'ウワ", "キョウ'ワ", "キョウワ'"]
    assert [accent_from_label(label, moras) for label in labels] == [1, 2, 3]
    assert accent_from_label("キョ'ウワ", ("キ", "ョウ", "ワ")) is None


def test_pattern_criteria_has_one_option_per_mora():
    criteria = pattern_criteria(("ア", "メ", "ガ"))
    assert list(criteria) == ["ア'メガ", "アメ'ガ", "アメガ'"]
    assert criteria["アメガ'"] == "最後まで下がらない (平板型または尾高型)"


def test_cases_skip_single_mora_phrases():
    cases = cases_from_query("雨が", query(phrase(["ア", "メ", "ガ"], 1), phrase(["エ"], 1)))
    assert [c.phrase_index for c in cases] == [0]
    assert cases[0].gold_accent == 1
    assert cases[0].gold_category == ATAMADAKA


def test_build_questions_fans_out_two_framings_per_phrase():
    cases = cases_from_query("雨が", query(phrase(["ア", "メ", "ガ"], 1)))
    questions = build_questions(cases)
    assert set(questions) == {"p1_pattern", "p1_type"}
    assert set(questions["p1_pattern"]["criteria"]) == {"ア'メガ", "アメ'ガ", "アメガ'"}
    assert set(questions["p1_type"]["criteria"]) == {ATAMADAKA, NAKADAKA, HEIBAN_OR_ODAKA}


def test_score_cases_and_summary():
    cases = cases_from_query("雨が", query(phrase(["ア", "メ", "ガ"], 1), phrase(["フ", "ッ", "タ"], 1)))
    answers = {
        "p1_pattern": ChoiceAnswer(type="choice", choice="ア'メガ", confidence=0.9, probabilities={"ア'メガ": 0.9}),
        "p1_type": ChoiceAnswer(type="choice", choice=NAKADAKA, confidence=0.5, probabilities={NAKADAKA: 0.5}),
        "p2_pattern": ChoiceAnswer(type="choice", choice="フッタ'", confidence=0.7, probabilities={"フッタ'": 0.7}),
    }
    results = score_cases(cases, answers)
    assert [r.pattern_hit for r in results] == [True, False]
    assert [r.category_hit for r in results] == [False, False]
    assert results[1].category_answer is None
    summary = render_summary(results)
    assert "pattern (exact accent) hits: 1/2 = 0.50" in summary
    assert "type (3-way category) hits: 0/2 = 0.00" in summary
    assert "baseline always-heiban hits: 0/2 = 0.00" in summary
