import pytest

from voicevox_jev_proxy.edits import apply_edits
from voicevox_jev_proxy.laughter import (
    build_laugh_questions,
    find_laughs,
    laugh_edits,
    laugh_phrases,
    overlaps,
    voiced_spans,
)
from voicevox_jev_proxy.typesafe import ChoiceAnswer
from voicevox_jev_proxy.voicevox import AccentPhrase, AudioQuery, Mora


def choice(value: str, confidence: float = 0.9, probabilities: dict[str, float] | None = None) -> ChoiceAnswer:
    return ChoiceAnswer(
        type="choice", choice=value, confidence=confidence, probabilities=probabilities or {value: confidence}
    )


def phrase(reading: str) -> AccentPhrase:
    moras = [Mora(text=ch, vowel="a", vowel_length=0.1, pitch=5.5) for ch in reading]
    return AccentPhrase(moras=moras, accent=1)


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
        return query(phrase("ア" * mora_counts[text]))

    return read


def test_find_laughs_takes_standalone_marks_in_text_order():
    text = "それガチで草なんだがwww、無理ゲーでしょ笑。まじ(笑)\uff57"
    assert [(laugh.surface, laugh.start) for laugh in find_laughs(text)] == [
        ("草", 5),
        ("www", 10),
        ("笑", 21),
        ("(笑)", 25),
        ("\uff57", 28),
    ]


@pytest.mark.parametrize("text", ["笑顔で笑う", "微笑む", "草原と雑草", "Twitterでwow", "Wiiを買った"])
def test_find_laughs_leaves_words_that_contain_the_marks(text: str):
    assert find_laughs(text) == []


def test_build_laugh_questions_show_each_reading_in_its_clause():
    text = "それガチで草なんだがwww"
    questions = build_laugh_questions(text, find_laughs(text))
    assert list(questions) == ["l5_laugh", "l10_laugh"]
    kusa = questions["l5_laugh"]["criteria"]
    assert kusa["silent"] == {"read_as": "（読まない）", "context": "それガチでなんだがwww"}
    assert kusa["kusa"] == {"read_as": "くさ", "context": "それガチでくさなんだがwww"}
    assert list(questions["l10_laugh"]["criteria"]) == ["silent", "none"]


def test_laugh_edits_replace_or_remove_the_mark_and_skip_unsure_answers():
    text = "草なんだがwww笑"
    laughs = find_laughs(text)
    unsure = choice("silent", 0.1, {"silent": 0.55, "none": 0.45})
    answers = {"l0_laugh": choice("kusa"), "l5_laugh": choice("silent"), "l8_laugh": unsure}
    edits = laugh_edits(laughs, answers, 0.6)
    assert [(e.start, e.end, e.replacement, e.change.field) for e in edits] == [
        (0, 1, "くさ", "laugh"),
        (5, 8, "", "laugh"),
    ]
    assert apply_edits(text, edits)[0] == "くさなんだが笑"
    assert laugh_edits(laughs, {"l0_laugh": choice("none")}, 0.6) == []


def test_laugh_edits_gate_on_the_mark_being_a_laugh_not_on_one_reading():
    text = "わかりみが深すぎるｗ"
    hesitant = choice("kusa", 0.54, {"none": 0.02, "kusa": 0.69, "silent": 0.29})
    edits = laugh_edits(find_laughs(text + "草"), {"l10_laugh": hesitant}, 0.6)
    assert [(e.start, e.end, e.replacement) for e in edits] == [(10, 11, "くさ")]


def test_voiced_spans_follow_earlier_edits_and_skip_silent_marks():
    text = "wwwそれは草だ笑"
    answers = {"l0_laugh": choice("silent"), "l6_laugh": choice("kusa"), "l8_laugh": choice("silent")}
    edits = laugh_edits(find_laughs(text), answers, 0.6)
    edited = apply_edits(text, edits)[0]
    spans = voiced_spans(edits)
    assert edited == "それはくさだ"
    assert spans == [(3, 5)]


def test_laugh_phrases_are_phrases_made_only_of_laugh_readings():
    text = "無理ゲーでしょくさ"
    read = reader({"無理ゲーでしょ": 7, text: 9})
    separate = query(phrase("ムリ"), phrase("ゲエデショ"), phrase("クサ"))
    joined = query(phrase("ムリ"), phrase("ゲエデショクサ"))
    assert laugh_phrases(separate, text, [(7, 9)], read) == {2}
    assert laugh_phrases(joined, text, [(7, 9)], read) == frozenset()
    assert laugh_phrases(separate, text, [], read) == frozenset()


def test_overlaps_detects_spans_touching_a_mark():
    laughs = find_laughs("それガチで草なんだが")
    assert overlaps(5, 10, laughs)
    assert not overlaps(0, 5, laughs)
