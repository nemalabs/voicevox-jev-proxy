import pytest

from voicevox_jev_proxy.edits import apply_edits
from voicevox_jev_proxy.laughter import build_laugh_questions, find_laughs, laugh_edits, overlaps
from voicevox_jev_proxy.typesafe import ChoiceAnswer


def choice(value: str, confidence: float = 0.9, probabilities: dict[str, float] | None = None) -> ChoiceAnswer:
    return ChoiceAnswer(
        type="choice", choice=value, confidence=confidence, probabilities=probabilities or {value: confidence}
    )


def test_find_laughs_takes_standalone_marks_in_text_order():
    text = "それガチで草なんだがwww、無理ゲーでしょ笑。まじ(笑)\uff57"
    assert [(laugh.surface, laugh.start) for laugh in find_laughs(text)] == [
        ("www", 10),
        ("笑", 21),
        ("(笑)", 25),
        ("\uff57", 28),
    ]


@pytest.mark.parametrize(
    "text", ["笑顔で笑う", "微笑む", "草原と雑草", "言ってて草", "庭の草を刈った。", "Twitterでwow", "Wiiを買った"]
)
def test_find_laughs_leaves_words_that_contain_the_marks(text: str):
    assert find_laughs(text) == []


def test_find_laughs_leaves_words_written_with_the_marks_to_jev():
    assert [laugh.surface for laugh in find_laughs("W杯の決勝を見た。")] == ["W"]


def test_build_laugh_questions_ask_whether_each_mark_is_a_laugh_in_its_clause():
    text = "それガチでwwwなんだが。まじ笑"
    questions = build_laugh_questions(text, find_laughs(text))
    assert list(questions) == ["l5_laugh", "l15_laugh"]
    www = questions["l5_laugh"]
    assert www["instructions"]["mark"] == "www"
    assert www["instructions"]["clause"] == "それガチでwwwなんだが"
    assert list(www["criteria"]) == ["laugh", "word"]
    assert questions["l15_laugh"]["instructions"]["clause"] == "まじ笑"


def test_laugh_edits_remove_the_marks_jev_takes_for_laughs():
    text = "W杯なんだがwww笑"
    laughs = find_laughs(text)
    word = choice("word", 0.9, {"word": 0.9, "laugh": 0.1})
    unsure = choice("laugh", 0.1, {"laugh": 0.55, "word": 0.45})
    answers = {"l0_laugh": word, "l6_laugh": choice("laugh"), "l9_laugh": unsure}
    edits = laugh_edits(laughs, answers, 0.6)
    assert [(e.start, e.end, e.replacement, e.reading, e.change.field) for e in edits] == [(6, 9, "", "", "laugh")]
    assert apply_edits(text, edits)[0] == "W杯なんだが笑"


def test_laugh_edits_gate_on_the_probability_of_a_laugh():
    text = "わかりみが深すぎるｗ"
    hesitant = choice("laugh", 0.3, {"laugh": 0.7, "word": 0.3})
    edits = laugh_edits(find_laughs(text), {"l9_laugh": hesitant}, 0.6)
    assert [(e.start, e.end, e.replacement) for e in edits] == [(9, 10, "")]
    assert edits[0].change.reason == "laugh confidence=0.30 p(laugh)=0.70"


def test_overlaps_detects_spans_touching_a_mark():
    laughs = find_laughs("それガチでwwwなんだが")
    assert overlaps(5, 10, laughs)
    assert not overlaps(0, 5, laughs)
