import pytest

from voicevox_jev_proxy.cli import REGROUPED, Corrector, Voice, carry_answers
from voicevox_jev_proxy.intonation import (
    LATE_PEAK,
    Height,
    build_height_questions,
    find_heights,
    lower_heights,
    move_late_nuclei,
)
from voicevox_jev_proxy.phrase_accents import Token
from voicevox_jev_proxy.prosody import Policy
from voicevox_jev_proxy.typesafe import ChoiceAnswer, SystemOneRequest, SystemOneResponse
from voicevox_jev_proxy.voicevox import AccentPhrase, AudioQuery, Mora

PAUSE = Mora(text="、", vowel="pau", vowel_length=0.3, pitch=0.0)


def phrase(moras: list[tuple[str, float]], accent: int) -> AccentPhrase:
    return AccentPhrase(
        moras=[Mora(text=text, vowel="a", vowel_length=0.1, pitch=pitch) for text, pitch in moras], accent=accent
    )


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


def token(start: int, surface: str, pron: str | None = None) -> Token:
    return Token(start, surface, ("名詞", "普通名詞"), surface if pron is None else pron, (0,), "")


def sora() -> AccentPhrase:
    return phrase([("ソ", 5.0), ("ラ", 5.6), ("ガ", 5.6)], 3)


def hareru() -> AccentPhrase:
    return phrase([("ハ", 5.4), ("レ", 6.1), ("ル", 5.3)], 2)


SORA_TOKENS = [token(0, "ソラ"), token(2, "ガ"), token(3, "ハレル")]
SORA_HEIGHT = Height(0, 0, 3, 6, "ソラガ", "ハレル")


def kana_reader(calls: list[str] | None = None):
    """VOICEVOX reading a kana text as one mora per character."""

    def read(text: str) -> AudioQuery:
        if calls is not None:
            calls.append(text)
        return query(phrase([(kana, 5.5) for kana in text], len(text))) if text else query()

    return read


def test_find_heights_names_a_flat_phrase_and_the_phrase_rising_after_it():
    heights = find_heights("ソラガハレル", query(sora(), hareru()), SORA_TOKENS, kana_reader())
    assert heights == [SORA_HEIGHT]
    assert heights[0].key == "p1_head"


def test_find_heights_skips_pairs_that_do_not_rise_again():
    falling = phrase([("ソ", 5.6), ("ラ", 5.0), ("ガ", 4.8)], 1)
    paused = sora()
    paused.pause_mora = PAUSE
    level = phrase([("ハ", 5.4), ("レ", 5.65), ("ル", 5.3)], 2)
    calls: list[str] = []
    for first, second in ((falling, hareru()), (paused, hareru()), (sora(), level)):
        assert find_heights("ソラガハレル", query(first, second), SORA_TOKENS, kana_reader(calls)) == []
    assert calls == []


def test_find_heights_skips_phrases_the_words_do_not_spell():
    tokens = [token(0, "ソラ", "ソレ"), token(2, "ガ"), token(3, "ハレル")]
    assert find_heights("ソラガハレル", query(sora(), hareru()), tokens, kana_reader()) == []


def test_build_height_questions_ask_what_the_first_phrase_modifies():
    questions = build_height_questions([SORA_HEIGHT])
    assert set(questions) == {"p1_head"}
    assert questions["p1_head"]["instructions"]["phrase"] == "ソラガ"
    assert questions["p1_head"]["instructions"]["following"] == "ハレル"
    assert set(questions["p1_head"]["criteria"]) == {"following", "beyond", "none"}


def test_lower_heights_brings_the_second_peak_down_to_the_first():
    second = hareru()
    second.moras.append(Mora(text="ツ", vowel="U", vowel_length=0.05, pitch=0.0))
    q = query(sora(), second)
    height = Height(0, 0, 3, 7, "ソラガ", "ハレルツ")
    updated, changes = lower_heights(q, [height], {"p1_head": choice("following", 0.85)}, 0.8)
    assert [m.pitch for m in updated.accent_phrases[1].moras] == pytest.approx([4.9, 5.6, 4.8, 0.0])
    assert [m.pitch for m in q.accent_phrases[1].moras] == [5.4, 6.1, 5.3, 0.0]
    assert [(c.phrase_id, c.field, c.before, c.after, c.reason) for c in changes] == [
        ("p2", "height", "6.10", "5.60", "head=following confidence=0.85")
    ]


@pytest.mark.parametrize("answer", [choice("following", 0.79), choice("beyond", 0.95), choice("none", 0.95)])
def test_lower_heights_needs_a_confident_following_answer(answer: ChoiceAnswer):
    updated, changes = lower_heights(query(sora(), hareru()), [SORA_HEIGHT], {"p1_head": answer}, 0.8)
    assert [m.pitch for m in updated.accent_phrases[1].moras] == [5.4, 6.1, 5.3]
    assert changes == []


def test_lower_heights_skips_pairs_that_changed_since_the_question():
    joined = phrase([*((m.text, m.pitch) for m in sora().moras), *((m.text, m.pitch) for m in hareru().moras)], 6)
    falling = sora()
    falling.accent = 1
    answers = {"p1_head": choice("following")}
    for q, reason in (
        (query(joined), "the two phrases were joined with others"),
        (query(falling, hareru()), "the second phrase no longer rises"),
    ):
        updated, changes = lower_heights(q, [SORA_HEIGHT], answers, 0.8)
        assert updated == q
        assert [(c.phrase_id, c.field, c.reason) for c in changes] == [("p2", "height_skipped", reason)]


def test_move_late_nuclei_moves_a_nucleus_on_the_second_to_last_mora_before_another_phrase():
    q = query(phrase([("ジ", 5.4), ("ツ", 5.9), ("ニ", 5.9)], 2), phrase([("ボ", 5.5), ("ク", 5.8), ("ワ", 5.6)], 1))
    updated, changes = move_late_nuclei(q)
    assert [p.accent for p in updated.accent_phrases] == [1, 1]
    assert q.accent_phrases[0].accent == 2
    assert [(c.phrase_id, c.field, c.before, c.after, c.reason) for c in changes] == [
        ("p1", "accent", "2", "1", LATE_PEAK)
    ]


def test_move_late_nuclei_leaves_the_last_phrase_a_pause_and_a_first_mora_nucleus():
    paused = hareru()
    paused.pause_mora = PAUSE
    short = phrase([("ア", 5.5), ("メ", 5.0)], 1)
    q = query(paused, short, sora(), hareru())
    updated, changes = move_late_nuclei(q)
    assert [p.accent for p in updated.accent_phrases] == [2, 1, 3, 2]
    assert changes == []


class RereadingVoicevox:
    """VOICEVOX for ソラガ|ハレル|ヒモ; /mora_pitch puts back its own pitches whatever the query carried."""

    def __init__(self) -> None:
        self.repitched: list[list[AccentPhrase]] = []
        self._read = kana_reader()

    def voiced(self) -> AudioQuery:
        return query(sora(), hareru(), phrase([("ヒ", 5.6), ("モ", 5.1)], 1))

    def audio_query(self, text: str, speaker: int) -> AudioQuery:
        return self.voiced() if text == "ソラガハレルヒモ" else self._read(text)

    def mora_pitch(self, accent_phrases: list[AccentPhrase], speaker: int) -> list[AccentPhrase]:
        self.repitched.append(accent_phrases)
        pitches = {p.reading: [m.pitch for m in p.moras] for p in self.voiced().accent_phrases}
        out = [p.model_copy(deep=True) for p in accent_phrases]
        for p in out:
            for m, pitch in zip(p.moras, pitches[p.reading], strict=True):
                m.pitch = pitch
        return out


class KanaTokenizer:
    def tokens(self, text: str) -> list[Token]:
        return [*SORA_TOKENS, token(6, "ヒモ")]


def test_corrector_lowers_heights_after_the_last_repitch():
    stub = RereadingVoicevox()
    sent: list[SystemOneRequest] = []

    def ask(request: SystemOneRequest) -> SystemOneResponse:
        sent.append(request)
        answers = {"p1_head": choice("following")} if "p1_head" in request.questions else {}
        return SystemOneResponse(model="jev-1.13.0", answers=answers)

    corrector = Corrector(None, KanaTokenizer(), "jev-latest", Policy())
    correction = corrector.correct("ソラガハレルヒモ", Voice(stub, 3), ask)
    assert "p1_head" in sent[-1].questions
    assert [p.accent for p in stub.repitched[0]] == [3, 1, 1]
    final = correction.query.accent_phrases
    assert [p.accent for p in final] == [3, 1, 1]
    assert [m.pitch for m in final[1].moras] == pytest.approx([4.9, 5.6, 4.8])
    assert [m.pitch for m in correction.original.accent_phrases[1].moras] == [5.4, 6.1, 5.3]
    assert [(c.phrase_id, c.field) for c in correction.changes] == [("p2", "accent"), ("p2", "height")]


def test_corrector_requests_include_the_height_questions():
    corrector = Corrector(None, KanaTokenizer(), "jev-latest", Policy())
    requests = corrector.requests("ソラガハレルヒモ", Voice(RereadingVoicevox(), 3))
    assert "p1_head" in requests[-1].questions


def spelled(kana: list[str], accent: int) -> AccentPhrase:
    return phrase([(text, 5.5) for text in kana], accent)


def test_carry_answers_moves_answers_to_the_phrases_with_the_same_words():
    original = query(
        spelled(["ダ", "ブ", "リュ", "ウ"], 1),
        spelled(["ア", "メ", "ガ"], 1),
        spelled(["ガ", "ン", "ブ", "タ", "ヲ"], 1),
        spelled(["マ", "ガ", "ッ", "タ"], 1),
        spelled(["ム", "ラ", "ノ", "モ"], 3),
        spelled(["ヘ", "イ", "ガ"], 1),
    )
    edited = query(
        spelled(["ア", "メ", "ガ"], 1),
        spelled(["マ", "ブ", "タ", "ヲ"], 1),
        spelled(["マ", "ガ", "ッ", "タ"], 1),
        spelled(["ム", "ラ", "ノ"], 3),
        spelled(["モ", "ヘ", "イ", "ガ"], 1),
    )
    answers = {
        "sentence_type": choice("question"),
        **{f"p{index}_unit": choice("independent", index / 10) for index in range(2, 7)},
        "p3_head": choice("following"),
        "p5_head": choice("beyond"),
    }
    heights = [Height(2, 7, 12, 16, "眼ぶたを", "曲がった"), Height(4, 16, 20, 23, "村の茂", "平が")]
    carried, moved, skipped = carry_answers(answers, original, edited, heights)
    assert carried == {
        "sentence_type": answers["sentence_type"],
        "p2_unit": answers["p3_unit"],
        "p3_unit": answers["p4_unit"],
        "p2_head": answers["p3_head"],
    }
    assert moved == [Height(1, 3, 7, 11, "眼ぶたを", "曲がった")]
    assert [(c.phrase_id, c.field, c.reason) for c in skipped] == [
        ("p2", "unit_skipped", REGROUPED),
        ("p5", "unit_skipped", REGROUPED),
        ("p6", "unit_skipped", REGROUPED),
        ("p6", "height_skipped", REGROUPED),
    ]


class LaughingVoicevox(RereadingVoicevox):
    """VOICEVOX spelling out the w of ソラガハレルwヒモ as an accent phrase of its own."""

    def audio_query(self, text: str, speaker: int) -> AudioQuery:
        if text == "ソラガハレルwヒモ":
            himo = phrase([("ヒ", 5.6), ("モ", 5.1)], 1)
            return query(sora(), hareru(), spelled(["ダ", "ブ", "リュ", "ウ"], 1), himo)
        return super().audio_query(text, speaker)


def silencing_ask(sent: list[SystemOneRequest]):
    """Jev dropping the w, saying ソラガ modifies ハレル and joining ヒモ to the phrase before it."""

    def ask(request: SystemOneRequest) -> SystemOneResponse:
        sent.append(request)
        laugh = ChoiceAnswer(type="choice", choice="silent", confidence=0.9, probabilities={"silent": 0.9, "none": 0.1})
        answers = {"l6_laugh": laugh, "p1_head": choice("following"), "p4_unit": choice("attached")}
        return SystemOneResponse(model="jev-1.13.0", answers=answers)

    return ask


def test_one_request_corrector_asks_about_the_text_and_voicevox_phrases_at_once():
    stub = LaughingVoicevox()
    sent: list[SystemOneRequest] = []
    corrector = Corrector(None, KanaTokenizer(), "jev-latest", Policy(), one_request=True)
    corrector.correct("ソラガハレルwヒモ", Voice(stub, 3), silencing_ask(sent))
    assert corrector.requests("ソラガハレルwヒモ", Voice(stub, 3)) == sent
    request = sent[0]
    assert {"l6_laugh", "sentence_type", "p2_unit", "p3_unit", "p4_unit", "p1_head"} <= set(request.questions)
    assert request.state == {
        "sentence": "ソラガハレルwヒモ",
        "phrases": [
            {"id": "p1", "reading": "ソラガ"},
            {"id": "p2", "reading": "ハレル"},
            {"id": "p3", "reading": "ダブリュウ"},
            {"id": "p4", "reading": "ヒモ"},
        ],
    }


def test_one_request_corrector_lowers_the_pair_it_asked_about_after_a_laugh_is_dropped():
    sent: list[SystemOneRequest] = []
    corrector = Corrector(None, KanaTokenizer(), "jev-latest", Policy(), one_request=True)
    correction = corrector.correct("ソラガハレルwヒモ", Voice(LaughingVoicevox(), 3), silencing_ask(sent))
    assert len(sent) == 1
    assert correction.text == "ソラガハレルヒモ"
    final = correction.query.accent_phrases
    assert [p.reading for p in final] == ["ソラガ", "ハレル", "ヒモ"]
    assert [p.accent for p in final] == [3, 1, 1]
    assert [m.pitch for m in final[1].moras] == pytest.approx([4.9, 5.6, 4.8])
    assert [(c.phrase_id, c.field) for c in correction.changes] == [
        ("l6", "laugh"),
        ("p4", "unit_skipped"),
        ("p2", "accent"),
        ("p2", "height"),
    ]
