import json
from pathlib import Path

import httpx
import pytest

from vovovo import cli
from vovovo.accents import Word
from vovovo.readings import ReadingCandidate, ReadingOption
from vovovo.settings import Settings
from vovovo.typesafe import (
    ChoiceAnswer,
    NoulAnswer,
    Pacer,
    RequestCapError,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    TypeSafeError,
    Usage,
    render_request,
    send,
)
from vovovo.voicevox import AccentPhrase, AudioQuery, Mora, VoicevoxClient

SAMPLE_PHRASE = {
    "moras": [
        {"text": "ア", "consonant": None, "consonant_length": None, "vowel": "a", "vowel_length": 0.1, "pitch": 5.7},
        {"text": "メ", "consonant": "m", "consonant_length": 0.05, "vowel": "e", "vowel_length": 0.1, "pitch": 6.0},
    ],
    "accent": 1,
    "pause_mora": None,
    "is_interrogative": False,
}

SAMPLE_QUERY = {
    "accent_phrases": [SAMPLE_PHRASE],
    "speedScale": 1.0,
    "pitchScale": 0.0,
    "intonationScale": 1.0,
    "volumeScale": 1.0,
    "prePhonemeLength": 0.1,
    "postPhonemeLength": 0.1,
    "pauseLength": None,
    "pauseLengthScale": 1.0,
    "outputSamplingRate": 24000,
    "outputStereo": False,
    "kana": "ア'メ",
    "futureField": "kept",
}

REPITCHED = 9.0


def repitch(phrases: list[AccentPhrase]) -> list[AccentPhrase]:
    out = [phrase.model_copy(deep=True) for phrase in phrases]
    for phrase in out:
        for mora in phrase.moras:
            mora.pitch = REPITCHED
    return out


def test_voicevox_client_round_trips_query_with_aliases_and_extras():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/audio_query":
            return httpx.Response(200, json=SAMPLE_QUERY)
        return httpx.Response(200, content=b"RIFF")

    client = VoicevoxClient("http://voicevox.test/", httpx.Client(transport=httpx.MockTransport(handler)))
    query = client.audio_query("雨", 3)
    assert query.accent_phrases[0].reading == "アメ"
    assert query.speed_scale == 1.0
    assert client.synthesis(query, 3) == b"RIFF"
    assert seen[0].url.params["text"] == "雨"
    assert seen[0].url.params["speaker"] == "3"
    assert seen[1].url.path == "/synthesis"
    assert json.loads(seen[1].content) == SAMPLE_QUERY


@pytest.mark.parametrize("method", ["mora_pitch", "mora_data"])
def test_voicevox_client_recomputes_morae_from_the_phrase_list(method: str):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        for phrase in body:
            for mora in phrase["moras"]:
                mora["pitch"] = REPITCHED
        return httpx.Response(200, json=body)

    client = VoicevoxClient("http://voicevox.test", httpx.Client(transport=httpx.MockTransport(handler)))
    phrases = [AccentPhrase.model_validate(SAMPLE_PHRASE)]
    phrases[0].accent = 2
    result = getattr(client, method)(phrases, 3)
    assert seen[0].url.path == f"/{method}"
    assert seen[0].url.params["speaker"] == "3"
    assert json.loads(seen[0].content)[0]["accent"] == 2
    assert [m.pitch for m in result[0].moras] == [REPITCHED, REPITCHED]
    assert phrases[0].moras[0].pitch == 5.7


def test_voicevox_client_raises_on_http_error():
    client = VoicevoxClient(
        "http://voicevox.test", httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.audio_query("雨", 3)


def request_fixture() -> SystemOneRequest:
    return SystemOneRequest(
        state={"sentence": "雨"}, model="jev-latest", questions={"q": {"type": "noul", "instructions": "雨か"}}
    )


def test_render_request_hides_key_and_keeps_japanese():
    text = render_request("https://api.test/v1/systemone", request_fixture())
    assert text.startswith("POST https://api.test/v1/systemone\nAuthorization: Bearer ***")
    assert '"sentence": "雨"' in text


def test_send_sets_bearer_header_and_parses_answers():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-key"
        assert json.loads(request.content)["model"] == "jev-latest"
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {"q": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 12},
            },
        )

    response = send(
        "https://api.test/v1/systemone",
        "secret-key",
        request_fixture(),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert response.model == "jev-1.13.0"
    assert isinstance(response.answers["q"], NoulAnswer)
    assert response.usage is not None
    assert response.usage.input_tokens == 12


def test_send_raises_with_status_and_body():
    handler = httpx.MockTransport(lambda _: httpx.Response(422, text='{"detail":"bad"}'))
    with pytest.raises(TypeSafeError, match=r"422.*bad"):
        send("https://api.test/v1/systemone", "k", request_fixture(), httpx.Client(transport=handler))


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_pacer_spaces_requests_start_to_start():
    clock = FakeClock()
    pacer = Pacer(3.0, clock=clock, sleep=clock.sleep)
    pacer.wait()
    clock.now += 1.0
    pacer.wait()
    clock.now += 5.0
    pacer.wait()
    assert clock.slept == [2.0]
    assert pacer.sent == 3


def test_pacer_refuses_requests_past_the_cap():
    clock = FakeClock()
    pacer = Pacer(0.0, cap=2, clock=clock, sleep=clock.sleep)
    pacer.wait()
    pacer.wait()
    with pytest.raises(RequestCapError, match="2 TypeSafe requests sent, the cap is 2"):
        pacer.wait()
    assert pacer.sent == 2


def test_render_answers_covers_all_answer_types():
    response = SystemOneResponse(
        model="jev-1.13.0",
        usage=Usage(input_tokens=10, output_tokens=0),
        answers={
            "a": ChoiceAnswer(type="choice", choice="rain", confidence=0.9, probabilities={"rain": 0.9, "candy": 0.1}),
            "b": ScoreAnswer(type="score", score=1.5, confidence=0.7, probabilities={0: 0.1, 1: 0.3, 2: 0.6}),
            "c": NoulAnswer(type="noul", noul=0.2),
        },
    )
    text = cli.render_answers(response)
    assert "a: rain (confidence 0.90) [rain=0.90, candy=0.10]" in text
    assert "b: score 1.50 (confidence 0.70)" in text
    assert "c: 0.20" in text


def test_render_changes_reports_none():
    assert cli.render_changes([]) == "changes: none"


class StubVoicevox:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.synthesized: list[AudioQuery] = []
        self.repitched: list[list[AccentPhrase]] = []
        self.remeasured: list[list[AccentPhrase]] = []
        self.queried: list[str] = []

    def audio_query(self, text: str, speaker: int) -> AudioQuery:
        self.queried.append(text)
        query = AudioQuery.model_validate(SAMPLE_QUERY)
        if not text:
            query.accent_phrases = []
        if "、" in text:
            query.accent_phrases[0].pause_mora = Mora(text="、", vowel="pau", vowel_length=0.3, pitch=0.0)
            query.accent_phrases[0].moras[0].text = "ジュ"
        return query

    def mora_pitch(self, accent_phrases: list[AccentPhrase], speaker: int) -> list[AccentPhrase]:
        self.repitched.append(accent_phrases)
        return repitch(accent_phrases)

    def mora_data(self, accent_phrases: list[AccentPhrase], speaker: int) -> list[AccentPhrase]:
        self.remeasured.append(accent_phrases)
        return repitch(accent_phrases)

    def synthesis(self, query: AudioQuery, speaker: int) -> bytes:
        self.synthesized.append(query)
        return b"RIFF" + str(query.accent_phrases[0].accent).encode()


@pytest.fixture
def offline_settings(monkeypatch):
    def factory(**overrides):
        monkeypatch.setattr(
            cli, "Settings", lambda: Settings(_env_file=None, sudachi_dict_path=None, jmdict_path=None, **overrides)
        )

    return factory


def test_main_dry_run_prints_request_and_sends_nothing(monkeypatch, capsys, offline_settings):
    offline_settings()
    monkeypatch.setattr(cli, "VoicevoxClient", StubVoicevox)
    assert cli.main(["雨が降る", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "POST https://api.typesafe.ai/v1/systemone" in out
    assert '"sentence_type"' in out
    assert "_sense" not in out
    assert "[dry-run] request not sent" in out


def test_main_without_key_fails_before_sending(monkeypatch, capsys, offline_settings):
    offline_settings()
    monkeypatch.setattr(cli, "VoicevoxClient", StubVoicevox)
    assert cli.main(["雨が降る"]) == 1
    assert "TYPESAFE_API_KEY is not set" in capsys.readouterr().err


def run_main_with(
    monkeypatch, offline_settings, answers: dict, argv: list[str], split_answers: dict | None = None
) -> tuple[StubVoicevox, int]:
    offline_settings(typesafe_api_key="k")
    stub = StubVoicevox("unused")
    monkeypatch.setattr(cli, "VoicevoxClient", lambda _url: stub)
    monkeypatch.setattr(cli, "REQUEST_INTERVAL_SEC", 0.0)
    responses = [SystemOneResponse(model="jev-1.13.0", answers=answers)]
    if split_answers is not None:
        responses.insert(0, SystemOneResponse(model="jev-1.13.0", answers=split_answers))
    monkeypatch.setattr(cli, "send", lambda _url, _key, _request: responses.pop(0))
    return stub, cli.main(argv)


def test_main_splits_text_before_phrase_questions(monkeypatch, capsys, offline_settings):
    split_answers = {
        "s0_split": ChoiceAnswer(type="choice", choice="split_2", confidence=0.8, probabilities={"split_2": 0.8}),
    }
    answers = {
        "sentence_type": ChoiceAnswer(
            type="choice", choice="statement", confidence=0.9, probabilities={"statement": 0.9}
        ),
    }
    out_path = Path(".build/tests/split.wav")
    before_path = Path(".build/tests/split.before.wav")
    try:
        stub, code = run_main_with(
            monkeypatch, offline_settings, answers, ["十里はなれた", "--out", str(out_path)], split_answers
        )
        assert code == 0
    finally:
        out_path.unlink(missing_ok=True)
        before_path.unlink(missing_ok=True)
    assert stub.queried == ["十里はなれた", "十里、はなれた", "十里、はなれた", "十里"]
    assert [[p.pause_mora for p in phrases] for phrases in stub.remeasured] == [[None]]
    assert stub.synthesized[1].accent_phrases[0].pause_mora is None
    assert {m.pitch for m in stub.synthesized[1].accent_phrases[0].moras} == {REPITCHED}
    printed = capsys.readouterr().out
    assert "s0.split: 十里はなれた -> 十里、はなれた (split confidence=0.80)" in printed
    assert '"sentence": "十里、はなれた"' in printed


def test_main_skips_repitch_when_accent_is_unchanged(monkeypatch, capsys, offline_settings):
    answers = {
        "sentence_type": ChoiceAnswer(
            type="choice", choice="question", confidence=0.95, probabilities={"question": 0.95}
        ),
    }
    out_path = Path(".build/tests/q.wav")
    before_path = Path(".build/tests/q.before.wav")
    try:
        stub, code = run_main_with(monkeypatch, offline_settings, answers, ["降るの", "--out", str(out_path)], {})
        assert code == 0
    finally:
        out_path.unlink(missing_ok=True)
        before_path.unlink(missing_ok=True)
    assert stub.repitched == []
    assert stub.synthesized[1].accent_phrases[0].is_interrogative is True
    assert "p1.is_interrogative: False -> True" in capsys.readouterr().out


class KanaHomographLexicon:
    def __init__(self, dict_path: Path, accents: object, glosses: object = None) -> None:
        self.dict_path = dict_path
        self.glosses = glosses

    def words(self, text: str) -> list[Word]:
        return []

    def candidates(self, text: str, *, minimum: int = 2) -> list[ReadingCandidate]:
        common = ("名詞-普通名詞-一般",)
        options = {
            "アメ(飴)": ReadingOption("アメ", (0,), ("飴",), common, ("飴",)),
            "アメ(雨)": ReadingOption("アメ", (1,), ("雨",), common, ("雨",)),
        }
        return [ReadingCandidate(0, 2, "あめ", "名詞", options, 1)]


def test_main_applies_the_accent_of_the_chosen_word_and_repitches(monkeypatch, capsys, offline_settings):
    monkeypatch.setattr(cli, "ReadingLexicon", KanaHomographLexicon)
    monkeypatch.setattr(cli, "AccentDictionary", lambda: None)
    word_answers = {
        "r0_reading": ChoiceAnswer(type="choice", choice="アメ(飴)", confidence=0.9, probabilities={"アメ(飴)": 0.9}),
    }
    answers = {
        "sentence_type": ChoiceAnswer(
            type="choice", choice="statement", confidence=0.9, probabilities={"statement": 0.9}
        ),
    }
    out_path = Path(".build/tests/accent.wav")
    before_path = Path(".build/tests/accent.before.wav")
    argv = ["あめが降る", "--out", str(out_path), "--sudachi-dict", "unused.dic"]
    try:
        stub, code = run_main_with(monkeypatch, offline_settings, answers, argv, word_answers)
        assert code == 0
        assert before_path.read_bytes() == b"RIFF1"
        assert out_path.read_bytes() == b"RIFF2"
    finally:
        out_path.unlink(missing_ok=True)
        before_path.unlink(missing_ok=True)
    printed = capsys.readouterr().out
    assert '"r0_reading"' in printed
    assert "r0.accent: アメ(1) -> アメ(2) (reading=アメ(飴) confidence=0.90 accent_type=0)" in printed
    assert ".pitch:" not in printed
    assert '"p1_focus"' not in printed
    assert '"tone"' not in printed
    assert stub.queried == ["あめが降る", "", "", "あめ"]
    assert [p[0].accent for p in stub.repitched] == [2]
    final = stub.synthesized[1].accent_phrases[0]
    assert [m.pitch for m in final.moras] == pytest.approx([REPITCHED, REPITCHED])
    assert [m.pitch for m in stub.synthesized[0].accent_phrases[0].moras] == [5.7, 6.0]
