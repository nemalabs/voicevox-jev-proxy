import json
import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from voicevox_jev_proxy import server
from voicevox_jev_proxy.cli import Correction
from voicevox_jev_proxy.prosody import Change
from voicevox_jev_proxy.settings import Settings
from voicevox_jev_proxy.typesafe import (
    ChoiceAnswer,
    Pacer,
    RequestCapError,
    SystemOneRequest,
    SystemOneResponse,
    TypeSafeError,
)
from voicevox_jev_proxy.voicevox import AudioQuery

QUERY = {
    "accent_phrases": [
        {
            "moras": [
                {
                    "text": "ア",
                    "consonant": None,
                    "consonant_length": None,
                    "vowel": "a",
                    "vowel_length": 0.1,
                    "pitch": 5.7,
                },
                {
                    "text": "メ",
                    "consonant": "m",
                    "consonant_length": 0.05,
                    "vowel": "e",
                    "vowel_length": 0.1,
                    "pitch": 6.0,
                },
            ],
            "accent": 1,
            "pause_mora": None,
            "is_interrogative": False,
        }
    ],
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
}
AUDIO_QUERY = "/audio_query?text=%E9%9B%A8&speaker=3"


class Corrections:
    def __init__(self, *failures: Exception) -> None:
        self.calls: list[tuple[str, int]] = []
        self._failures = list(failures)

    def __call__(self, text: str, speaker: int) -> AudioQuery:
        self.calls.append((text, speaker))
        if self._failures:
            raise self._failures.pop(0)
        return AudioQuery.model_validate(QUERY)


class Upstream:
    def __init__(self, response: httpx.Response | None = None, error: httpx.HTTPError | None = None) -> None:
        self.seen: list[httpx.Request] = []
        self._response = response
        self._error = error

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        if self._error is not None:
            raise self._error
        return self._response or httpx.Response(200, json=[{"name": "ずんだもん"}])


def app_with(correct, upstream: Upstream | None = None, origins: frozenset[str] = frozenset()) -> server.App:
    client = httpx.Client(transport=httpx.MockTransport(upstream or Upstream()))
    return server.App(correct, client, "http://voicevox.test/", origins)


def test_audio_query_answers_with_the_corrected_query_in_voicevox_field_names():
    correct, upstream = Corrections(), Upstream()
    reply = app_with(correct, upstream).handle("POST", AUDIO_QUERY, {}, b"")
    assert reply.status == 200
    assert reply.headers == (("Content-Type", "application/json"),)
    assert json.loads(reply.body) == QUERY
    assert correct.calls == [("雨", 3)]
    assert upstream.seen == []


@pytest.mark.parametrize("target", ["/audio_query?text=&speaker=3", "/audio_query?speaker=3", "/audio_query?text=+"])
def test_audio_query_without_text_goes_to_voicevox(target: str):
    correct, upstream = Corrections(), Upstream()
    reply = app_with(correct, upstream).handle("POST", target, {}, b"")
    assert reply.status == 200
    assert correct.calls == []
    assert [str(request.url) for request in upstream.seen] == [f"http://voicevox.test{target}"]


@pytest.mark.parametrize("target", ["/audio_query?text=%E9%9B%A8", "/audio_query?text=%E9%9B%A8&speaker=zundamon"])
def test_audio_query_needs_an_integer_speaker(target: str):
    correct = Corrections()
    reply = app_with(correct).handle("POST", target, {}, b"")
    assert reply.status == 422
    assert correct.calls == []


def test_audio_query_answers_only_the_web_pages_allowed():
    correct = Corrections()
    app = app_with(correct, origins=frozenset({"http://localhost:3000"}))
    refused = app.handle("POST", AUDIO_QUERY, {"origin": "https://example.com"}, b"")
    allowed = app.handle("POST", AUDIO_QUERY, {"origin": "http://localhost:3000"}, b"")
    assert refused.status == 403
    assert allowed.status == 200
    assert ("Access-Control-Allow-Origin", "http://localhost:3000") in allowed.headers
    assert correct.calls == [("雨", 3)]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (RequestCapError("20 TypeSafe requests sent, the cap is 20"), 503),
        (TypeSafeError("TypeSafe API returned 500: down"), 502),
        (httpx.ConnectError("VOICEVOX is not running"), 502),
        (ValueError("unexpected"), 500),
    ],
)
def test_audio_query_reports_a_failed_correction_and_keeps_serving(error: Exception, status: int, capsys):
    app = app_with(Corrections(error))
    failed = app.handle("POST", AUDIO_QUERY, {}, b"")
    assert failed.status == status
    assert str(error) in json.loads(failed.body)["detail"]
    assert str(error) in capsys.readouterr().err
    assert app.handle("POST", AUDIO_QUERY, {}, b"").status == 200


def test_other_requests_go_to_voicevox_unchanged():
    response = httpx.Response(
        200,
        content=b"RIFF",
        headers={"content-type": "audio/wav", "server": "uvicorn", "x-voicevox": "kept"},
    )
    upstream = Upstream(response)
    headers = {"content-type": "application/json", "accept": "audio/wav", "host": "127.0.0.1:50121", "cookie": "c"}
    reply = app_with(Corrections(), upstream).handle("POST", "/synthesis?speaker=3", headers, b'{"kana": "x"}')
    assert reply.status == 200
    assert reply.body == b"RIFF"
    assert dict(reply.headers) == {"content-type": "audio/wav", "x-voicevox": "kept"}
    [seen] = upstream.seen
    assert seen.method == "POST"
    assert str(seen.url) == "http://voicevox.test/synthesis?speaker=3"
    assert seen.content == b'{"kana": "x"}'
    assert seen.headers["content-type"] == "application/json"
    assert seen.headers["accept"] == "audio/wav"
    assert "cookie" not in seen.headers


def test_forward_keeps_voicevox_errors_and_reports_voicevox_being_down():
    missing = app_with(Corrections(), Upstream(httpx.Response(404, json={"detail": "Not Found"})))
    assert missing.handle("GET", "/nothing", {}, b"").status == 404
    down = app_with(Corrections(), Upstream(error=httpx.ConnectError("refused")))
    reply = down.handle("GET", "/speakers", {}, b"")
    assert reply.status == 502
    assert "refused" in json.loads(reply.body)["detail"]


def test_forward_sends_only_paths():
    upstream = Upstream()
    reply = app_with(Corrections(), upstream).handle("GET", "http://example.com/speakers", {}, b"")
    assert reply.status == 400
    assert upstream.seen == []


def test_handler_serves_the_app_over_http():
    correct, upstream = Corrections(), Upstream(httpx.Response(200, content=b"RIFF"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.handler_for(app_with(correct, upstream)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        with httpx.Client(trust_env=False) as client:
            corrected = client.post(f"{base}/audio_query", params={"text": "雨", "speaker": 3})
            synthesized = client.post(f"{base}/synthesis", params={"speaker": 3}, json=corrected.json())
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()
    assert corrected.status_code == 200
    assert corrected.json() == QUERY
    assert synthesized.content == b"RIFF"
    assert json.loads(upstream.seen[0].content) == QUERY
    assert correct.calls == [("雨", 3)]


class FakeCorrector:
    def __init__(self) -> None:
        self.voices: list[object] = []

    def requests(self, text: str, voice) -> list[SystemOneRequest]:
        return [SystemOneRequest(state={"sentence": text}, model="jev-latest", questions={})]

    def correct(self, text: str, voice, ask) -> Correction:
        self.voices.append(voice)
        ask(SystemOneRequest(state={"sentence": text}, model="jev-latest", questions={}))
        query = AudioQuery.model_validate(QUERY)
        change = Change("p2", "height", "6.10", "5.60", "head=following confidence=0.90")
        return Correction(text, query, query, [change])


class FakeClient:
    def __init__(self) -> None:
        self.queried: list[tuple[str, int]] = []

    def audio_query(self, text: str, speaker: int) -> AudioQuery:
        self.queried.append((text, speaker))
        return AudioQuery.model_validate(QUERY)


def test_live_correction_logs_answers_and_changes_and_shares_the_cap_across_texts(monkeypatch, capsys):
    answer = ChoiceAnswer(type="choice", choice="following", confidence=0.9, probabilities={"following": 0.9})
    typesafe = httpx.Client()
    used: list[object] = []

    def fake_send(_url, _key, _request, client):
        used.append(client)
        return SystemOneResponse(model="jev-1.13.0", answers={"p1_head": answer})

    monkeypatch.setattr(server, "send", fake_send)
    corrector = FakeCorrector()
    session = server.TypeSafeSession("https://typesafe.test/v1/systemone", "k", Pacer(0.0, cap=1), typesafe)
    correct = server.live_correction(corrector, FakeClient(), session)
    assert correct("雨", 3) == AudioQuery.model_validate(QUERY)
    assert used == [typesafe]
    err = capsys.readouterr().err
    assert "===== speaker 3: 雨" in err
    assert "p1_head: following (confidence 0.90)" in err
    assert "p2.height: 6.10 -> 5.60 (head=following confidence=0.90)" in err
    assert "TypeSafe requests so far: 1" in err
    with pytest.raises(RequestCapError):
        correct("飴", 3)


def test_dry_correction_logs_requests_and_returns_voicevox_query(monkeypatch, capsys):
    def refuse(*_args):
        raise AssertionError

    monkeypatch.setattr(server, "send", refuse)
    client = FakeClient()
    correct = server.dry_correction(FakeCorrector(), client, Settings(_env_file=None))
    assert correct("雨", 3) == AudioQuery.model_validate(QUERY)
    assert client.queried == [("雨", 3)]
    err = capsys.readouterr().err
    assert "POST https://api.typesafe.ai/v1/systemone\nAuthorization: Bearer ***" in err
    assert '"sentence": "雨"' in err
    assert "[dry-run] requests not sent" in err


def test_main_without_key_fails_before_serving(monkeypatch, capsys):
    monkeypatch.setattr(server, "Settings", lambda: Settings(_env_file=None, sudachi_dict_path=None, jmdict_path=None))
    assert server.main([]) == 1
    assert "TYPESAFE_API_KEY is not set" in capsys.readouterr().err


def test_main_needs_a_request_cap_of_at_least_one():
    with pytest.raises(SystemExit):
        server.main(["--request-cap", "0"])


def test_main_needs_a_request_interval_of_at_least_zero():
    with pytest.raises(SystemExit):
        server.main(["--request-interval", "-1"])


def test_main_needs_intonation_for_requests():
    with pytest.raises(SystemExit):
        server.main(["--requests", "1"])


class StoppedServer:
    def __init__(self, _address, _handler) -> None:
        pass

    def serve_forever(self) -> None:
        raise KeyboardInterrupt

    def server_close(self) -> None:
        pass


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], (0.0, 20)), (["--request-interval", "1.5", "--request-cap", "4"], (1.5, 4))],
)
def test_main_paces_typesafe_requests_as_the_flags_say(monkeypatch, argv: list[str], expected: tuple[float, int]):
    pacers: list[tuple[float, int]] = []

    def recording_pacer(interval: float, cap: int) -> Pacer:
        pacers.append((interval, cap))
        return Pacer(interval, cap)

    settings = Settings(_env_file=None, typesafe_api_key="k", sudachi_dict_path=None, jmdict_path=None)
    monkeypatch.setattr(server, "Settings", lambda: settings)
    monkeypatch.setattr(server, "build_corrector", lambda _args, _settings: FakeCorrector())
    monkeypatch.setattr(server, "Pacer", recording_pacer)
    monkeypatch.setattr(server, "ThreadingHTTPServer", StoppedServer)
    assert server.main(argv) == 0
    assert pacers == [expected]


@pytest.mark.parametrize(("argv", "expected"), [([], [False]), (["--intonation"], [True])])
def test_main_corrects_intonation_only_when_asked(monkeypatch, argv: list[str], expected: list[bool]):
    seen: list[bool] = []

    def recording_corrector(args, _settings) -> FakeCorrector:
        seen.append(args.intonation)
        return FakeCorrector()

    settings = Settings(_env_file=None, typesafe_api_key="k", sudachi_dict_path=None, jmdict_path=None)
    monkeypatch.setattr(server, "Settings", lambda: settings)
    monkeypatch.setattr(server, "build_corrector", recording_corrector)
    monkeypatch.setattr(server, "ThreadingHTTPServer", StoppedServer)
    assert server.main(argv) == 0
    assert seen == expected
