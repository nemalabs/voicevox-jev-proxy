"""A VOICEVOX-compatible server that corrects readings and intonation with Jev before VOICEVOX speaks.

POST /audio_query returns the corrected query; every other request goes to VOICEVOX unchanged, so a
tool that talks to VOICEVOX needs only this server's address. Corrections run one at a time, since the
tokenizers are not meant to be shared across threads.

A request with an Origin header comes from a web page. Any page could otherwise make this server spend
TypeSafe requests, so /audio_query refuses origins that were not allowed on the command line. A page can
also point its own domain at this machine (DNS rebinding) and call the server as its own origin, so every
request whose Host is not this server's own address is refused.
"""

import argparse
import ipaddress
import json
import socket
import sys
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import httpx

from voicevox_jev_proxy.cli import (
    Corrector,
    Voice,
    add_correction_arguments,
    build_corrector,
    check_correction_arguments,
    render_answers,
    render_changes,
)
from voicevox_jev_proxy.settings import Settings
from voicevox_jev_proxy.typesafe import (
    Pacer,
    RequestCapError,
    SystemOneRequest,
    SystemOneResponse,
    TypeSafeError,
    render_request,
    send,
)
from voicevox_jev_proxy.voicevox import AudioQuery, VoicevoxClient

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50121
DEFAULT_REQUEST_CAP = 20
DEFAULT_REQUEST_INTERVAL_SEC = 0.0
DEFAULT_MAX_TEXT_LENGTH = 1000
MAX_BODY_BYTES = 1024 * 1024
MAX_CONNECTIONS = 32
CLIENT_TIMEOUT_SEC = 30.0
LOOPBACK_NAMES = ("localhost", "127.0.0.1", "[::1]")
WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})  # noqa: S104 - recognised, not bound
UPSTREAM_TIMEOUT_SEC = 120.0
TYPESAFE_TIMEOUT_SEC = 30.0
AUDIO_QUERY_PATH = "/audio_query"
JSON_TYPE = "application/json"
ORIGIN = "origin"
HOST = "host"
FORWARDED_REQUEST_HEADERS = (
    "content-type",
    "accept",
    ORIGIN,
    "access-control-request-method",
    "access-control-request-headers",
)
DROPPED_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "content-encoding",
        "server",
        "date",
    }
)

Correct = Callable[[str, int], AudioQuery]


@dataclass(frozen=True)
class Reply:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


def json_reply(status: int, payload: object, headers: tuple[tuple[str, str], ...] = ()) -> Reply:
    body = json.dumps(payload, ensure_ascii=False).encode()
    return Reply(status, (("Content-Type", JSON_TYPE), *headers), body)


def error_reply(status: int, detail: str) -> Reply:
    return json_reply(status, {"detail": detail})


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def host_header(host: str, port: int) -> str:
    """Write `host` and `port` as a Host header does, with an IPv6 address in brackets."""
    try:
        bracketed = ipaddress.ip_address(host).version == 6  # noqa: PLR2004
    except ValueError:
        bracketed = False
    return f"[{host}]:{port}" if bracketed else f"{host}:{port}"


def own_hosts(host: str, port: int, extra: list[str]) -> frozenset[str]:
    """Return the Host headers that name this server: the loopback names, the bound address and those allowed."""
    names = [f"{name}:{port}" for name in LOOPBACK_NAMES]
    if host not in WILDCARD_HOSTS:
        names.append(host_header(host, port))
    return frozenset(name.lower() for name in [*names, *extra])


class App:
    """Routes one HTTP request: corrects POST /audio_query and forwards the rest to VOICEVOX."""

    def __init__(  # noqa: PLR0913
        self,
        correct: Correct,
        upstream: httpx.Client,
        voicevox_url: str,
        allowed_origins: frozenset[str],
        allowed_hosts: frozenset[str],
        *,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
    ) -> None:
        self._correct = correct
        self._upstream = upstream
        self._voicevox_url = voicevox_url.rstrip("/")
        self._allowed_origins = allowed_origins
        self._allowed_hosts = allowed_hosts
        self._max_text_length = max_text_length
        self._lock = threading.Lock()

    def handle(self, method: str, target: str, headers: Mapping[str, str], body: bytes) -> Reply:
        """Answer one request; `headers` has lower-case names.

        A request without Host is let through: browsers always send one, so it cannot come from a page.
        """
        host = headers.get(HOST)
        if host is not None and host.lower() not in self._allowed_hosts:
            return error_reply(403, f"host {host} is not this server")
        parts = urlsplit(target)
        if method == "POST" and parts.path == AUDIO_QUERY_PATH:
            params = parse_qs(parts.query, keep_blank_values=True)
            text = params.get("text", [""])[0]
            if len(text) > self._max_text_length:
                return error_reply(413, f"text is longer than {self._max_text_length} characters")
            if text.strip():
                return self._audio_query(text, params.get("speaker", [""])[0], headers.get(ORIGIN))
        return self._forward(method, target, headers, body)

    def _audio_query(self, text: str, speaker: str, origin: str | None) -> Reply:
        if origin is not None and origin not in self._allowed_origins:
            return error_reply(403, f"origin {origin} is not allowed to request corrections")
        try:
            speaker_id = int(speaker)
        except ValueError:
            return error_reply(422, "speaker must be an integer")
        with self._lock:
            try:
                query = self._correct(text, speaker_id)
            except RequestCapError as error:
                log(f"correction refused: {error}")
                return error_reply(503, str(error))
            except (TypeSafeError, httpx.HTTPError) as error:
                log(f"correction failed: {error}")
                return error_reply(502, str(error))
            except Exception as error:  # noqa: BLE001 - the client gets the error instead of a dropped connection
                log(traceback.format_exc())
                return error_reply(500, f"correction failed: {error!r}")
        allow = () if origin is None else (("Access-Control-Allow-Origin", origin),)
        return json_reply(200, query.to_payload(), allow)

    def _forward(self, method: str, target: str, headers: Mapping[str, str], body: bytes) -> Reply:
        if not target.startswith("/"):
            return error_reply(400, "the request target must be a path")
        forwarded = {name: value for name in FORWARDED_REQUEST_HEADERS if (value := headers.get(name)) is not None}
        try:
            response = self._upstream.request(method, f"{self._voicevox_url}{target}", headers=forwarded, content=body)
        except httpx.HTTPError as error:
            log(f"VOICEVOX request failed: {error}")
            return error_reply(502, f"VOICEVOX request failed: {error}")
        kept = tuple(
            (name, value) for name, value in response.headers.items() if name.lower() not in DROPPED_RESPONSE_HEADERS
        )
        return Reply(response.status_code, kept, response.content)


def body_length(header: str | None) -> int | Reply:
    """Read Content-Length, or the reply refusing it when it is not an integer or is too large."""
    try:
        length = int(header or 0)
    except ValueError:
        return error_reply(400, "Content-Length must be an integer")
    if length > MAX_BODY_BYTES:
        return error_reply(413, f"the body is larger than {MAX_BODY_BYTES} bytes")
    return length


class LimitedHTTPServer(ThreadingHTTPServer):
    """Serves at most `limit` connections at once and closes the ones past that."""

    def __init__(
        self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], limit: int = MAX_CONNECTIONS
    ) -> None:
        self._slots = threading.BoundedSemaphore(limit)
        super().__init__(address, handler)

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:  # type: ignore[override]
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: tuple[str, int]) -> None:  # type: ignore[override]
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def handler_for(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        timeout = CLIENT_TIMEOUT_SEC

        def do_GET(self) -> None:
            self._serve()

        def do_POST(self) -> None:
            self._serve()

        def do_PUT(self) -> None:
            self._serve()

        def do_DELETE(self) -> None:
            self._serve()

        def do_OPTIONS(self) -> None:
            self._serve()

        def _serve(self) -> None:
            length = body_length(self.headers.get("Content-Length"))
            if isinstance(length, Reply):
                # The body is left unread, so the connection cannot carry another request.
                self.close_connection = True
                self._send(length)
                return
            body = self.rfile.read(length) if length > 0 else b""
            headers = {name.lower(): value for name, value in self.headers.items()}
            self._send(app.handle(self.command, self.path, headers, body))

        def _send(self, reply: Reply) -> None:
            self.send_response(reply.status)
            for name, value in reply.headers:
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)

    return Handler


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class TypeSafeSession:
    """Every TypeSafe request of the process goes through one pacer and one connection pool."""

    url: str
    api_key: str
    pacer: Pacer
    client: httpx.Client

    def ask(self, request: SystemOneRequest) -> SystemOneResponse:
        self.pacer.wait()
        response = send(self.url, self.api_key, request, self.client)
        log(render_answers(response))
        return response


def live_correction(corrector: Corrector, client: VoicevoxClient, typesafe: TypeSafeSession) -> Correct:
    def correct(text: str, speaker: int) -> AudioQuery:
        log(f"===== speaker {speaker}: {text}")
        correction = corrector.correct(text, Voice(client, speaker), typesafe.ask)
        sent = typesafe.pacer.sent
        log(f"read as: {correction.text}\n{render_changes(correction.changes)}\nTypeSafe requests so far: {sent}")
        return correction.query

    return correct


def dry_correction(corrector: Corrector, client: VoicevoxClient, settings: Settings) -> Correct:
    def correct(text: str, speaker: int) -> AudioQuery:
        voice = Voice(client, speaker)
        log(f"===== speaker {speaker}: {text}")
        for request in corrector.requests(text, voice):
            log(render_request(settings.typesafe_url, request))
        log("[dry-run] requests not sent; VOICEVOX's own query is returned")
        return voice.read(text)

    return correct


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voicevox-jev-proxy",
        description="VOICEVOX-compatible server that corrects readings with Jev in at most one TypeSafe request per "
        "text, and intonation too with --intonation",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dry-run", action="store_true", help="log the TypeSafe requests instead of sending them")
    parser.add_argument(
        "--request-cap",
        type=int,
        default=DEFAULT_REQUEST_CAP,
        help="TypeSafe requests this process may send; /audio_query answers 503 after that",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=DEFAULT_REQUEST_INTERVAL_SEC,
        help="least seconds between the starts of two TypeSafe requests",
    )
    parser.add_argument(
        "--allow-origin", action="append", default=[], help="web page origin allowed to call /audio_query"
    )
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="Host header (name:port) that also names this server, such as its LAN address; the loopback names "
        "and the --host address are always allowed",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=DEFAULT_MAX_TEXT_LENGTH,
        help="longest text /audio_query corrects, in characters; longer ones get 413",
    )
    add_correction_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    check_correction_arguments(parser, args)
    if args.request_cap < 1:
        parser.error("--request-cap must be at least 1")
    if args.request_interval < 0:
        parser.error("--request-interval must not be negative")
    if args.max_text_length < 1:
        parser.error("--max-text-length must be at least 1")
    settings = Settings()
    api_key = settings.typesafe_api_key
    if api_key is None and not args.dry_run:
        print("TYPESAFE_API_KEY is not set", file=sys.stderr)
        return 1
    corrector = build_corrector(args, settings)
    client = VoicevoxClient(settings.voicevox_url)
    if api_key is None or args.dry_run:
        correct = dry_correction(corrector, client, settings)
    else:
        pacer = Pacer(args.request_interval, args.request_cap)
        typesafe = TypeSafeSession(settings.typesafe_url, api_key, pacer, httpx.Client(timeout=TYPESAFE_TIMEOUT_SEC))
        correct = live_correction(corrector, client, typesafe)
    app = App(
        correct,
        httpx.Client(timeout=UPSTREAM_TIMEOUT_SEC),
        settings.voicevox_url,
        frozenset(args.allow_origin),
        own_hosts(args.host, args.port, args.allow_host),
        max_text_length=args.max_text_length,
    )
    if not is_loopback(args.host):
        log(
            f"warning: listening on {args.host}, not loopback; anyone who can reach this port can spend the "
            "TypeSafe requests and use every VOICEVOX endpoint, since the server has no authentication. "
            "Clients on other machines also need their Host allowed with --allow-host."
        )
    server = LimitedHTTPServer((args.host, args.port), handler_for(app))
    pacing = f"at most {args.request_cap} TypeSafe requests, {args.request_interval}s apart"
    mode = "dry run" if args.dry_run else pacing
    print(
        f"voicevox-jev-proxy on http://{args.host}:{args.port} for VOICEVOX at {settings.voicevox_url} ({mode})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
