import json
import threading
import time
from collections.abc import Callable
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, Field, JsonValue


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    probabilities: dict[int, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


Answer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage | None = None


class SystemOneRequest(BaseModel):
    state: JsonValue
    model: str
    questions: dict[str, JsonValue]


class TypeSafeError(RuntimeError):
    pass


class RequestCapError(RuntimeError):
    pass


class Pacer:
    """Keeps TypeSafe requests `interval` seconds apart, start to start, and refuses any past `cap`.

    One pacer is shared by every thread that sends, so the spacing holds for the host as a whole.
    """

    def __init__(
        self,
        interval: float,
        cap: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = interval
        self._cap = cap
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last: float | None = None
        self._sent = 0

    @property
    def sent(self) -> int:
        return self._sent

    def wait(self) -> None:
        """Return once the next request may go out; raise RequestCapError when the cap is used up."""
        with self._lock:
            if self._cap is not None and self._sent >= self._cap:
                msg = f"{self._sent} TypeSafe requests sent, the cap is {self._cap}"
                raise RequestCapError(msg)
            if self._last is not None:
                delay = self._interval - (self._clock() - self._last)
                if delay > 0:
                    self._sleep(delay)
            self._last = self._clock()
            self._sent += 1


def render_request(url: str, request: SystemOneRequest) -> str:
    body = json.dumps(request.model_dump(), ensure_ascii=False, indent=2)
    return "\n".join([f"POST {url}", "Authorization: Bearer ***", "Content-Type: application/json", "", body])


def send(url: str, api_key: str, request: SystemOneRequest, client: httpx.Client | None = None) -> SystemOneResponse:
    http = client or httpx.Client(timeout=30.0)
    response = http.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=request.model_dump(),
    )
    if response.status_code != httpx.codes.OK:
        msg = f"TypeSafe API returned {response.status_code}: {response.text}"
        raise TypeSafeError(msg)
    return SystemOneResponse.model_validate(response.json())
