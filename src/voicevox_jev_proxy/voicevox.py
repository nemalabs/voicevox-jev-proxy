import httpx
from pydantic import BaseModel, ConfigDict, Field


class Mora(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    consonant: str | None = None
    consonant_length: float | None = None
    vowel: str
    vowel_length: float
    pitch: float


class AccentPhrase(BaseModel):
    model_config = ConfigDict(extra="allow")

    moras: list[Mora]
    accent: int
    pause_mora: Mora | None = None
    is_interrogative: bool = False

    @property
    def reading(self) -> str:
        return "".join(mora.text for mora in self.moras)


class AudioQuery(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    accent_phrases: list[AccentPhrase]
    speed_scale: float = Field(alias="speedScale")
    pitch_scale: float = Field(alias="pitchScale")
    intonation_scale: float = Field(alias="intonationScale")
    volume_scale: float = Field(alias="volumeScale")
    pre_phoneme_length: float = Field(alias="prePhonemeLength")
    post_phoneme_length: float = Field(alias="postPhonemeLength")
    pause_length: float | None = Field(default=None, alias="pauseLength")
    pause_length_scale: float = Field(default=1.0, alias="pauseLengthScale")
    output_sampling_rate: int = Field(alias="outputSamplingRate")
    output_stereo: bool = Field(alias="outputStereo")
    kana: str | None = None

    def to_payload(self) -> dict[str, object]:
        return self.model_dump(by_alias=True)


class VoicevoxClient:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=60.0)

    def audio_query(self, text: str, speaker: int) -> AudioQuery:
        response = self._client.post(f"{self._base_url}/audio_query", params={"text": text, "speaker": speaker})
        response.raise_for_status()
        return AudioQuery.model_validate(response.json())

    def synthesis(self, query: AudioQuery, speaker: int) -> bytes:
        response = self._client.post(
            f"{self._base_url}/synthesis", params={"speaker": speaker}, json=query.to_payload()
        )
        response.raise_for_status()
        return response.content

    def mora_pitch(self, accent_phrases: list[AccentPhrase], speaker: int) -> list[AccentPhrase]:
        return self._recompute("mora_pitch", accent_phrases, speaker)

    def mora_data(self, accent_phrases: list[AccentPhrase], speaker: int) -> list[AccentPhrase]:
        """Recompute both the lengths and the pitches of the morae."""
        return self._recompute("mora_data", accent_phrases, speaker)

    def _recompute(self, path: str, accent_phrases: list[AccentPhrase], speaker: int) -> list[AccentPhrase]:
        response = self._client.post(
            f"{self._base_url}/{path}",
            params={"speaker": speaker},
            json=[phrase.model_dump() for phrase in accent_phrases],
        )
        response.raise_for_status()
        return [AccentPhrase.model_validate(item) for item in response.json()]
