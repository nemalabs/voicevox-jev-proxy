"""Heights of accent phrases: where the next phrase rises again, and where VOICEVOX hides a fall.

After a phrase that does not fall, VOICEVOX often raises the next phrase above it. Jev says what the
first phrase modifies; when it modifies the next phrase (首へ|まきつきました), the next phrase is lowered
so its peak does not pass the first's. Only confident answers are used (`Policy.height_threshold`).

VOICEVOX puts the pitch peak one mora after the nucleus (ココロワ with accent 2 peaks on ロ). A phrase
whose nucleus is its second-to-last mora therefore peaks on its last mora, and a phrase right after it
hides the fall (ジツニ with accent 2, then ボクワ). Moving the nucleus one mora earlier lets the fall be
heard inside the phrase.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from voicevox_jev_proxy.accents import mora_total, phrase_starts
from voicevox_jev_proxy.edits import spoken
from voicevox_jev_proxy.phrase_accents import Token
from voicevox_jev_proxy.prosody import ACCENT_FIELD, Change, phrase_id
from voicevox_jev_proxy.typesafe import Answer, ChoiceAnswer
from voicevox_jev_proxy.voicevox import AccentPhrase, AudioQuery

HEIGHT_FIELD = "height"
HEIGHT_SKIPPED_FIELD = f"{HEIGHT_FIELD}_skipped"
HEAD_SUFFIX = "head"
MIN_RISE = 0.1
SECOND_MORA = 2
FOLLOWING_KEY = "following"
HEAD_QUESTION = "`sentence` の中で、`phrase` はどの語句にかかるか"
HEAD_NOTE = (
    "かかるとは、係り受けで `phrase` がその語句を修飾すること、またはその語句の主語・目的語などになること。"
    "`following` は `phrase` のすぐ後の語句"
)
HEAD_CRITERIA: dict[str, JsonValue] = {
    FOLLOWING_KEY: "すぐ後の `following` にかかる",
    "beyond": "`following` より後の語句にかかる",
    "none": "後ろのどの語句にもかからない",
}
LATE_PEAK = "VOICEVOX peaks one mora after the nucleus, so the next phrase would hide this fall"


@dataclass(frozen=True)
class Height:
    """Two phrases with no pause between them, the first flat and the second rising above it.

    index is the first phrase's place in the query asked about. start, boundary and end are mora
    offsets: where the first phrase starts, where the second starts and where it ends. They find the
    pair again after other phrases were joined.
    """

    index: int
    start: int
    boundary: int
    end: int
    phrase: str
    following: str

    @property
    def key(self) -> str:
        return f"{phrase_id(self.index)}_{HEAD_SUFFIX}"


def peak(phrase: AccentPhrase) -> float:
    return max(mora.pitch for mora in phrase.moras)


def falls(phrase: AccentPhrase) -> bool:
    """VOICEVOX writes a flat phrase as accent == mora count."""
    return phrase.accent < len(phrase.moras)


def find_heights(
    text: str, query: AudioQuery, tokens: Sequence[Token], read: Callable[[str], AudioQuery]
) -> list[Height]:
    """Pairs whose second phrase rises more than MIN_RISE above a flat first phrase.

    Both phrases must be spelled by whole words, found by reading the text before each word and
    counting morae, so the question can name them.
    """
    phrases = query.accent_phrases
    rising = [index for index in range(1, len(phrases)) if _rises(phrases[index - 1], phrases[index])]
    if not rising:
        return []
    marks = {token.start: mora_total(read(text[: token.start])) if token.start else 0 for token in tokens}
    words = _phrase_words(query, tokens, marks)
    starts = phrase_starts(query)
    found: list[Height] = []
    for index in rising:
        before, after = words[index - 1], words[index]
        if before is None or after is None:
            continue
        end = starts[index] + len(phrases[index].moras)
        found.append(
            Height(index - 1, starts[index - 1], starts[index], end, _surface(text, before), _surface(text, after))
        )
    return found


def build_height_questions(heights: Sequence[Height]) -> dict[str, JsonValue]:
    return {
        height.key: {
            "type": "choice",
            "instructions": {
                "phrase": height.phrase,
                "following": height.following,
                "question": HEAD_QUESTION,
                "note": HEAD_NOTE,
            },
            "criteria": HEAD_CRITERIA,
        }
        for height in heights
    }


def lower_heights(
    query: AudioQuery, heights: Sequence[Height], answers: Mapping[str, Answer], threshold: float
) -> tuple[AudioQuery, list[Change]]:
    """Lower the second phrase of each pair Jev says belongs together, down to the first phrase's peak.

    Pitches are moved directly, so this comes after the last /mora_pitch. Voiceless morae (pitch 0)
    stay as they are.
    """
    updated = query.model_copy(deep=True)
    changes: list[Change] = []
    starts = phrase_starts(updated)
    ends = [start + len(phrase.moras) for start, phrase in zip(starts, updated.accent_phrases, strict=True)]
    for height in heights:
        answer = answers.get(height.key)
        if not isinstance(answer, ChoiceAnswer) or answer.choice != FOLLOWING_KEY or answer.confidence < threshold:
            continue
        index = _pair(starts, ends, height)
        pid = phrase_id(height.index + 1)
        if index is None:
            changes.append(Change(pid, HEIGHT_SKIPPED_FIELD, "", "", "the two phrases were joined with others"))
            continue
        left, right = updated.accent_phrases[index], updated.accent_phrases[index + 1]
        excess = peak(right) - peak(left)
        if falls(left) or excess <= 0:
            changes.append(Change(pid, HEIGHT_SKIPPED_FIELD, "", "", "the second phrase no longer rises"))
            continue
        before = peak(right)
        for mora in right.moras:
            if mora.pitch > 0:
                mora.pitch -= excess
        reason = f"{HEAD_SUFFIX}={answer.choice} confidence={answer.confidence:.2f}"
        changes.append(Change(pid, HEIGHT_FIELD, f"{before:.2f}", f"{peak(right):.2f}", reason))
    return updated, changes


def move_late_nuclei(query: AudioQuery) -> tuple[AudioQuery, list[Change]]:
    """Move a nucleus on the second-to-last mora one mora earlier when another phrase follows at once.

    A nucleus on the first mora cannot move. The caller recomputes the pitches with /mora_pitch.
    """
    updated = query.model_copy(deep=True)
    changes: list[Change] = []
    last = len(updated.accent_phrases) - 1
    for index, phrase in enumerate(updated.accent_phrases):
        if index == last or phrase.pause_mora is not None:
            continue
        if phrase.accent != len(phrase.moras) - 1 or phrase.accent < SECOND_MORA:
            continue
        phrase.accent -= 1
        changes.append(Change(phrase_id(index), ACCENT_FIELD, str(phrase.accent + 1), str(phrase.accent), LATE_PEAK))
    return updated, changes


def _rises(left: AccentPhrase, right: AccentPhrase) -> bool:
    return left.pause_mora is None and not falls(left) and peak(right) - peak(left) > MIN_RISE


def _phrase_words(query: AudioQuery, tokens: Sequence[Token], marks: Mapping[int, int]) -> list[list[Token] | None]:
    """Return the words of each phrase, or None where their pronunciations do not spell the phrase."""
    bounds = [*phrase_starts(query), mora_total(query)]
    found: list[list[Token] | None] = []
    for phrase, begin, end in zip(query.accent_phrases, bounds, bounds[1:], strict=False):
        inside = [token for token in tokens if begin <= marks[token.start] < end]
        aligned = bool(inside) and marks[inside[0].start] == begin
        if aligned and spoken("".join(token.pron for token in inside)) == spoken(phrase.reading):
            found.append(inside)
        else:
            found.append(None)
    return found


def _surface(text: str, words: Sequence[Token]) -> str:
    return text[words[0].start : words[-1].end]


def _pair(starts: Sequence[int], ends: Sequence[int], height: Height) -> int | None:
    if height.start not in starts:
        return None
    index = starts.index(height.start)
    if index + 1 >= len(starts) or starts[index + 1] != height.boundary or ends[index + 1] != height.end:
        return None
    return index
