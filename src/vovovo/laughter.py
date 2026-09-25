"""Chat laughter marks (www, 笑, 草) and how to voice them, chosen by Jev per sentence."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from vovovo.accents import mora_total, phrase_starts
from vovovo.edits import Edit
from vovovo.prosody import Change
from vovovo.readings import clause_around, to_katakana
from vovovo.typesafe import Answer, ChoiceAnswer
from vovovo.voicevox import AudioQuery

LAUGH_FIELD = "laugh"
SILENT_KEY = "silent"
SILENT_TEXT = "（読まない）"
KEEP_TEXT = "どれも当てはまらない"
KEEP_KEY = "none"
QUESTION = "`sentence` を声に出して読み上げるとき、`mark` はどう読むか"
NOTE = "`context` は、`mark` をその読み方に置き換えた文節。読まない場合は `mark` を取り除いてある"
LATIN = "A-Za-z\uff21-\uff3a\uff41-\uff5a"
KANJI = "一-龯々〆ヶ"
W_MARKS = "wW\uff57\uff37"
W_RUN = re.compile(rf"(?<![{LATIN}])[{W_MARKS}]+(?![{LATIN}])")
WARA = re.compile(rf"[（(]笑[）)]|(?<![{KANJI}])笑(?![ぁ-ゖ{KANJI}])")
KUSA = re.compile(rf"(?<![{KANJI}])草(?![{KANJI}])")
READINGS: dict[str, dict[str, str]] = {
    "w": {SILENT_KEY: ""},
    "笑": {SILENT_KEY: ""},
    "草": {SILENT_KEY: "", "kusa": "くさ"},
}
PATTERNS = ((W_RUN, "w"), (WARA, "笑"), (KUSA, "草"))


@dataclass(frozen=True)
class Laugh:
    """A laughter mark in the text and the replacements it may be voiced as ("" reads nothing)."""

    start: int
    end: int
    surface: str
    readings: Mapping[str, str]

    @property
    def id(self) -> str:
        return f"l{self.start}"


def find_laughs(text: str) -> list[Laugh]:
    """Laughter marks that stand on their own: not part of a word such as 笑顔, 草原 or Twitter."""
    found = [
        Laugh(match.start(), match.end(), match.group(), READINGS[kind])
        for pattern, kind in PATTERNS
        for match in pattern.finditer(text)
    ]
    return sorted(found, key=lambda laugh: laugh.start)


def overlaps(start: int, end: int, laughs: Sequence[Laugh]) -> bool:
    return any(start < laugh.end and laugh.start < end for laugh in laughs)


def build_laugh_questions(text: str, laughs: Sequence[Laugh]) -> dict[str, JsonValue]:
    questions: dict[str, JsonValue] = {}
    for laugh in laughs:
        before, after = clause_around(text, laugh.start, laugh.end)
        criteria: dict[str, JsonValue] = {
            key: {"read_as": replacement or SILENT_TEXT, "context": f"{before}{replacement}{after}"}
            for key, replacement in laugh.readings.items()
        }
        criteria[KEEP_KEY] = KEEP_TEXT
        questions[f"{laugh.id}_{LAUGH_FIELD}"] = {
            "type": "choice",
            "instructions": {"mark": laugh.surface, "question": QUESTION, "note": NOTE},
            "criteria": criteria,
        }
    return questions


def laugh_edits(laughs: Sequence[Laugh], answers: Mapping[str, Answer], threshold: float) -> list[Edit]:
    """Replace each mark Jev reads as a laugh with the reading it chose.

    The threshold applies to Jev's belief that the mark is a laugh at all, 1 - P(none), and not to its
    confidence in one reading: a mark left as written is spelled out (w as ダブリュウ), which is worse
    than either reading Jev hesitated between.
    """
    edits: list[Edit] = []
    for laugh in laughs:
        answer = answers.get(f"{laugh.id}_{LAUGH_FIELD}")
        if not isinstance(answer, ChoiceAnswer):
            continue
        replacement = laugh.readings.get(answer.choice)
        laughing = 1 - answer.probabilities.get(KEEP_KEY, 0.0)
        if replacement is None or laughing < threshold:
            continue
        reason = f"{LAUGH_FIELD}={answer.choice} confidence={answer.confidence:.2f} p(laugh)={laughing:.2f}"
        change = Change(laugh.id, LAUGH_FIELD, laugh.surface, replacement, reason)
        edits.append(Edit(laugh.start, laugh.end, replacement, change, to_katakana(replacement)))
    return edits


def voiced_spans(edits: Sequence[Edit]) -> list[tuple[int, int]]:
    """Where the laugh readings stand in the text after `edits`, a sorted, non-overlapping list."""
    spans: list[tuple[int, int]] = []
    for edit in edits:
        if edit.change.field != LAUGH_FIELD or not edit.replacement:
            continue
        start = edit.start + sum(len(e.replacement) - (e.end - e.start) for e in edits if e.end <= edit.start)
        spans.append((start, start + len(edit.replacement)))
    return spans


def laugh_phrases(
    query: AudioQuery, text: str, spans: Sequence[tuple[int, int]], read: Callable[[str], AudioQuery]
) -> frozenset[int]:
    """Return the indices of the phrases whose every mora belongs to a laugh reading in `spans`."""
    covered: set[int] = set()
    for start, end in spans:
        covered.update(range(mora_total(read(text[:start])), mora_total(read(text[:end]))))
    return frozenset(
        index
        for index, (first, phrase) in enumerate(zip(phrase_starts(query), query.accent_phrases, strict=True))
        if phrase.moras and covered.issuperset(range(first, first + len(phrase.moras)))
    )
