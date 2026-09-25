"""Chat laughter marks (www, 笑), which Jev tells apart from the same letters written as words.

草 is not taken for a mark: it stays as written, and VOICEVOX reads it クサ.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from voicevox_jev_proxy.edits import Edit
from voicevox_jev_proxy.prosody import Change
from voicevox_jev_proxy.readings import clause_around
from voicevox_jev_proxy.typesafe import Answer, ChoiceAnswer

LAUGH_FIELD = "laugh"
LAUGH_KEY = "laugh"
WORD_KEY = "word"
QUESTION = "`sentence` の中の `mark` は、笑っていることを表す記号か、語や文字として書かれたものか"
NOTE = "`clause` は `mark` を含む文節"
CRITERIA: dict[str, JsonValue] = {
    LAUGH_KEY: "笑っていることを表す、ネットでの書き方 (www、笑 など)",
    WORD_KEY: "笑いとは関係なく、語や文字として意味を持って書かれている",
}
LATIN = "A-Za-z\uff21-\uff3a\uff41-\uff5a"
KANJI = "一-龯々〆ヶ"
W_MARKS = "wW\uff57\uff37"
W_RUN = re.compile(rf"(?<![{LATIN}])[{W_MARKS}]+(?![{LATIN}])")
WARA = re.compile(rf"[（(]笑[）)]|(?<![{KANJI}])笑(?![ぁ-ゖ{KANJI}])")
PATTERNS = (W_RUN, WARA)


@dataclass(frozen=True)
class Laugh:
    """A mark in the text that may be a laugh."""

    start: int
    end: int
    surface: str

    @property
    def id(self) -> str:
        return f"l{self.start}"


def find_laughs(text: str) -> list[Laugh]:
    """Find the marks that may be laughs: not part of a word such as 笑顔 or Twitter.

    The letters alone do not tell a laugh from a word (W杯), so Jev decides.
    """
    found = [
        Laugh(match.start(), match.end(), match.group()) for pattern in PATTERNS for match in pattern.finditer(text)
    ]
    return sorted(found, key=lambda laugh: laugh.start)


def overlaps(start: int, end: int, laughs: Sequence[Laugh]) -> bool:
    return any(start < laugh.end and laugh.start < end for laugh in laughs)


def build_laugh_questions(text: str, laughs: Sequence[Laugh]) -> dict[str, JsonValue]:
    questions: dict[str, JsonValue] = {}
    for laugh in laughs:
        before, after = clause_around(text, laugh.start, laugh.end)
        questions[f"{laugh.id}_{LAUGH_FIELD}"] = {
            "type": "choice",
            "instructions": {
                "mark": laugh.surface,
                "clause": f"{before}{laugh.surface}{after}",
                "question": QUESTION,
                "note": NOTE,
            },
            "criteria": CRITERIA,
        }
    return questions


def laugh_edits(laughs: Sequence[Laugh], answers: Mapping[str, Answer], threshold: float) -> list[Edit]:
    """Remove each mark Jev takes for a laugh, which VOICEVOX would otherwise spell out (w as ダブリュウ).

    The threshold applies to P(laugh), Jev's belief that the mark is a laugh.
    """
    edits: list[Edit] = []
    for laugh in laughs:
        answer = answers.get(f"{laugh.id}_{LAUGH_FIELD}")
        if not isinstance(answer, ChoiceAnswer):
            continue
        laughing = answer.probabilities.get(LAUGH_KEY, 0.0)
        if laughing < threshold:
            continue
        reason = f"{LAUGH_FIELD} confidence={answer.confidence:.2f} p(laugh)={laughing:.2f}"
        change = Change(laugh.id, LAUGH_FIELD, laugh.surface, "", reason)
        edits.append(Edit(laugh.start, laugh.end, "", change, ""))
    return edits
