from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher

from pydantic import JsonValue

from voicevox_jev_proxy.typesafe import Answer, ChoiceAnswer
from voicevox_jev_proxy.voicevox import AccentPhrase, AudioQuery

ACCENT_FIELD = "accent"
MERGE_FIELD = "merge"
REPITCH_FIELDS = frozenset({ACCENT_FIELD, MERGE_FIELD})
SENTENCE_TYPE_KEY = "sentence_type"
UNIT_SUFFIX = "unit"

SENTENCE_TYPES: dict[str, JsonValue] = {
    "statement": "ふつうの言い切り",
    "question": "聞き手に答えを求める疑問",
    "confirmation": "聞き手に同意を求める念押し (〜よね、〜でしょ)",
    "trailing": "言い切らずに終わる言いさし (〜けど、〜て)",
    "exclamation": "驚きや感嘆",
}
UNIT_TYPES: dict[str, JsonValue] = {
    "independent": "独立した語で始まる。名詞・動詞・形容詞・副詞など、それ自体に内容がある語",
    "attached": "前の句に付く語で始まる。助詞・助動詞・補助動詞 (〜ている、〜てくる)・形式名詞 (こと、もの、の)",
    "fixed": "前の句と合わせて一つの決まり文句になる (かもしれない、なければならない、に違いない)",
}
MERGE_UNITS = frozenset({"attached", "fixed"})
RISING_TYPES = frozenset({"question", "confirmation"})


@dataclass(frozen=True)
class PhraseRoles:
    """What the text itself says about phrases, whatever Jev answers.

    attachable: phrases whose first word the morphology allows to join the phrase before it.
    laughs: phrases that only voice a laughter mark (草 read as くさ). They join no phrase and do not
    rise: in 無理ゲーでしょ|くさ the question rises on でしょ.
    """

    attachable: frozenset[int] = frozenset()
    laughs: frozenset[int] = frozenset()


NO_ROLES = PhraseRoles()


@dataclass(frozen=True)
class Policy:
    threshold: float = 0.6
    height_threshold: float = 0.8


@dataclass(frozen=True)
class Change:
    phrase_id: str
    field: str
    before: str
    after: str
    reason: str


@dataclass(frozen=True)
class Counterpart:
    """The phrase of an edited query that covers the words a phrase of the query before the edits did."""

    index: int
    same_reading: bool


def phrase_id(index: int) -> str:
    return f"p{index + 1}"


def unit_key(index: int) -> str:
    return f"{phrase_id(index)}_{UNIT_SUFFIX}"


def match_phrases(before: AudioQuery, after: AudioQuery) -> dict[int, Counterpart]:
    """Map each phrase of `before` to the phrase of `after` that covers the same words, where one does.

    A phrase read the same keeps its words. So does a lone phrase read anew between two such phrases (or
    the text's ends), since nothing else lies between them. Phrases the edits split, joined or
    regrouped have no counterpart.
    """
    old = [phrase.reading for phrase in before.accent_phrases]
    new = [phrase.reading for phrase in after.accent_phrases]
    matched: dict[int, Counterpart] = {}
    for tag, old_start, old_end, new_start, new_end in SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            for offset in range(old_end - old_start):
                matched[old_start + offset] = Counterpart(new_start + offset, same_reading=True)
        elif tag == "replace" and old_end - old_start == 1 and new_end - new_start == 1:
            matched[old_start] = Counterpart(new_start, same_reading=False)
    return matched


def build_state(sentence: str, query: AudioQuery) -> dict[str, JsonValue]:
    phrases: list[JsonValue] = [
        {"id": phrase_id(index), "reading": phrase.reading} for index, phrase in enumerate(query.accent_phrases)
    ]
    return {"sentence": sentence, "phrases": phrases}


def build_questions(query: AudioQuery) -> dict[str, JsonValue]:
    """Questions about standard intonation only: accent phrase units and the sentence type."""
    questions: dict[str, JsonValue] = {
        SENTENCE_TYPE_KEY: {
            "type": "choice",
            "instructions": "`sentence` を声に出すとき、文末はどの型か",
            "criteria": SENTENCE_TYPES,
        },
    }
    phrases = query.accent_phrases
    for index in range(len(phrases) - 1):
        questions[unit_key(index + 1)] = {
            "type": "choice",
            "instructions": {
                "target": f"`phrases[{index + 1}]` (「{phrases[index + 1].reading}」)",
                "previous": f"`phrases[{index}]` (「{phrases[index].reading}」)",
                "question": "`sentence` の中で、この句は前の句に対してどの関係か",
            },
            "criteria": UNIT_TYPES,
        }
    return questions


def apply_structure(
    query: AudioQuery,
    answers: Mapping[str, Answer],
    policy: Policy,
    roles: PhraseRoles = NO_ROLES,
) -> tuple[AudioQuery, list[Change]]:
    """Phrase-merge and interrogative edits.

    A phrase joins the one before it only when Jev calls it attached and `roles` allows it.
    VOICEVOX synthesizes from mora pitch, so the accent of a joined phrase only takes effect after
    /mora_pitch recomputes the pitches; the caller does that.
    """
    updated = query.model_copy(deep=True)
    changes, origins = _apply_merge(updated, answers, policy, roles)
    rising = [index for index, origin in enumerate(origins) if origin not in roles.laughs]
    changes += _apply_rise(updated, _choice(answers, SENTENCE_TYPE_KEY, policy), rising[-1] if rising else None)
    return updated, changes


def needs_repitch(changes: list[Change]) -> bool:
    return any(change.field in REPITCH_FIELDS for change in changes)


def _choice(answers: Mapping[str, Answer], key: str, policy: Policy) -> tuple[str, float] | None:
    answer = answers.get(key)
    if not isinstance(answer, ChoiceAnswer) or answer.confidence < policy.threshold:
        return None
    return answer.choice, answer.confidence


def joined_accent(left: AccentPhrase, right: AccentPhrase) -> int:
    """Nucleus of two phrases read as one unit: the first real nucleus wins, otherwise flat.

    VOICEVOX writes accent == mora count for flat phrases, so such a phrase carries no nucleus into
    the join even when the word itself is 尾高.
    """
    if left.accent < len(left.moras):
        return left.accent
    if right.accent < len(right.moras):
        return len(left.moras) + right.accent
    return len(left.moras) + len(right.moras)


def join_phrases(left: AccentPhrase, right: AccentPhrase) -> None:
    """Append `right` to `left` so VOICEVOX reads them as one accent phrase."""
    accent = joined_accent(left, right)
    left.moras += right.moras
    left.accent = accent
    left.pause_mora = right.pause_mora
    left.is_interrogative = right.is_interrogative


def _apply_merge(
    query: AudioQuery,
    answers: Mapping[str, Answer],
    policy: Policy,
    roles: PhraseRoles,
) -> tuple[list[Change], list[int]]:
    """Join attached phrases; also return, for each phrase left, the index it had before the join."""
    merged: list[AccentPhrase] = []
    origins: list[int] = []
    changes: list[Change] = []
    for index, phrase in enumerate(query.accent_phrases):
        unit = _choice(answers, unit_key(index), policy)
        joinable = (
            index in roles.attachable
            and index not in roles.laughs
            and bool(merged)
            and origins[-1] not in roles.laughs
            and merged[-1].pause_mora is None
        )
        if not joinable or unit is None or unit[0] not in MERGE_UNITS:
            merged.append(phrase)
            origins.append(index)
            continue
        left = merged[-1]
        before = f"{left.reading}({left.accent})+{phrase.reading}({phrase.accent})"
        join_phrases(left, phrase)
        reason = f"unit={unit[0]} confidence={unit[1]:.2f}"
        changes.append(
            Change(phrase_id(len(merged) - 1), MERGE_FIELD, before, f"{left.reading}({left.accent})", reason)
        )
    query.accent_phrases = merged
    return changes, origins


def _apply_rise(query: AudioQuery, sentence_type: tuple[str, float] | None, index: int | None) -> list[Change]:
    if sentence_type is None or sentence_type[0] not in RISING_TYPES or index is None:
        return []
    last = query.accent_phrases[index]
    if last.is_interrogative:
        return []
    last.is_interrogative = True
    pid = phrase_id(index)
    reason = f"{SENTENCE_TYPE_KEY}={sentence_type[0]} confidence={sentence_type[1]:.2f}"
    return [Change(pid, "is_interrogative", "False", "True", reason)]
