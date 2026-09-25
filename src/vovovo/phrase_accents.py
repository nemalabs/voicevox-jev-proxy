"""Accent nuclei of noun phrases by UniDic's accent combination rules, for Jev to choose from.

A phrase of a noun and the particles and auxiliaries after it gets its accent type from the noun's
aType and each attached word's aConType, by table 12 of the UniDic 1.3.9 manual (「アクセント結合型分類
（助詞・助動詞）」), with N the morae so far, A their accent type (0 is flat) and M, L the numbers after @:

    F1: A    F2@M: N+M if A is 0, else A    F3@M: A if A is 0, else N+M
    F4@M: N+M    F5: 0    F6@M,L: N+M if A is 0, else N+L
"""

import re
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import fugashi
from pydantic import JsonValue

from vovovo.accents import mora_total, phrase_starts
from vovovo.edits import spoken
from vovovo.prosody import ACCENT_FIELD, Change, Policy, phrase_id
from vovovo.readings import accent_types, mora_count
from vovovo.typesafe import Answer, ChoiceAnswer
from vovovo.voicevox import AccentPhrase, AudioQuery

NONE_KEY = "none"
NOUN_RULES = "名詞"
RULE = re.compile(r"(名詞|動詞|形容詞)%F(\d)(?:@(-?\d+)(?:,(-?\d+))?)?")
HEAD_POS = frozenset({"名詞", "代名詞"})
ATTACHED_POS = frozenset({"助詞", "助動詞"})
COMPOUND_POS = frozenset({"名詞", "代名詞", "接頭辞"})
SILENT_POS = frozenset({"補助記号", "空白"})
NUMERAL = "数詞"
AUXILIARY_STEM = "助動詞語幹"
DROP = "\N{FULLWIDTH REVERSE SOLIDUS}"
HIGH, LOW = "高", "低"
QUESTION = "`word` (読み `reading`) を東京の共通語で言うとき、アクセント辞典に載る型はどれか"
NOTE = (
    "[ ] の数字は、声が下がる直前の拍の位置。0 は下がり目が無い。"
    f"`word` の {DROP} は下がり目、`pitch` は語の各拍と、あとに付く助詞 1 拍 (括弧内) の高低。"
    "`phrase` はこの文でその語を含む句"
)
NONE_TEXT = "どれも当てはまらない"


@dataclass(frozen=True)
class Token:
    """A UniDic word: where it starts in the text, its pronunciation and accent attributes."""

    start: int
    surface: str
    pos: tuple[str, str]
    pron: str
    accents: tuple[int, ...]
    combination: str

    @property
    def end(self) -> int:
        return self.start + len(self.surface)


class UnidicTokenizer:
    """Words of a text with their UniDic aType and aConType; punctuation and spaces are left out.

    A word UniDic does not know keeps no pronunciation or accent, so no phrase with it is asked about.
    """

    def __init__(self) -> None:
        self._tagger = fugashi.Tagger()

    def tokens(self, text: str) -> list[Token]:
        found: list[Token] = []
        cursor = 0
        for node in self._tagger(text):
            start = text.index(node.surface, cursor)
            cursor = start + len(node.surface)
            feature = node.feature
            pos = (feature.pos1, feature.pos2)
            if feature.pos1 in SILENT_POS:
                continue
            if node.is_unk:
                found.append(Token(start, node.surface, pos, "", (), ""))
            else:
                accents = accent_types(feature.aType)
                found.append(Token(start, node.surface, pos, feature.pron, accents, feature.aConType))
        return found


def combine(accent: int, moras: int, rules: str) -> int | None:  # noqa: PLR0911
    """Accent type once a particle or auxiliary with the aConType `rules` follows `moras` morae of type `accent`.

    The rule for a noun before the word applies, also after other particles (兵十|は|だけ). None when the
    word has no such rule or the rule puts the nucleus before the phrase.
    """
    match = next((found for found in RULE.finditer(rules) if found.group(1) == NOUN_RULES), None)
    if match is None:
        return None
    first, second = match.group(3), match.group(4)
    match match.group(2):
        case "1":
            return accent
        case "2":
            return _shifted(moras, first) if accent == 0 else accent
        case "3":
            return accent if accent == 0 else _shifted(moras, first)
        case "4":
            return _shifted(moras, first)
        case "5":
            return 0
        case "6":
            return _shifted(moras, first) if accent == 0 else _shifted(moras, second)
    return None


def _shifted(moras: int, shift: str | None) -> int | None:
    """N+M of the table; None when the rule gives no M or the nucleus falls before the phrase."""
    if shift is None:
        return None
    nucleus = moras + int(shift)
    return nucleus if nucleus > 0 else None


def phrase_type(word_type: int, head: Token, attached: Sequence[Token]) -> int | None:
    """Accent type (0 is flat) of the phrase when its noun has `word_type`, by the rules of the attached words."""
    accent, moras = word_type, mora_count(head.pron)
    for token in attached:
        combined = combine(accent, moras, token.combination)
        if combined is None:
            return None
        accent = combined
        moras += mora_count(token.pron)
    return accent


@dataclass(frozen=True)
class AccentOption:
    """A dictionary accent type of the noun and the VOICEVOX accent of the phrase it gives.

    VOICEVOX writes the nucleus mora, or the mora count for a flat phrase.
    """

    word_type: int
    nucleus: int


@dataclass(frozen=True)
class AccentQuestion:
    """A phrase whose VOICEVOX nucleus none of the noun's UniDic accent types give.

    reading holds the noun's morae as VOICEVOX reads them. options maps a label such as 頭高型[1] to
    each accent type UniDic lists for the noun and each type that gives VOICEVOX's nucleus.
    """

    index: int
    words: tuple[str, ...]
    reading: tuple[str, ...]
    options: Mapping[str, AccentOption]


def marked(phrase: AccentPhrase, accent: int) -> str:
    moras = [mora.text for mora in phrase.moras]
    if accent >= len(moras):
        return "".join(moras)
    return "".join(moras[:accent]) + DROP + "".join(moras[accent:])


def pitch(moras: int, accent: int) -> str:
    """Tokyo-dialect high and low of each mora: after a first low mora the pitch stays high up to the nucleus."""
    return "".join(
        HIGH if (accent == 1 if position == 1 else position <= accent) else LOW for position in range(1, moras + 1)
    )


def type_label(word_type: int, moras: int) -> str:
    """Name an accent type the dictionary way, with its number: 平板型[0], 頭高型[1], 中高型[2], 尾高型[n]."""
    if word_type == 0:
        name = "平板型"
    elif word_type == 1:
        name = "頭高型"
    elif word_type == moras:
        name = "尾高型"
    else:
        name = "中高型"
    return f"{name}[{word_type}]"


def word_marked(reading: Sequence[str], word_type: int) -> str:
    """Return the word with a drop mark after its nucleus, which is its last mora for 尾高型."""
    return "".join(reading[:word_type]) + (DROP if word_type else "") + "".join(reading[word_type:])


def word_pitch(moras: int, word_type: int) -> str:
    """High and low of the word's morae and, in brackets, of one particle after it (尾高型 低高(低))."""
    levels = pitch(moras + 1, word_type or moras + 1)
    return f"{levels[:-1]}({levels[-1]})"


@dataclass(frozen=True)
class Settled:
    """What other answers already decide: phrase indices, and text spans an edit wrote."""

    phrases: frozenset[int] = frozenset()
    spans: Sequence[tuple[int, int]] = ()


NOTHING_SETTLED = Settled()


def accent_questions(
    query: AudioQuery,
    text: str,
    tokens: Sequence[Token],
    read: Callable[[str], AudioQuery],
    settled: Settled = NOTHING_SETTLED,
) -> list[AccentQuestion]:
    """Phrases of a noun and its particles whose VOICEVOX accent differs from every rule result.

    The noun must start the phrase, found by reading the text before it and counting morae, and the
    pronunciations of the words must spell the phrase. A noun right after another noun or a prefix
    is part of a compound, whose rules this does not cover, and the stem of an auxiliary (そう of
    降るそうです) follows a verb. Settled phrases and words that overlap a settled span are left alone.
    """
    starts = phrase_starts(query)
    found: list[AccentQuestion] = []
    for position, head in enumerate(tokens):
        if head.pos[0] not in HEAD_POS or head.pos[1] in {NUMERAL, AUXILIARY_STEM}:
            continue
        previous = tokens[position - 1] if position else None
        if previous is not None and previous.end == head.start and previous.pos[0] in COMPOUND_POS:
            continue
        begin = mora_total(read(text[: head.start])) if head.start else 0
        index = bisect_right(starts, begin) - 1
        if index < 0 or starts[index] != begin or index in settled.phrases:
            continue
        phrase = query.accent_phrases[index]
        words = _filling(tokens[position:], len(phrase.moras))
        if any(_overlaps(token, settled.spans) for token in words):
            continue
        question = _question(phrase, index, words)
        if question is not None:
            found.append(question)
    return found


def _filling(tokens: Sequence[Token], size: int) -> list[Token]:
    """Return the words from tokens[0] on whose pronunciations reach `size` morae."""
    words: list[Token] = []
    moras = 0
    for token in tokens:
        if moras >= size:
            break
        words.append(token)
        moras += mora_count(token.pron)
    return words


def _question(phrase: AccentPhrase, index: int, words: Sequence[Token]) -> AccentQuestion | None:
    head, attached = words[0], words[1:]
    if any(token.pos[0] not in ATTACHED_POS for token in attached):
        return None
    if spoken("".join(token.pron for token in words)) != spoken(phrase.reading):
        return None
    size, word_moras = len(phrase.moras), mora_count(head.pron)
    nuclei: dict[int, int] = {}
    for word_type in range(word_moras + 1):
        accent = phrase_type(word_type, head, attached)
        if accent is not None and accent <= size:
            nuclei[word_type] = accent or size
    dictionary = set(head.accents)
    if not dictionary or not dictionary <= nuclei.keys():
        return None
    if phrase.accent in {nuclei[word_type] for word_type in dictionary}:
        return None
    voicevox = {word_type for word_type, nucleus in nuclei.items() if nucleus == phrase.accent}
    if not voicevox:
        return None
    options = {
        type_label(word_type, word_moras): AccentOption(word_type, nuclei[word_type])
        for word_type in sorted(dictionary | voicevox)
    }
    reading = tuple(mora.text for mora in phrase.moras[:word_moras])
    return AccentQuestion(index, tuple(token.surface for token in words), reading, options)


def _overlaps(token: Token, spans: Sequence[tuple[int, int]]) -> bool:
    return any(token.start < end and start < token.end for start, end in spans)


def build_accent_questions(query: AudioQuery, questions: Sequence[AccentQuestion]) -> dict[str, JsonValue]:
    built: dict[str, JsonValue] = {}
    for question in questions:
        phrase = query.accent_phrases[question.index]
        moras = len(question.reading)
        criteria: dict[str, JsonValue] = {
            key: {
                "word": word_marked(question.reading, option.word_type),
                "pitch": word_pitch(moras, option.word_type),
            }
            for key, option in question.options.items()
        }
        criteria[NONE_KEY] = NONE_TEXT
        built[f"{phrase_id(question.index)}_{ACCENT_FIELD}"] = {
            "type": "choice",
            "instructions": {
                "word": question.words[0],
                "reading": "".join(question.reading),
                "phrase": f"`phrases[{question.index}]` (「{phrase.reading}」)",
                "question": QUESTION,
                "note": NOTE,
            },
            "criteria": criteria,
        }
    return built


def apply_accent_answers(
    query: AudioQuery, answers: Mapping[str, Answer], questions: Sequence[AccentQuestion], policy: Policy
) -> tuple[AudioQuery, list[Change]]:
    """Set the nucleus Jev chose; the caller recomputes the pitches with /mora_pitch."""
    updated = query.model_copy(deep=True)
    changes: list[Change] = []
    for question in questions:
        pid = phrase_id(question.index)
        answer = answers.get(f"{pid}_{ACCENT_FIELD}")
        if not isinstance(answer, ChoiceAnswer) or answer.confidence < policy.threshold:
            continue
        option = question.options.get(answer.choice)
        phrase = updated.accent_phrases[question.index]
        if option is None or option.nucleus == phrase.accent:
            continue
        before = marked(phrase, phrase.accent)
        phrase.accent = option.nucleus
        reason = f"{ACCENT_FIELD}={answer.choice} confidence={answer.confidence:.2f}"
        changes.append(Change(pid, ACCENT_FIELD, before, marked(phrase, option.nucleus), reason))
    return updated, changes
