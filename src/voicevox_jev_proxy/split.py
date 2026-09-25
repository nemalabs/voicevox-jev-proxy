import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from voicevox_jev_proxy.accents import DEPENDENT, Word, attaches, mora_total, phrase_starts
from voicevox_jev_proxy.edits import SEPARATOR, Edit, Rephrase, shifted
from voicevox_jev_proxy.prosody import MERGE_FIELD, Change, join_phrases
from voicevox_jev_proxy.readings import CONTENT_POS, NUMERAL
from voicevox_jev_proxy.typesafe import Answer, ChoiceAnswer
from voicevox_jev_proxy.voicevox import AudioQuery

WHOLE_KEY = "whole"
NONE_KEY = "none"
SPLIT_FIELD = "split"
BOUNDARY_FIELD = "boundary"
SPAN_PATTERN = re.compile(r"[一-龯々〆ヶ]+[ぁ-ん]{2,}")
SILENT_POS = frozenset({"補助記号", "空白"})
NOUN_POS = frozenset({"名詞", "代名詞"})
PREFIX_POS = "接頭辞"
AUXILIARY_STEM = "助動詞語幹"
DEPENDENT_ADJECTIVE = ("形容詞", DEPENDENT)
BAR = " | "


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def id(self) -> str:
        return f"s{self.start}"


def find_spans(text: str) -> list[Span]:
    return [Span(match.start(), match.end()) for match in SPAN_PATTERN.finditer(text)]


def split_key(offset: int) -> str:
    return f"split_{offset}"


def build_split_questions(text: str, spans: list[Span]) -> dict[str, JsonValue]:
    questions: dict[str, JsonValue] = {}
    for span in spans:
        surface = text[span.start : span.end]
        criteria: dict[str, JsonValue] = {WHOLE_KEY: surface}
        for offset in range(1, len(surface)):
            criteria[split_key(offset)] = f"{surface[:offset]} | {surface[offset:]}"
        criteria[NONE_KEY] = "どの切り方も正しくない"
        questions[f"{span.id}_split"] = {
            "type": "choice",
            "instructions": {
                "span": surface,
                "question": "`sentence` の中の `span` を語に分けるとき、正しい切れ目はどれか",
                "note": f"| が語の切れ目。切れ目が無い一語なら {WHOLE_KEY}",
            },
            "criteria": criteria,
        }
    return questions


def choose_splits(text: str, spans: list[Span], answers: Mapping[str, Answer], threshold: float) -> list[Edit]:
    edits: list[Edit] = []
    for span in spans:
        answer = answers.get(f"{span.id}_split")
        if not isinstance(answer, ChoiceAnswer) or answer.confidence < threshold:
            continue
        if answer.choice in (WHOLE_KEY, NONE_KEY) or not answer.choice.startswith("split_"):
            continue
        offset = int(answer.choice.removeprefix("split_"))
        surface = text[span.start : span.end]
        after = f"{surface[:offset]}{SEPARATOR}{surface[offset:]}"
        change = Change(span.id, SPLIT_FIELD, surface, after, f"split confidence={answer.confidence:.2f}")
        position = span.start + offset
        edits.append(Edit(position, position, SEPARATOR, change))
    return edits


@dataclass(frozen=True)
class Stretch:
    """Text around an accent phrase that cuts across the 文節 of its words.

    Offsets count from the start of the stretch. current is where VOICEVOX starts the phrase, None
    when the phrase starts with the stretch; cuts are the 文節 starts the phrase runs across.
    """

    start: int
    end: int
    current: int | None
    cuts: tuple[int, ...]

    @property
    def id(self) -> str:
        return f"b{self.start}"


@dataclass(frozen=True)
class _Layout:
    """The spoken words of a text, the mora each starts at, the word before each, the 文節 starts, and the marks."""

    words: list[Word]
    marks: dict[int, int]
    previous: dict[int, Word | None]
    heads: list[Word]
    silent: list[int]


def find_stretches(
    text: str, query: AudioQuery, words: Sequence[Word], read: Callable[[str], AudioQuery]
) -> list[Stretch]:
    """Stretches where VOICEVOX's accent phrases cut across the 文節 of the words, for Jev to judge.

    Two kinds are found. A phrase that starts inside a 文節, at a particle or mid-word, and runs into
    the next one (日もごんは read ヒ|モゴンワ, 見ていたごんは read ミテ|イ|タゴンワ). And a phrase that runs
    one noun into the next (そのとき兵十は read ソノ|トキヘエジュウワ), which only the meaning tells from
    a compound (秋祭). Other 文節 VOICEVOX joins (小さくなって, 坂の上で) are left alone.
    Word positions are found by reading the text before each word and counting morae.
    """
    layout = _layout(text, words, read)
    found: list[Stretch] = []
    for begin, phrase in zip(phrase_starts(query), query.accent_phrases, strict=True):
        stretch = _stretch(text, layout, begin, begin + len(phrase.moras), read)
        if stretch is not None:
            found.append(stretch)
    return found


def _layout(text: str, words: Sequence[Word], read: Callable[[str], AudioQuery]) -> _Layout:
    spoken = [word for word in words if word.pos[0] not in SILENT_POS]
    previous = dict(zip((word.start for word in spoken), [None, *spoken], strict=False))
    marks = {word.start: mora_total(read(text[: word.start])) for word in spoken}
    heads = [word for word in spoken if _opens(previous[word.start], word)]
    return _Layout(spoken, marks, previous, heads, [word.start for word in words if word.pos[0] in SILENT_POS])


def _opens(previous: Word | None, word: Word) -> bool:
    """Whether the word starts a 文節.

    A prefix does unless it follows another. An independent word does unless it follows a prefix or a
    numeral (a counter), attaches to the word before, or is an auxiliary stem (みたい) or the
    dependent adjective ない (じゃない, ではなく).
    """
    if word.pos[0] == PREFIX_POS:
        return previous is None or previous.pos[0] != PREFIX_POS
    if word.pos[0] not in CONTENT_POS or word.pos[1] == AUXILIARY_STEM or word.pos[:2] == DEPENDENT_ADJECTIVE:
        return False
    if previous is None:
        return True
    return previous.pos[0] != PREFIX_POS and previous.pos[1] != NUMERAL and not attaches(previous, word)


def _stretch(text: str, layout: _Layout, begin: int, end: int, read: Callable[[str], AudioQuery]) -> Stretch | None:
    inside = [word for word in layout.heads if begin < layout.marks[word.start] < end]
    opening = [word for word in layout.heads if layout.marks[word.start] <= begin]
    if not inside or not opening:
        return None
    head = opening[-1]
    last = [word for word in layout.words if layout.marks[word.start] < end][-1]
    stop = last.start + len(last.surface)
    if any(head.start <= position < stop for position in layout.silent):
        return None
    if layout.marks[head.start] == begin:
        cuts = tuple(word.start - head.start for word in inside if _nouns(layout.previous[word.start], word))
        return Stretch(head.start, stop, None, cuts) if cuts else None
    current = next(
        (position for position in range(head.start + 1, inside[0].start) if mora_total(read(text[:position])) == begin),
        None,
    )
    if current is None:
        return None
    return Stretch(head.start, stop, current - head.start, tuple(word.start - head.start for word in inside))


def _nouns(previous: Word | None, word: Word) -> bool:
    return (
        previous is not None
        and previous.pos[0] in NOUN_POS
        and word.pos[0] in NOUN_POS
        and NUMERAL not in (previous.pos[1], word.pos[1])
    )


def boundary_key(offset: int | None) -> str:
    return WHOLE_KEY if offset is None else f"cut_{offset}"


def marked(surface: str, offset: int | None) -> str:
    return surface if offset is None else f"{surface[:offset]}{BAR}{surface[offset:]}"


def build_boundary_questions(text: str, stretches: list[Stretch]) -> dict[str, JsonValue]:
    """Ask which of VOICEVOX's phrasing and the 文節 starts it runs across is right, VOICEVOX's first."""
    questions: dict[str, JsonValue] = {}
    for stretch in stretches:
        surface = text[stretch.start : stretch.end]
        criteria: dict[str, JsonValue] = {
            boundary_key(offset): marked(surface, offset) for offset in (stretch.current, *stretch.cuts)
        }
        criteria[NONE_KEY] = "どの区切り方も正しくない"
        questions[f"{stretch.id}_boundary"] = {
            "type": "choice",
            "instructions": {
                "span": surface,
                "question": "`sentence` を声に出して読むとき、`span` の区切り方として正しいのはどれか",
                "note": (
                    "| は文節の切れ目。文節は、名詞・動詞などの自立語と、その後に付く助詞・助動詞のまとまり。"
                    "| の無い候補は全体で一つの文節"
                ),
            },
            "criteria": criteria,
        }
    return questions


def choose_boundaries(
    text: str, stretches: list[Stretch], answers: Mapping[str, Answer], threshold: float
) -> list[Edit]:
    """Separators at the 文節 starts Jev picks; VOICEVOX's own phrasing and none change nothing."""
    edits: list[Edit] = []
    for stretch in stretches:
        answer = answers.get(f"{stretch.id}_boundary")
        if not isinstance(answer, ChoiceAnswer) or answer.confidence < threshold:
            continue
        cut = next((offset for offset in stretch.cuts if boundary_key(offset) == answer.choice), None)
        if cut is None:
            continue
        surface = text[stretch.start : stretch.end]
        reason = f"boundary confidence={answer.confidence:.2f}"
        change = Change(stretch.id, BOUNDARY_FIELD, marked(surface, stretch.current), marked(surface, cut), reason)
        position = stretch.start + cut
        moves_from = None if stretch.current is None else stretch.start + stretch.current
        rephrase = Rephrase(stretch.start, stretch.end, moves_from)
        edits.append(Edit(position, position, SEPARATOR, change, rephrase=rephrase))
    return edits


def join_moved_boundaries(
    query: AudioQuery, text: str, edits: list[Edit], read: Callable[[str], AudioQuery]
) -> tuple[AudioQuery, list[Change]]:
    """Join the phrases VOICEVOX still breaks where a phrasing separator moved the boundary away from.

    VOICEVOX reads そのまま、横っとびに as ソノ|ママ|ヨコットビニ, keeping the break Jev put after まま.
    `text` is the text after the edits.
    """
    updated = query.model_copy(deep=True)
    changes: list[Change] = []
    for edit in edits:
        moves_from = None if edit.rephrase is None else edit.rephrase.moves_from
        position = None if moves_from is None else shifted(moves_from, edits)
        if position is None:
            continue
        starts = phrase_starts(updated)
        begin = mora_total(read(text[:position]))
        index = starts.index(begin) if begin in starts else 0
        if index == 0 or updated.accent_phrases[index - 1].pause_mora is not None:
            continue
        left, right = updated.accent_phrases[index - 1], updated.accent_phrases[index]
        before = f"{left.reading}({left.accent})+{right.reading}({right.accent})"
        join_phrases(left, right)
        del updated.accent_phrases[index]
        after = f"{left.reading}({left.accent})"
        changes.append(Change(edit.change.phrase_id, MERGE_FIELD, before, after, "VOICEVOX kept the moved break"))
    return updated, changes


def strip_inserted_pauses(
    query: AudioQuery, text: str, inserted: list[int], read: Callable[[str], AudioQuery]
) -> tuple[AudioQuery, list[int]]:
    """Remove the pauses VOICEVOX puts at the separators we inserted.

    Each separator is found by reading the text before it and counting morae, since VOICEVOX's own
    rules for which marks pause (runs of ！？, ・, ♪, a leading 、) are not ours to copy. Returns the
    separators no pausing phrase ends at; their pause, if any, stays.
    """
    updated = query.model_copy(deep=True)
    phrases = updated.accent_phrases
    ends = {
        start + len(phrase.moras): index
        for index, (start, phrase) in enumerate(zip(phrase_starts(updated), phrases, strict=True))
    }
    unmatched: list[int] = []
    for position in inserted:
        index = ends.get(mora_total(read(text[:position])))
        if index is None or phrases[index].pause_mora is None:
            unmatched.append(position)
            continue
        phrases[index].pause_mora = None
    return updated, unmatched
