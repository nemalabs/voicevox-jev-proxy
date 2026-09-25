from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

from vovovo.edits import Edit
from vovovo.prosody import ACCENT_FIELD, Change, join_phrases
from vovovo.voicevox import AccentPhrase, AudioQuery

SKIPPED_FIELD = f"{ACCENT_FIELD}_skipped"
ATTACHING_POS = frozenset({"助詞", "助動詞", "接尾辞"})
HELPER_POS = frozenset({"動詞", "形容詞"})
DEPENDENT = "非自立可能"
CONJUNCTIVE = ("助詞", "接続助詞")
TE_FORMS = frozenset({"て", "で"})
RUN_ON = "the accent phrase goes on past the particles after the word"


@dataclass(frozen=True)
class Word:
    """A token of the text: its first character, surface and Sudachi part of speech."""

    start: int
    surface: str
    pos: tuple[str, ...]


@dataclass(frozen=True)
class AccentTarget:
    """A word whose dictionary accent type decides the nucleus of the accent phrase it starts.

    accents follow the dictionary convention: 0 is 平板型, n puts the nucleus on the n-th mora. The first
    is the main one; the rest are variants the word is also read with.
    attached_moras counts the morae of the particles and auxiliaries that directly follow the word,
    the only words the phrase may contain besides it.
    cut tells whether a phrase that runs on past them may be cut there: the word is used as an
    adverb and an independent word follows (その時|兵十は), not a suffix (中山|さま) or the rest of a
    compound (金|文字).
    """

    id: str
    start: int
    end: int
    accents: tuple[int, ...]
    moras: int
    attached_moras: int
    reason: str
    cut: bool = False


def moved(target: AccentTarget, edits: Sequence[Edit]) -> AccentTarget | None:
    """Return the target's span in the text after `edits`, or None when an edit falls inside the word."""
    shift = 0
    length = target.end - target.start
    for edit in edits:
        if edit.start == target.start and edit.end == target.end and edit.start < edit.end:
            length = len(edit.replacement)
        elif edit.end <= target.start:
            shift += len(edit.replacement) - (edit.end - edit.start)
        elif edit.start < target.end:
            return None
    start = target.start + shift
    return replace(target, start=start, end=start + length)


def mora_total(query: AudioQuery) -> int:
    return sum(len(phrase.moras) for phrase in query.accent_phrases)


def phrase_accent(phrase: AccentPhrase, word_moras: int, accents: Sequence[int]) -> int:
    """Nucleus of a phrase whose first word has the given accent types.

    VOICEVOX's nucleus stays when it agrees with any of the types; otherwise the first type decides.
    """
    nuclei = [nucleus(phrase, word_moras, accent) for accent in accents]
    return phrase.accent if phrase.accent in nuclei else nuclei[0]


def nucleus(phrase: AccentPhrase, word_moras: int, accent: int) -> int:
    """Nucleus of a phrase whose first word has one accent type.

    The first nucleus wins in a Tokyo-dialect phrase, so an accented word decides it outright. A flat
    word leaves the phrase flat unless VOICEVOX put a nucleus on an attached word after it (です, ます).
    VOICEVOX writes a flat phrase as accent == mora count.
    """
    if accent > 0:
        return accent
    if word_moras < phrase.accent < len(phrase.moras):
        return phrase.accent
    return len(phrase.moras)


def apply_accents(
    query: AudioQuery, text: str, targets: Sequence[AccentTarget], read: Callable[[str], AudioQuery]
) -> tuple[AudioQuery, list[Change], frozenset[int]]:
    """Set the nucleus of every phrase that starts with a target word.

    A word is found by reading the text before it and counting morae, since VOICEVOX reports no
    character offsets. A word VOICEVOX split across phrases is joined back into one phrase first.
    A phrase that runs on past the word and its particles into the next word (そのとき兵十は read
    ソノ|トキヘエジュウワ) is cut there when the target allows it and VOICEVOX's nucleus lies in the
    rest, which keeps it.
    Returns the indices of the phrases the targets landed on, changed or not.
    """
    updated = query.model_copy(deep=True)
    changes: list[Change] = []
    handled: set[int] = set()
    for target in sorted(targets, key=lambda target: target.start):
        begin = mora_total(read(text[: target.start]))
        end = mora_total(read(text[: target.end]))
        span, problem = _locate(updated, target, begin, end)
        if span is None:
            changes.append(Change(target.id, SKIPPED_FIELD, text[target.start : target.end], "", problem))
            continue
        first, last, cut = span
        before = "+".join(f"{p.reading}({p.accent})" for p in updated.accent_phrases[first : last + 1])
        split = cut < len(updated.accent_phrases[last].moras)
        if split:
            _split(updated, last, cut)
        phrase = _join(updated, first, last)
        handled.add(first)
        accent = phrase_accent(phrase, target.moras, target.accents)
        if accent == phrase.accent and first == last and not split:
            continue
        phrase.accent = accent
        shown = updated.accent_phrases[first : first + (2 if split else 1)]
        after = "+".join(f"{p.reading}({p.accent})" for p in shown)
        changes.append(Change(target.id, ACCENT_FIELD, before, after, target.reason))
    return updated, changes, frozenset(handled)


def attachable_phrases(
    query: AudioQuery, text: str, words: Sequence[Word], read: Callable[[str], AudioQuery]
) -> frozenset[int]:
    """Return the indices of the phrases whose first word may join the phrase before it.

    Those are particles, auxiliaries and suffixes, and a dependent verb or adjective (いる, くれる,
    みる) right after the て/で of the verb it helps. The same verb after anything else is a verb of
    its own (連絡を|くれる), so its phrase stays apart whatever Jev says.
    """
    starts = {start: index for index, start in enumerate(phrase_starts(query))}
    found: set[int] = set()
    for previous, word in pairwise(words):
        if not attaches(previous, word):
            continue
        index = starts.get(mora_total(read(text[: word.start])))
        if index:
            found.add(index)
    return frozenset(found)


def attaches(previous: Word, word: Word) -> bool:
    if word.pos[0] in ATTACHING_POS:
        return True
    helper = word.pos[0] in HELPER_POS and word.pos[1] == DEPENDENT
    return helper and previous.pos[:2] == CONJUNCTIVE and previous.surface in TE_FORMS


def phrase_starts(query: AudioQuery) -> list[int]:
    starts: list[int] = []
    total = 0
    for phrase in query.accent_phrases:
        starts.append(total)
        total += len(phrase.moras)
    return starts


def _join(query: AudioQuery, first: int, last: int) -> AccentPhrase:
    phrase = query.accent_phrases[first]
    for right in query.accent_phrases[first + 1 : last + 1]:
        join_phrases(phrase, right)
    del query.accent_phrases[first + 1 : last + 1]
    return phrase


def _split(query: AudioQuery, index: int, cut: int) -> None:
    """Cut a phrase after `cut` morae whose nucleus lies past them: the rest keeps it, the head is flat."""
    phrase = query.accent_phrases[index]
    rest = AccentPhrase(
        moras=phrase.moras[cut:],
        accent=phrase.accent - cut,
        pause_mora=phrase.pause_mora,
        is_interrogative=phrase.is_interrogative,
    )
    phrase.moras = phrase.moras[:cut]
    phrase.accent = cut
    phrase.pause_mora = None
    phrase.is_interrogative = False
    query.accent_phrases.insert(index + 1, rest)


def _locate(query: AudioQuery, target: AccentTarget, begin: int, end: int) -> tuple[tuple[int, int, int] | None, str]:
    """First and last phrase the word covers and how many morae of the last belong to the word and its particles.

    None with the reason when the word cannot be placed.
    """
    problem = _misfit(target, begin, end)
    if problem:
        return None, problem
    starts = phrase_starts(query)
    if begin not in starts:
        return None, "the word does not start an accent phrase"
    first = starts.index(begin)
    ends = [start + len(phrase.moras) for start, phrase in zip(starts, query.accent_phrases, strict=True)]
    last = next((index for index in range(first, len(ends)) if ends[index] >= end), None)
    if last is None:
        return None, "the word runs past the last accent phrase"
    if any(phrase.pause_mora is not None for phrase in query.accent_phrases[first:last]):
        return None, "VOICEVOX pauses inside the word"
    phrase = query.accent_phrases[last]
    cut = len(phrase.moras) - max(ends[last] - end - target.attached_moras, 0)
    if cut < len(phrase.moras) and (not target.cut or phrase.accent <= cut):
        return None, RUN_ON if not target.cut else f"{RUN_ON}, with its nucleus before them"
    return (first, last, cut), ""


def _misfit(target: AccentTarget, begin: int, end: int) -> str:
    if end - begin != target.moras:
        return f"VOICEVOX reads {end - begin} moras here, the chosen reading has {target.moras}"
    if not target.accents or max(target.accents) > target.moras:
        return f"accent types {target.accents} do not fit {target.moras} moras"
    return ""
