from collections.abc import Callable
from dataclasses import dataclass, replace

from voicevox_jev_proxy.prosody import Change
from voicevox_jev_proxy.voicevox import AudioQuery

SEPARATOR = "、"
SKIPPED_SUFFIX = "_skipped"
UNCHANGED = "reading unchanged"
VOWEL_ROWS = {
    "ア": "アカガサザタダナハバパマヤラワァャヮ",
    "イ": "イキギシジチヂニヒビピミリィ",
    "ウ": "ウクグスズツヅヌフブプムユルゥュヴ",
    "エ": "エケゲセゼテデネヘベペメレェ",
    "オ": "オコゴソゾトドノホボポモヨロヲォョ",
}
VOWEL_OF = {kana: vowel for vowel, row in VOWEL_ROWS.items() for kana in row}
YOTSUGANA = str.maketrans("ヂヅ", "ジズ")


@dataclass(frozen=True)
class Rephrase:
    """The stretch of text a separator rephrases, whose reading alone may change with it.

    moves_from is where in the text VOICEVOX had the accent phrase boundary instead, None when it had none.
    """

    start: int
    end: int
    moves_from: int | None = None


@dataclass(frozen=True)
class Edit:
    """A replacement of `text[start:end]`; an insertion when start == end.

    reading is the katakana VOICEVOX should read the replacement as, None when only the change of
    reading matters (an inserted separator).
    rephrase is set on a separator that moves an accent phrase boundary.
    """

    start: int
    end: int
    replacement: str
    change: Change
    reading: str | None = None
    rephrase: Rephrase | None = None


def apply_edits(text: str, edits: list[Edit]) -> tuple[str, list[int]]:
    """Apply non-overlapping edits in text order.

    Returns the new text and the positions in it of every inserted SEPARATOR.
    """
    pieces: list[str] = []
    separators: list[int] = []
    cursor = 0
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        if edit.start < cursor:
            message = f"edit {edit.change.phrase_id} overlaps an earlier edit"
            raise ValueError(message)
        pieces.append(text[cursor : edit.start])
        if edit.start == edit.end and edit.replacement == SEPARATOR:
            separators.append(sum(len(piece) for piece in pieces))
        pieces.append(edit.replacement)
        cursor = edit.end
    pieces.append(text[cursor:])
    return "".join(pieces), separators


def drop_overlaps(edits: list[Edit]) -> list[Edit]:
    kept: list[Edit] = []
    cursor = 0
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        if edit.start < cursor:
            continue
        kept.append(edit)
        cursor = max(cursor, edit.end)
    return kept


def shifted(position: int, edits: list[Edit]) -> int | None:
    """Where a position of the text stands after all the edits, None when one replaces the text around it."""
    if any(edit.start < position < edit.end for edit in edits):
        return None
    return position + sum(len(edit.replacement) - (edit.end - edit.start) for edit in edits if edit.end <= position)


def placed_spans(edits: list[Edit]) -> list[tuple[int, int]]:
    """Where the words the edits wrote with a reading stand in the text after all of them."""
    spans: list[tuple[int, int]] = []
    for edit in edits:
        if edit.reading is None or not edit.replacement:
            continue
        shift = sum(len(e.replacement) - (e.end - e.start) for e in edits if e is not edit and e.end <= edit.start)
        spans.append((edit.start + shift, edit.start + shift + len(edit.replacement)))
    return spans


def reading_of(query: AudioQuery) -> str:
    return "".join(mora.text for phrase in query.accent_phrases for mora in phrase.moras)


def spoken(katakana: str) -> str:
    """Spell a reading as it sounds, so that dictionary and VOICEVOX spellings compare equal.

    Dictionaries write long vowels as ウ, イ or ー (トウ, センセイ); VOICEVOX mostly repeats the vowel
    (トオ, センセエ). ヂ and ヅ sound as ジ and ズ.
    """
    sounds: list[str] = []
    for kana in katakana.translate(YOTSUGANA):
        vowel = VOWEL_OF.get(sounds[-1]) if sounds else None
        lengthens = kana == "ー" or (kana == "ウ" and vowel == "オ") or (kana == "イ" and vowel == "エ")
        sounds.append(vowel if lengthens and vowel else kana)
    return "".join(sounds)


def effective_edits(
    text: str, edits: list[Edit], original: AudioQuery, read: Callable[[str], AudioQuery]
) -> tuple[list[Edit], list[Change]]:
    """Keep only the edits that make VOICEVOX read the text the chosen way.

    An edit at a place VOICEVOX already reads that way would only churn the phrase structure, so it
    is reported as skipped. A replacement with a reading must also be read as it, in accent phrases of
    its own: kana in the text can fuse with the words around it (どうもはらが is read ドオモワラガ, は
    as the particle; むらのもへい is phrased ムラノモ|ヘイ, も as the particle), so it is retried in
    katakana and skipped when VOICEVOX misreads or misphrases that too.
    A separator that moves a phrase boundary must make VOICEVOX start a phrase at it, reading the
    text outside the stretch it rephrases the same.
    """
    kept: list[Edit] = []
    skipped: list[Change] = []
    for edit in edits:
        settled, reason = _settle(text, edit, original, read)
        if settled is not None:
            kept.append(settled)
            continue
        change = edit.change
        field = f"{change.field}{SKIPPED_SUFFIX}"
        skipped.append(Change(change.phrase_id, field, change.before, change.after, reason))
    return kept, skipped


def _settle(text: str, edit: Edit, original: AudioQuery, read: Callable[[str], AudioQuery]) -> tuple[Edit | None, str]:
    if edit.rephrase is not None:
        return _settle_boundary(text, edit, edit.rephrase, original, read)
    before = reading_of(original)
    for attempt in _attempts(edit):
        candidate, _ = apply_edits(text, [attempt])
        query = read(candidate)
        after = reading_of(query)
        if spoken(after) == spoken(before):
            return None, UNCHANGED
        if attempt.reading is None:
            return attempt, ""
        start = _reading_start(before, after, attempt.reading)
        if start is not None and _keeps_phrasing(original, query, start):
            return attempt, ""
    return None, f"VOICEVOX does not read the replacement as {edit.reading} in its own phrasing"


def _settle_boundary(
    text: str, edit: Edit, rephrase: Rephrase, original: AudioQuery, read: Callable[[str], AudioQuery]
) -> tuple[Edit | None, str]:
    """Keep a separator when VOICEVOX starts a phrase at it and reads the text around the stretch the same.

    The stretch itself may be read differently: the new phrasing can mend a misreading VOICEVOX made
    from the wrong one (どうかお入り read ドオ|カオイリ, with the separator ドオカ|オハイリ).
    """
    candidate, _ = apply_edits(text, [edit])
    query = read(candidate)
    before, after = spoken(reading_of(original)), spoken(reading_of(query))
    if before != after:
        first = len(spoken(reading_of(read(text[: rephrase.start]))))
        last = len(spoken(reading_of(read(text[: rephrase.end]))))
        tail = _common_length(before[::-1], after[::-1])
        if _common_length(before, after) < first or len(before) - tail > last:
            return None, "VOICEVOX reads the text outside the stretch differently with the separator"
    head = spoken(reading_of(read(text[: edit.start])))
    if not after.startswith(head):
        return None, "VOICEVOX reads the text before the separator differently on its own"
    if len(head) not in _phrase_offsets(query):
        return None, "VOICEVOX does not start a phrase at the separator"
    return edit, ""


def _attempts(edit: Edit) -> list[Edit]:
    if edit.reading is None or edit.replacement == edit.reading:
        return [edit]
    katakana = replace(edit, replacement=edit.reading, change=replace(edit.change, after=edit.reading))
    return [edit, katakana]


def _reading_start(before: str, after: str, expected: str) -> int | None:
    """Where the stretch of `after` that sounds as `expected` starts, if it covers all that changed.

    `expected` is compared with what comes before it in place: a long vowel only exists after the
    mora it lengthens, and なかへ|いり is not the long エ of センセイ.
    """
    head = _common_length(before, after)
    tail = _common_length(before[head:][::-1], after[head:][::-1])
    changed_end = len(after) - tail
    return next(
        (
            start
            for start in range(max(0, head - len(expected)), head + 1)
            if changed_end <= start + len(expected) <= len(after)
            and spoken(after[: start + len(expected)]) == spoken(after[:start] + expected)
        ),
        None,
    )


def _keeps_phrasing(original: AudioQuery, edited: AudioQuery, start: int) -> bool:
    """Whether the replaced word still starts an accent phrase if it did.

    A word that joins the phrase before it breaks the phrasing (ムラノ|モヘイ as ムラノモ|ヘイ). One
    that starts a phrase of its own does not (ソノカン as ソノ|アイダ), nor does a boundary that moves
    to where it belongs (ナッタ|シフクワ as ナッタシ|ハラワ).
    """
    before = _phrase_offsets(original)
    return start not in before or start in _phrase_offsets(edited)


def _phrase_offsets(query: AudioQuery) -> list[int]:
    """Character offsets in reading_of(query) at which each accent phrase starts."""
    offsets: list[int] = []
    total = 0
    for phrase in query.accent_phrases:
        offsets.append(total)
        total += sum(len(mora.text) for mora in phrase.moras)
    return offsets


def _common_length(left: str, right: str) -> int:
    length = 0
    while length < min(len(left), len(right)) and left[length] == right[length]:
        length += 1
    return length
