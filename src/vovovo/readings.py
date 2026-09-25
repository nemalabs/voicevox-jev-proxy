import gzip
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import fugashi
from pydantic import JsonValue
from sudachipy import Dictionary, Morpheme, SplitMode

from vovovo.accents import SKIPPED_FIELD, AccentTarget, Word, mora_total, phrase_accent, phrase_starts
from vovovo.edits import YOTSUGANA, Edit, spoken
from vovovo.prosody import Change
from vovovo.typesafe import Answer, ChoiceAnswer
from vovovo.voicevox import AccentPhrase, AudioQuery

NONE_KEY = "none"
READING_FIELD = "reading"
LITERARY_PREFIX = "文語"
PROPER_NOUN = "固有名詞"
KANJI = re.compile(r"[一-龯々〆ヶ]")
HIRAGANA_WORD = re.compile(r"[ぁ-ゖー]+")
ACCENT_TYPE = re.compile(r"\d+")
CONTENT_POS = frozenset({"名詞", "代名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞", "接続詞", "感動詞"})
UNINFLECTED_POS = frozenset({"名詞", "代名詞", "形状詞", "副詞", "連体詞", "接続詞", "感動詞"})
ATTACHED_POS = frozenset({"助詞", "助動詞"})
SMALL_KANA = frozenset("ァィゥェォャュョヮ")
KATAKANA_START, KATAKANA_END, KANA_OFFSET = ord("ァ"), ord("ヶ"), ord("ァ") - ord("ぁ")
CLAUSE_BREAKS = frozenset("、。！？!?「」『』（）()\n")
HOW_TO_READ = "`sentence` の文脈では、`word` はどう読むか"
WHICH_WORD = "`sentence` の文脈では、`word` はどの語か"
READING_CONTEXT = "`context` は、その読みをひらがなで当てはめた文節"
WORD_CONTEXT = "`context` は、その語の表記を当てはめた文節"
NONE_TEXT = "どの候補も当てはまらない"
HOW_TO_READ_EN = "In the context of `sentence`, which reading of `word` is correct?"
WHICH_WORD_EN = "In the context of `sentence`, which word is `word`?"
OTHER_MEANING_EN = "; `other_meaning` lists the senses the dictionary gives after those in `meaning`"
READING_CONTEXT_EN = (
    "`context` is the phrase with the reading written in hiragana; `meaning` is the English dictionary gloss of that"
    f" reading{OTHER_MEANING_EN}"
)
WORD_CONTEXT_EN = (
    "`context` is the phrase with the word written as that lemma; `meaning` is the English dictionary gloss of that"
    f" word{OTHER_MEANING_EN}"
)
NONE_TEXT_EN = "none of the candidates fits"
VOICEVOX_KIND = "VOICEVOX"
VOICING = str.maketrans("カキクケコサシスセソタチツテトハヒフヘホ", "ガギグゲゴザジズゼゾダヂヅデドバビブベボ")
COMPOUND_HEAD = re.compile(r"[一-龯々〆ヶァ-ヺー]")
ARCHAIC = "archaic"
TERM_SUFFIX = " term"
NUMERAL = "数詞"
ADVERBIAL = "副詞可能"
BASE_FORMS = frozenset({"*", "基本形"})


def pos_label(pos: Sequence[str]) -> str:
    depth = 4 if pos[1] == PROPER_NOUN else 3
    return "-".join(p for p in pos[:depth] if p != "*")


def to_hiragana(katakana: str) -> str:
    return "".join(chr(ord(ch) - KANA_OFFSET) if KATAKANA_START <= ord(ch) <= KATAKANA_END else ch for ch in katakana)


def to_katakana(hiragana: str) -> str:
    start, end = KATAKANA_START - KANA_OFFSET, KATAKANA_END - KANA_OFFSET
    return "".join(chr(ord(ch) + KANA_OFFSET) if start <= ord(ch) <= end else ch for ch in hiragana)


def mora_count(katakana: str) -> int:
    return sum(1 for ch in katakana if ch not in SMALL_KANA)


def accent_types(value: str) -> tuple[int, ...]:
    """Accent types in a UniDic aType field such as "0", "2,1" or "*", the main one first."""
    return tuple(int(number) for number in ACCENT_TYPE.findall(value))


@dataclass(frozen=True)
class Entry:
    """A dictionary reading of a surface; base is the reading of the lemma, None when unknown."""

    reading: str
    lemma: str
    pos: str
    accents: tuple[int, ...]
    base: str | None = None

    @property
    def kind(self) -> str:
        """What tells this entry apart from others with the same reading: the kind of name, else the lemma."""
        parts = self.pos.split("-")
        if len(parts) > 2 and parts[1] == PROPER_NOUN:  # noqa: PLR2004
            return parts[2]
        return self.lemma


@dataclass(frozen=True)
class ReadingOption:
    """One reading of a word; meanings is None when no gloss dictionary was consulted.

    meanings are the first senses of its dictionary entries and later_meanings the other senses. Jev sees
    the two apart: in one list, the second sense of 市場 read しじょう, "(street) market", drew Jev from
    いちば where fish are bought. common tells whether JMdict marks the reading as a common word (re_pri).
    """

    reading: str
    accents: tuple[int, ...]
    lemmas: tuple[str, ...]
    pos: tuple[str, ...]
    kinds: tuple[str, ...]
    meanings: tuple[str, ...] | None = None
    usage: tuple[str, ...] = ()
    lemma_readings: tuple[tuple[str, str], ...] = ()
    common: bool = False
    later_meanings: tuple[str, ...] = ()

    @property
    def is_name(self) -> bool:
        return all(pos.split("-")[1:2] == [PROPER_NOUN] for pos in self.pos)

    def describe(self, context: list[str]) -> JsonValue:
        described: dict[str, JsonValue] = {
            "reading": self.reading,
            "lemma": list(self.lemmas),
            "pos": list(self.pos),
            "context": list(context),
        }
        if self.meanings is not None:
            described["meaning"] = list(self.meanings)
        if self.later_meanings:
            described["other_meaning"] = list(self.later_meanings)
        if self.usage:
            described["usage"] = list(self.usage)
        return described


def group_options(entries: Sequence[Entry], *, by_accent: bool) -> dict[str, ReadingOption]:
    """One option per reading, split further by accent type when the accent will be applied.

    Keys are the reading, qualified by the kind of word when two options share a reading.
    """
    grouped: dict[tuple[str, tuple[int, ...]], list[Entry]] = {}
    for entry in entries:
        grouped.setdefault((entry.reading, entry.accents if by_accent else ()), []).append(entry)
    shared = [reading for reading, _ in grouped]
    options: dict[str, ReadingOption] = {}
    for (reading, accents), members in grouped.items():
        option = ReadingOption(
            reading,
            accents,
            _unique(entry.lemma for entry in members),
            _unique(entry.pos for entry in members),
            _unique(entry.kind for entry in members),
            lemma_readings=_unique((entry.lemma, entry.base) for entry in members if entry.base is not None),
        )
        base = reading if shared.count(reading) == 1 else f"{reading}({'・'.join(option.kinds)})"
        key, suffix = base, 2
        while key in options:
            key, suffix = f"{base}#{suffix}", suffix + 1
        options[key] = option
    return options


def _unique[T](values: Iterable[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(values))


def single_sense(options: Mapping[str, ReadingOption], *, inflected: bool = False) -> bool:
    """Whether every reading means the same thing, so picking one is a matter of style Jev cannot judge.

    The readings of an inflected word also mean the same when the gloss terms of one are all among
    those of the other: 入る read いる is "to enter; to go in; to get in; to come in", read はいる
    "to enter; to come in; to go in; to get in; to arrive", and Jev picked the literary いる. Nouns are
    held to equal glosses, since the first sense JMdict gives may be a minor one (方 read かた is
    "direction; way", not the "person" of あの方).
    Only the first senses count: the later ones of 入る read いる ("to set (of the sun or moon)") would
    part it from はいる again.
    A name, or a reading the gloss dictionary does not know, counts as a sense of its own.
    """
    senses: list[frozenset[str]] = []
    for option in options.values():
        if option.is_name or not option.meanings:
            return False
        senses.append(frozenset(term.strip() for meaning in option.meanings for term in meaning.split(";")))
    if inflected:
        return all(left <= right or right <= left for left, right in combinations(senses, 2))
    return len(set(senses)) < 2  # noqa: PLR2004


def common_only(options: Mapping[str, ReadingOption]) -> dict[str, ReadingOption]:
    """Drop the readings JMdict does not mark as common when some reading of the word is; names stay."""
    if not any(option.common for option in options.values()):
        return dict(options)
    return {key: option for key, option in options.items() if option.common or option.is_name}


@dataclass(frozen=True)
class ReadingCandidate:
    """A word with the readings to ask about.

    followed_by is the part of speech of the first word after its particles and auxiliaries, "" at
    the end of the text.
    """

    start: int
    end: int
    surface: str
    pos: str
    options: Mapping[str, ReadingOption]
    attached_moras: int = 0
    followed_by: str = ""

    @property
    def id(self) -> str:
        return f"r{self.start}"

    @property
    def kana_only(self) -> bool:
        return not KANJI.search(self.surface)


class AccentDictionary:
    """UniDic entries of a surface form, each with its accent type (aType)."""

    def __init__(self) -> None:
        self._tagger = fugashi.Tagger("-a")

    def entries(self, surface: str, pos1: str) -> list[Entry]:
        """Entries of the surface as a word of its own.

        Initial and final sound changes (連濁 as in 人→ビト, 三→ミッ) only happen inside compounds, so
        UniDic's changed forms are left out.
        """
        found: list[Entry] = []
        for feature in self._features(surface, pos1):
            if not _changed(feature):
                pos = (feature.pos1, feature.pos2, feature.pos3, feature.pos4)
                accents = accent_types(feature.aType)
                found.append(Entry(feature.kana, feature.lemma, pos_label(pos), accents, feature.lForm))
        return found

    def changed_readings(self, surface: str, pos1: str) -> frozenset[str]:
        """Return the readings UniDic knows for the surface only as a sound-changed form, ヂ/ヅ as ジ/ズ."""
        return frozenset(same_sound(feature.kana) for feature in self._features(surface, pos1) if _changed(feature))

    def _features(self, surface: str, pos1: str) -> list[fugashi.UnidicFeatures26]:
        return [
            node.feature
            for node in self._tagger(surface)
            if node.surface == surface
            and not node.is_unk
            and node.feature.pos1 == pos1
            and not node.feature.cType.startswith(LITERARY_PREFIX)
        ]


def _changed(feature: fugashi.UnidicFeatures26) -> bool:
    return feature.iForm not in BASE_FORMS or feature.fForm not in BASE_FORMS


def same_sound(katakana: str) -> str:
    return katakana.translate(YOTSUGANA)


@dataclass(frozen=True)
class Gloss:
    """The senses one JMdict entry gives a written form with a reading.

    text is the first sense and usage its register tags; later are the other senses, each with its own
    register tags in parentheses.
    """

    text: str
    usage: tuple[str, ...]
    common: bool = False
    later: tuple[str, ...] = ()


def load_jmdict(path: Path) -> dict[tuple[str, str], list[Gloss]]:
    """Map (written form, hiragana reading) to the senses of each JMdict entry that has them.

    The text may use a sense other than the first (空く read すく is "to become less crowded" before "to
    be hungry"), so every sense that fits the form and reading is kept.
    JMdict writes the readings of loanwords in katakana (硝子, ガラス); they are indexed in hiragana
    like the rest. Only register tags such as "archaic" or "dated term" are kept, since they say how
    likely a reading is. Those of a later sense hold for that sense only, so they stay with its text.
    """
    index: dict[tuple[str, str], list[Gloss]] = {}
    with gzip.open(path) as source:
        # JMdict declares its tags as internal DTD entities, which defusedxml refuses; the stdlib parser
        # does not fetch external entities, and expat >= 2.4.1 limits entity expansion.
        for _, element in ET.iterparse(source, events=("end",)):  # noqa: S314
            if element.tag == "entry":
                _index_entry(element, index)
                element.clear()
    return index


def _index_entry(entry: ET.Element, index: dict[tuple[str, str], list[Gloss]]) -> None:
    forms = [form.findtext("keb") or "" for form in entry.findall("k_ele")]
    senses = entry.findall("sense")
    for reading_element in entry.findall("r_ele"):
        reading = reading_element.findtext("reb") or ""
        restricted = _texts(reading_element, "re_restr")
        for form in forms:
            if restricted and form not in restricted:
                continue
            fitting = [
                sense
                for sense in senses
                if _allows(_texts(sense, "stagk"), form) and _allows(_texts(sense, "stagr"), reading)
            ]
            if not fitting:
                continue
            first, *later = fitting
            common = reading_element.find("re_pri") is not None
            gloss = Gloss(_sense_text(first), _register(first), common, tuple(_tagged_text(sense) for sense in later))
            index.setdefault((form, to_hiragana(reading)), []).append(gloss)


def _sense_text(sense: ET.Element) -> str:
    return "; ".join(gloss.text or "" for gloss in sense.findall("gloss"))


def _register(sense: ET.Element) -> tuple[str, ...]:
    return tuple(tag for tag in _texts(sense, "misc") if tag == ARCHAIC or tag.endswith(TERM_SUFFIX))


def _tagged_text(sense: ET.Element) -> str:
    register = _register(sense)
    return f"{_sense_text(sense)} ({', '.join(register)})" if register else _sense_text(sense)


def _texts(element: ET.Element, tag: str) -> list[str]:
    return [child.text or "" for child in element.findall(tag)]


def _allows(restriction: list[str], value: str) -> bool:
    return not restriction or value in restriction


class GlossDictionary:
    """English glosses from JMdict, so Jev can tell readings apart by meaning."""

    def __init__(self, path: Path) -> None:
        self._index = load_jmdict(path)

    def lookup(self, form: str, reading: str) -> list[Gloss]:
        return self._index.get((form, to_hiragana(reading)), [])

    def gloss(self, options: Mapping[str, ReadingOption], surface: str, pos1: str) -> dict[str, ReadingOption]:
        """Attach meanings to options, looked up by lemma and its reading.

        An uninflected kanji word is also looked up by its surface. An inflected one is not: its
        surface with the inflected reading (降り, ふり) names a different, nominal entry.
        Names get none: JMdict lists common words, so a name would borrow the gloss of its surface.
        """
        by_surface = pos1 in UNINFLECTED_POS and bool(KANJI.search(surface))
        glossed: dict[str, ReadingOption] = {}
        for key, option in options.items():
            if option.is_name:
                glossed[key] = replace(option, meanings=())
                continue
            keys = _unique([*([(surface, option.reading)] if by_surface else []), *option.lemma_readings])
            found = [gloss for form, reading in keys for gloss in self.lookup(form, reading)]
            usage = tuple(sorted({tag for gloss in found for tag in gloss.usage}))
            common = any(gloss.common for gloss in found)
            glossed[key] = replace(
                option,
                meanings=_unique(gloss.text for gloss in found),
                later_meanings=_unique(text for gloss in found for text in gloss.later),
                usage=usage,
                common=common,
            )
        return glossed


class ReadingLexicon:
    """Reading and accent candidates for the words of a text.

    Readings come from Sudachi, which knows recent words; accent types come from UniDic where it has
    the same reading, and stay unknown otherwise. Meanings come from JMdict when it is given.
    """

    def __init__(self, dict_path: Path, accents: AccentDictionary, glosses: GlossDictionary | None = None) -> None:
        self._dictionary = Dictionary(dict=str(dict_path))
        self._tokenizer = self._dictionary.create()
        self._accents = accents
        self._glosses = glosses

    def candidates(self, text: str, *, minimum: int = 2) -> list[ReadingCandidate]:
        """Words with at least `minimum` dictionary readings.

        Numerals and the counter after one are left to VOICEVOX: their reading comes from the pair
        (三 + 時 → サンジ, 三 + 本 → サンボン), not from the meaning of either.
        """
        morphemes = list(self._tokenizer.tokenize(text, SplitMode.C))
        found: list[ReadingCandidate] = []
        for index, morpheme in enumerate(morphemes):
            surface = morpheme.surface()
            pos = morpheme.part_of_speech()
            after_numeral = index > 0 and morphemes[index - 1].part_of_speech()[1] == NUMERAL
            if pos[0] not in CONTENT_POS or pos[1] == NUMERAL or after_numeral:
                continue
            pos1 = pos[0]
            options = group_options(self._entries(surface, pos1), by_accent=pos1 in UNINFLECTED_POS)
            if len(options) < max(minimum, 1):
                continue
            if self._glosses is not None:
                options = self._glosses.gloss(options, surface, pos1)
                if len(options) > 1 and single_sense(options, inflected=pos1 not in UNINFLECTED_POS):
                    continue
            following = morphemes[index + 1 :]
            found.append(
                ReadingCandidate(
                    morpheme.begin(),
                    morpheme.end(),
                    surface,
                    pos1,
                    options,
                    attached_moras(following),
                    next_word_pos(following),
                )
            )
        return found

    def words(self, text: str) -> list[Word]:
        return [
            Word(morpheme.begin(), morpheme.surface(), tuple(morpheme.part_of_speech()))
            for morpheme in self._tokenizer.tokenize(text, SplitMode.C)
        ]

    def _entries(self, surface: str, pos1: str) -> list[Entry]:
        if HIRAGANA_WORD.fullmatch(surface):
            return self._accents.entries(surface, pos1)
        if not KANJI.search(surface):
            return []
        known = self._accents.entries(surface, pos1)
        labelled = {(entry.reading, entry.pos) for entry in known}
        changed = self._accents.changed_readings(surface, pos1)
        extra = [
            entry
            for entry in self._sudachi_entries(surface, pos1)
            if (entry.reading, entry.pos) not in labelled and same_sound(entry.reading) not in changed
        ]
        return known + extra

    def _sudachi_entries(self, surface: str, pos1: str) -> list[Entry]:
        entries: list[Entry] = []
        for entry in self._dictionary.lookup(surface):
            pos = entry.part_of_speech()
            if pos[0] != pos1 or pos[4].startswith(LITERARY_PREFIX):
                continue
            base = entry.reading_form() if pos1 in UNINFLECTED_POS else None
            entries.append(Entry(entry.reading_form(), entry.normalized_form(), pos_label(pos), (), base))
        return entries


def voicevox_keys(
    candidate: ReadingCandidate, text: str, query: AudioQuery, read: Callable[[str], AudioQuery]
) -> frozenset[str]:
    """Return the options VOICEVOX reads the word as in `query`.

    The word is found by reading the text before it and counting morae, since VOICEVOX reports no
    character offsets. Of the options whose reading starts there, the longest win, so that ハイ does
    not also claim a word VOICEVOX reads ハイロオ.
    The lemmas of a kana word share the reading on the page, so there the accent tells them apart:
    when the word starts an accent phrase, only the lemmas whose accent type gives VOICEVOX's nucleus
    count. VOICEVOX accents とき on its first mora, as 鴇 is, where the text means 時.
    """
    moras = [mora.text for phrase in query.accent_phrases for mora in phrase.moras]
    begin = mora_total(read(text[: candidate.start]))
    ahead = spoken("".join(moras[begin:]))
    lengths = {
        key: len(option.reading)
        for key, option in candidate.options.items()
        if ahead.startswith(spoken(option.reading))
    }
    longest = max(lengths.values(), default=0)
    keys = frozenset(key for key, length in lengths.items() if length == longest)
    if not candidate.kana_only or candidate.pos not in UNINFLECTED_POS:
        return keys
    return _accented_as(candidate, keys, query, begin)


def _accented_as(candidate: ReadingCandidate, keys: frozenset[str], query: AudioQuery, begin: int) -> frozenset[str]:
    starts = phrase_starts(query)
    if begin not in starts or not all(candidate.options[key].accents for key in keys):
        return keys
    phrase = query.accent_phrases[starts.index(begin)]
    return frozenset(key for key in keys if _fits(phrase, candidate.options[key]))


def _fits(phrase: AccentPhrase, option: ReadingOption) -> bool:
    return phrase_accent(phrase, mora_count(option.reading), option.accents) == phrase.accent


def with_voicevox_reading(
    candidate: ReadingCandidate, text: str, query: AudioQuery, read: Callable[[str], AudioQuery]
) -> ReadingCandidate | None:
    """Add VOICEVOX's own reading of a kanji word it reads as none of the dictionary readings.

    Such a word is often misread (云い as ゆい, 互に as かたみに), yet VOICEVOX is sometimes right where
    the dictionary lacks the form, so Jev chooses between them. The option has no lemma or meaning,
    since no dictionary knows the reading.
    A voiced first mora after a kanji or katakana word is the sound change of a compound (西洋造り as
    せいようづくり, 裏戸口 as うらとぐち), which Jev does not hear: it chose つくり and くち. Such a word
    gets no question (None), and VOICEVOX reads it alone.
    """
    moras = [mora.text for phrase in query.accent_phrases for mora in phrase.moras]
    begin = mora_total(read(text[: candidate.start]))
    reading = "".join(moras[begin : mora_total(read(text[: candidate.end]))])
    if not reading or candidate.kana_only or reading in candidate.options:
        return candidate
    compound = candidate.start > 0 and COMPOUND_HEAD.match(text[candidate.start - 1]) is not None
    voiced = {
        spoken(option.reading[:1].translate(VOICING) + option.reading[1:]) for option in candidate.options.values()
    }
    if compound and spoken(reading) in voiced:
        return None
    option = ReadingOption(reading, (), (), (), (VOICEVOX_KIND,), meanings=())
    return replace(candidate, options={**candidate.options, reading: option})


def narrow(candidate: ReadingCandidate, current: frozenset[str]) -> ReadingCandidate | None:
    """Keep the readings worth asking Jev about, or None when no question is left.

    When VOICEVOX already reads the word as a common reading, the rare readings the dictionaries list
    are dropped: Jev picked them where VOICEVOX was right (家 as や, 硝子 as しょうし). When VOICEVOX
    reads it some other way (腹 as ふく, 気がつく as けがつく), every reading stays for Jev to choose from.
    A kana word is read the same way as all its lemmas, so `current` holds the lemmas its VOICEVOX
    accent fits: accented as a rare lemma (ところ flat, as 野老) it keeps every lemma for Jev to pick
    by meaning.
    """
    common = common_only(candidate.options)
    options = common if current & common.keys() else dict(candidate.options)
    inflected = candidate.pos not in UNINFLECTED_POS
    if len(options) < 2 or single_sense(options, inflected=inflected):  # noqa: PLR2004
        return None
    return replace(candidate, options=options)


def next_word_pos(following: Sequence[Morpheme]) -> str:
    """Return the part of speech of the first word after the particles and auxiliaries, "" when none follows."""
    return next((pos for pos in (m.part_of_speech()[0] for m in following) if pos not in ATTACHED_POS), "")


def attached_moras(following: Sequence[Morpheme]) -> int:
    total = 0
    for morpheme in following:
        if morpheme.part_of_speech()[0] not in ATTACHED_POS:
            break
        total += mora_count(morpheme.reading_form())
    return total


def clause_around(text: str, start: int, end: int) -> tuple[str, str]:
    """Return the parts of the clause around text[start:end] that come before and after it."""
    head = start
    while head > 0 and text[head - 1] not in CLAUSE_BREAKS:
        head -= 1
    tail = end
    while tail < len(text) and text[tail] not in CLAUSE_BREAKS:
        tail += 1
    return text[head:start], text[end:tail]


def option_context(text: str, candidate: ReadingCandidate, option: ReadingOption) -> list[str]:
    """Write out the clause around the word with the word as the option would have it.

    Kanji words get the reading in hiragana; kana words get each lemma, since their reading is
    already on the page.
    """
    before, after = clause_around(text, candidate.start, candidate.end)
    words = option.lemmas if candidate.kana_only else (to_hiragana(option.reading),)
    return [f"{before}{word}{after}" for word in words]


def build_reading_questions(text: str, candidates: list[ReadingCandidate]) -> dict[str, JsonValue]:
    questions: dict[str, JsonValue] = {}
    for candidate in candidates:
        criteria: dict[str, JsonValue] = {
            key: option.describe(option_context(text, candidate, option)) for key, option in candidate.options.items()
        }
        question, note, none = _wording(candidate)
        criteria[NONE_KEY] = none
        questions[f"{candidate.id}_reading"] = {
            "type": "choice",
            "instructions": {"word": candidate.surface, "question": question, "note": note},
            "criteria": criteria,
        }
    return questions


def _wording(candidate: ReadingCandidate) -> tuple[str, str, str]:
    """Question, note and none-text; English once meanings are in, since Jev reads English glosses best."""
    if any(option.meanings is not None for option in candidate.options.values()):
        if candidate.kana_only:
            return WHICH_WORD_EN, WORD_CONTEXT_EN, NONE_TEXT_EN
        return HOW_TO_READ_EN, READING_CONTEXT_EN, NONE_TEXT_EN
    if candidate.kana_only:
        return WHICH_WORD, WORD_CONTEXT, NONE_TEXT
    return HOW_TO_READ, READING_CONTEXT, NONE_TEXT


@dataclass(frozen=True)
class ReadingChoice:
    candidate: ReadingCandidate
    key: str
    option: ReadingOption
    confidence: float

    @property
    def reason(self) -> str:
        return f"reading={self.key} confidence={self.confidence:.2f}"


def choose_options(
    candidates: list[ReadingCandidate], answers: Mapping[str, Answer], threshold: float
) -> list[ReadingChoice]:
    choices: list[ReadingChoice] = []
    for candidate in candidates:
        answer = answers.get(f"{candidate.id}_reading")
        if not isinstance(answer, ChoiceAnswer) or answer.confidence < threshold:
            continue
        option = candidate.options.get(answer.choice)
        if option is None:
            continue
        choices.append(ReadingChoice(candidate, answer.choice, option, answer.confidence))
    return choices


def reading_edits(choices: list[ReadingChoice]) -> list[Edit]:
    edits: list[Edit] = []
    for choice in choices:
        candidate = choice.candidate
        if candidate.kana_only:
            continue
        replacement = to_hiragana(choice.option.reading)
        change = Change(candidate.id, READING_FIELD, candidate.surface, replacement, choice.reason)
        edits.append(Edit(candidate.start, candidate.end, replacement, change, choice.option.reading))
    return edits


def accent_targets(choices: list[ReadingChoice]) -> tuple[list[AccentTarget], list[Change]]:
    """Accent targets for the chosen words whose dictionary accent is known and does not inflect.

    The phrase of a noun used as an adverb (時, 今日) may be cut before the independent word after it.
    """
    targets: list[AccentTarget] = []
    skipped: list[Change] = []
    for choice in choices:
        candidate, option = choice.candidate, choice.option
        if candidate.pos not in UNINFLECTED_POS:
            continue
        if not option.accents:
            skipped.append(Change(candidate.id, SKIPPED_FIELD, candidate.surface, "", "no dictionary accent"))
            continue
        reason = f"{choice.reason} accent_type={','.join(map(str, option.accents))}"
        adverbial = any(pos.split("-")[2:3] == [ADVERBIAL] for pos in option.pos)
        targets.append(
            AccentTarget(
                candidate.id,
                candidate.start,
                candidate.end,
                option.accents,
                mora_count(option.reading),
                candidate.attached_moras,
                reason,
                cut=adverbial and candidate.followed_by in CONTENT_POS,
            )
        )
    return targets, skipped
