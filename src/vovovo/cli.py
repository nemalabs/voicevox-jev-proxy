import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from vovovo.accents import SKIPPED_FIELD, AccentTarget, apply_accents, attachable_phrases, moved, phrase_starts
from vovovo.edits import SEPARATOR, Edit, apply_edits, drop_overlaps, effective_edits
from vovovo.intonation import (
    HEIGHT_SKIPPED_FIELD,
    Height,
    build_height_questions,
    find_heights,
    lower_heights,
    move_late_nuclei,
)
from vovovo.laughter import (
    Laugh,
    build_laugh_questions,
    find_laughs,
    laugh_edits,
    laugh_phrases,
    overlaps,
    voiced_spans,
)
from vovovo.phrase_accents import Token, UnidicTokenizer
from vovovo.prosody import (
    SENTENCE_TYPE_KEY,
    UNIT_SUFFIX,
    Change,
    PhraseRoles,
    Policy,
    apply_structure,
    build_questions,
    build_state,
    match_phrases,
    needs_repitch,
    phrase_id,
    unit_key,
)
from vovovo.readings import (
    AccentDictionary,
    GlossDictionary,
    ReadingCandidate,
    ReadingChoice,
    ReadingLexicon,
    accent_targets,
    build_reading_questions,
    choose_options,
    narrow,
    reading_edits,
    voicevox_keys,
    with_voicevox_reading,
)
from vovovo.settings import Settings
from vovovo.split import (
    Span,
    Stretch,
    build_boundary_questions,
    build_split_questions,
    choose_boundaries,
    choose_splits,
    find_spans,
    find_stretches,
    join_moved_boundaries,
    strip_inserted_pauses,
)
from vovovo.typesafe import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    Pacer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    render_request,
    send,
)
from vovovo.voicevox import AccentPhrase, AudioQuery, VoicevoxClient

DEFAULT_SPEAKER = 3
DEFAULT_THRESHOLD = 0.6
DEFAULT_HEIGHT_THRESHOLD = 0.8
REQUEST_INTERVAL_SEC = 3.0
DEFAULT_REQUESTS = 2
PAUSE_KEPT_FIELD = "pause_kept"
SKIPPED_SUFFIX = "_skipped"
REGROUPED = "asked about the phrase before the text edits, which regrouped it"

Ask = Callable[[SystemOneRequest], SystemOneResponse]


class Tokenizer(Protocol):
    def tokens(self, text: str) -> list[Token]: ...


@dataclass(frozen=True)
class Voice:
    client: VoicevoxClient
    speaker: int

    def read(self, text: str) -> AudioQuery:
        return self.client.audio_query(text, self.speaker)

    def repitch(self, phrases: list[AccentPhrase]) -> list[AccentPhrase]:
        return self.client.mora_pitch(phrases, self.speaker)

    def remeasure(self, phrases: list[AccentPhrase]) -> list[AccentPhrase]:
        return self.client.mora_data(phrases, self.speaker)

    def synthesize(self, query: AudioQuery) -> bytes:
        return self.client.synthesis(query, self.speaker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vovovo", description="Synthesize with VOICEVOX after Jev prosody judgments")
    parser.add_argument("text")
    parser.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER)
    parser.add_argument("--out", type=Path, default=Path(".build/out.wav"))
    parser.add_argument("--dry-run", action="store_true", help="print the TypeSafe requests and stop")
    add_correction_arguments(parser)
    return parser


def add_correction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--height-threshold",
        type=float,
        default=DEFAULT_HEIGHT_THRESHOLD,
        help="least confidence of an answer that lowers a phrase",
    )
    parser.add_argument(
        "--sudachi-dict", type=Path, default=None, help="Sudachi system dictionary for reading candidates"
    )
    parser.add_argument("--jmdict", type=Path, default=None, help="JMdict_e.gz for the meanings of reading candidates")
    parser.add_argument(
        "--requests",
        type=int,
        choices=(1, 2),
        default=DEFAULT_REQUESTS,
        help="TypeSafe requests per text: 1 asks about the text and its accent phrases together",
    )


def render_answers(response: SystemOneResponse) -> str:
    lines = [f"model: {response.model}"]
    if response.usage is not None:
        lines.append(f"usage: input_tokens={response.usage.input_tokens} output_tokens={response.usage.output_tokens}")
    for key, answer in response.answers.items():
        if isinstance(answer, ChoiceAnswer):
            probabilities = ", ".join(f"{k}={v:.2f}" for k, v in answer.probabilities.items())
            lines.append(f"{key}: {answer.choice} (confidence {answer.confidence:.2f}) [{probabilities}]")
        elif isinstance(answer, ScoreAnswer):
            probabilities = ", ".join(f"{k}={v:.2f}" for k, v in answer.probabilities.items())
            lines.append(f"{key}: score {answer.score:.2f} (confidence {answer.confidence:.2f}) [{probabilities}]")
        elif isinstance(answer, NoulAnswer):
            lines.append(f"{key}: {answer.noul:.2f}")
    return "\n".join(lines)


def render_changes(changes: list[Change]) -> str:
    if not changes:
        return "changes: none"
    return "\n".join(
        f"{change.phrase_id}.{change.field}: {change.before} -> {change.after} ({change.reason})" for change in changes
    )


@dataclass(frozen=True)
class TextPlan:
    """What the first request asks about the text.

    Where to split, where accent phrases break, how to read words and how to voice laughs. Splits,
    breaks and readings that touch a laughter mark are left out, so the mark's own answer decides it.
    """

    text: str
    spans: list[Span]
    candidates: list[ReadingCandidate]
    laughs: list[Laugh]
    stretches: list[Stretch]


def plan_text(
    text: str, lexicon: ReadingLexicon | None, original: AudioQuery, read: Callable[[str], AudioQuery]
) -> TextPlan:
    laughs = find_laughs(text)
    candidates: list[ReadingCandidate] = []
    for candidate in [] if lexicon is None else lexicon.candidates(text, minimum=1):
        current = voicevox_keys(candidate, text, original, read)
        widened = candidate if current else with_voicevox_reading(candidate, text, original, read)
        narrowed = None if widened is None else narrow(widened, current)
        if narrowed is not None:
            candidates.append(narrowed)
    stretches = [] if lexicon is None else find_stretches(text, original, lexicon.words(text), read)
    return TextPlan(
        text,
        [span for span in find_spans(text) if not overlaps(span.start, span.end, laughs)],
        [candidate for candidate in candidates if not overlaps(candidate.start, candidate.end, laughs)],
        laughs,
        [stretch for stretch in stretches if not overlaps(stretch.start, stretch.end, laughs)],
    )


def text_request(plan: TextPlan, model: str) -> SystemOneRequest | None:
    questions = (
        build_split_questions(plan.text, plan.spans)
        | build_boundary_questions(plan.text, plan.stretches)
        | build_reading_questions(plan.text, plan.candidates)
        | build_laugh_questions(plan.text, plan.laughs)
    )
    if not questions:
        return None
    return SystemOneRequest(state={"sentence": plan.text}, model=model, questions=questions)


def phrase_request(text: str, query: AudioQuery, model: str, heights: Sequence[Height] = ()) -> SystemOneRequest:
    questions = build_questions(query) | build_height_questions(heights)
    return SystemOneRequest(state=build_state(text, query), model=model, questions=questions)


def combined_request(plan: TextPlan, original: AudioQuery, model: str, heights: Sequence[Height]) -> SystemOneRequest:
    """Ask the text questions and the phrase questions about VOICEVOX's own phrases in one request."""
    first = text_request(plan, model)
    second = phrase_request(plan.text, original, model, heights=heights)
    questions = second.questions if first is None else first.questions | second.questions
    return SystemOneRequest(state=second.state, model=model, questions=questions)


def carry_answers(
    answers: Mapping[str, Answer], original: AudioQuery, query: AudioQuery, heights: Sequence[Height]
) -> tuple[dict[str, Answer], list[Height], list[Change]]:
    """Move the answers about the phrases of `original` to the phrases of `query` that cover the same words.

    `query` is the query after the text edits. A unit answer needs its phrase and the one before it to
    stay neighbours, and a height answer both phrases of its pair. Answers that cannot move are reported
    as skipped.
    """
    matched = match_phrases(original, query)
    carried: dict[str, Answer] = {}
    skipped: list[Change] = []
    if (sentence_type := answers.get(SENTENCE_TYPE_KEY)) is not None:
        carried[SENTENCE_TYPE_KEY] = sentence_type
    for index in range(len(original.accent_phrases)):
        target = matched.get(index)
        if (unit := answers.get(unit_key(index))) is not None:
            previous = matched.get(index - 1)
            if target is not None and previous is not None and previous.index == target.index - 1:
                carried[unit_key(target.index)] = unit
            else:
                skipped.append(Change(phrase_id(index), f"{UNIT_SUFFIX}{SKIPPED_SUFFIX}", "", "", REGROUPED))
    starts = phrase_starts(query)
    moved_heights: list[Height] = []
    for height in heights:
        left, right = matched.get(height.index), matched.get(height.index + 1)
        if left is None or right is None or right.index != left.index + 1:
            skipped.append(Change(phrase_id(height.index + 1), HEIGHT_SKIPPED_FIELD, "", "", REGROUPED))
            continue
        second = left.index + 1
        end = starts[second] + len(query.accent_phrases[second].moras)
        where = replace(height, index=left.index, start=starts[left.index], boundary=starts[second], end=end)
        if (answer := answers.get(height.key)) is not None:
            carried[where.key] = answer
        moved_heights.append(where)
    return carried, moved_heights, skipped


def apply_text_edits(
    text: str, original: AudioQuery, edits: list[Edit], voice: Voice
) -> tuple[str, AudioQuery, list[Change], list[Edit]]:
    kept, changes = effective_edits(text, edits, original, voice.read)
    kept = drop_overlaps(kept)
    if not kept:
        return text, original, changes, []
    edited, inserted = apply_edits(text, kept)
    query, unmatched = strip_inserted_pauses(voice.read(edited), edited, inserted, voice.read)
    changes += [
        Change(f"s{position}", PAUSE_KEPT_FIELD, "", SEPARATOR, "no pausing phrase ends at the inserted separator")
        for position in unmatched
    ]
    query, joins = join_moved_boundaries(query, edited, kept, voice.read)
    if inserted or joins:
        # VOICEVOX lengthens the mora before a pause and pitches the phrase after it anew; without the pause
        # both are measured again.
        query.accent_phrases = voice.remeasure(query.accent_phrases)
    return edited, query, [*changes, *(edit.change for edit in kept), *joins], kept


def apply_word_accents(
    text: str, query: AudioQuery, choices: list[ReadingChoice], kept: list[Edit], voice: Voice
) -> tuple[AudioQuery, list[Change]]:
    targets, changes = accent_targets(choices)
    placed: list[AccentTarget] = []
    for target in targets:
        where = moved(target, kept)
        if where is None:
            changes.append(Change(target.id, SKIPPED_FIELD, "", "", "another edit falls inside the word"))
        else:
            placed.append(where)
    updated, accent_changes, _ = apply_accents(query, text, placed, voice.read)
    if needs_repitch(accent_changes):
        updated.accent_phrases = voice.repitch(updated.accent_phrases)
    return updated, [*changes, *accent_changes]


def choose_text_edits(
    plan: TextPlan, answers: Mapping[str, Answer], policy: Policy
) -> tuple[list[ReadingChoice], list[Edit]]:
    choices = choose_options(plan.candidates, answers, policy.threshold)
    edits = [
        *choose_splits(plan.text, plan.spans, answers, policy.threshold),
        *choose_boundaries(plan.text, plan.stretches, answers, policy.threshold),
        *reading_edits(choices),
        *laugh_edits(plan.laughs, answers, policy.threshold),
    ]
    return choices, edits


@dataclass(frozen=True)
class EditedText:
    """The text after the text answers, VOICEVOX's query for it and what changed on the way.

    kept are the edits made.
    """

    text: str
    query: AudioQuery
    changes: list[Change]
    kept: list[Edit]


def edit_text(
    plan: TextPlan, original: AudioQuery, answers: Mapping[str, Answer], policy: Policy, voice: Voice
) -> EditedText:
    choices, edits = choose_text_edits(plan, answers, policy)
    edited, query, changes, kept = apply_text_edits(plan.text, original, edits, voice)
    query, accent_changes = apply_word_accents(edited, query, choices, kept, voice)
    return EditedText(edited, query, [*changes, *accent_changes], kept)


@dataclass(frozen=True)
class Correction:
    """The text as edited, VOICEVOX's query for the text as given, the corrected query and what changed."""

    text: str
    original: AudioQuery
    query: AudioQuery
    changes: list[Change]


@dataclass(frozen=True)
class Corrector:
    """Corrects a text with TypeSafe requests about the text and about its accent phrases.

    In two requests, the phrase questions come second and are about the text as the first answers
    edited it. In one, they are about VOICEVOX's phrases of the text as given, and each answer moves to
    the phrase covering the same words after the edits (`carry_answers`).
    The phrase questions also ask how phrases hang together; those answers lower phrases VOICEVOX raises
    after a phrase that modifies them. Nuclei VOICEVOX would hide are moved earlier without asking.
    """

    lexicon: ReadingLexicon | None
    tokenizer: Tokenizer
    model: str
    policy: Policy
    one_request: bool = False

    def requests(self, text: str, voice: Voice) -> list[SystemOneRequest]:
        """Return the requests a correction would start with, before any answers shape a second one."""
        original = voice.read(text)
        plan = plan_text(text, self.lexicon, original, voice.read)
        heights = find_heights(text, original, self.tokenizer.tokens(text), voice.read)
        if self.one_request:
            return [combined_request(plan, original, self.model, heights)]
        first = text_request(plan, self.model)
        second = phrase_request(text, original, self.model, heights=heights)
        return [second] if first is None else [first, second]

    def correct(self, text: str, voice: Voice, ask: Ask) -> Correction:
        original = voice.read(text)
        plan = plan_text(text, self.lexicon, original, voice.read)
        if self.one_request:
            asked = find_heights(text, original, self.tokenizer.tokens(text), voice.read)
            answers = ask(combined_request(plan, original, self.model, asked)).answers
            done = edit_text(plan, original, answers, self.policy, voice)
            carried, heights, skipped = carry_answers(answers, original, done.query, asked)
            return self._finish(original, replace(done, changes=[*done.changes, *skipped]), carried, heights, voice)
        first = text_request(plan, self.model)
        done = EditedText(text, original, [], [])
        if first is not None:
            done = edit_text(plan, original, ask(first).answers, self.policy, voice)
        heights = find_heights(done.text, done.query, self.tokenizer.tokens(done.text), voice.read)
        answers = ask(phrase_request(done.text, done.query, self.model, heights)).answers
        return self._finish(original, done, answers, heights, voice)

    def _finish(
        self,
        original: AudioQuery,
        done: EditedText,
        answers: Mapping[str, Answer],
        heights: Sequence[Height],
        voice: Voice,
    ) -> Correction:
        edited, query = done.text, done.query
        words = [] if self.lexicon is None else self.lexicon.words(edited)
        attachable = frozenset() if self.lexicon is None else attachable_phrases(query, edited, words, voice.read)
        roles = PhraseRoles(attachable, laugh_phrases(query, edited, voiced_spans(done.kept), voice.read))
        updated, structure_changes = apply_structure(query, answers, self.policy, roles)
        updated, late_changes = move_late_nuclei(updated)
        if needs_repitch([*structure_changes, *late_changes]):
            updated.accent_phrases = voice.repitch(updated.accent_phrases)
        updated, height_changes = lower_heights(updated, heights, answers, self.policy.height_threshold)
        changes = [*done.changes, *structure_changes, *late_changes, *height_changes]
        return Correction(edited, original, updated, changes)


def build_lexicon(args: argparse.Namespace, settings: Settings) -> ReadingLexicon | None:
    dict_path = args.sudachi_dict or settings.sudachi_dict_path
    if dict_path is None:
        return None
    jmdict_path = args.jmdict or settings.jmdict_path
    glosses = None if jmdict_path is None else GlossDictionary(jmdict_path)
    return ReadingLexicon(dict_path, AccentDictionary(), glosses)


def build_corrector(args: argparse.Namespace, settings: Settings) -> Corrector:
    policy = Policy(threshold=args.threshold, height_threshold=args.height_threshold)
    lexicon = build_lexicon(args, settings)
    return Corrector(lexicon, UnidicTokenizer(), settings.typesafe_model, policy, one_request=args.requests == 1)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    voice = Voice(VoicevoxClient(settings.voicevox_url), args.speaker)
    corrector = build_corrector(args, settings)
    if args.dry_run:
        for request in corrector.requests(args.text, voice):
            print(render_request(settings.typesafe_url, request))
            print()
        print("[dry-run] request not sent")
        return 0
    api_key = settings.typesafe_api_key
    if api_key is None:
        print("TYPESAFE_API_KEY is not set", file=sys.stderr)
        return 1
    pacer = Pacer(REQUEST_INTERVAL_SEC)

    def ask(request: SystemOneRequest) -> SystemOneResponse:
        pacer.wait()
        print(render_request(settings.typesafe_url, request))
        response = send(settings.typesafe_url, api_key, request)
        print()
        print(render_answers(response))
        print()
        return response

    correction = corrector.correct(args.text, voice, ask)
    print(render_changes(correction.changes))
    write_outputs(args.out, voice.synthesize(correction.original), voice.synthesize(correction.query))
    return 0


def write_outputs(out: Path, before: bytes, after: bytes) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    before_path = out.with_name(f"{out.stem}.before{out.suffix}")
    before_path.write_bytes(before)
    out.write_bytes(after)
    print(f"wrote {before_path} and {out}")


if __name__ == "__main__":
    sys.exit(main())
