"""Measure whether Jev can pick Japanese pitch-accent patterns without a dictionary.

Gold accents come from VOICEVOX's own analysis of kanji input, so the score is agreement with OpenJTalk's
dictionary, not with a human-checked accent dictionary.
"""

import argparse
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import JsonValue

from vovovo.settings import Settings
from vovovo.typesafe import Answer, ChoiceAnswer, SystemOneRequest, render_request, send
from vovovo.voicevox import AccentPhrase, AudioQuery, VoicevoxClient

MAX_REQUESTS = 40
MIN_MORAS = 2
REQUEST_INTERVAL_SEC = 3.0
ATAMADAKA = "atamadaka"
NAKADAKA = "nakadaka"
HEIBAN_OR_ODAKA = "heiban_or_odaka"
CATEGORY_CRITERIA: dict[str, JsonValue] = {
    ATAMADAKA: "頭高型。1 モーラ目だけ高く、2 モーラ目で下がる",
    NAKADAKA: "中高型。途中のモーラまで高く、句の中で下がる",
    HEIBAN_OR_ODAKA: "平板型または尾高型。句の最後のモーラまで下がらない",
}

SENTENCES = [
    "雨が降ったから、飴を買った",
    "橋を渡って、箸を買った",
    "花が咲いたので、鼻がむずむずする",
    "牡蠣を食べて、柿も食べた",
    "雲が晴れて、蜘蛛が出てきた",
    "酒を飲みながら、鮭を焼いた",
    "今日は天気がいいから、公園に行こう",
    "電車が遅れて、会議に遅刻した",
    "猫が窓の外を見ている",
    "明日の朝、駅で待ち合わせよう",
    "私は毎日、コーヒーを飲む",
    "図書館で本を借りた",
    "山の上から、海が見えた",
    "妹が新しい自転車を買った",
    "先生が質問に答えた",
    "東京から大阪まで新幹線で行く",
    "冷蔵庫に牛乳が入っている",
    "春になると、桜が咲く",
    "財布を家に忘れた",
    "昨日の夜、友達と映画を見た",
    "犬が庭で走り回っている",
    "日本の夏は暑くて湿気が多い",
    "パソコンの電源を入れた",
    "音楽を聞きながら勉強する",
    "病院で薬をもらった",
    "兄は銀行で働いている",
    "このリンゴは甘くておいしい",
    "寒いので、窓を閉めてください",
    "彼女は英語と中国語を話す",
    "空港までタクシーで向かった",
]


@dataclass(frozen=True)
class PhraseCase:
    sentence: str
    phrase_index: int
    moras: tuple[str, ...]
    gold_accent: int

    @property
    def reading(self) -> str:
        return "".join(self.moras)

    @property
    def gold_category(self) -> str:
        return category_for(self.gold_accent, len(self.moras))

    @property
    def question_id(self) -> str:
        return f"p{self.phrase_index + 1}"


@dataclass(frozen=True)
class CaseResult:
    sentence: str
    reading: str
    gold_accent: int
    pattern_accent: int | None
    pattern_confidence: float | None
    category_gold: str
    category_answer: str | None
    category_confidence: float | None

    @property
    def pattern_hit(self) -> bool:
        return self.pattern_accent == self.gold_accent

    @property
    def category_hit(self) -> bool:
        return self.category_answer == self.category_gold

    @property
    def heiban_baseline_hit(self) -> bool:
        return self.category_gold == HEIBAN_OR_ODAKA


def category_for(accent: int, mora_count: int) -> str:
    if accent == 1:
        return ATAMADAKA
    if accent == mora_count:
        return HEIBAN_OR_ODAKA
    return NAKADAKA


def pattern_label(moras: tuple[str, ...], accent: int) -> str:
    return "".join(moras[:accent]) + "'" + "".join(moras[accent:])


def accent_from_label(label: str, moras: tuple[str, ...]) -> int | None:
    head = label.split("'", 1)[0]
    consumed = ""
    for index, text in enumerate(moras):
        consumed += text
        if consumed == head:
            return index + 1
    return None


def pattern_criteria(moras: tuple[str, ...]) -> dict[str, JsonValue]:
    criteria: dict[str, JsonValue] = {}
    for accent in range(1, len(moras) + 1):
        if accent == len(moras):
            description = "最後まで下がらない (平板型または尾高型)"
        else:
            description = f"{accent} モーラ目の「{moras[accent - 1]}」のあとで下がる"
        criteria[pattern_label(moras, accent)] = description
    return criteria


def cases_from_query(sentence: str, query: AudioQuery) -> list[PhraseCase]:
    cases: list[PhraseCase] = []
    for index, phrase in enumerate(query.accent_phrases):
        moras = tuple(mora.text for mora in phrase.moras)
        if len(moras) < MIN_MORAS:
            continue
        cases.append(PhraseCase(sentence, index, moras, phrase.accent))
    return cases


def build_state(sentence: str, phrases: list[AccentPhrase]) -> dict[str, JsonValue]:
    readings: list[JsonValue] = [
        {"id": f"p{index + 1}", "reading": phrase.reading} for index, phrase in enumerate(phrases)
    ]
    return {"sentence": sentence, "phrases": readings}


def build_questions(cases: list[PhraseCase]) -> dict[str, JsonValue]:
    questions: dict[str, JsonValue] = {}
    for case in cases:
        target = f"`phrases[{case.phrase_index}]` の「{case.reading}」"
        questions[f"{case.question_id}_pattern"] = {
            "type": "choice",
            "instructions": {
                "target": target,
                "question": "東京式アクセントで読むとき、この句の音の高さはどこで下がるか。' の直前のモーラまで高い",
            },
            "criteria": pattern_criteria(case.moras),
        }
        questions[f"{case.question_id}_type"] = {
            "type": "choice",
            "instructions": {"target": target, "question": "東京式アクセントでこの句はどの型か"},
            "criteria": CATEGORY_CRITERIA,
        }
    return questions


def score_cases(cases: list[PhraseCase], answers: Mapping[str, Answer]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        pattern = answers.get(f"{case.question_id}_pattern")
        category = answers.get(f"{case.question_id}_type")
        pattern_accent = None
        pattern_confidence = None
        if isinstance(pattern, ChoiceAnswer):
            pattern_accent = accent_from_label(pattern.choice, case.moras)
            pattern_confidence = pattern.confidence
        category_answer = None
        category_confidence = None
        if isinstance(category, ChoiceAnswer):
            category_answer = category.choice
            category_confidence = category.confidence
        results.append(
            CaseResult(
                sentence=case.sentence,
                reading=case.reading,
                gold_accent=case.gold_accent,
                pattern_accent=pattern_accent,
                pattern_confidence=pattern_confidence,
                category_gold=case.gold_category,
                category_answer=category_answer,
                category_confidence=category_confidence,
            )
        )
    return results


def render_summary(results: list[CaseResult]) -> str:
    total = len(results)
    if total == 0:
        return "no phrases scored"
    pattern_hits = sum(r.pattern_hit for r in results)
    category_hits = sum(r.category_hit for r in results)
    baseline_hits = sum(r.heiban_baseline_hit for r in results)
    lines = [
        f"phrases: {total}",
        f"pattern (exact accent) hits: {pattern_hits}/{total} = {pattern_hits / total:.2f}",
        f"type (3-way category) hits: {category_hits}/{total} = {category_hits / total:.2f}",
        f"baseline always-heiban hits: {baseline_hits}/{total} = {baseline_hits / total:.2f}",
    ]
    return "\n".join(lines)


def render_rows(results: list[CaseResult]) -> str:
    rows = []
    for r in results:
        pattern_mark = "o" if r.pattern_hit else "x"
        category_mark = "o" if r.category_hit else "x"
        pattern_conf = f"{r.pattern_confidence:.2f}" if r.pattern_confidence is not None else "-"
        category_conf = f"{r.category_confidence:.2f}" if r.category_confidence is not None else "-"
        rows.append(
            f"{pattern_mark}{category_mark} {r.reading}: gold={r.gold_accent} jev={r.pattern_accent} ({pattern_conf})"
            f" | {r.category_gold} -> {r.category_answer} ({category_conf}) | {r.sentence}"
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vovovo-pilot", description="Accent-knowledge pilot for Jev")
    parser.add_argument("--speaker", type=int, default=3)
    parser.add_argument("--limit", type=int, default=len(SENTENCES), help="number of sentences to send")
    parser.add_argument("--dry-run", action="store_true", help="print the first request and stop")
    parser.add_argument("--out", type=Path, default=Path(".build/pilot/results.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    voicevox = VoicevoxClient(settings.voicevox_url)
    sentences = SENTENCES[: args.limit]
    if len(sentences) > MAX_REQUESTS:
        msg = f"{len(sentences)} sentences exceed MAX_REQUESTS={MAX_REQUESTS}"
        raise RuntimeError(msg)
    results: list[CaseResult] = []
    requests_sent = 0
    for index, sentence in enumerate(sentences):
        if requests_sent != index:
            msg = f"cursor did not advance: index={index} requests_sent={requests_sent}"
            raise RuntimeError(msg)
        query = voicevox.audio_query(sentence, args.speaker)
        cases = cases_from_query(sentence, query)
        request = SystemOneRequest(
            state=build_state(sentence, query.accent_phrases),
            model=settings.typesafe_model,
            questions=build_questions(cases),
        )
        if args.dry_run:
            print(render_request(settings.typesafe_url, request))
            print(
                f"\n[dry-run] would send {len(sentences)} requests, {REQUEST_INTERVAL_SEC}s apart, cap {MAX_REQUESTS}"
            )
            return 0
        if settings.typesafe_api_key is None:
            print("TYPESAFE_API_KEY is not set", file=sys.stderr)
            return 1
        response = send(settings.typesafe_url, settings.typesafe_api_key, request)
        results += score_cases(cases, response.answers)
        requests_sent = index + 1
        print(f"[{requests_sent}/{len(sentences)}] {sentence} ({len(cases)} phrases)")
        if requests_sent < len(sentences):
            time.sleep(REQUEST_INTERVAL_SEC)
    print()
    print(render_rows(results))
    print()
    print(render_summary(results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
