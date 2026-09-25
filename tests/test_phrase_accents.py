import pytest

from vovovo.phrase_accents import (
    DROP,
    AccentOption,
    AccentQuestion,
    Settled,
    Token,
    UnidicTokenizer,
    accent_questions,
    apply_accent_answers,
    build_accent_questions,
    combine,
    phrase_type,
    pitch,
    type_label,
    word_marked,
    word_pitch,
)
from vovovo.prosody import Policy
from vovovo.typesafe import ChoiceAnswer
from vovovo.voicevox import AccentPhrase, AudioQuery, Mora

POLICY = Policy(threshold=0.6)
GA = "動詞%F2@0,名詞%F1"
MADE = "名詞%F2@1,形容詞%F2@1,動詞%F2@1"


def phrase(texts: list[str], accent: int) -> AccentPhrase:
    return AccentPhrase(moras=[Mora(text=t, vowel="a", vowel_length=0.1, pitch=5.5) for t in texts], accent=accent)


def query(*phrases: AccentPhrase) -> AudioQuery:
    return AudioQuery(
        accent_phrases=list(phrases),
        speedScale=1.0,
        pitchScale=0.0,
        intonationScale=1.0,
        volumeScale=1.0,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        outputSamplingRate=24000,
        outputStereo=False,
    )


def reader(mora_counts: dict[str, int]):
    def read(text: str) -> AudioQuery:
        count = mora_counts[text]
        return query(phrase(["ア"] * count, count)) if count else query()

    return read


def noun(start: int, surface: str, pron: str, *accents: int, pos2: str = "普通名詞") -> Token:
    return Token(start, surface, ("名詞", pos2), pron, accents, "C1")


def particle(start: int, surface: str, pron: str, rules: str = GA) -> Token:
    return Token(start, surface, ("助詞", "格助詞"), pron, (), rules)


def choice(value: str, confidence: float = 0.9) -> ChoiceAnswer:
    return ChoiceAnswer(type="choice", choice=value, confidence=confidence, probabilities={value: confidence})


@pytest.mark.parametrize(
    ("accent", "moras", "rules", "expected"),
    [
        (1, 2, GA, 1),
        (0, 2, GA, 0),
        (0, 2, MADE, 3),
        (1, 2, MADE, 1),
        (0, 3, "動詞%F3@1,名詞%F3@1", 0),
        (2, 3, "動詞%F3@1,名詞%F3@1", 4),
        (0, 3, "名詞%F4@1,動詞%F4@1,形容詞%F4@1", 4),
        (2, 3, "名詞%F4@1,動詞%F4@1,形容詞%F4@1", 4),
        (2, 3, "名詞%F5", 0),
        (0, 3, "名詞%F6@1,-1", 4),
        (2, 3, "名詞%F6@1,-1", 2),
        (0, 2, "名詞%F2@1,形容詞%F2@-1動詞%F2@0", 3),
    ],
)
def test_combine_follows_the_unidic_table_for_a_noun_before(accent: int, moras: int, rules: str, expected: int):
    assert combine(accent, moras, rules) == expected


@pytest.mark.parametrize(
    ("accent", "moras", "rules"),
    [
        (0, 2, "動詞%F2@1,形容詞%F4@-2"),
        (0, 2, "*"),
        (0, 2, "名詞%F2"),
        (0, 1, "名詞%F4@-1"),
    ],
)
def test_combine_gives_nothing_without_a_usable_noun_rule(accent: int, moras: int, rules: str):
    assert combine(accent, moras, rules) is None


def test_phrase_type_chains_particles_after_the_noun():
    head = noun(0, "今", "イマ", 1, 0)
    attached = [particle(1, "まで", "マデ", MADE), particle(3, "は", "ワ")]
    assert phrase_type(1, head, attached) == 1
    assert phrase_type(0, head, attached) == 3
    assert phrase_type(0, head, [particle(1, "た", "タ", "動詞%F2@1")]) is None


@pytest.mark.parametrize(("moras", "accent", "expected"), [(3, 1, "高低低"), (3, 2, "低高低"), (3, 3, "低高高")])
def test_pitch_writes_tokyo_high_and_low(moras: int, accent: int, expected: str):
    assert pitch(moras, accent) == expected


@pytest.mark.parametrize(
    ("word_type", "moras", "label", "word", "levels"),
    [
        (0, 2, "平板型[0]", "ダレ", "低高(高)"),
        (1, 2, "頭高型[1]", f"ダ{DROP}レ", "高低(低)"),
        (2, 2, "尾高型[2]", f"ダレ{DROP}", "低高(低)"),
        (2, 3, "中高型[2]", f"ダレ{DROP}カ", "低高低(低)"),
    ],
)
def test_word_types_are_written_the_dictionary_way(word_type: int, moras: int, label: str, word: str, levels: str):
    assert type_label(word_type, moras) == label
    assert word_marked(["ダ", "レ", "カ"][:moras], word_type) == word
    assert word_pitch(moras, word_type) == levels


DAREGA = ["ダ", "レ", "ガ"]
DARE = f"ダ{DROP}レガ"
FLAT = "平板型[0]"
HEAD_HIGH = "頭高型[1]"
DARE_OPTIONS = {FLAT: AccentOption(0, 3), HEAD_HIGH: AccentOption(1, 1)}


def darega_tokens() -> list[Token]:
    return [
        Token(3, "誰", ("代名詞", "*"), "ダレ", (1,), "*"),
        particle(4, "が", "ガ"),
        Token(5, "来", ("動詞", "非自立可能"), "キ", (1,), "C1"),
    ]


def test_accent_questions_offer_the_rule_nucleus_beside_voicevox():
    q = query(phrase(["ソ", "レ", "デ"], 3), phrase(DAREGA, 3), phrase(["キ", "タ"], 1))
    (question,) = accent_questions(q, "それで誰が来た", darega_tokens(), reader({"それで": 3}))
    assert question == AccentQuestion(1, ("誰", "が"), ("ダ", "レ"), DARE_OPTIONS)


def test_accent_questions_need_a_word_type_that_gives_the_voicevox_nucleus():
    tokens = [Token(0, "誰", ("代名詞", "*"), "ダレ", (1,), "*"), particle(1, "から", "カラ", "名詞%F1")]
    q = query(phrase(["ダ", "レ", "カ", "ラ"], 3))
    assert accent_questions(q, "誰から", tokens, reader({})) == []


def test_accent_questions_skip_phrases_that_agree_or_are_settled_elsewhere():
    text = "それで誰が来た"
    read = reader({"それで": 3})
    agreeing = query(phrase(["ソ", "レ", "デ"], 3), phrase(DAREGA, 1), phrase(["キ", "タ"], 1))
    assert accent_questions(agreeing, text, darega_tokens(), read) == []
    q = query(phrase(["ソ", "レ", "デ"], 3), phrase(DAREGA, 3), phrase(["キ", "タ"], 1))
    assert accent_questions(q, text, darega_tokens(), read, Settled(phrases=frozenset({1}))) == []
    assert accent_questions(q, text, darega_tokens(), read, Settled(spans=[(3, 4)])) == []


def test_accent_questions_need_the_words_to_spell_the_phrase():
    q = query(phrase(["ソ", "レ", "デ"], 3), phrase(["ダ", "レ", "モ"], 3), phrase(["キ", "タ"], 1))
    assert accent_questions(q, "それで誰が来た", darega_tokens(), reader({"それで": 3})) == []


def test_accent_questions_leave_compounds_and_nouns_inside_a_phrase():
    compound = [noun(0, "百姓", "ヒャクショー", 1), noun(2, "家", "ヤ", 1), particle(3, "の", "ノ", "名詞%F1")]
    q = query(phrase(["ヒャ", "ク", "ショ", "オ"], 1), phrase(["ヤ", "ノ"], 2))
    assert accent_questions(q, "百姓家の", compound, reader({"百姓": 4})) == []
    inside = [particle(0, "を", "オ"), noun(1, "今", "イマ", 1), particle(2, "が", "ガ")]
    assert accent_questions(query(phrase(["オ", "イ", "マ", "ガ"], 4)), "を今が", inside, reader({"を": 1})) == []
    stem = [noun(0, "そう", "ソー", 1, pos2="助動詞語幹"), Token(2, "です", ("助動詞", "*"), "デス", (), "名詞%F2@1")]
    assert accent_questions(query(phrase(["ソ", "オ", "デ", "ス"], 3)), "そうです", stem, reader({})) == []


def test_build_accent_questions_offer_dictionary_types_of_the_word():
    q = query(phrase(DAREGA, 3))
    questions = build_accent_questions(q, [AccentQuestion(0, ("誰", "が"), ("ダ", "レ"), DARE_OPTIONS)])
    built = questions["p1_accent"]
    assert built["criteria"] == {
        FLAT: {"word": "ダレ", "pitch": "低高(高)"},
        HEAD_HIGH: {"word": f"ダ{DROP}レ", "pitch": "高低(低)"},
        "none": "どれも当てはまらない",
    }
    assert (built["instructions"]["word"], built["instructions"]["reading"]) == ("誰", "ダレ")


def test_apply_accent_answers_sets_the_chosen_nucleus_only_when_confident():
    q = query(phrase(DAREGA, 3))
    question = AccentQuestion(0, ("誰", "が"), ("ダ", "レ"), DARE_OPTIONS)
    updated, changes = apply_accent_answers(q, {"p1_accent": choice(HEAD_HIGH)}, [question], POLICY)
    assert updated.accent_phrases[0].accent == 1
    assert q.accent_phrases[0].accent == 3
    assert [(c.phrase_id, c.field, c.before, c.after) for c in changes] == [("p1", "accent", "ダレガ", DARE)]
    for answer in (choice(HEAD_HIGH, 0.5), choice("none"), choice(FLAT)):
        updated, changes = apply_accent_answers(q, {"p1_accent": answer}, [question], POLICY)
        assert updated.accent_phrases[0].accent == 3
        assert changes == []


def test_unidic_tokenizer_keeps_offsets_and_drops_punctuation():
    tokens = UnidicTokenizer().tokens("「誰が、来た。」")
    assert [(t.start, t.surface, t.pos[0]) for t in tokens] == [
        (1, "誰", "代名詞"),
        (2, "が", "助詞"),
        (4, "来", "動詞"),
        (5, "た", "助動詞"),
    ]
    assert (tokens[0].pron, tokens[0].accents) == ("ダレ", (1,))
    assert "名詞%F1" in tokens[1].combination
