from dataclasses import replace

import pytest

from voicevox_jev_proxy.edits import Edit, apply_edits, drop_overlaps, effective_edits, placed_spans
from voicevox_jev_proxy.prosody import Change
from voicevox_jev_proxy.voicevox import AccentPhrase, AudioQuery, Mora


def edit(start: int, end: int, replacement: str) -> Edit:
    return Edit(start, end, replacement, Change(f"e{start}", "x", "", replacement, ""))


def query(*readings: list[str]) -> AudioQuery:
    phrases = [
        AccentPhrase(moras=[Mora(text=kana, vowel="a", vowel_length=0.1, pitch=5.5) for kana in reading], accent=1)
        for reading in readings
    ]
    return AudioQuery(
        accent_phrases=phrases,
        speedScale=1.0,
        pitchScale=0.0,
        intonationScale=1.0,
        volumeScale=1.0,
        prePhonemeLength=0.1,
        postPhonemeLength=0.1,
        outputSamplingRate=24000,
        outputStereo=False,
    )


WARETA = ["ワ", "レ", "タ"]
W3 = ["ダ", "ブ", "リュ", "ウ"] * 3


def test_apply_edits_mixes_insertions_and_replacements():
    text = "王城を出て、都に十里はなれた"
    edits = [edit(10, 10, "、"), edit(6, 7, "みやこ")]
    new_text, separators = apply_edits(text, edits)
    assert new_text == "王城を出て、みやこに十里、はなれた"
    assert separators == [12]
    assert new_text[12] == "、"


def test_apply_edits_rejects_overlap_and_drop_overlaps_keeps_first():
    edits = [edit(0, 3, "あ"), edit(2, 2, "、"), edit(3, 4, "い")]
    with pytest.raises(ValueError, match="overlaps"):
        apply_edits("一二三四", edits)
    kept = drop_overlaps(edits)
    assert [(e.start, e.end) for e in kept] == [(0, 3), (3, 4)]


def test_placed_spans_follow_earlier_edits_and_skip_separators_and_silent_marks():
    text = "王城を出て、都に十里はなれたw"
    edits = [edit(3, 3, "、"), replace(edit(6, 7, "みやこ"), reading="ミヤコ"), replace(edit(14, 15, ""), reading="")]
    edited, _ = apply_edits(text, edits)
    assert placed_spans(edits) == [(7, 10)]
    assert edited[7:10] == "みやこ"


def test_effective_edits_keep_a_word_removed_from_the_end_of_the_text():
    removal = replace(edit(3, 6, ""), reading="")
    kept, skipped = effective_edits("割れたwww", [removal], query(WARETA, W3), {"割れた": query(WARETA)}.__getitem__)
    assert (kept, skipped) == ([removal], [])


def test_effective_edits_still_skip_a_word_at_the_end_that_joins_the_phrase_before():
    reads = {"割れたぐさ": query([*WARETA, "グ", "サ"]), "割れたグサ": query([*WARETA, "グ", "サ"])}
    kusa = replace(edit(3, 4, "ぐさ"), reading="グサ")
    kept, skipped = effective_edits("割れた草", [kusa], query(WARETA, ["ク", "サ"]), reads.__getitem__)
    assert kept == []
    assert [change.field for change in skipped] == ["x_skipped"]
