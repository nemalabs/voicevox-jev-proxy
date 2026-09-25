from dataclasses import replace

import pytest

from voicevox_jev_proxy.edits import Edit, apply_edits, drop_overlaps, placed_spans
from voicevox_jev_proxy.prosody import Change


def edit(start: int, end: int, replacement: str) -> Edit:
    return Edit(start, end, replacement, Change(f"e{start}", "x", "", replacement, ""))


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
