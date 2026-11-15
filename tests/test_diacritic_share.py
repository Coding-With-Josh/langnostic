"""Tests for the h7 diacritics-stripped proxy. Stdlib plus pytest only."""
import json

import pytest

from speechgap.diacritic_share import (bootstrap_share, check_reference_agreement,
                                       main, score_cache, select_group,
                                       to_markdown, validate_records)
from speechgap.report import score


def _rec(uid, ref="a b", hyp="a b", speaker=None, group="g"):
    return {
        "utt_id": uid,
        "speaker": speaker or uid,
        "group": group,
        "ref": ref,
        "hyp": hyp,
    }


def test_share_of_errors_that_vanish_with_marks():
    recs = [_rec(f"u{i}", ref="à b", hyp="a b") for i in range(4)]
    out = score_cache(recs, n_boot=100)["all"]
    assert out["wer_with_marks"]["rate"] == 0.5
    assert out["wer_stripped"]["rate"] == 0.0
    assert out["share_vanished"]["rate"] == 1.0
    assert out["errors_with_marks"] == 4
    assert out["errors_stripped"] == 0


def test_share_is_partial_when_a_real_error_remains():
    recs = [_rec(f"u{i}", ref="à x", hyp="a y") for i in range(4)]
    out = score_cache(recs, n_boot=100)["all"]
    assert out["wer_with_marks"]["rate"] == 1.0
    assert out["wer_stripped"]["rate"] == 0.5
    assert out["share_vanished"]["rate"] == 0.5


def test_non_looped_uses_the_existing_loop_rule():
    recs = [_rec(f"n{i}", ref="à b", hyp="a b") for i in range(4)]
    recs.append(_rec("loop", ref="à b", hyp="a" * 5000))
    out = score_cache(recs, n_boot=50)
    assert out["all"]["n_clips"] == 5
    assert out["non_looped"]["n_clips"] == 4
    assert out["non_looped"]["share_vanished"]["rate"] == 1.0


def test_zero_errors_yields_no_share_not_zero():
    recs = [_rec(f"u{i}") for i in range(4)]
    out = score_cache(recs, n_boot=50)["all"]
    assert out["errors_with_marks"] == 0
    assert out["share_vanished"] == {
        "rate": None, "lo": None, "hi": None, "n_speakers": 4,
    }


def test_single_speaker_gets_point_only():
    recs = [_rec(f"u{i}", ref="à b", hyp="a b", speaker="only")
            for i in range(3)]
    out = score_cache(recs, n_boot=50)["all"]["share_vanished"]
    assert out["rate"] == 1.0
    assert out["lo"] == out["hi"] == out["rate"]
    assert out["n_speakers"] == 1


def test_bootstrap_share_drops_undefined_resamples():
    # Resamples that pick only the perfect speaker have a zero denominator.
    # Counting them as 0 would pull the lower bound to 0.
    out = bootstrap_share([0, 1], [0, 0], ["p", "b"], n_boot=100)
    assert out["rate"] == out["lo"] == out["hi"] == 1.0


def test_score_is_deterministic():
    recs = [_rec(f"u{i}", ref="à b", hyp="a b", speaker=f"s{i % 3}")
            for i in range(9)]
    assert score_cache(recs, n_boot=200) == score_cache(recs, n_boot=200)


def test_wer_matches_the_existing_report():
    recs = [_rec(f"u{i}", ref="à b", hyp="a b", speaker=f"s{i % 3}")
            for i in range(9)]
    rep = score(recs, n_boot=100)["g"]
    out = score_cache(recs, n_boot=100)["all"]
    assert out["wer_with_marks"]["rate"] == rep["wer"]["rate"]
    assert out["wer_with_marks"]["n_speakers"] == rep["wer"]["n_speakers"]


@pytest.mark.parametrize("records", [
    [],
    [{"utt_id": "a", "speaker": "s", "group": "g", "ref": "x"}],
    [_rec("a", ref=None)],
    [{**_rec("a"), "speaker": ""}],
    [_rec("a"), _rec("a")],
    ["not a record"],
])
def test_validation_rejects_malformed_records(records):
    with pytest.raises(ValueError):
        validate_records(records)


def test_multiple_groups_must_be_selected():
    recs = [_rec("a", group="x"), _rec("b", group="y")]
    with pytest.raises(ValueError):
        score_cache(recs, n_boot=10)
    assert len(select_group(recs, "x")) == 1
    with pytest.raises(ValueError):
        select_group(recs, "missing")


def test_reference_agreement_ignores_case_and_punctuation():
    a = [_rec("1", ref="Àwọn, b")]
    b = [_rec("2", ref="àwọn b")]
    check_reference_agreement([("a", a), ("b", b)])


def test_reference_agreement_stops_on_a_real_mismatch():
    a = [_rec("1", ref="Àwọn b")]
    b = [_rec("2", ref="àwọn c")]
    with pytest.raises(ValueError):
        check_reference_agreement([("a", a), ("b", b)])


def test_reference_agreement_skips_different_lengths():
    a = [_rec("1", ref="one")]
    b = [_rec("2", ref="two"), _rec("3", ref="three")]
    check_reference_agreement([("a", a), ("b", b)])


def test_markdown_escapes_labels_and_groups():
    recs = [_rec(f"u{i}", ref="à b", hyp="a b") for i in range(3)]
    md = to_markdown({
        "bad|label": {**score_cache(recs, n_boot=20), "group": "g|x"},
    })
    assert "bad\\|label" in md
    assert "g\\|x" in md


def test_main_writes_json_and_prints_table(tmp_path, capsys):
    rows = [_rec(f"u{i}", ref="à b", hyp="a b", speaker=f"s{i}")
            for i in range(4)]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    for path in (first, second):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                        encoding="utf-8")
    out = tmp_path / "out.json"
    main([str(first), str(second), "--out", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["caches"]) == {"first", "second"}
    assert payload["caches"]["first"]["all"]["share_vanished"]["rate"] == 1.0
    assert "share vanished" in capsys.readouterr().out


def test_main_filters_one_group(tmp_path):
    rows = ([_rec(f"u{i}", group="yo") for i in range(4)]
            + [_rec("e0", group="en")])
    cache = tmp_path / "mixed.jsonl"
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows),
                     encoding="utf-8")
    results = main([str(cache), "--group-for", f"{cache}=yo"])
    assert results["mixed"]["group"] == "yo"
    assert results["mixed"]["all"]["n_clips"] == 4


if __name__ == "__main__":  # run without pytest: python tests/test_diacritic_share.py
    import sys
    sys.path.insert(0, ".")
