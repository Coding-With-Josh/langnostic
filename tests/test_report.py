"""Stdlib-only tests for the worst-example section of the report."""
import unicodedata

from speechgap.report import (LOOP_CHARS, LOOP_MARKER, diacritic_retention,
                              score, to_markdown, truncate_hyp, worst_examples,
                              wrong_script_rate)


def _rec(uid, group="g", ref="one two three", hyp="one two three"):
    return {"utt_id": uid, "speaker": uid, "group": group, "ref": ref, "hyp": hyp}


def test_short_hyp_is_untouched():
    assert truncate_hyp("a" * LOOP_CHARS) == "a" * LOOP_CHARS
    assert truncate_hyp("bawo ni") == "bawo ni"


def test_long_hyp_is_cut_and_marked():
    out = truncate_hyp("a" * 5000)
    assert out.startswith("a" * LOOP_CHARS)
    assert out.endswith(LOOP_MARKER.strip())
    assert len(out) <= LOOP_CHARS + len(LOOP_MARKER)


def test_worst_examples_order_by_wer_then_utt_id():
    recs = [
        _rec("b", ref="one two", hyp="one two"),        # 0.0
        _rec("a", ref="one two", hyp="one"),            # 0.5, deletion
        _rec("c", ref="one two", hyp="one two three"),  # 0.5, insertion
        _rec("d", ref="one two", hyp="x y z"),          # 1.5, two subs + insertion
    ]
    top = worst_examples(recs, "g", k=3)
    assert [e["utt_id"] for e in top] == ["d", "a", "c"]
    assert top[0]["wer"] == 1.5


def test_worst_examples_truncates_a_loop():
    recs = [_rec(str(i), hyp=" ".join(["a"] * 5000)) for i in range(3)]
    out = worst_examples(recs, "g", k=1)
    assert out[0]["hyp"].endswith(LOOP_MARKER.strip())
    assert len(out[0]["hyp"]) <= LOOP_CHARS + len(LOOP_MARKER)


def test_markdown_neutralises_pipes_and_newlines():
    # Phase 3: a transcript must not be able to forge table structure.
    recs = [_rec("x", ref="a | b", hyp="c | d\ne")]
    rep = score(recs, n_boot=50)
    md = to_markdown(rep, {"g": worst_examples(recs, "g", k=1)})
    assert "a \\| b" in md and "c \\| d e" in md
    row = [line for line in md.splitlines() if line.startswith("| x |")][0]
    assert row.count("|") - row.count("\\|") == 5  # 5 real cell boundaries


def test_markdown_shows_five_worst_per_group():
    recs = [_rec(f"u{i}", ref="a b c d", hyp="a b c") for i in range(7)]
    recs[0]["hyp"] = "x" * 5000
    rep = score(recs, n_boot=50)
    md = to_markdown(rep, {"g": worst_examples(recs, "g")})
    assert "### g: 5 worst clips" in md
    body = md.split("### g: 5 worst clips")[1]
    rows = [l for l in body.splitlines() if l.startswith("| u") and "utt_id" not in l]
    assert len(rows) == 5
    assert "[loop]" in body


def test_wrong_script_khmer_is_one():
    assert wrong_script_rate("កខគ") == 1.0


def test_wrong_script_yoruba_underdots_is_zero():
    assert wrong_script_rate("ẹ̀yà ọ̀nà ṣe") == 0.0
    assert wrong_script_rate("Àwọn èyàn ti mọ̀ nípa ọ̀nà") == 0.0


def test_wrong_script_english_is_zero():
    assert wrong_script_rate("hello world") == 0.0


def test_wrong_script_mixed_counts_only_script_bearing_chars():
    # 123, punctuation and space carry no script; one Khmer letter of two counts.
    assert wrong_script_rate("a ក 123 !?") == 0.5
    assert wrong_script_rate("") == 0.0
    assert wrong_script_rate("42 --") == 0.0


def test_summary_table_has_wrong_script_and_diacritic_columns():
    recs = [_rec("x", hyp="កខគ")]
    md = to_markdown(score(recs, n_boot=50))
    header = md.splitlines()[0]
    assert "wrong script" in header and "diacritic retention" in header
    row = [l for l in md.splitlines() if l.startswith("| g |")][0]
    assert "| 100.0% | n/a |" in row


def test_diacritic_retention_precomposed_equals_decomposed():
    ref = "ẹ̀yà ọ̀nà ṣe"
    hyp = unicodedata.normalize("NFD", ref)
    assert diacritic_retention([ref], [hyp]) == 1.0
    assert diacritic_retention([hyp], [ref]) == 1.0


def test_diacritic_retention_partial_and_no_inflation():
    # ọ́ = o + dot below + acute. Keeping only the underdot retains 1 of 2.
    assert diacritic_retention(["ọ́"], ["ọ"]) == 0.5
    assert diacritic_retention(["ọ́"], ["o"]) == 0.0
    # Extra hypothesis marks are capped at the reference count.
    assert diacritic_retention(["à"], ["àáàáà"]) == 1.0
    # Tone and underdot are distinct marks; one does not stand in for the other.
    assert diacritic_retention(["ẹ̀"], ["ẹ"]) == 0.5


def test_diacritic_retention_ignores_marks_on_non_latin_bases():
    # Greek ἄ = alpha + psili + acute: the acute is U+0301, but not on Latin.
    assert diacritic_retention(["á"], ["ἄ"]) == 0.0
    # Khmer loops carry combining marks with other code points.
    assert diacritic_retention(["ẹ̀"], ["កខគ"]) == 0.0


def test_diacritic_retention_is_undefined_without_reference_marks():
    assert diacritic_retention(["hello"], ["hẹllo"]) is None
    assert diacritic_retention([], []) is None


def test_score_includes_diacritic_retention_for_yoruba():
    recs = [_rec("x", group="yoruba_read", ref="àwọn", hyp="awon")]
    rep = score(recs, n_boot=50)
    assert rep["yoruba_read"]["diacritic_retention"] == 0.0


if __name__ == "__main__":  # run without pytest: python tests/test_report.py
    import sys
    sys.path.insert(0, ".")
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
