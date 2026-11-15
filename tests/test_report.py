"""Stdlib-only tests for the worst-example section of the report."""
import unicodedata

from speechgap.metrics import char_loop, loop_flags
from speechgap.report import (LOOP_CHARS, LOOP_MARKER, diacritic_retention,
                              diacritic_retention_all, diacritic_retention_detail,
                              loop_rates, score, to_markdown, truncate_hyp,
                              worst_examples, wrong_script_rate)


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
    assert "loop rate" in header and "retention (all)" in header
    row = [l for l in md.splitlines() if l.startswith("| g |")][0]
    assert "| 0.0% | 100.0% | 100.0% | n/a | n/a | 0 |" in row
    # The thresholds that define "looped" travel with the table.
    assert "character 4-gram" in md and "no pilot gate uses them" in md


def test_diacritic_retention_precomposed_equals_decomposed():
    ref = "ẹ̀yà ọ̀nà ṣe"
    hyp = unicodedata.normalize("NFD", ref)
    assert diacritic_retention([ref], [hyp]) == 1.0
    assert diacritic_retention([hyp], [ref]) == 1.0


def test_diacritic_retention_partial_and_no_inflation():
    # ọ́ = o + dot below + acute. Keeping only the underdot retains 1 of 2.
    assert diacritic_retention(["ọ́"], ["ọ"]) == 0.5
    assert diacritic_retention(["ọ́"], ["o"]) == 0.0
    # Tone and underdot are distinct marks; one does not stand in for the other.
    assert diacritic_retention(["ẹ̀"], ["ẹ"]) == 0.5
    # A hypothesis is read only to twice its reference's character length, so
    # a mark that appears only in an over-long tail is not counted.
    assert diacritic_retention(["à"], ["xxà"]) == 0.0


def test_char_loop_catches_space_free_repetition():
    # The word detector sees one token and no word n-gram here.
    assert char_loop("à" * 100)
    assert char_loop("ḍ" * 5000)
    assert not char_loop("bawo ni ọmọ mi")


def test_loop_flags_report_each_test():
    long_ref = "x" * 1000
    flags = loop_flags(long_ref, "a" * 100)
    assert flags["char"] and not flags["word"] and not flags["length"]
    assert flags["any"]
    assert loop_flags(long_ref, "bawo ni " * 20)["word"]
    assert loop_flags("short ref", "z" * 200)["length"]


def test_loop_rates_report_components_and_union():
    refs = ["x" * 1000] * 3
    hyps = ["a" * 100, "bawo ni " * 20, "clean text"]
    rates = loop_rates(refs, hyps)
    assert rates["n_looped"] == 2
    assert rates["loop_rate"] == 2 / 3
    assert rates["loop_rate_char"] == 2 / 3  # both repeated texts
    assert rates["loop_rate_word"] == 1 / 3  # only the phrase loop
    assert rates["loop_rate_length"] == 0.0

    length = loop_rates(["short ref"], ["z" * 200])
    assert length["loop_rate_length"] == 1.0 and length["loop_rate"] == 1.0


def test_diacritic_retention_excludes_looped_hypotheses():
    loop = "à" * 100
    assert diacritic_retention(["à"], [loop]) is None
    # The old pooled value still shows the inflation this fixes.
    assert diacritic_retention_all(["à"], [loop]) == 1.0
    detail = diacritic_retention_detail(["à"], [loop])
    assert detail["retention"] is None
    assert detail["retention_all"] == 1.0
    assert detail["n_looped_excluded"] == 1


def test_diacritic_retention_is_per_clip_not_pooled():
    # Clip 1's loop must not cover clip 2's missing acute.
    refs = ["à", "á"]
    hyps = ["à" * 100, "a"]
    detail = diacritic_retention_detail(refs, hyps)
    assert detail["retention"] == 0.0
    assert detail["retention_all"] == 0.5  # old pooled: 1 of 2 marks
    assert detail["n_looped_excluded"] == 1


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


def test_score_reports_wrong_script_on_non_looped_hyps():
    # The Khmer hyp is a character loop; it must not set the primary column.
    recs = [_rec("loop", hyp="កខគ" * 20), _rec("empty", hyp="")]
    rep = score(recs, n_boot=50)["g"]
    assert rep["wrong_script_rate"] == 0.0
    assert rep["wrong_script_rate_all"] == 1.0
    assert rep["n_looped"] == 1 and rep["n_looped_excluded"] == 1


def test_score_reports_loop_columns():
    recs = [_rec("a", ref="x" * 1000, hyp="a" * 100),
            _rec("b", ref="x" * 1000, hyp="clean text")]
    rep = score(recs, n_boot=10)["g"]
    assert rep["loop_rate"] == 0.5 and rep["n_looped"] == 1
    assert rep["loop_rate_char"] == 0.5


def test_score_reports_fair_and_pooled_retention():
    recs = [
        _rec("loop", ref="àwọn", hyp="à" * 100),
        _rec("flat", ref="àwọn", hyp="awon"),
    ]
    rep = score(recs, n_boot=50)["g"]
    assert rep["n_looped"] == 1 and rep["n_looped_excluded"] == 1
    assert rep["diacritic_retention"] == 0.0
    assert rep["diacritic_retention_all"] == 0.5


def test_score_records_thresholds():
    rep = score([_rec("x")], n_boot=10)["g"]
    assert rep["thresholds"]["char_ngram"] == {"n": 4, "min_repeats": 8}
    assert rep["thresholds"]["word_ngram"] == {"n": 3, "min_repeats": 4}
    assert rep["thresholds"]["length_ratio"] == 10.0
    assert rep["thresholds"]["retention_hyp_cap"] == 2


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
