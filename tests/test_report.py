"""Stdlib-only tests for the worst-example section of the report."""
from speechgap.report import (LOOP_CHARS, LOOP_MARKER, score, to_markdown,
                              truncate_hyp, worst_examples)


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
