import math

from speechgap.metrics import (bootstrap_rate, breakdown, is_hallucination,
                               paired_bootstrap_diff, rate, utt_counts)
from speechgap.normalize import normalize
from speechgap.report import score


def test_wer_known_case():
    c = utt_counts("a b c d", "a x c", "word")   # b->x substitution, d deleted
    assert (c["s"], c["d"], c["i"], c["n"]) == (1, 1, 0, 4)
    assert rate([c]) == 0.5


def test_insertions_counted():
    c = utt_counts("a b", "a b b b", "word")
    assert (c["s"], c["d"], c["i"]) == (0, 0, 2)
    assert breakdown([c])["ins"] == 1.0


def test_cer_ignores_whitespace():
    assert rate([utt_counts("ab cd", "abcd", "char")]) == 0.0
    assert rate([utt_counts("abcd", "abxd", "char")]) == 0.25


def test_perfect_and_empty():
    assert rate([utt_counts("same words", "same words", "word")]) == 0.0
    assert math.isnan(rate([utt_counts("", "", "word")]))


def test_normalize():
    assert normalize("Hello,  World!") == "hello world"
    assert normalize("Café", strip_diacritics=True) == "cafe"
    assert normalize("Café") == "café"


def test_bootstrap_interval_contains_point_and_is_deterministic():
    counts, speakers = [], []
    for sp in range(10):
        for k in range(5):
            counts.append({"s": 1 if (sp + k) % 4 == 0 else 0, "d": 0, "i": 0, "n": 5})
            speakers.append(f"s{sp}")
    a = bootstrap_rate(counts, speakers, n_boot=500, seed=1)
    b = bootstrap_rate(counts, speakers, n_boot=500, seed=1)
    assert a == b
    assert a["lo"] <= a["rate"] <= a["hi"]
    assert a["n_speakers"] == 10


def test_single_speaker_has_no_interval():
    counts = [{"s": 1, "d": 0, "i": 0, "n": 4}] * 3
    r = bootstrap_rate(counts, ["only"] * 3)
    assert r["lo"] == r["hi"] == r["rate"] and r["n_speakers"] == 1


def test_paired_diff_zero_for_identical_systems():
    counts = [{"s": 1, "d": 0, "i": 0, "n": 4}, {"s": 0, "d": 0, "i": 0, "n": 4}] * 4
    speakers = ["a", "a", "b", "b", "c", "c", "d", "d"]
    d = paired_bootstrap_diff(counts, counts, speakers, n_boot=200)
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_hallucination_flags():
    assert is_hallucination("short ref", "no no no no no no no no no no no no no no no")
    assert not is_hallucination("a normal sentence here", "a normal sentence here")


def test_report_scores_groups():
    recs = [
        {"utt_id": str(i), "speaker": f"s{i % 3}", "group": g, "ref": "one two three", "hyp": h}
        for i, (g, h) in enumerate([("good", "one two three")] * 6 + [("bad", "one two")] * 6)
    ]
    rep = score(recs, n_boot=100)
    assert rep["good"]["wer"]["rate"] == 0.0
    assert abs(rep["bad"]["wer"]["rate"] - 1 / 3) < 1e-9


if __name__ == "__main__":  # run without pytest: python tests/test_metrics.py
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
