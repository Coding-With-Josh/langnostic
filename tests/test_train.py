"""Stdlib-only tests for the LoRA trainer's pre-flight guards."""
import os
import tempfile

from speechgap.train import (cap_rows, check_disjoint, check_homogeneous,
                             probe_durations, select_clips)


def _row(i, speaker="s0", group="yoruba_read", lang="yoruba"):
    return {"utt_id": f"u{i}", "audio": f"audio/test/u{i}.wav",
            "speaker": speaker, "group": group, "ref": "bawo ni", "language": lang}


def test_check_disjoint_accepts_clean_splits_and_names_the_leak():
    check_disjoint({"train": [_row(0, "a"), _row(1, "b")],
                    "val": [_row(2, "c")],
                    "test": [_row(3, "d")]})
    try:
        check_disjoint({"train": [_row(0, "a")], "test": [_row(1, "a")]})
    except ValueError as e:
        assert "speaker leakage" in str(e) and "'a'" in str(e)
    else:
        raise AssertionError("overlapping speakers must be refused")


def test_check_homogeneous_rejects_mixed_group_and_language():
    check_homogeneous([_row(0), _row(1)], "yoruba_read", "yo", "train")
    for kwargs, needle in (({"group": "hausa_read"}, "group"),
                           ({"lang": "hausa"}, "language")):
        try:
            check_homogeneous([_row(0), _row(1, **kwargs)], "yoruba_read", "yo", "train")
        except ValueError as e:
            assert needle in str(e)
        else:
            raise AssertionError(f"mixed {needle} must be refused")


def test_select_clips_is_deterministic_and_respects_the_budget():
    rows = [_row(i) for i in range(50)]
    durations = {r["utt_id"]: 6.0 for r in rows}
    a = select_clips(rows, durations, minutes=1.0, seed=3)
    b = select_clips(list(reversed(rows)), durations, minutes=1.0, seed=3)
    assert [r["utt_id"] for r in a[0]] == [r["utt_id"] for r in b[0]]
    assert a[1] <= 60.0 + 1e-9
    assert 54.0 <= a[1] <= 60.0
    assert a[2] == 0


def test_select_clips_drops_clips_longer_than_the_window():
    rows = [_row(0), _row(1)]
    durations = {"u0": 45.0, "u1": 5.0}
    chosen, seconds, too_long = select_clips(rows, durations, minutes=30)
    assert [r["utt_id"] for r in chosen] == ["u1"]
    assert seconds == 5.0 and too_long == 1


def test_cap_rows_is_deterministic():
    rows = [_row(i) for i in range(20)]
    a = [r["utt_id"] for r in cap_rows(rows, 5, seed=1)]
    b = [r["utt_id"] for r in cap_rows(list(reversed(rows)), 5, seed=1)]
    assert a == b and len(a) == 5


def test_probe_durations_rejects_missing_and_non_16k():
    try:
        import soundfile as sf
    except ImportError:  # CI has no audio stack; this test runs on the laptop/Colab
        return
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "audio", "test", "u0.wav")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        import numpy as np

        sf.write(p, np.zeros(16000, dtype="float32"), 16000, subtype="PCM_16", format="WAV")
        rows = [_row(0)]
        assert probe_durations(rows, d)["u0"] == 1.0
        p8 = os.path.join(d, "audio", "test", "u1.wav")
        sf.write(p8, np.zeros(8000, dtype="float32"), 8000, subtype="PCM_16", format="WAV")
        try:
            probe_durations([_row(1)], d)
        except ValueError as e:
            assert "16 kHz" in str(e)
        else:
            raise AssertionError("non-16k audio must be refused")


if __name__ == "__main__":
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
