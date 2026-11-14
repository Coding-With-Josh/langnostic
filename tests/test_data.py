"""Stdlib-only tests for manifest building. No datasets/audio download in CI."""
import json
import os
import re
import tempfile
from collections import Counter

from speechgap.data import (Allocator, l2arctic_candidates, make_record,
                            select_parquet_files, split_for, utt_id,
                            verify_manifests)


def test_split_for_is_deterministic_and_speaker_level():
    assert split_for("voxpopuli:42") == split_for("voxpopuli:42")
    seen = Counter(split_for(f"naijavoices:{i}") for i in range(20000))
    assert 0.66 < seen["train"] / 20000 < 0.74
    assert 0.12 < seen["val"] / 20000 < 0.18
    assert 0.12 < seen["test"] / 20000 < 0.18


def test_allocator_blocks_cross_group_and_respects_caps():
    a = Allocator({"train": 3, "val": 3, "test": 3})
    assert a.admit("voxpopuli:1", "english_l2_nl", "test")
    # Same person in another group must be dropped, not double-counted.
    assert not a.admit("voxpopuli:1", "english_read", "test")
    assert a.dropped["conflict:english_l2_nl->english_read"] == 1
    for i in range(2, 4):
        assert a.admit(f"voxpopuli:{i}", "english_l2_nl", "test")
    assert not a.admit("voxpopuli:9", "english_l2_nl", "test")
    assert a.counts["english_l2_nl"]["test"] == 3
    assert len(a.speakers[("english_l2_nl", "test")]) == 3


def test_allocator_release_frees_the_speaker_and_the_slot():
    a = Allocator({"train": 1, "val": 1, "test": 1})
    assert a.admit("s", "yoruba_read", "test")
    a.release("s", "yoruba_read", "test")
    assert not a.counts["yoruba_read"]["test"]
    assert a.admit("s", "yoruba_read", "test")


def test_utt_id_is_safe_and_collision_resistant():
    a = utt_id("naijavoices", "yoruba-0-0-1")
    b = utt_id("naijavoices", "yoruba-0-0/1")
    assert re.fullmatch(r"[A-Za-z0-9._-]+", a)
    assert a != b


def test_select_parquet_files_handles_every_source_layout():
    naija = ["data/train-00000-of-00010.parquet",
             "data/train-00001-of-00010.parquet", "README.md"]
    assert select_parquet_files(naija, None, "train", max_files=1) == [
        "data/train-00000-of-00010.parquet"
    ]
    voxpopuli = ["en/test-00000-of-00001.parquet",
                 "en/train-00000-of-00030.parquet",
                 "de/test-00000-of-00001.parquet"]
    assert select_parquet_files(voxpopuli, "en", "test") == [
        "en/test-00000-of-00001.parquet"
    ]
    fleurs = ["parquet-data/yo_ng/test-00000-of-00001.parquet",
              "parquet-data/yo_ng/train-00000-of-00001.parquet"]
    assert select_parquet_files(fleurs, "yo_ng", "test") == [
        "parquet-data/yo_ng/test-00000-of-00001.parquet"
    ]
    l2 = ["data/train-00000-of-00001.parquet",
          "data/validation-00000-of-00001.parquet",
          "data/test-00000-of-00001.parquet",
          "data/suitcase-00000-of-00001.parquet"]
    assert select_parquet_files(l2, "default", "validation") == [
        "data/validation-00000-of-00001.parquet"
    ]
    try:
        select_parquet_files(["README.md"], "en", "test")
    except ValueError as e:
        assert "refusing to build an empty group" in str(e)
    else:
        raise AssertionError("an unknown layout must raise, not yield zero files")


def test_l2arctic_candidates_pool_all_splits_into_one_test_group():
    def row(speaker, utt, split):
        return {"speaker_id": speaker, "transcript": f"text {utt}",
                "utterance_id": utt, "corpus_split": split, "audio": {"bytes": b""}}

    rows = {
        "train": [row("A", "a1", "train"), row("B", "b1", "train")],
        "validation": [row("C", "c1", "validation")],
        "test": [row("D", "d1", "test")],
        "suitcase": [row("E", "e1", "suitcase")],
    }
    cands = l2arctic_candidates(rows)
    assert len(cands) == 5
    assert {c["split"] for c in cands} == {"test"}
    assert {c["group"] for c in cands} == {"english_l2_arctic"}
    assert {c["speaker_key"] for c in cands} == {
        "l2arctic:A", "l2arctic:B", "l2arctic:C", "l2arctic:D", "l2arctic:E"
    }
    # Pooling must not depend on the order files happened to be read in.
    shuffled = {k: list(reversed(v)) for k, v in rows.items()}
    assert [c["tag"] for c in cands] == [c["tag"] for c in l2arctic_candidates(shuffled)]


def test_l2arctic_tags_disambiguate_speaker_scoped_utterance_ids():
    # L2-ARCTIC reuses utterance ids across speakers (arctic_a0136 is one file
    # per speaker). A tag without the speaker collapsed them to one utt_id and
    # the second write overwrote the first's audio.
    def row(speaker, utt):
        return {"speaker_id": speaker, "transcript": "text",
                "utterance_id": utt, "audio": {"bytes": b""}}

    rows = {"train": [row("A", "arctic_a0136"), row("B", "arctic_a0136")],
            "test": [row("C", "arctic_a0136")]}
    cands = l2arctic_candidates(rows)
    assert len({c["tag"] for c in cands}) == 3
    assert len({utt_id(c["source"], c["tag"]) for c in cands}) == 3


def test_l2arctic_candidates_skip_rows_without_a_speaker_or_reference():
    rows = {"train": [{"speaker_id": "", "transcript": "text", "utterance_id": "a"},
                      {"speaker_id": "A", "transcript": "  ", "utterance_id": "b"},
                      {"speaker_id": "A", "transcript": "kept", "utterance_id": "c"}]}
    cands = l2arctic_candidates(rows)
    assert [c["tag"] for c in cands] == ["train-A-c"]


def test_make_record_rejects_empty_fields():
    rec = make_record("u1", "audio/test/u1.wav", "naijavoices:X",
                      "yoruba_read", "bawo ni", "yoruba")
    assert set(rec) == {"utt_id", "audio", "speaker", "group", "ref", "language"}
    try:
        make_record("u1", "a.wav", "s", "g", "   ", "yoruba")
    except ValueError:
        pass
    else:
        raise AssertionError("empty ref must be rejected")


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(i, speaker, group="yoruba_read"):
    return {"utt_id": f"u{i}", "audio": f"audio/test/u{i}.wav",
            "speaker": speaker, "group": group,
            "ref": "bawo ni", "language": "yoruba"}


def test_verify_accepts_a_clean_build():
    with tempfile.TemporaryDirectory() as d:
        rows = [_row(i, f"naijavoices:s{i % 11}") for i in range(220)]
        for r in rows:
            p = os.path.join(d, r["audio"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").close()
        _write(os.path.join(d, "manifest_test.jsonl"), rows)
        open(os.path.join(d, "manifest_train.jsonl"), "w").close()
        open(os.path.join(d, "manifest_val.jsonl"), "w").close()
        summary, problems = verify_manifests(d)
        assert not problems, problems
        assert summary["groups"]["yoruba_read"]["test"]["speakers"] == 11


def test_verify_catches_split_leak_and_thin_group():
    with tempfile.TemporaryDirectory() as d:
        rows = [_row(i, "naijavoices:s0") for i in range(5)]
        for r in rows:
            p = os.path.join(d, r["audio"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").close()
        _write(os.path.join(d, "manifest_test.jsonl"), rows)
        _write(os.path.join(d, "manifest_train.jsonl"),
               [dict(rows[0], utt_id="leak", audio="audio/test/u0.wav")])
        open(os.path.join(d, "manifest_val.jsonl"), "w").close()
        _, problems = verify_manifests(d)
        assert any("both train and test" in p for p in problems), problems
        assert any("test clips < 200" in p for p in problems), problems
        assert any("test speakers < 10" in p for p in problems), problems


def test_verify_catches_missing_audio_and_bad_keys():
    with tempfile.TemporaryDirectory() as d:
        bad = _row(1, "naijavoices:s1")
        bad["extra"] = "nope"
        _write(os.path.join(d, "manifest_test.jsonl"), [bad])
        open(os.path.join(d, "manifest_train.jsonl"), "w").close()
        open(os.path.join(d, "manifest_val.jsonl"), "w").close()
        _, problems = verify_manifests(d)
        assert any("wrong keys" in p for p in problems), problems


def test_build_publishes_only_after_a_full_pass():
    """End-to-end orchestration with the network and decoder stubbed out."""
    import argparse
    from speechgap import data as D

    def fake_iter(language, shard, stats, max_scan=None, revision=None):
        for i in range(2000):
            spk = f"naijavoices:s{i % 300}"
            stats["rows"] += 1
            yield {
                "source": "naijavoices", "speaker_key": spk,
                "group": f"{language}_read", "split": D.split_for(spk),
                "language": language, "ref": "bawo ni", "tag": f"{language}-{shard}-{i}",
                "audio": None,
            }

    def fake_wav(value, dest, target_sr=D.TARGET_SR):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"RIFF")
        return 1.0

    saved = (D.iter_naija, D.write_wav, D._resolve_revision)
    D.iter_naija, D.write_wav, D._resolve_revision = fake_iter, fake_wav, lambda repo: "deadbeef"
    try:
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(
                out_dir=d, naija=["yoruba:0-0"], voxpopuli=[], fleurs=[],
                l2arctic=False,
                accents="", allow_clip_level=False, allow_thin_groups=False,
                max_train=900, max_val=100, max_test=300,
                min_test_clips=200, min_test_speakers=10, min_seconds=0.1,
                max_scan=None, max_decode_failures=10, force=True, dry_run=False,
            )
            assert D.build(args) == 0
            summary, problems = verify_manifests(d)
            assert not problems, problems
            assert summary["groups"]["yoruba_read"]["test"]["clips"] >= 200
            assert summary["groups"]["yoruba_read"]["test"]["speakers"] >= 10
            assert os.path.exists(os.path.join(d, "audit.json"))
            assert not os.path.exists(os.path.join(d, ".staging"))
    finally:
        D.iter_naija, D.write_wav, D._resolve_revision = saved


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
