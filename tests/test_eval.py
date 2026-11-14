"""Stdlib-only tests for manifest validation and cache resume. No model needed."""
import json
import os
import tempfile

from speechgap.eval import _read_partial, load_manifest


def _rec(i=0, **over):
    r = {"utt_id": f"u{i}", "audio": f"audio/test/u{i}.wav",
         "speaker": "naijavoices:s1", "group": "yoruba_read",
         "ref": "bawo ni", "language": "yoruba"}
    r.update(over)
    return r


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_manifest_accepts_the_contract():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.jsonl")
        _write(p, [_rec(0), _rec(1)])
        assert [r["utt_id"] for r in load_manifest(p)] == ["u0", "u1"]


def test_load_manifest_rejects_extra_key():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.jsonl")
        _write(p, [_rec(0, phase="phase 1")])
        try:
            load_manifest(p)
        except ValueError as e:
            assert "extra" in str(e)
        else:
            raise AssertionError("extra keys must be rejected")


def test_load_manifest_rejects_duplicate_id_and_empty_ref():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.jsonl")
        _write(p, [_rec(0), _rec(0)])
        try:
            load_manifest(p)
        except ValueError as e:
            assert "duplicate" in str(e)
        else:
            raise AssertionError("duplicate utt_id must be rejected")
        _write(p, [_rec(0, ref=" ")])
        try:
            load_manifest(p)
        except ValueError:
            pass
        else:
            raise AssertionError("empty ref must be rejected")


def test_resume_keeps_valid_lines_and_drops_a_torn_tail():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "out.jsonl.part")
        with open(p, "wb") as f:
            f.write((json.dumps({"utt_id": "u0", "hyp": "a"}) + "\n").encode())
            f.write((json.dumps({"utt_id": "u1", "hyp": "b"}) + "\n").encode())
            f.write(b'{"utt_id": "u2", "hyp": "trunc')
        done = _read_partial(p, {"u0", "u1", "u2"})
        assert done == {"u0", "u1"}
        with open(p, "rb") as f:
            assert f.read().endswith(b"\n")
        assert _read_partial(p, {"u0", "u1", "u2"}) == {"u0", "u1"}


def test_resume_refuses_a_partial_from_another_manifest():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "out.jsonl.part")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"utt_id": "foreign", "hyp": "x"}) + "\n")
        try:
            _read_partial(p, {"u0"})
        except ValueError as e:
            assert "not in this manifest" in str(e)
        else:
            raise AssertionError("foreign partial must be refused")


def test_supported_languages_strips_whisper_tokens():
    from types import SimpleNamespace

    from speechgap.eval import supported_languages

    pipe = SimpleNamespace(
        model=SimpleNamespace(generation_config=SimpleNamespace(
            lang_to_id={"<|en|>": 1, "<|yo|>": 2})),
        tokenizer=None,
    )
    assert supported_languages(pipe) == {"en", "yo"}


def test_report_refuses_a_partial_cache():
    from speechgap.report import load_jsonl

    try:
        load_jsonl("results/cache/zero_shot.jsonl.part")
    except ValueError as e:
        assert ".part" in str(e)
    else:
        raise AssertionError("a .part cache must not be scored")


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
