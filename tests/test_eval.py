"""Tests for manifest validation, cache resume and the eval call shape.

two tests need a third-party module: one builds fake tensors with torch, one
inspects the installed transformers pipeline signature. no model is loaded and
no inference runs. when the module is genuinely absent those two tests print a
skip line and return, so the stdlib-only ci stays green; every other test here
runs with no packages installed.
"""
import json
import os
import tempfile

from speechgap.eval import _read_partial, load_manifest, select_rows


def _missing(module, test_name):
    """true when `module` is not installed; prints the skip line and returns true.

    only a ModuleNotFoundError naming this exact top-level module counts as
    missing. a module that is present but broken (its own dependency absent, a
    bad shared library) re-raises and fails the run, so a broken install cannot
    silently drop coverage. the try wraps this __import__ and nothing else: an
    import error raised inside a test body must fail, not skip.
    """
    try:
        __import__(module)
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != module:
            raise
        print(f"skip {test_name} needs {module}")
        return True
    return False


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


def test_select_rows_filters_group_before_limit():
    rows = [_rec(0, group="english_read"), _rec(1, group="fleurs_yo"),
            _rec(2, group="fleurs_yo"), _rec(3, group="fleurs_yo")]
    # limit=2 must be "first 2 of fleurs_yo", not "first 2 of the manifest"
    got = select_rows(rows, group="fleurs_yo", limit=2)
    assert [r["utt_id"] for r in got] == ["u1", "u2"]
    assert len(select_rows(rows, group="fleurs_yo")) == 3
    assert len(select_rows(rows, limit=2)) == 2


def test_select_rows_refuses_an_empty_selection():
    rows = [_rec(0, group="english_read")]
    for kwargs in ({"group": "fleurs_yo"}, {"group": " "}, {"group": ""},
                   {"limit": 0}, {"limit": -3}):
        try:
            select_rows(rows, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"must refuse {kwargs}: an empty run would score as nan")


def test_transcribe_chunk_keeps_the_decoding_call_and_detects_language_separately():
    if _missing("torch", "test_transcribe_chunk_keeps_the_decoding_call_and_detects_language_separately"):
        return
    import torch
    from types import SimpleNamespace

    from speechgap.eval import _transcribe_chunk

    class FakeModel:
        device = "cpu"
        generation_config = SimpleNamespace(lang_to_id={"<|en|>": 1, "<|yo|>": 2})

        def detect_language(self, input_features):
            assert tuple(input_features.shape) == (2, 80, 3000)
            return torch.tensor([2, 1])

    class FakePipe:
        model = FakeModel()

        def __init__(self):
            self.calls = []

        def __call__(self, paths, **kwargs):
            self.calls.append((list(paths), kwargs))
            return [{"text": "hyp-a"}, {"text": "hyp-b"}]

        def preprocess(self, path, chunk_length_s=0):
            yield {"input_features": torch.zeros(1, 80, 3000)}

    rows = [_rec(0), _rec(1)]
    pipe = FakePipe()
    recs = _transcribe_chunk(pipe, rows, ["a.wav", "b.wav"], None, False)
    assert pipe.calls == [(["a.wav", "b.wav"],
                           {"batch_size": 2,
                            "generate_kwargs": {"task": "transcribe"}})]
    assert "return_language" not in pipe.calls[0][1]
    assert "return_timestamps" not in pipe.calls[0][1]["generate_kwargs"]
    assert [r["language"] for r in recs] == ["yo", "en"]
    assert [r["hyp"] for r in recs] == ["hyp-a", "hyp-b"]


def test_transcribe_chunk_records_the_forced_language_without_detection():
    from speechgap.eval import _transcribe_chunk

    class FakePipe:
        model = object()

        def __init__(self):
            self.calls = []

        def __call__(self, paths, **kwargs):
            self.calls.append(kwargs)
            return [{"text": "hyp"}]

        def preprocess(self, path, chunk_length_s=0):
            raise AssertionError("forced runs must not run language detection")

    pipe = FakePipe()
    recs = _transcribe_chunk(pipe, [_rec(0)], ["a.wav"], "yo", True)
    assert pipe.calls == [{"batch_size": 1,
                           "generate_kwargs": {"task": "transcribe",
                                               "language": "yo"}}]
    assert recs[0]["language"] == "yo"
    assert recs[0]["hyp"] == "hyp"


def test_the_real_pipeline_accepts_the_kwargs_the_eval_path_uses():
    if _missing("transformers", "test_the_real_pipeline_accepts_the_kwargs_the_eval_path_uses"):
        return
    import inspect

    from transformers.pipelines.automatic_speech_recognition import (
        AutomaticSpeechRecognitionPipeline,
    )
    from transformers.pipelines.base import Pipeline

    assert "batch_size" in inspect.signature(Pipeline.__call__).parameters
    preprocess = inspect.signature(AutomaticSpeechRecognitionPipeline.preprocess)
    assert "chunk_length_s" in preprocess.parameters
    sanitize = inspect.signature(
        AutomaticSpeechRecognitionPipeline._sanitize_parameters)
    assert "return_language" in sanitize.parameters


def test_cache_record_prefers_the_detected_language():
    from speechgap.eval import _cache_record

    row = _rec(0)
    rec = _cache_record(row, {"text": "bawo"}, "yo", "en")
    assert rec["language"] == "en"
    assert rec["hyp"] == "bawo"
    assert set(rec) == {"utt_id", "speaker", "group", "ref", "hyp", "language"}
    assert _cache_record(row, {"text": "x"}, "yo")["language"] == "yo"
    assert _cache_record(row, {"text": "x"}, None, None)["language"] is None


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
