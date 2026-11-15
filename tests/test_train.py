"""Stdlib-only tests for the LoRA trainer's pre-flight guards."""
import os
import tempfile

from speechgap.train import (GenerateGuard, _publish_adapter, assert_language_kwargs,
                             build_lora_config, cap_rows, check_disjoint,
                             check_homogeneous, decoder_embedding,
                             degenerate_reasons, example_lines,
                             force_generation_kwargs, package_versions, pilot_gates,
                             probe_durations, prompt_language_check, prompt_prefix_ok,
                             select_clips, validate_failed_target)


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


def test_assert_language_kwargs_accepts_code_and_token_forms():
    assert_language_kwargs({"language": "yo", "task": "transcribe"}, "yo", "transcribe")
    assert_language_kwargs({"language": "<|yo|>", "task": "transcribe"}, "yo", "transcribe")


def test_assert_language_kwargs_rejects_every_unforced_call():
    bad = [
        {"task": "transcribe"},                    # language omitted -> auto-detect
        {"language": None, "task": "transcribe"},  # explicit auto-detect
        {"language": "en", "task": "transcribe"},  # wrong language
        {"language": "yo"},                        # task omitted
        {"language": "yo", "task": "translate"},   # wrong task
    ]
    for kwargs in bad:
        try:
            assert_language_kwargs(kwargs, "yo", "transcribe")
        except ValueError:
            continue
        raise AssertionError(f"must refuse {kwargs}")


def test_lora_config_does_not_use_the_seq2seq_wrapper():
    # Regression: PeftModelForSeq2SeqLM.forward passes input_ids=None to a Whisper
    # forward that has no such parameter, which crashes the first training step
    # with "WhisperDecoder() got multiple values for keyword argument 'input_ids'".
    # A generic PeftModel (task_type=None) forwards input_features untouched.
    cfg = build_lora_config()
    assert cfg.task_type is None
    assert set(cfg.target_modules) == {"q_proj", "v_proj"}
    assert (cfg.r, cfg.lora_alpha, cfg.lora_dropout) == (8, 16, 0.05)


def test_force_generation_kwargs_installs_and_survives_the_trainer_rebuild():
    # evaluate() rebuilds _gen_kwargs from scratch on every call, so the helper
    # must be idempotent and safe on a dict that is already forced.
    gen = {}
    assert force_generation_kwargs(gen, "yo", "transcribe") is gen
    assert gen == {"language": "yo", "task": "transcribe"}
    force_generation_kwargs(gen, "yo", "transcribe")
    assert gen == {"language": "yo", "task": "transcribe"}
    force_generation_kwargs({"language": "<|yo|>", "task": "transcribe"}, "yo", "transcribe")


def test_force_generation_kwargs_refuses_conflicts_instead_of_overwriting():
    for gen, needle in [
        ({"language": "en", "task": "transcribe"}, "language"),
        ({"language": "yo", "task": "translate"}, "task"),
    ]:
        try:
            force_generation_kwargs(gen, "yo", "transcribe")
        except ValueError as e:
            assert needle in str(e)
        else:
            raise AssertionError(f"must refuse {gen}")


def test_degenerate_reasons_flags_structure_not_badness():
    refs = ["bawo ni", "mo wa", "o dabo", "e kaaro", "kini"]
    assert degenerate_reasons(refs, list(refs)) == []
    # Bad but real output is the WER gate's job, not this one.
    assert degenerate_reasons(refs, ["", "mo wa", "o dabo", "e kaaro", "kini"]) == []
    assert "empty" in " ".join(degenerate_reasons(refs, [""] * 5))
    assert "identical" in " ".join(degenerate_reasons(refs, ["bawo"] * 5))
    collapsed = ["a", "b"] * 10
    assert "distinct" in " ".join(degenerate_reasons(["x"] * 20, collapsed))
    loop = "bawo ni " * 40  # 320 chars against 8-char references
    assert "loops" in " ".join(
        degenerate_reasons(refs, [loop + str(i) for i in range(5)])
    )


def test_pilot_gates_pass_on_a_clean_run():
    refs = ["bawo ni", "mo wa", "o dabo"]
    gates = pilot_gates(refs, list(refs))
    assert gates["ok"] and gates["val_wer"] == 0.0 and gates["failures"] == []


def test_pilot_gates_fail_above_100_percent_wer():
    gates = pilot_gates(["one"], ["x y z"])  # 3 insertions for 1 reference word
    assert not gates["ok"] and "val wer" in " ".join(gates["failures"])


def test_pilot_gates_fail_on_wrong_script_without_calling_it_degenerate():
    refs = ["one two three four five six seven eight"] * 5
    hyps = ["កខគ", "ខគក", "គកខ", "កគខ", "ខកគ"]  # 2 tokens each, 1/4 wer
    gates = pilot_gates(refs, hyps)
    assert not gates["ok"]
    assert "wrong script" in " ".join(gates["failures"])
    assert gates["degenerate"] == []


def test_example_lines_cap_at_three_and_mark_loops():
    refs = ["a", "b", "c", "d"]
    hyps = ["a", "b", "c", "x" * 5000]
    lines = example_lines(refs, hyps, ["u0", "u1", "u2", "u3"])
    assert len(lines) == 3 and lines[0].startswith("[u0]")
    long_lines = example_lines(refs, hyps, k=4)
    assert "[loop]" in long_lines[-1] and len(long_lines[-1]) < 5000


class _FakeModel:
    def __init__(self, sequences):
        self.sequences = sequences
        self.seen = []

    def generate(self, **kwargs):
        self.seen.append(kwargs)
        return self.sequences


def test_generate_guard_requires_the_forced_kwargs():
    model = _FakeModel([[50258, 50325, 50359, 42]])
    guard = GenerateGuard(model, "yo", "transcribe", decoder_start_id=50258,
                          language_id=50325, task_id=50359).install(model)
    model.generate(language="yo", task="transcribe")
    assert guard.calls == 1 and guard.prompt_verified
    try:
        model.generate(task="transcribe")  # auto-detect path
    except ValueError:
        pass
    else:
        raise AssertionError("an unforced generate must be refused")


def test_generate_guard_rejects_a_prompt_with_the_wrong_language_token():
    model = _FakeModel([[50258, 50259, 50359, 42]])  # <|en|>, not <|yo|>
    GenerateGuard(model, "yo", "transcribe", decoder_start_id=50258,
                  language_id=50325, task_id=50359).install(model)
    try:
        model.generate(language="yo", task="transcribe")
    except RuntimeError as e:
        assert "did not take effect" in str(e)
    else:
        raise AssertionError("a prompt without <|yo|> must be refused")


def test_generate_guard_records_an_unobservable_prompt_without_passing_it():
    model = _FakeModel([[42, 43]])  # prompt stripped by this version
    guard = GenerateGuard(model, "yo", "transcribe", decoder_start_id=50258,
                          language_id=50325, task_id=50359).install(model)
    model.generate(language="yo", task="transcribe")
    assert guard.calls == 1
    assert not guard.prompt_observable and not guard.prompt_verified


def test_prompt_prefix_ok_requires_the_exact_three_tokens():
    assert prompt_prefix_ok([50258, 50325, 50359, 42], 50258, 50325, 50359)
    assert not prompt_prefix_ok([50258, 50259, 50359], 50258, 50325, 50359)
    assert not prompt_prefix_ok([50258], 50258, 50325, 50359)
    assert not prompt_prefix_ok([], 50258, 50325, 50359)


class _FakeEmbed:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, fn):
        self.hooks.append(fn)

    def __call__(self, ids):
        for hook in self.hooks:
            hook(self, (ids,))
        return ids


class _Node:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakePeft:
    """PeftModel -> LoraModel -> WhisperForConditionalGeneration -> WhisperModel -> decoder."""

    def __init__(self, fed_ids, sequences):
        self.fed_ids = fed_ids
        self.sequences = sequences
        self.seen = []
        self.embed = _FakeEmbed()
        self.decoder = _Node(embed_tokens=self.embed)
        self.model = _Node(decoder=self.decoder)
        self.base_model = _Node(model=_Node(model=self.model))

    def generate(self, **kwargs):
        self.seen.append(kwargs)
        self.embed(self.fed_ids)
        return self.sequences


def test_guard_verifies_from_the_decoder_input_when_the_prompt_is_stripped():
    model = _FakePeft([50258, 50325, 50359], sequences=[[42, 43]])
    guard = GenerateGuard(model, "yo", "transcribe", decoder_start_id=50258,
                          language_id=50325, task_id=50359).install(model)
    model.generate(language="yo", task="transcribe")
    assert guard.calls == 1
    assert guard.prompt_verified and guard.prompt_source == "decoder_input"


def test_guard_refuses_a_decoder_prompt_with_the_wrong_language():
    model = _FakePeft([50258, 50259, 50359], sequences=[[42, 43]])
    GenerateGuard(model, "yo", "transcribe", decoder_start_id=50258,
                  language_id=50325, task_id=50359).install(model)
    try:
        model.generate(language="yo", task="transcribe")
    except RuntimeError as e:
        assert "did not take effect" in str(e)
    else:
        raise AssertionError("a decoder prompt without <|yo|> must be refused")


def test_decoder_embedding_returns_none_without_an_embedding():
    assert decoder_embedding(_Node()) is None


def test_validate_failed_target_requires_the_suffix_and_a_distinct_path():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "yo15")
        good = os.path.join(d, "yo15.failed")
        assert validate_failed_target(good, out_path=out) == os.path.abspath(good)
        assert validate_failed_target(None) is None
        for path, out_path in ((out, out), ("", out), (good, good), ("   ", out)):
            try:
                validate_failed_target(path, out_path=out_path)
            except ValueError:
                pass
            else:
                raise AssertionError(f"must reject {path!r} with out {out_path!r}")
        os.makedirs(good)
        try:
            validate_failed_target(good, out_path=out)
        except ValueError as e:
            assert "exists" in str(e)
        else:
            raise AssertionError("must refuse an existing destination without force")
        assert validate_failed_target(good, out_path=out, force=True) == os.path.abspath(good)


def test_validate_failed_target_refuses_a_symlink():
    with tempfile.TemporaryDirectory() as d:
        real = os.path.join(d, "real")
        os.makedirs(real)
        link = os.path.join(d, "yo15.failed")
        os.symlink(real, link)
        try:
            validate_failed_target(link, out_path=os.path.join(d, "yo15"), force=True)
        except ValueError as e:
            assert "symlink" in str(e)
        else:
            raise AssertionError("a symlink destination must be refused")


class _SavingModel:
    def __init__(self, fail=False):
        self.fail = fail

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "adapter_model.safetensors"), "w",
                  encoding="utf-8") as f:
            f.write("x")
        if self.fail:
            raise RuntimeError("boom")


def test_publish_adapter_writes_atomically_and_cleans_up_on_failure():
    with tempfile.TemporaryDirectory() as d:
        dest = os.path.join(d, "yo15.failed")
        _publish_adapter(_SavingModel(), dest, {"publishable": False})
        assert os.path.isfile(os.path.join(dest, "train_meta.json"))
        assert os.path.isfile(os.path.join(dest, "adapter_model.safetensors"))
        try:
            _publish_adapter(_SavingModel(), dest, {})
        except FileExistsError:
            pass
        else:
            raise AssertionError("must refuse an existing destination without --force")
        broken = os.path.join(d, "broken.failed")
        try:
            _publish_adapter(_SavingModel(fail=True), broken, {})
        except RuntimeError:
            pass
        else:
            raise AssertionError("a failed save must raise")
        assert not os.path.exists(broken)
        assert not [p for p in os.listdir(d) if p.startswith(".publish-")]


def test_package_versions_names_the_three_packages():
    versions = package_versions()
    assert set(versions) == {"transformers", "torch", "peft"}
    assert isinstance(versions["transformers"], str)
    assert isinstance(versions["torch"], str)
    assert versions["peft"] is None or isinstance(versions["peft"], str)


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
