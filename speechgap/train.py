"""Train a small LoRA adapter on one group, speaker-disjoint from validation and test.

Runs on a free Colab T4. The adapter goes to ``checkpoints/<name>`` and is never
committed (``.gitignore``); only its metrics and the cache it produces travel back.

What this file refuses to do, on purpose:
  * train if a speaker appears in more than one split (manifests are re-checked,
    not trusted from build time);
  * train on a manifest that mixes groups or languages;
  * call a run "30 minutes" when the manifest cannot supply 30 minutes;
  * overwrite an existing adapter without ``--force``;
  * publish a half-written adapter after a disconnect;
  * publish an adapter whose validation output is degenerate, above 100% WER,
    or written in the wrong script (the pilot gates, below);
  * score a run whose ``generate()`` call did not receive the forced language.

Usage (Colab):
  python -m speechgap.train --train data/manifest_train.jsonl \
      --val data/manifest_val.jsonl --minutes 30 \
      --out checkpoints/yo-30min --group yoruba_read --language yo

  # pre-flight without a GPU or model download
  python -m speechgap.train ... --dry-run

  # pilot: 2 minutes of audio, one epoch, 20 validation clips
  python -m speechgap.train --train data/manifest_train.jsonl \
      --val data/manifest_val.jsonl --minutes 2 --epochs 1 --max-val 20 \
      --patience 1 --out checkpoints/yo-pilot --group yoruba_read --language yo

  # evaluate the adapter with the same scorer as the baseline
  python -m speechgap.eval --manifest data/manifest_test.jsonl --device cuda \
      --adapter checkpoints/yo-30min --out results/cache/yo_lora30.jsonl
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from itertools import combinations

from .eval import (DEFAULT_MODEL, DEFAULT_REVISION, load_manifest,
                   manifest_sha256, resolve_device)
from .languages import resolve_language
from .metrics import rate, utt_counts
from .normalize import normalize
from .report import diacritic_retention, truncate_hyp, wrong_script_rate

# Fixed analysis-plan defaults. Changing these is a deviations-log change.
LORA_TARGETS = ("q_proj", "v_proj")

# Deliberately None, not "SEQ_2_SEQ_LM". With that task type peft wraps Whisper in
# PeftModelForSeq2SeqLM, whose forward passes ``input_ids=None`` to the base model.
# Whisper's forward has no ``input_ids`` parameter, so the value falls into
# ``**kwargs`` and is forwarded to ``self.decoder(input_ids=decoder_input_ids,
# ..., **kwargs)`` -- a duplicate keyword, raised as a TypeError on the first
# training step. A generic PeftModel forwards ``input_features`` untouched; the LoRA
# layers still inject, and ``adapter_config.json`` reloads as PeftModel, which is
# what ``eval.py --adapter`` merges. Regression-tested in tests/test_train.py.
LORA_TASK_TYPE = None

# Pilot pass conditions (ANALYSIS_PLAN.md). A run that misses any of them
# publishes nothing, so its number can never be quoted as an adapter result.
PILOT_MAX_VAL_WER = 1.0
PILOT_MAX_WRONG_SCRIPT = 0.10
PILOT_EXAMPLES = 3

# A token loop is not "bad ASR", it is a score that stops meaning anything:
# WER fills up with insertions. Ten times the reference length is far outside
# anything observed in a healthy run, including the 367% WER yoruba baseline.
LOOP_RATIO = 10.0


# ---------------------------------------------------------------------------
# pure helpers (stdlib only: unit-tested in CI)
# ---------------------------------------------------------------------------

def split_speakers(splits):
    return {name: {r["speaker"] for r in rows} for name, rows in splits.items() if rows}


def check_disjoint(splits):
    """Raise if any speaker appears in two splits.

    Threat (Phase 3): a stale or hand-edited manifest pair can leak a voice into
    training, which turns a held-out WER into a memorization score. This is
    re-checked at train time because the build-time audit is not a security
    boundary by itself.
    """
    spk = split_speakers(splits)
    problems = []
    for a, b in combinations(sorted(spk), 2):
        overlap = spk[a] & spk[b]
        if overlap:
            problems.append(
                f"{len(overlap)} speaker(s) in both {a} and {b}: {sorted(overlap)[:5]}"
            )
    if problems:
        raise ValueError("speaker leakage between splits: " + "; ".join(problems))
    return spk


def check_homogeneous(rows, group, language_code, split_name):
    """Every row must be the same group and the same whisper language."""
    if not rows:
        raise ValueError(f"{split_name} manifest is empty")
    groups = Counter(r["group"] for r in rows)
    if set(groups) != {group}:
        raise ValueError(
            f"{split_name}: expected only group {group!r}, saw {dict(groups)}"
        )
    langs = Counter(resolve_language(r["language"]) for r in rows)
    if set(langs) != {language_code}:
        raise ValueError(
            f"{split_name}: expected only language {language_code!r}, saw {dict(langs)}"
        )


def select_clips(rows, durations, minutes, seed=0, max_seconds=30.0):
    """Pick clips up to a seconds budget in a deterministic, hash-shuffled order.

    Clips longer than whisper's 30 s window are excluded: the feature extractor
    would truncate the audio but keep the full label, which teaches hallucination.
    Returns (chosen, actual_seconds, n_too_long).
    """
    budget = float(minutes) * 60.0
    chosen, total, too_long = [], 0.0, 0
    order = sorted(
        rows, key=lambda r: hashlib.sha256(f"{seed}|{r['utt_id']}".encode()).hexdigest()
    )
    for r in order:
        d = float(durations[r["utt_id"]])
        if d <= 0 or d > max_seconds:
            too_long += 1
            continue
        if total + d > budget:
            continue  # keep looking: a shorter clip may still fit
        chosen.append(r)
        total += d
    return chosen, total, too_long


def cap_rows(rows, limit, seed=0):
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    return sorted(
        rows, key=lambda r: hashlib.sha256(f"cap|{seed}|{r['utt_id']}".encode()).hexdigest()
    )[:limit]


def probe_durations(rows, audio_root):
    """Read every audio header before model load; missing/corrupt audio aborts."""
    import soundfile as sf

    durations = {}
    for r in rows:
        p = r["audio"]
        if not os.path.isabs(p):
            p = os.path.join(audio_root, p)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{r['utt_id']}: audio not found at {p}")
        info = sf.info(p)
        if info.samplerate != 16000:
            raise ValueError(f"{r['utt_id']}: expected 16 kHz, got {info.samplerate}")
        durations[r["utt_id"]] = float(info.duration)
    return durations


def wer_of(refs, hyps):
    counts = [utt_counts(normalize(a), normalize(b), "word") for a, b in zip(refs, hyps)]
    return rate(counts)


def degenerate_reasons(refs, hyps):
    """Why these hypotheses cannot support a WER claim (empty list = usable).

    Four structural failures, each seen in whisper runs:
      * nothing decoded at all;
      * a collapse to one string (a 2-minute adapter can memorize one line);
      * a collapse to a handful of strings across the whole set;
      * token loops, which inflate WER with insertions instead of measuring ASR.

    Thresholds are deliberately loose: this gate asks "is the score meaningless",
    not "is the score bad". A bad-but-real transcript passes here and is caught
    by the WER and wrong-script gates instead, so a weak pilot is still reported
    honestly rather than being misfiled as a broken run.
    """
    if not hyps:
        return ["no hypotheses"]
    n = len(hyps)
    reasons = []
    empty = sum(1 for h in hyps if not h.strip())
    if empty == n:
        reasons.append("every hypothesis is empty")
    elif empty * 2 >= n:
        reasons.append(f"{empty}/{n} hypotheses are empty")
    distinct = len(set(hyps))
    if n >= 2 and distinct == 1:
        reasons.append("every hypothesis is identical")
    elif n >= 5 and distinct * 5 < n:
        reasons.append(f"only {distinct}/{n} distinct hypotheses")
    loops = sum(
        1 for h, r in zip(hyps, refs) if len(h) > LOOP_RATIO * max(len(r), 1)
    )
    if loops * 2 >= n:
        reasons.append(
            f"{loops}/{n} hypotheses exceed {LOOP_RATIO:.0f}x their reference (loops)"
        )
    return reasons


def pilot_gates(refs, hyps):
    """Evaluate the pilot gates. Never raises: the caller prints, then decides.

    Measured on raw text, exactly as ``speechgap.report`` measures them. WER is
    normalised inside ``wer_of``; the script and diacritic columns are not, so
    lowercasing or stripping punctuation cannot make the output look better than
    the cache a reviewer would score later.
    """
    if not refs or len(refs) != len(hyps):
        raise ValueError(
            f"gate inputs must be parallel and non-empty; got {len(refs)} refs, "
            f"{len(hyps)} hypotheses"
        )
    val_wer = wer_of(refs, hyps)
    wrong_script = wrong_script_rate(" ".join(hyps))
    reasons = degenerate_reasons(refs, hyps)
    failures = list(reasons)
    if val_wer > PILOT_MAX_VAL_WER:
        failures.append(f"val wer {val_wer:.1%} > {PILOT_MAX_VAL_WER:.0%}")
    if wrong_script > PILOT_MAX_WRONG_SCRIPT:
        failures.append(
            f"wrong script {wrong_script:.1%} > {PILOT_MAX_WRONG_SCRIPT:.0%}"
        )
    return {
        "n_clips": len(hyps),
        "val_wer": val_wer,
        "wrong_script_rate": wrong_script,
        "diacritic_retention": diacritic_retention(refs, hyps),
        "degenerate": reasons,
        "failures": failures,
        "ok": not failures,
    }


def force_generation_kwargs(gen, language, task):
    """Force ``language``/``task`` into a generation-kwargs dict, in place.

    A conflicting value is an error, not something to overwrite: the caller
    asked for one language and the harness is about to generate in another. This
    is where the value is installed; ``assert_language_kwargs`` is what proves it
    survived all the way to ``generate()``.
    """
    current = gen.get("language")
    if current not in (None, language, f"<|{language}|>"):
        raise ValueError(
            f"generation language {current!r} conflicts with the forced {language!r}"
        )
    current_task = gen.get("task")
    if current_task not in (None, task):
        raise ValueError(
            f"generation task {current_task!r} conflicts with the forced {task!r}"
        )
    gen["language"] = language
    gen["task"] = task
    return gen


def assert_language_kwargs(kwargs, language, task):
    """Prove the forced language and task are present at the ``generate()`` call.

    Threat (Phase 3): whisper auto-detects when it is not told otherwise, so a
    missing kwarg silently turns the run into a different experiment, and the
    validation WER stops describing the adapter that ``speechgap.eval`` scores.
    This raises instead of logging, because there is no safe default: guessing
    is exactly the failure mode being measured.
    """
    value = kwargs.get("language")
    if value not in (language, f"<|{language}|>"):
        raise ValueError(
            f"generate() received language={value!r}, expected {language!r}; "
            "refusing to score an unforced run"
        )
    got_task = kwargs.get("task")
    if got_task != task:
        raise ValueError(
            f"generate() received task={got_task!r}, expected {task!r}; "
            "refusing to score an unforced run"
        )


def prompt_language_check(sequences, decoder_start_id, language_id, task_id):
    """Is the forced prompt visible in the returned decoder sequence?

    True: the sequence starts with ``<|startoftranscript|><|lang|><|task|>``.
    False: the prompt is present but the language/task tokens are not ours.
    None: the prompt was stripped by this transformers version, so the call
    cannot be judged from its output. ``None`` is recorded, never treated as
    a pass -- the kwarg assertion above remains the control in that case.
    """
    if not sequences or not isinstance(sequences[0], list):
        return None
    first = sequences[0]
    if first[:1] != [decoder_start_id]:
        return None
    if len(first) < 3:
        return False
    return first[1] == language_id and first[2] == task_id


class GenerateGuard:
    """Wrap ``model.generate`` so every call proves it was language-forced."""

    def __init__(self, model, language, task, decoder_start_id=None,
                 language_id=None, task_id=None):
        self.language = language
        self.task = task
        self.decoder_start_id = decoder_start_id
        self.language_id = language_id
        self.task_id = task_id
        self.calls = 0
        self.prompt_verified = False
        self.prompt_observable = False
        self._original = model.generate
        self.checked = self._make_checked()

    def _make_checked(self):
        def checked(*args, **kwargs):
            assert_language_kwargs(kwargs, self.language, self.task)
            self.calls += 1
            out = self._original(*args, **kwargs)
            if self.language_id is not None and self.task_id is not None:
                sequences = getattr(out, "sequences", out)
                if hasattr(sequences, "tolist"):
                    sequences = sequences.tolist()
                verdict = prompt_language_check(
                    sequences, self.decoder_start_id, self.language_id, self.task_id
                )
                if verdict is False:
                    # Phase 2: a kwarg that reached generate but did not reach
                    # the decoder prompt is the same silent experiment change.
                    raise RuntimeError(
                        f"generate() returned a prompt without <|{self.language}|>"
                        f"/<|{self.task}|>; the language kwarg did not take effect"
                    )
                if verdict is not None:
                    self.prompt_observable = True
                    self.prompt_verified = self.prompt_verified or verdict
            return out

        return checked

    def install(self, model):
        # Set on the instance: PeftModel forwards unknown attributes to the base
        # model, and prediction_step reads ``model.generate`` at call time.
        model.generate = self.checked
        return self

    def report(self):
        return {
            "language": self.language,
            "task": self.task,
            "generate_calls": self.calls,
            "prompt_tokens_observable": self.prompt_observable,
            "prompt_tokens_verified": self.prompt_verified,
        }


def example_lines(refs, hyps, utt_ids=None, k=PILOT_EXAMPLES):
    """The ref/hyp pairs a human has to read before trusting any number."""
    lines = []
    for i in range(min(k, len(refs), len(hyps))):
        uid = utt_ids[i] if utt_ids and i < len(utt_ids) else str(i)
        lines.append(f"[{uid}]\n  ref: {refs[i]}\n  hyp: {truncate_hyp(hyps[i])}")
    return lines


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _dataset_classes(processor):
    """Build the torch Dataset/collator lazily so CI never imports torch."""
    import torch
    import soundfile as sf

    class ManifestDataset(torch.utils.data.Dataset):
        def __init__(self, rows, audio_root):
            self.rows = rows
            self.audio_root = audio_root

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            p = r["audio"]
            if not os.path.isabs(p):
                p = os.path.join(self.audio_root, p)
            audio, sr = sf.read(p, dtype="float32", always_2d=True)
            if sr != 16000:
                raise ValueError(f"{r['utt_id']}: expected 16 kHz, got {sr}")
            audio = audio.mean(axis=1)
            feats = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features[0]
            labels = processor.tokenizer(r["ref"]).input_ids
            return {"input_features": feats, "labels": labels}

    class Collator:
        def __init__(self, proc):
            self.proc = proc

        def __call__(self, features):
            batch = self.proc.feature_extractor.pad(
                [{"input_features": f["input_features"]} for f in features],
                return_tensors="pt",
            )
            labels = self.proc.tokenizer.pad(
                [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
            )
            ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
            if (ids[:, 0] == self.proc.tokenizer.bos_token_id).all().item():
                ids = ids[:, 1:]
            batch["labels"] = ids
            return batch

    return ManifestDataset, Collator


def build_lora_config(r=8, alpha=16, dropout=0.05):
    """The adapter config, in one place so its task type cannot drift.

    ``task_type`` must stay ``None``; see ``LORA_TASK_TYPE`` for the failure it
    avoids. Kept as a function so a test can assert it without downloading a
    model.
    """
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(LORA_TARGETS),
        bias="none",
        task_type=LORA_TASK_TYPE,
    )


def _trainer_class():
    import torch
    from transformers import Seq2SeqTrainer

    class LanguageForcedTrainer(Seq2SeqTrainer):
        """Force one language during generation, including through a PeftModel."""

        def __init__(self, *a, language=None, task="transcribe", **kw):
            super().__init__(*a, **kw)
            self._force_language = language
            self._force_task = task

        def training_step(self, model, inputs, num_items_in_batch=None):
            # Phase 3/Phase 2: a non-finite loss means the fp16 run has diverged;
            # publishing that adapter would be worse than failing the run.
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
            if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
                raise RuntimeError(
                    "non-finite training loss; refusing to continue (retry with --fp32 "
                    "or a smaller --batch-size)"
                )
            return loss

        def _force_gen_kwargs(self):
            """Put the forced language/task where prediction_step reads them.

            transformers 5.x accepts extra kwargs in ``prediction_step`` but 4.x
            does not, so the supported path is ``self._gen_kwargs`` -- the same
            dict the trainer copies into ``generate()``. ``evaluate()`` rebuilds
            that dict on every call, hence this runs per call, not once in
            ``__init__``. If a future version renames the attribute, this becomes
            a no-op and the generate guard is what refuses the unforced call.
            """
            if not isinstance(getattr(self, "_gen_kwargs", None), dict):
                self._gen_kwargs = {}
            force_generation_kwargs(
                self._gen_kwargs, self._force_language, self._force_task
            )

        def prediction_step(self, model, inputs, prediction_loss_only,
                            ignore_keys=None, **gen_kwargs):
            self._force_gen_kwargs()
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
            )

    return LanguageForcedTrainer


def train(args):
    train_path = os.path.abspath(args.train)
    val_path = os.path.abspath(args.val)
    out_path = os.path.abspath(args.out)
    audio_root = os.path.dirname(train_path)

    train_rows = load_manifest(train_path)
    val_rows = load_manifest(val_path)
    language_code = resolve_language(args.language)
    check_homogeneous(train_rows, args.group, language_code, "train")
    check_homogeneous(val_rows, args.group, language_code, "val")

    splits = {"train": train_rows, "val": val_rows}
    test_path = args.test_manifest or os.path.join(audio_root, "manifest_test.jsonl")
    if os.path.exists(test_path):
        # The test manifest is deliberately multi-group (yoruba + english + accents),
        # so it is checked for speaker leakage, not for homogeneity.
        test_rows = load_manifest(test_path)
        splits["test"] = test_rows
    else:
        test_rows = []
        print(f"warning: no test manifest at {test_path}; disjointness check covers train/val only")

    speakers = check_disjoint(splits)
    test_group_speakers = {
        r["speaker"] for r in test_rows if r["group"] == args.group
    }

    durations = probe_durations(train_rows, audio_root)
    selected, seconds, too_long = select_clips(
        train_rows, durations, args.minutes, seed=args.seed, max_seconds=args.max_seconds
    )
    val_subset = cap_rows(val_rows, args.max_val, seed=args.seed)
    if not selected:
        raise ValueError("no training clip fits the duration budget")
    if seconds < 0.8 * args.minutes * 60 and not args.allow_short:
        raise ValueError(
            f"only {seconds / 60:.1f} minutes of usable audio for a {args.minutes}-minute run; "
            "pass --allow-short to train anyway and report the real duration"
        )

    summary = {
        "group": args.group,
        "language": language_code,
        "train_clips": len(selected),
        "train_minutes": round(seconds / 60, 3),
        "train_speakers": len(speakers["train"]),
        "val_clips": len(val_subset),
        "val_speakers": len(speakers["val"]),
        "test_speakers": len(test_group_speakers),
        "skipped_too_long": too_long,
        "out": out_path,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    if os.path.exists(out_path) and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to replace it")

    import torch
    import transformers
    from peft import get_peft_model
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor,
                              set_seed)

    dev = resolve_device(args.device)
    set_seed(args.seed)

    revision = args.revision
    if revision is None and args.model == DEFAULT_MODEL:
        revision = DEFAULT_REVISION

    print(f"loading {args.model}@{revision} on {dev}")
    processor = WhisperProcessor.from_pretrained(args.model, revision=revision)
    base = WhisperForConditionalGeneration.from_pretrained(args.model, revision=revision)
    base.to(dev)
    if not hasattr(base.generation_config, "lang_to_id"):
        # Compatibility shim if a checkpoint ships an outdated generation config.
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        base.generation_config.lang_to_id = {
            f"<|{code}|>": processor.tokenizer.convert_tokens_to_ids(f"<|{code}|>")
            for code in LANGUAGES
        }
        base.generation_config.task_to_id = {
            "transcribe": processor.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
            "translate": processor.tokenizer.convert_tokens_to_ids("<|translate|>"),
        }
        base.generation_config.is_multilingual = True

    lora = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    import numpy as np

    tokenizer = processor.tokenizer

    # Resolve the prompt ids from the tokenizer, not from generation_config:
    # transformers 5.x drops lang_to_id, and convert_tokens_to_ids is stable.
    # An unknown id means this checkpoint cannot express the language we need,
    # so refuse before training rather than let whisper auto-detect.
    language_token_id = tokenizer.convert_tokens_to_ids(f"<|{language_code}|>")
    task_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    unk = tokenizer.unk_token_id
    if language_token_id in (None, unk) or task_token_id in (None, unk):
        raise SystemExit(
            f"{args.model} has no <|{language_code}|>/<|transcribe|> token "
            f"(ids {language_token_id}/{task_token_id}); cannot force the language"
        )
    decoder_start_id = getattr(base.generation_config, "decoder_start_token_id", None)

    def new_guard(target):
        # Phase 3: load_best_model_at_end may hand back a different model object,
        # so the guard is rebuilt on whatever the trainer will actually call.
        return GenerateGuard(
            target, language_code, "transcribe",
            decoder_start_id=decoder_start_id,
            language_id=language_token_id,
            task_id=task_token_id,
        ).install(target)

    guard = new_guard(model)
    last_eval = {}

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = np.array(pred.label_ids, copy=True)
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        hyp = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        ref = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # Keep the raw strings: the gates measure the script and the diacritics,
        # which normalisation for WER would erase.
        last_eval["ref"], last_eval["hyp"] = ref, hyp
        refs = [normalize(x) for x in ref]
        hyps = [normalize(x) for x in hyp]
        counts_w = [utt_counts(a, b, "word") for a, b in zip(refs, hyps)]
        counts_c = [utt_counts(a, b, "char") for a, b in zip(refs, hyps)]
        return {"wer": rate(counts_w), "cer": rate(counts_c)}

    parent = os.path.dirname(out_path) or "."
    os.makedirs(parent, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix=".train-", dir=parent)
    adapter_tmp = tempfile.mkdtemp(prefix=".adapter-", dir=parent)
    ManifestDataset, Collator = _dataset_classes(processor)
    TrainerClass = _trainer_class()

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(run_dir, "trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        fp16=dev.startswith("cuda") and not args.fp32,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_num_workers=0 if dev == "mps" else 2,
    )

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=ManifestDataset(selected, audio_root),
        eval_dataset=ManifestDataset(val_subset, audio_root),
        data_collator=Collator(processor),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        language=language_code,
        task="transcribe",
    )

    try:
        trainer.train()
        # Phase 3: the guard must be on the object the final evaluation calls.
        # If load_best_model_at_end swapped it, the old wrapper is gone and the
        # assertion would be vacuous; rebuilding closes that and re-covers the
        # published weights rather than the last epoch's.
        guard = new_guard(trainer.model)
        trainer.evaluate()
        refs = last_eval.get("ref", [])
        hyps = last_eval.get("hyp", [])
        if not hyps:
            raise RuntimeError("the final evaluation produced no hypotheses")
        if guard.calls == 0:
            # A guard that was never invoked proves nothing; treat it as failure.
            raise RuntimeError(
                "generate() never ran through the language guard; "
                "cannot confirm the forced language reached the model"
            )

        print("\n--- validation examples: read these before quoting the number ---")
        for line in example_lines(refs, hyps, [r["utt_id"] for r in val_subset]):
            print(line)
        gates = pilot_gates(refs, hyps)
        forced = guard.report()
        print("\n" + json.dumps({"pilot_gates": gates, "language_forced": forced},
                                indent=2, ensure_ascii=False))
        if not gates["ok"]:
            # Phase 2: fail closed. Nothing is published, so no test cache can be
            # produced from an adapter that failed the pre-registered gates.
            raise SystemExit("pilot gates failed: " + "; ".join(gates["failures"]))

        trainer.model.save_pretrained(adapter_tmp)
        history = [h["eval_wer"] for h in trainer.state.log_history if "eval_wer" in h]
        if not history:
            raise RuntimeError("no validation WER was recorded; refusing to publish the adapter")
        meta = {
            "base_model": args.model,
            "revision": revision,
            "group": args.group,
            "language": language_code,
            "language_forced": forced,
            "pilot_gates": gates,
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                     "dropout": args.lora_dropout, "targets": list(LORA_TARGETS),
                     "task_type": LORA_TASK_TYPE},
            "minutes_requested": args.minutes,
            "train_minutes_actual": round(seconds / 60, 3),
            "train_clips": len(selected),
            "train_speakers": len(speakers["train"]),
            "val_clips": len(val_subset),
            "val_speakers": len(speakers["val"]),
            "test_speakers_checked": len(test_group_speakers),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "seed": args.seed,
            "device": dev,
            "eval_wer_history": history,
            "best_val_wer": min(history),
            "manifest_sha256": {
                "train": manifest_sha256(train_path),
                "val": manifest_sha256(val_path),
                **( {"test": manifest_sha256(test_path)} if test_rows else {} ),
            },
            "versions": {
                "transformers": transformers.__version__,
                "torch": torch.__version__,
            },
        }
        with open(os.path.join(adapter_tmp, "train_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(adapter_tmp, out_path)
        print(f"wrote {out_path} (best val wer {min(history):.2%})")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        if os.path.isdir(adapter_tmp):
            shutil.rmtree(adapter_tmp, ignore_errors=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test-manifest", default=None,
                    help="defaults to manifest_test.jsonl next to --train")
    ap.add_argument("--out", required=True, help="adapter directory, e.g. checkpoints/yo-30min")
    ap.add_argument("--group", default="yoruba_read")
    ap.add_argument("--language", default="yo")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--max-val", type=int, default=120)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return train(a)


if __name__ == "__main__":
    raise SystemExit(main())
