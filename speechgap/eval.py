"""Transcribe a manifest with whisper (optionally plus a lora adapter) and cache the output.

Runs on a free Colab T4 (``--device cuda``), Kaggle T4, or Apple silicon
(``--device mps``). The cache is the only artifact that needs to come back to
the laptop; ``speechgap.report`` scores it with no model installed.

Manifest (jsonl), one record per clip, exactly these six keys:
  {"utt_id": "a1", "audio": "audio/test/a1.wav", "speaker": "voxpopuli:42",
   "group": "yoruba_read", "ref": "reference text", "language": "yoruba"}

``audio`` is resolved relative to the manifest's own directory, so a manifest
plus its ``audio/`` tree is portable between Colab and the laptop.

Usage (Colab):
  python -m speechgap.eval --manifest data/manifest_test.jsonl --device cuda \
      --out results/cache/zero_shot.jsonl
  python -m speechgap.eval --manifest data/manifest_test.jsonl --device cuda \
      --adapter checkpoints/yo15 --out results/cache/yo_lora15.jsonl
Usage (laptop, scoring only):
  python -m speechgap.report results/cache/zero_shot.jsonl

Resilience: output is written to ``<out>.part`` and renamed only after every
row is transcribed. On a disconnect the ``.part`` file survives; rerun with
``--resume`` to continue without retranscribing. A final cache file is never
partially written, so ``report`` can never score a truncated run.
"""
import argparse
import hashlib
import json
import os

REQUIRED_KEYS = ("utt_id", "audio", "speaker", "group", "ref", "language")
DEFAULT_MODEL = "openai/whisper-small"
# Pinned so a Hub-side change cannot silently move the experiment.
DEFAULT_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"


def manifest_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    """Parse and validate a manifest. Raises instead of skipping bad lines.

    Threat (Phase 3): a hand-edited or half-transferred manifest must not be
    silently repaired into a different experiment. Exactly six keys, no extras,
    non-empty values, and globally unique ids are required.
    """
    rows = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid json: {e}") from e
            if not isinstance(r, dict):
                raise ValueError(f"{path}:{lineno}: expected an object")
            missing = [k for k in REQUIRED_KEYS if k not in r]
            extra = sorted(set(r) - set(REQUIRED_KEYS))
            if missing or extra:
                raise ValueError(
                    f"{path}:{lineno}: manifest contract is exactly {list(REQUIRED_KEYS)}; "
                    f"missing={missing} extra={extra}"
                )
            for k in REQUIRED_KEYS:
                if not isinstance(r[k], str) or not r[k].strip():
                    raise ValueError(f"{path}:{lineno}: {k!r} must be a non-empty string")
            if r["utt_id"] in seen:
                # Duplicate ids double-weight a speaker and corrupt the bootstrap.
                raise ValueError(f"{path}:{lineno}: duplicate utt_id {r['utt_id']!r}")
            seen.add(r["utt_id"])
            rows.append(r)
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def resolve_device(requested):
    """Resolve 'auto' honestly; never fall back from an explicit cuda request."""
    import torch

    if requested in (None, "", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but CUDA is not available. "
                "Refusing to fall back silently; use --device cpu for a deliberate CPU run."
            )
        return requested
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("--device mps was requested but MPS is not available")
        return requested
    if requested == "cpu":
        return requested
    raise ValueError(f"unknown device {requested!r}")


def supported_languages(pipe):
    """Codes the checkpoint actually exposes, or None if the API changed."""
    raw = None
    gen = getattr(getattr(pipe, "model", None), "generation_config", None)
    if gen is not None:
        raw = getattr(gen, "lang_to_id", None)
    if not raw:
        tok = getattr(pipe, "tokenizer", None)
        raw = getattr(tok, "lang_code_to_id", None)
    if not raw:
        return None
    codes = set()
    for key in raw:
        k = str(key)
        if k.startswith("<|") and k.endswith("|>"):
            k = k[2:-2]
        codes.add(k)
    return codes or None


def _read_partial(part_path, valid_ids):
    """Return utt_ids already cached in ``.part`` and drop a torn final line.

    Threat (Phase 3): a stale ``.part`` from a different manifest must not be
    merged into this run. Unknown or duplicate ids raise.
    """
    done = set()
    good_bytes = 0
    if not os.path.exists(part_path):
        return done
    with open(part_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                print(f"resume: dropping truncated final line at byte {pos}")
                break
            try:
                rec = json.loads(line)
                uid = rec["utt_id"]
                if not isinstance(rec.get("hyp"), str):
                    raise ValueError("missing hyp")
            except Exception:
                print(f"resume: dropping unreadable line at byte {pos}")
                break
            if uid not in valid_ids:
                raise ValueError(
                    f"{part_path} contains utt_id {uid!r} that is not in this manifest; "
                    "refusing to mix two different runs. Delete it or pass --force."
                )
            if uid in done:
                raise ValueError(f"{part_path} contains duplicate utt_id {uid!r}")
            done.add(uid)
            good_bytes = f.tell()
    with open(part_path, "r+b") as f:
        f.truncate(good_bytes)
    return done


def _write_meta(out_path, meta):
    """Sidecar audit trail: what model/device/manifest produced this cache."""
    meta_path = out_path + ".meta.json"
    tmp = meta_path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, meta_path)


def transcribe(manifest, out_path, model_name=DEFAULT_MODEL, adapter=None,
               device=None, force_language=True, force=False, resume=False,
               batch_size=None, revision=None, limit=None, fp32=False):
    if force:
        for p in (out_path, out_path + ".part"):
            if os.path.exists(p):
                os.remove(p)
    elif os.path.exists(out_path):
        print(f"{out_path} exists; reuse it (pass --force to redo)")
        return out_path

    part_path = out_path + ".part"
    if not force and not resume and os.path.exists(part_path):
        raise RuntimeError(
            f"{part_path} exists from an interrupted run; pass --resume to continue it "
            "or --force to discard it"
        )

    rows = load_manifest(manifest)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        rows = rows[:limit]
    ids = {r["utt_id"] for r in rows}
    done = _read_partial(part_path, ids) if resume else set()
    if done:
        print(f"resume: {len(done)} clips already cached")

    # Resolve and check every pending audio path before loading the model: a
    # broken manifest should not cost a 1 GB download first.
    audio_root = os.path.dirname(os.path.abspath(manifest))
    pending_paths = {}
    for r in rows:
        if r["utt_id"] in done:
            continue
        p = r["audio"]
        if not os.path.isabs(p):
            p = os.path.join(audio_root, p)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{r['utt_id']}: audio not found at {p}")
        pending_paths[r["utt_id"]] = p

    import torch
    from transformers import pipeline

    from .languages import resolve_language

    dev = resolve_device(device)
    if batch_size is None:
        batch_size = 8 if dev.startswith("cuda") else 1
    dtype = torch.float16 if (dev.startswith("cuda") and not fp32) else torch.float32
    if revision is None and model_name == DEFAULT_MODEL:
        revision = DEFAULT_REVISION

    print(f"loading {model_name}@{revision} on {dev} ({str(dtype).replace('torch.', '')})")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        revision=revision,
        chunk_length_s=30,
        device=dev,
        torch_dtype=dtype,
    )
    if adapter:
        from peft import PeftModel

        pipe.model = PeftModel.from_pretrained(pipe.model, adapter).merge_and_unload()
        # PeftModel loads on CPU; move the merged model back before inference.
        pipe.model.to(dev)
        pipe.model.eval()

    supported = supported_languages(pipe)

    # Resolve every language before the first inference: an unsupported language
    # is a configuration error, not a reason to let whisper auto-detect.
    resolved = {}
    for r in rows:
        if r["utt_id"] in done:
            continue
        lang = resolve_language(r["language"], supported=supported) if force_language else None
        resolved[r["utt_id"]] = lang
    unknown = {l for l in resolved.values() if l is not None and supported and l not in supported}
    if unknown:
        raise ValueError(f"checkpoint does not expose languages: {sorted(unknown)}")

    ordered = [r for r in rows if r["utt_id"] not in done]
    # Whisper takes one language per call, so batch within language buckets.
    buckets = {}
    for r in ordered:
        buckets.setdefault(resolved[r["utt_id"]], []).append(r)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(part_path, "a", encoding="utf-8") as out:
        for lang in sorted(buckets, key=lambda x: (x is not None, x or "")):
            bucket = buckets[lang]
            for start in range(0, len(bucket), batch_size):
                chunk = bucket[start:start + batch_size]
                paths = [pending_paths[r["utt_id"]] for r in chunk]
                kwargs = {"task": "transcribe"}
                if lang:
                    kwargs["language"] = lang
                preds = pipe(paths, batch_size=len(paths), generate_kwargs=kwargs)
                if isinstance(preds, dict):
                    preds = [preds]
                if len(preds) != len(chunk):
                    raise RuntimeError(
                        f"expected {len(chunk)} transcripts, got {len(preds)}"
                    )
                for r, pred in zip(chunk, preds):
                    rec = {k: r[k] for k in ("utt_id", "speaker", "group", "ref")}
                    rec["hyp"] = pred["text"]
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                # Flush per batch: a Colab disconnect must not lose the batch.
                out.flush()

    os.replace(part_path, out_path)
    meta = {
        "manifest": os.path.abspath(manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "model": model_name,
        "revision": revision,
        "adapter": adapter,
        "device": dev,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": batch_size,
        "force_language": bool(force_language),
        "n_manifest_rows": len(rows),
        "n_transcribed_this_run": written,
        "n_resumed": len(done),
        "transformers_version": getattr(__import__("transformers"), "__version__", None),
        "torch_version": getattr(torch, "__version__", None),
    }
    _write_meta(out_path, meta)
    print(f"wrote {out_path} ({written} new, {len(done)} resumed)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None,
                    help="hub revision/sha; defaults to the pinned whisper-small sha")
    ap.add_argument("--adapter")
    ap.add_argument("--device", default="auto",
                    help="auto, cpu, mps, cuda, or cuda:0")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test on the first N manifest rows")
    ap.add_argument("--fp32", action="store_true",
                    help="disable fp16 on cuda (use if you see NaN/empty output)")
    ap.add_argument("--no-force-language", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted .part cache")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    transcribe(a.manifest, a.out, model_name=a.model, adapter=a.adapter,
               device=a.device, force_language=not a.no_force_language,
               force=a.force, resume=a.resume, batch_size=a.batch_size,
               revision=a.revision, limit=a.limit, fp32=a.fp32)
