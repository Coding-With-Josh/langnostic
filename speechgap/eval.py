"""Transcribe a manifest with whisper (optionally plus a lora adapter) and cache the output.

NOT RUN YET: written without gpu or network access. Expect to adjust it on first use.

Manifest (jsonl), one record per clip:
  {"utt_id": "a1", "audio": "data/a1.wav", "speaker": "s01", "group": "yoruba_read",
   "ref": "reference text", "language": "yoruba"}

Usage (local; apple silicon: add --device mps):
  python -m speechgap.eval --manifest data/manifest.jsonl --out results/cache/zero_shot.jsonl
  python -m speechgap.eval --manifest data/manifest.jsonl --adapter checkpoints/yo15 \
      --out results/cache/yo_lora15.jsonl
  python -m speechgap.eval --manifest data/manifest.jsonl --device mps \
      --out results/cache/zero_shot_mps.jsonl
Existing output is reused unless --force is given, so reruns are cheap.
"""
import argparse
import json
import os


def transcribe(manifest, out_path, model_name="openai/whisper-small", adapter=None,
               device=None, force_language=True, force=False):
    if os.path.exists(out_path) and not force:
        print(f"{out_path} exists; reuse it or pass --force")
        return
    from transformers import pipeline  # imported lazily so the rest works without it

    with open(manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pipe = pipeline("automatic-speech-recognition", model=model_name,
                    chunk_length_s=30, device=device)
    if adapter:
        from peft import PeftModel
        pipe.model = PeftModel.from_pretrained(pipe.model, adapter).merge_and_unload()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for r in rows:
            kwargs = {"task": "transcribe"}
            if force_language and r.get("language"):
                kwargs["language"] = r["language"]
            hyp = pipe(r["audio"], generate_kwargs=kwargs)["text"]
            rec = {k: r[k] for k in ("utt_id", "speaker", "group", "ref")}
            rec["hyp"] = hyp
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--adapter")
    ap.add_argument("--device", default=None,
                    help="cpu, mps, cuda, or a cuda index; default cpu")
    ap.add_argument("--no-force-language", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    transcribe(a.manifest, a.out, model_name=a.model, adapter=a.adapter,
               device=a.device, force_language=not a.no_force_language, force=a.force)
