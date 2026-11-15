# langnostic

[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Coding-With-Josh/langnostic/blob/main/notebooks/colab_run.ipynb)

> working name; rename the repo and the `speechgap` package if it changes.

does speech recognition hear every language and accent equally well? this project measures whisper-small across languages and accents, looks at where the gaps come from, and tests whether a small lora adapter trained on 15 to 30 minutes of audio narrows them, and what that costs on other groups.

built for the veritas ai international high school hackathon 2026 (track: ai + languages).

**status:** early. results go in `results/` and this file as they land. nothing here is a finding yet.

## repo layout

| path | what it is |
| --- | --- |
| `ANALYSIS_PLAN.md` | hypotheses, splits, metrics and decision rules, committed before any experiment |
| `speechgap/normalize.py` | text normalization (fixed per language) |
| `speechgap/metrics.py` | wer / cer, error mix, speaker-clustered bootstrap intervals, hallucination check |
| `speechgap/languages.py` | language names → whisper codes; refuses codes the checkpoint lacks |
| `speechgap/data.py` | build speaker-disjoint manifests from public datasets (runs where the data is: colab) |
| `speechgap/eval.py` | transcribe a manifest with whisper (+ optional adapter), cache the output; colab t4 or mps |
| `speechgap/train.py` | train a rank-8 lora adapter on one group, speaker-disjoint from val/test, cached to `checkpoints/` |
| `speechgap/report.py` | score cached transcripts per group; no model needed |
| `tests/` | stdlib tests: metrics, manifest contract, split/cap invariants, cache resume |
| `tests/fixtures/` | a tiny transcript file the report cli can run against without a model |
| `.github/workflows/tests.yml` | cpu-only ci: all stdlib tests plus a report smoke test |
| `results/cache/` | cached transcripts, committed so scoring and the demo run without a gpu |
| `data/` | local audio and manifests, git-ignored |
| `notebooks/` | the colab runner (build → transcribe → report) |

## quick start (laptop: no model, no downloads)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg           # only for local audio decode

PYTHONPATH=. python tests/test_metrics.py
PYTHONPATH=. python tests/test_data.py
PYTHONPATH=. python tests/test_eval.py
PYTHONPATH=. python -m speechgap.report tests/fixtures/tiny_report.jsonl

PYTHONPATH=. python -m speechgap.data build --dry-run \
    --naija yoruba:0-0 --voxpopuli en:test --l2arctic \
    --fleurs yo_ng:test --allow-clip-level
```

## compute

model work runs in a free **colab t4** (kaggle t4 as backup): dataset download and decode, whisper transcription, and lora training. the laptop only runs tests, scores cached jsonl, and develops the gradio demo. `speechgap.eval --device cuda` refuses to fall back silently if cuda is absent.

the cache is the only artifact that comes back from colab: `results/cache/*.jsonl` plus its `.meta.json` sidecar. audio and manifests stay in the colab session (or drive); they are regenerable and git-ignored.

```bash
# inside colab, from the repo root (about 2.6 gb of pinned parquet)
python -m speechgap.data build --out-dir data --naija yoruba:0-0 \
    --voxpopuli en:test --l2arctic --fleurs yo_ng:test --allow-clip-level
python -m speechgap.data verify --out-dir data
python -m speechgap.eval --manifest data/manifest_test.jsonl --device cuda \
    --out results/cache/zero_shot.jsonl

# lora: 15 and 30 minutes, then the same scorer with --adapter
python -m speechgap.train --train data/manifest_train.jsonl \
    --val data/manifest_val.jsonl --minutes 30 --group yoruba_read --language yo \
    --out checkpoints/yo-30min
python -m speechgap.eval --manifest data/manifest_test.jsonl --device cuda \
    --adapter checkpoints/yo-30min --out results/cache/yo_lora30.jsonl
```

a transcription run writes `<out>.part` first and renames it only when every row is done. if colab disconnects, rerun the same command with `--resume`; it keeps validated rows and never lets a truncated file be scored.

github actions is cpu-only and dependency-free: it runs the stdlib tests and the report smoke test, not whisper, dataset streaming, or training.

## data and privacy

public datasets and own recordings only. audio is git-ignored and never committed. nobody else is recorded without permission. the demo processes uploaded audio in memory and never stores it. only cached transcripts and report json ship in the repo.

`data.py` never makes up speaker ids: sources without one (fleurs) are refused unless `--allow-clip-level` is passed, and the resulting groups are flagged in `audit.json`.

## sources and credits

tbd: datasets (fleurs, common voice), models (openai/whisper-small), libraries (transformers, peft, datasets, gradio), ai tools used (claude), each with a link.
