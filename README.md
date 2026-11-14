# langnostic

> working name; rename the repo and the `speechgap` package if the name changes.

Does speech recognition hear every language and accent equally well? This project measures
how whisper-small performs across languages and accents, explains where the gaps come from,
and tests whether a small LoRA adapter trained on 15 to 30 minutes of audio narrows them,
and what that costs on other groups.

Built for the Veritas AI International High School Hackathon 2026 (track: AI + Languages).

**status:** just started. results will appear in `results/` and this file as they come in.
Nothing here is a finding yet.

## repo layout

| path | what it is |
| --- | --- |
| `ANALYSIS_PLAN.md` | hypotheses, splits, metrics and decision rules, committed before any experiment |
| `speechgap/normalize.py` | text normalization (fixed per language) |
| `speechgap/metrics.py` | wer / cer, error mix, speaker-clustered bootstrap intervals, hallucination check |
| `speechgap/languages.py` | language names → whisper codes; refuses codes the checkpoint lacks |
| `speechgap/data.py` | build speaker-disjoint manifests from public datasets (runs where the data is: Colab) |
| `speechgap/eval.py` | transcribe a manifest with whisper (+ optional adapter), cache the output; Colab T4 or MPS |
| `speechgap/train.py` | train a rank-8 LoRA adapter on one group, speaker-disjoint from val/test, cached to `checkpoints/` |
| `speechgap/report.py` | score cached transcripts per group; no model needed |
| `tests/` | stdlib tests: metrics, manifest contract, split/cap invariants, cache resume |
| `tests/fixtures/` | a tiny transcript file the report CLI can run against without a model |
| `.github/workflows/tests.yml` | CPU-only CI: all stdlib tests plus a report smoke test |
| `results/cache/` | cached transcripts, committed so scoring and the demo run without a GPU |
| `data/` | local audio and manifests, git-ignored |
| `notebooks/` | the Colab runner (build → transcribe → report) |

## quick start (laptop: no model, no downloads)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg           # only needed if you decode audio locally

PYTHONPATH=. python tests/test_metrics.py
PYTHONPATH=. python tests/test_data.py
PYTHONPATH=. python tests/test_eval.py
PYTHONPATH=. python -m speechgap.report tests/fixtures/tiny_report.jsonl

PYTHONPATH=. python -m speechgap.data build --dry-run \
    --naija yoruba:0-0 --voxpopuli en_accented:test en:test
```

## compute

All model work happens in a free **Colab T4** (Kaggle T4 as backup): dataset streaming,
whisper transcription, and LoRA training. The laptop only runs tests, scores cached JSONL,
and develops the Gradio demo. `speechgap.eval --device cuda` refuses to fall back silently
if CUDA is absent.

The cache is the only artifact that comes back from Colab: `results/cache/*.jsonl` plus its
`.meta.json` sidecar. Audio and manifests stay in the Colab session (or Drive); they are
regenerable and git-ignored.

```bash
# inside Colab, from the repo root
python -m speechgap.data build --out-dir data --naija yoruba:0-0 \
    --voxpopuli en_accented:test en:test --accents en_nl,en_de,en_cs
python -m speechgap.data verify --out-dir data
python -m speechgap.eval --manifest data/manifest_test.jsonl --device cuda \
    --out results/cache/zero_shot.jsonl

# LoRA: 15 and 30 minutes, then the same scorer with --adapter
python -m speechgap.train --train data/manifest_train.jsonl \
    --val data/manifest_val.jsonl --minutes 30 --group yoruba_read --language yo \
    --out checkpoints/yo-30min
python -m speechgap.eval --manifest data/manifest_test.jsonl --device cuda \
    --adapter checkpoints/yo-30min --out results/cache/yo_lora30.jsonl
```

A transcription run writes `<out>.part` first and renames it only when every row is done.
If Colab disconnects, rerun the same command with `--resume`; it keeps validated rows and
never lets a truncated file be scored.

GitHub Actions is CPU-only and intentionally dependency-free: it runs the stdlib tests and
the report smoke test, not whisper, dataset streaming, or training.

## data and privacy

Public datasets and our own voice only. Audio is git-ignored and never committed. Nobody
else is recorded without their permission. The demo processes uploaded audio in memory and
never stores it. Only cached transcripts and report JSON are shipped in the repo.

`data.py` never fabricates speaker ids: sources without one (FLEURS) are refused unless
`--allow-clip-level` is passed, and the resulting groups are flagged in `audit.json`.

## sources and credits

TBD: datasets (fleurs, common voice), models (openai/whisper-small), libraries
(transformers, peft, datasets, gradio), AI tools used (Claude), each with a link.
