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
| `speechgap/eval.py` | transcribe a manifest with whisper (+ optional adapter), cache the output; runs on CPU or MPS. **not run yet** |
| `speechgap/report.py` | score cached transcripts per group |
| `tests/` | tests for the metrics (`python tests/test_metrics.py`, or pytest) |
| `tests/fixtures/` | a tiny transcript file the report CLI can run against without a model |
| `.github/workflows/tests.yml` | CPU-only CI: metrics tests plus a report smoke test |
| `results/cache/` | cached transcripts, committed so the notebook runs fast |
| `data/` | local audio and manifests, git-ignored |
| `notebooks/` | the submission notebook (loads cached results by default) |

## quick start

```bash
# apple silicon: install torch and an audio decoder first
pip install torch
brew install ffmpeg
pip install -r requirements.txt

PYTHONPATH=. python tests/test_metrics.py          # sanity check the metrics
PYTHONPATH=. python -m speechgap.report tests/fixtures/tiny_report.jsonl
```

## compute

whisper-small runs locally on the M2 with `--device mps` (CPU works but is slower).
Transcribing a few hundred clips is comfortable; training the LoRA adapter locally is
possible but slow, so use a rented GPU or Colab only for that step. `requirements.txt` is
unpinned, so record the versions that work once the pipeline first runs.

GitHub Actions is CPU-only and intentionally dependency-free: it runs the metrics tests and
the report smoke test, not whisper or training.

## data and privacy

Public datasets and our own voice only. Audio is git-ignored and never committed. Nobody
else is recorded without their permission. The demo processes uploaded audio in memory and
never stores it. Only cached transcripts and report JSON are shipped in the repo.

## sources and credits

TBD: datasets (fleurs, common voice), models (openai/whisper-small), libraries
(transformers, peft, datasets, gradio), AI tools used (Claude), each with a link.
