# langnostic

[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Coding-With-Josh/langnostic/blob/main/notebooks/colab_run.ipynb)

this project measures how well whisper-small transcribes yoruba, english and accented english, and tests whether a small lora adapter trained on 15 to 30 minutes of yoruba audio narrows the yoruba gap, and what that costs on english.

built for the veritas ai international high school hackathon 2026 (track: ai + languages). built by Joshua Idele, solo.

**site:** [langnostic.vercel.app](https://langnostic.vercel.app)

**status:** built, run and scored. the results are below.

## results

whisper-small is very bad at yoruba zero-shot: 940.0% wer on the 300-clip naija test set and 367.4% on the 100-clip fleurs test set, mostly repetition loops (77% of naija zero-shot hypotheses loop). on english it is fine: 10.6% wer on english read and 8.7% on l2-arctic.

a 30-minute lora adapter cuts yoruba wer on held-out speakers from 940.0% to 131.6% on naija and from 367.4% to 124.6% on fleurs (the fleurs baseline was run at batch size 1 and the adapter at batch size 8), but mostly by stopping those loops (the naija loop rate falls from 77% to 10.3%). on clips that do not loop, wer is still 92.1% on naija and 95.6% on fleurs. it does not yet transcribe yoruba accurately.

the adapter costs english: 17.6% wer on english read (from 10.6%) and 108.6% on accented l2-arctic english (from 8.7%).

h7 (tone marks) proxy: stripping tone marks and underdots removes only a minority of word errors: 13.8% of all-clip errors on the naija adapter and 14.5% on the fleurs adapter, and 21.0% and 23.7% on the non-looped clips; no cache or clip set exceeds 23.7%. it does not check whether the marks that are present are correct.

these are descriptive: one seed per run, no confirmatory claim, and fleurs has no speaker ids, so its intervals are clip-level and optimistic. every number above comes from the committed reports in `results/`; intervals, the deviations log and the rest of the report are on the [site](https://langnostic.vercel.app).

limits: one seed per run. the 15-minute adapter failed its validation gate twice and is not used. the 30-minute run passed validation (97.1% wer) but scored 131.6% on test, partly because validation decodes without chunking and test uses 30-second chunks. the language token is unverified for adapter runs, and the english rows are forced to english. dropping looped clips is a post-hoc view. fleurs intervals are clip-level. a native speaker corrected the reference for one example clip; no native speaker has scored the model outputs.

more caveats on decoding, the language token and scoring are in notes.md.

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
| `speechgap/diacritic_share.py` | h7 (tone marks) proxy: score with and without tone marks and underdots |
| `tests/` | stdlib tests: metrics, manifest contract, split/cap invariants, cache resume |
| `tests/fixtures/` | a tiny transcript file the report cli can run against without a model |
| `.github/workflows/tests.yml` | cpu-only ci: stdlib tests and a report smoke test run in ci; four torch tests skip in ci by design |
| `results/cache/` | cached transcripts, committed so scoring runs without a gpu |
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

model work runs on a free **kaggle t4** (colab t4 was the earlier machine): dataset download and decode, whisper transcription, and lora training. the laptop only runs tests, scores cached jsonl, and builds the static site (no gradio). `speechgap.eval --device cuda` refuses to fall back silently if cuda is absent.

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

stdlib tests and a report smoke test run in ci; four torch tests skip in ci by design.

## data and privacy

public datasets and own recordings only. audio is git-ignored and never committed. nobody else is recorded without permission. there is no live demo; the site shows precomputed results only. the adapters are not distributed. only cached transcripts and report json ship in the repo.

`data.py` never makes up speaker ids: sources without one (fleurs) are refused unless `--allow-clip-level` is passed, and the resulting groups are flagged in `audit.json`.

yoruba audio: i used naijavoices yoruba audio from a third-party mirror (huggingface.co/datasets/EYEDOL/naija-voices-yoruba-split_0-0) to save time during the hackathon. i did not have official access when the runs were made. on 16 november 2026, after the runs, i registered for a naijavoices explorer membership (academic and research use) and was granted official access on hugging face. the original is licensed cc by-nc-sa 4.0. this is non-commercial research. no audio is published and the adapters are not distributed. the repo holds the reference text and model output for the scored clips. the data includes child speakers. i have not checked that the mirror matches the original's yoruba-batch-0.

## sources and credits

- whisper (openai): https://github.com/openai/whisper
- peft (hugging face): https://github.com/huggingface/peft
- naijavoices: emezue and the naijavoices community, interspeech 2025, https://arxiv.org/abs/2505.20564. the licence is cc by-nc-sa 4.0 on the original's card (https://huggingface.co/datasets/naijavoices/naijavoices-dataset). the audio came from a third-party mirror (https://huggingface.co/datasets/EYEDOL/naija-voices-yoruba-split_0-0) that states no licence. the data includes child speakers.
- fleurs: https://arxiv.org/abs/2205.12446
- voxpopuli: https://arxiv.org/abs/2101.00390
- l2-arctic: https://psi.engr.tamu.edu/l2-arctic-corpus/
- compute: kaggle and google colab
- yoruba speaker: Diamond Olamide-OLMD recorded two voice notes and typed both transcripts, with consent. no result uses them. he also corrected the reference text for one example clip.
- ai tools: a coding agent did the repo work (speechgap, tests, ci) under my direction; claude was used for research, planning, code review and drafting the write-up.
- transformers: https://github.com/huggingface/transformers
- datasets: https://github.com/huggingface/datasets
- accelerate: https://github.com/huggingface/accelerate
- jiwer: https://github.com/jitsi/jiwer
- soundfile: https://github.com/bastibe/python-soundfile
- soxr: https://github.com/paulmuaddib/pysoxr
- librosa: https://github.com/librosa/librosa
- pandas: https://github.com/pandas-dev/pandas
- matplotlib: https://github.com/matplotlib/matplotlib
- gradio: https://github.com/gradio-app/gradio
- huggingface_hub: https://github.com/huggingface/huggingface_hub
- aiohttp: https://github.com/aio-libs/aiohttp
- pytest: https://github.com/pytest-dev/pytest