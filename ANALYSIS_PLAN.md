# analysis plan (written before any model is run)

**status:** PRE-REGISTERED. TBDs are filled; the thresholds below are fixed. This file was
committed before any model was run. After that, change it only by adding a dated line to the
deviations log at the bottom. The commit history is the proof that the plan came first.

## question

How well does whisper-small hear different languages and different accents, why do the
gaps exist, and does a small LoRA adapter trained on 15 to 30 minutes of audio narrow
them, and at what cost to other groups?

## groups

- reference language: english (voxpopuli `en` test rows, re-partitioned by speaker id)
- other high-resource language: german (voxpopuli `de` test rows)
- star language: yoruba (public NaijaVoices mirror, `EYEDOL/naija-voices-yoruba-*`)
- second low-resource language if time: hausa (same mirror)
- english accents: `en_nl`, `en_de`, `en_cs` from voxpopuli `en_accented` (L2 English, test only)
- spontaneous clips: ~40 minimum, own voice or people who agreed, star language, native-speaker checked

## data and splits

- sources: voxpopuli (language + L2 accent rows), the public NaijaVoices mirror (star language),
  our own recordings. cite each with a link; record the resolved mirror revision in `audit.json`.
- about 200 test clips per group where available; count speakers, not just clips.
- splits are speaker-disjoint: train / validation / test. voxpopuli is evaluation-only, so its
  rows are all test. NaijaVoices speakers are hashed (`sha256(speaker_key)`) into 70/15/15, and a
  speaker may appear in exactly one group. validation is for lora early stopping and for choosing
  settings. the test set is evaluated once per final system.
- our own spontaneous clips are test-only, except where a run is explicitly labeled otherwise.

## normalization (fixed per language, decided now)

- lowercase: yes. strip punctuation: yes. numerals: keep as written (no number expansion).
- diacritics: keep by default; also report with `--strip-diacritics` for languages where
  they carry meaning.
- the same normalization is applied to references and hypotheses in every run.

## metrics

- wer and cer (cer ignores whitespace). per group, never only overall.
- 95% bootstrap intervals, resampling speakers (not clips). paired bootstrap for
  before/after comparisons on the same clips.
- error mix: substitution / deletion / insertion shares. hallucination rate (output much
  longer than the reference, or a repeated loop).
- native-speaker error labels on ~30 errors: harmless / meaning-changing / unintelligible.

## hypotheses and what would count as support

Thresholds are drafts. Edit them now, before seeing any result.

- **h1** error rises as the language's share of whisper training data falls.
  support: descriptive only (3 or 4 languages cannot support a regression). report the
  ordering and compare it with the training-data shares in the whisper paper.
- **h2** the accent gap is smaller than the language gap.
  support: the interval of (max accent wer - min accent wer) lies below the interval of
  (star language wer - english wer), non-overlapping.
- **h3** a small LoRA adapter closes a large share of the star-language gap.
  support (draft): adapter closes at least 30% of the baseline gap to english on held-out
  speakers, and the paired-difference 95% interval excludes 0.
- **h4** that gain costs something elsewhere.
  support: at least one other group's wer rises, with the paired 95% interval excluding 0.
  report the largest change either way.
- **h5** for low-resource languages, much of the error is insertions and hallucination.
  support (draft): insertions are over 30% of errors, or hallucination rate is over 5%,
  for the lowest-resource language.
- **h6** spontaneous speech has higher error than read speech, and an adapter trained on
  read speech helps it less.
  support: spontaneous wer - read wer > 0 (interval excludes 0), and the adapter's gain on
  spontaneous clips is smaller than on read clips.

If a hypothesis is not supported we say so.

## lora protocol (defaults, edit if needed)

- peft LoRA, rank 8, on attention projections of encoder and decoder; encoder-only vs
  decoder-only only if time allows.
- training audio: 15 and 30 minutes (5 if time), speaker-disjoint from validation and test.
- early stopping on validation wer; one seed (two if time allows).
- turning the adapter off must reproduce the baseline exactly.
- forgetting: evaluate every adapter on every other group.

## rules we hold ourselves to

- no tuning on the test set. no peeking at test results to pick a model.
- no exclusion of clips after seeing results, except documented defects (corrupt audio,
  missing reference), listed below.
- we report weak and negative results.
- audio of other people is never committed without their permission.

## deviations log

| date (et) | change | why |
| --- | --- | --- |
| 2026-11-14 | Move every model run (dataset streaming, whisper, LoRA) to a free Colab T4, Kaggle T4 as backup. The laptop runs only tests, scoring on cached JSONL, and Gradio. The cache JSONL plus its `.meta.json` sidecar is the only artifact returned. Skip the local whisper smoke test. | 14 GB free disk and an unstable link make multi-GB local downloads and MPS runs unreliable; the deadline needs GPU time, and scoring is already model-free. |
| 2026-11-14 | Replace the planned FLEURS/Common Voice sources with voxpopuli `en`, `de`, `en_accented` (pinned `42f01879c780b4a2e90ec0b4f616c2ece526e4f1`) and the public NaijaVoices mirror `EYEDOL/naija-voices-yoruba-split_*` for the star-language group. NaijaVoices speakers are hash-split by speaker key; voxpopuli rows are evaluation-only. | FLEURS exposes no `speaker_id`; Common Voice audio is gated or unavailable; NaijaVoices proper is gated at 84 GB. The mirror is the only star-language audio reachable in this window. Its licence/provenance is unstated, so the resolved revision is recorded in `audit.json` and results are labelled as a public mirror. |
| 2026-11-14 | Any group below 200 test clips or 10 test speakers is reported as descriptive only, without a speaker-clustered interval, unless the build audit shows it cleared both thresholds. FLEURS, if used at all, is clip-level and labelled as such. | Speaker-clustered intervals require real, independent speaker ids; a thin or fabricated group would overstate confidence. |
