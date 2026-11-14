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

- peft LoRA, rank 8, alpha 16, dropout 0.05, on `q_proj` and `v_proj`; encoder-only vs
  decoder-only only if time allows.
- training audio: 15 and 30 minutes (5 if time), speaker-disjoint from validation and test.
  clips longer than whisper's 30 s window are excluded rather than truncated.
- optimisation: lr 1e-3, 3 epochs maximum, effective batch 16 (4 x 4), warmup 50 steps,
  early stopping on validation wer with patience 2, one seed (two if time allows).
- early stopping uses validation only; each final adapter is evaluated on the test set once.
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
| 2026-11-14 (sat) | FLEURS zero-shot baseline becomes the headline result (`results/cache/fleurs_zero_shot.jsonl`, committed as-is): `openai/whisper-small`, fp32, greedy decoding, `chunk_length_s=30`, language forced per clip, no repetition guard, first 100 `test` clips each of `en_us` and `yo_ng`, streamed from FLEURS. FLEURS has no speaker ids, so every clip is its own speaker and the bootstrap intervals are clip-level and optimistic, not speaker-clustered. Observed: english wer 6.5% [5.1, 7.9], cer 3.0%; yoruba wer 367.4% [307.9, 427.9], cer 145.6%, insertions 73% of errors, hallucination rate 49%; most yoruba hyps are one character repeated to the token limit. A sensitivity run (`no_repeat_ngram_size=4`, `max_new_tokens=128`, `chunk_length_s` removed, 20 yoruba clips) is reported separately and labelled post-hoc; it does not replace this row's numbers. This run is n=100 per group, below the 200-test-clip floor set earlier the same day, so it is labelled descriptive: the intervals are reported as observed but are not confirmatory. NaijaVoices caveat, restating and correcting the licence field of the earlier mirror row: `EYEDOL/naija-voices-yoruba-split_*` is an unofficial mirror of a gated dataset, listed cc by-nc-sa 4.0 (non-commercial), no audio is committed, and these results are not described as the original paper's test set. | FLEURS is the only public source with both `en_us` and `yo_ng` test audio here; the default run is reported exactly as it happened, loops included, rather than swapped for the more flattering sensitivity settings after seeing the result. The earlier "licence unstated" note was wrong and is superseded by this row. |
| 2026-11-14 (sat) | Post-hoc yoruba sensitivity run, 20 clips per variant (`results/cache/fleurs_yo_sensitivity.jsonl`), correcting the previous row's settings: the guard used `no_repeat_ngram_size=4` with `max_new_tokens=256` (not 128), and `guard_nochunk` removed `chunk_length_s` while keeping the guard. Removing chunking changed nothing: all 20 guarded hyps are byte-identical with and without it, measured on the file, because every FLEURS clip is under 30 s. Observed: default 327.5% wer, guard 127.1% wer / 75.4% cer, guard_nochunk identical; the table below carries the intervals and error mix. The guarded hyps are not yoruba: they are latin transliterations with diacritics dropped and, on some clips, khmer/georgian/greek (20.6% of guarded letters are non-latin). The wer drop is fewer insertions (70% to 24%), not better transcription. The 20 default clips are a subset of the 100-clip headline and their hyps are byte-identical there, so this is a re-read of the default condition, not a new result. | The run tests whether a repetition guard explains the loops. It reduces loops and insertions, but does not recover yoruba orthography. It is reported separately, post-hoc and descriptive, so it cannot displace the pre-registered default numbers. |

### 2026-11-14 (sat) — yoruba sensitivity, post-hoc (n=20 per variant)

Clip-level and descriptive; each clip is its own speaker, so the intervals are optimistic.

| variant | settings | wer [95% ci] | cer | insertions | hallucination | non-latin letters |
| --- | --- | --- | --- | --- | --- | --- |
| default | `chunk_length_s=30`, no repetition guard | 327.5% [217.7, 458.2] | 159.4% | 70% | 50% | 30.4% |
| guard | `no_repeat_ngram_size=4`, `max_new_tokens=256`, `chunk_length_s=30` | 127.1% [107.2, 150.6] | 75.4% | 24% | 10% | 20.6% |
| guard_nochunk | same guard, `chunk_length_s` removed | 127.1% [107.2, 150.6] | 75.4% | 24% | 10% | 20.6% |
