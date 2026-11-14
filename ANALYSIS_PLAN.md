# analysis plan (written before any model is run)

**status:** DRAFT. Fill every TBD and edit the draft thresholds below, then commit this
file *before* running any experiment. After that, change it only by adding a dated line
to the deviations log at the bottom. The commit history is the proof that the plan came
first.

## question

How well does whisper-small hear different languages and different accents, why do the
gaps exist, and does a small LoRA adapter trained on 15 to 30 minutes of audio narrow
them, and at what cost to other groups?

## groups

- reference language: english (read speech)
- other high-resource language: TBD
- lower-resource language(s) whisper supports: TBD (candidates: swahili, yoruba, hausa)
- star language (the one our team speaks): TBD
- english accents: TBD (2 or 3 labels from common voice with enough speakers), plus own voice
- spontaneous clips: ~40 minimum, own voice or people who agreed, star language, native-speaker checked

## data and splits

- sources: fleurs (languages), common voice (accents), our own recordings. cite each with a link.
- about 200 clips per group where available; count speakers, not just clips.
- splits are speaker-disjoint: train / validation / test. validation is for lora early
  stopping and for choosing settings. the test set is evaluated once per final system.
- our own spontaneous clips are test-only, except where a run is explicitly labeled otherwise.

## normalization (fixed per language, decided now)

- lowercase: yes. strip punctuation: yes. numerals: TBD (keep as written / spell out).
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
