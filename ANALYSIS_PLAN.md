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
| 2026-11-14 (sat) | Accent axis changes from voxpopuli `en_accented` (`en_nl`, `en_de`, `en_cs`) to L2-ARCTIC (`chikingsley/l2-arctic-manual-v5.0-16k`), **pooled only**: one `english_l2_arctic` group over all 24 speakers, evaluation-only, all four source splits (train/validation/test/suitcase) pooled into test. L2-ARCTIC is never trained on or used for validation. No per-L1-accent subgroup and no accent-specific claim will be made. | The voxpopuli accented test partitions are thin per accent and their labels come from uploader metadata. L2-ARCTIC provides 24 speakers with documented L1 backgrounds and real speaker ids, and pooling keeps the test interval speaker-clustered rather than splitting 24 people below the 10-test-speaker floor. |
| 2026-11-14 (sat) | Second dataset is FLEURS `yo_ng` **test only** (~0.87 GB), never trained on and never used for validation; German is dropped from the build. The reference English group remains voxpopuli `en` test. | The headline baseline is already FLEURS `yo_ng`; keeping the LoRA comparison on that distribution avoids a source change between baseline and adapter. `de` adds download and decode time without answering a pre-registered question. |
| 2026-11-14 (sat) | Licence correction, final for this build: the NaijaVoices mirror is an unofficial mirror of a gated dataset, listed **cc by-nc-sa 4.0**; use is **non-commercial**, **no adapter trained on this audio will be distributed**, and no audio is committed. L2-ARCTIC is **cc-by-nc-4.0**. FLEURS and VoxPopuli are evaluation-only under their published dataset terms. Results are not described as the original NaijaVoices paper's test set. | The earlier mirror row said the licence was unstated; that was wrong. Recording the restriction before any training prevents an adapter from being published under terms the audio does not allow. |
| 2026-11-14 (sat) | Trimmed build, sized before download: exactly one Naija mirror parquet file (`data/train-00000-of-00010.parquet`, ~0.47 GB), voxpopuli `en` test (~0.94 GB), L2-ARCTIC all four splits (~0.30 GB) and FLEURS `yo_ng` test (~0.87 GB), about 2.57 GB total. Bytes are downloaded at pinned source revisions with `hf_hub_download` and decoded only for rows the allocator admits; no per-row streaming. Resolved revisions are recorded in `audit.json`. | The first Naija file already contains 20,738 rows and 83 distinct speakers in its first 100 rows, so one file supplies the training budget. Download-then-decode removes the observed ~10 s/clip streaming bottleneck and fits inside the 13 GiB free disk alongside the 16 kHz wav output. |
| 2026-11-15 | Pilot gates are enforced by `speechgap.train`, not by eye. A run publishes an adapter only if its final `evaluate()` of the best checkpoint satisfies every condition: val WER ≤100%; `wrong_script_rate` ≤10% on the raw validation hypotheses; no degenerate output (every hypothesis empty, ≥half empty, one distinct hypothesis, fewer than 1 in 5 distinct, or ≥half longer than 10× their reference); and `generate()` provably received `language`/`task` with the forced tokens, with the prompt-token check recorded when the returned sequence exposes it. Three ref/hyp pairs are printed for a human to read. A failing run writes no adapter, so it cannot produce a test cache. | The pilot decides whether the 15/30-minute budget is worth spending; a gate that lives only in a notebook is not a gate. Fail-closed publishing also keeps a 300%-WER or wrong-script adapter from being scored on the test set and quoted. Degenerate thresholds are loose on purpose so a weak but real result is reported, not misfiled as a broken run. |
| 2026-11-15 | LoRA wrapping fix: train with peft's generic `PeftModel` (`task_type=None`) instead of `PeftModelForSeq2SeqLM`. Rank/alpha/dropout/targets, data, splits and metrics are unchanged; only the wrapper class differs. | `PeftModelForSeq2SeqLM.forward` passes `input_ids=None` to Whisper, which has no such parameter, so it reaches `self.decoder(input_ids=decoder_input_ids, ..., **kwargs)` and raises `TypeError: got multiple values for keyword argument 'input_ids'` on the first training step. Generic `PeftModel` forwards `input_features` untouched; the adapter still round-trips through `save_pretrained` -> `PeftModel.from_pretrained` -> `merge_and_unload`, which is what `eval.py --adapter` does. |
| 2026-11-15 | Pilot compute moves to a free Colab T4. The local MPS attempt is abandoned after 1 of 3 optimizer steps in 35 minutes (2,111 s/step; a full pilot plus 20-clip generation was tracking at 2-2.5 h). The pilot runs remotely with the same flags, so `train_meta.device` will read `cuda` rather than `mps`. | The plan already assigns model compute to a T4 and the laptop only to tests, cached-JSONL scoring and Gradio; MPS fp32 throughput made the local run impractical before the deadline. The move changes only the recorded device, not the data, flags, gates, or scorer. The cancelled run published nothing, so no partial adapter can be mistaken for a result. |
| 2026-11-15 | post-hoc forced vs auto-detect comparison on the same first 100 FLEURS `yo_ng` `test` clips. the auto-detect arm runs the existing eval path with `--no-force-language --group fleurs_yo --limit 100 --fp32 --batch-size 1` to `results/cache/fleurs_yo_autodetect.jsonl`; the group filter is applied before the limit and recorded in the `.meta.json` sidecar, and fp32 plus one clip per call match the headline's dtype and batching so language forcing is the only difference. the forced arm is the already-committed `results/cache/fleurs_zero_shot.jsonl`, which is not modified and remains the headline baseline. both arms are scored clip-level and printed side by side; there is no threshold, no confirmatory claim, and the auto-detect row cannot replace or revise the 367.4% baseline. | a single-clip language-token probe on the same pinned revision showed: the tokenizer prefix resolves to `<|yo|>`; forcing yoruba reproduces the `ḍ` loop; forcing english returns english; auto-detect returns phonetic latin yoruba-like text with tone marks dropped and is byte-different from the forced output. so the forced prompt does reach the decoder and the 367.4% baseline is a model result, not a setup bug, which makes forced vs auto-detect a genuine second condition on otherwise identical clips. the probe is one clip; the 100-clip cache is the measurement. |
| 2026-11-15 | model compute moves from the free Colab T4 to a Kaggle T4 (the Colab gpu quota ran out) for the pilot and every later run. no data, manifest, flag, gate, threshold, or scorer changes; `train_meta.device` still reads `cuda`. kaggle needs `pip uninstall -y torchao` before the run, because the preinstalled torchao 0.10.0 breaks the peft import. the first pilot ran with both kaggle T4s visible, so transformers wrapped the model in `nn.DataParallel` (the "gather along dimension 0" warning); later runs pin `CUDA_VISIBLE_DEVICES=0`. | the deadline needs a gpu and colab's free quota ran out; kaggle T4 was already the recorded backup. torchao is not used by this project, so removing it changes importability only, no numerics. two visible gpus double the dataloader batch to 8, so the 34-clip pilot is 5 micro-batches and `ceil(5/4)` is 2 optimizer steps; the single-gpu rerun on the same transformers version is 9 micro-batches and 3 steps, so the step difference is the data-parallel batch size, not a different transformers major version. single-gpu is pinned and recorded instead of left implicit. no run or result is selected or dropped by this move: the failed pilot published no adapter and its numbers stay reported as-is. |
| 2026-11-15 | add `--save-failed-adapter DIR` to `speechgap.train`, default off. when the pilot gates fail, the run writes the final adapter plus `train_meta.json` (`publishable: false`, `outcome: "failed_pilot_gates"`, `pilot_gates.ok: false`, and the `failures` list) to `DIR`, then exits nonzero. `DIR` must end in `.failed`, must differ from `--out`, must not be a symlink, and must not already exist unless `--force`. no flag means the previous behaviour: nothing is written. the pass path and the fail path both go through one `_publish_adapter`, which stages in the destination's parent and renames atomically. `checkpoints/` and `*.failed/` stay gitignored. | a failed 15/30-minute run needs to be evaluable, but it must never look like a published result. a separate, suffixed, opt-in destination keeps the pre-registered rule intact for `checkpoints/<name>` while making the failure label travel with the artifact. one publish function means a crash cannot leave a half-written adapter in either path. |
| 2026-11-15 | `GenerateGuard` now asserts on the tensor the decoder actually consumed: `decoder_embedding` finds `decoder.embed_tokens` through the peft wrapper, a forward pre-hook captures the first three ids of the first decoder call, and `prompt_prefix_ok` requires the sequence `sot`, language token, transcribe token. the returned-sequence check stays as the fallback when the hook does not fire. `language_forced.prompt_source` records `decoder_input` or `output_sequence`; `prompt_tokens_verified` is true only when one of them observed the right prefix. no flag, gate or threshold changes. | the kaggle pilot reported `prompt_tokens_verified: false` because that transformers version strips the prompt from the returned sequences, so the old check returned `None` and the only live control was the kwarg assertion. the decoder input is the tensor that decides the language, so it is observable even when the return is not. the hook raises on a wrong prefix and records `false` (never `true`) when it cannot observe one. |
| 2026-11-15 | the kaggle pilot failed the pre-registered gates: val wer 143.5%, `wrong_script_rate` 21.6%, degenerate `[]`, diacritic retention 0.295, hypotheses in khmer and devanagari. it published nothing. the 15-minute run proceeds with the preregistered settings unchanged (rank 8, alpha 16, dropout 0.05, lr 1e-3, 3 epochs, batch 4x4, patience 2), because the 2-minute pilot ran 2 optimizer steps and never left the linear warmup (`--warmup-steps 50`), so its learning rate was about 3e-5 rather than 1e-3; it tested plumbing, not the budget. `train_meta` and the eval `.meta.json` now record `transformers`, `peft` and `torch` versions so each run describes itself. the 15-minute run is not claimed to pass; it is run and its gates are reported. | a pilot that never reaches the stated learning rate cannot answer whether the 15-minute budget works, so treating its failure as a stop would test the warmup schedule, not the adapter. gates, thresholds and flags are unchanged, so this row only records that the run goes ahead with its failure visible. recording versions is what would have prevented inferring the trainer version from a step count. |

the pilot ran 2 optimizer steps at an effective learning rate near 3e-5, because `--warmup-steps 50` exceeds the whole run, so the stated 1e-3 is never reached. its hypotheses were khmer and devanagari, `wrong_script_rate` 21.6%, val wer 143.5%, diacritic retention 0.295, so it failed the pre-registered gates and published nothing. the forced language token could not be confirmed from the returned sequences (`prompt_tokens_verified: false`), so the kwarg assertion was the only control that ran.

### 2026-11-14 (sat) — yoruba sensitivity, post-hoc (n=20 per variant)

Clip-level and descriptive; each clip is its own speaker, so the intervals are optimistic.

| variant | settings | wer [95% ci] | cer | insertions | hallucination | non-latin letters |
| --- | --- | --- | --- | --- | --- | --- |
| default | `chunk_length_s=30`, no repetition guard | 327.5% [217.7, 458.2] | 159.4% | 70% | 50% | 30.4% |
| guard | `no_repeat_ngram_size=4`, `max_new_tokens=256`, `chunk_length_s=30` | 127.1% [107.2, 150.6] | 75.4% | 24% | 10% | 20.6% |
| guard_nochunk | same guard, `chunk_length_s` removed | 127.1% [107.2, 150.6] | 75.4% | 24% | 10% | 20.6% |
