# langnostic — single-file research handoff

Generated 2026-11-14. **Upload this one file; no other attachments are needed.**
The research brief is first. The repository files it refers to are reproduced
verbatim in the appendices at the bottom.

---

## How to use this file

Give the entire file to Claude or Manus and say:

> Do the research described in the brief below. Return the deliverables in the
> requested format. The repository files are in the appendices.

---

## The research brief

```
===== BEGIN PROMPT =====

You are doing data-source research for a 48-hour hackathon project. Do not write the
application. Produce a verified data plan and exact loading code.

PROJECT
"langnostic" measures how whisper-small transcribes different languages and accents,
then tests whether a small LoRA adapter (15-30 min of audio, peft) narrows the gap and
what it costs on other groups. Python only. Runs on an Apple M2 (MPS). Submission
deadline: Sunday Nov 15, 2026, 11:59pm ET. Current date: Nov 14, 2026.

WHAT THE ANALYSIS PLAN REQUIRES (Appendix A, ANALYSIS_PLAN.md — read it first)
- Per-group WER/CER with 95% bootstrap intervals that resample SPEAKERS, not clips.
- Splits must be speaker-disjoint: train / validation / test.
- About 200 clips per group, counting speakers, not just clips.
- English as the reference, one more high-resource language, one or two lower-resource
  languages whisper supports, a "star language" the team speaks, and 2-3 English
  accent groups with enough speakers.
- A separate spontaneous-speech test set (own recordings) is out of scope for you.
- A clean test set touched once.

KNOWN FACTS TO VERIFY, NOT RE-DERIVE
1. FLEURS on Hugging Face exposes no speaker_id. Schema:
   https://datasets-server.huggingface.co/info?dataset=google/fleurs
   Configs include yo_ng, ha_ng, sw_ke, ig_ng, en_us.
   Confirm whether ANY official FLEURS release exposes speaker IDs (original TSV,
   metadata, filenames). If not, FLEURS cannot support speaker-clustered intervals or
   verifiable speaker-disjoint splits. Say so plainly.
2. Common Voice 17.0 on HF is empty; Mozilla moved datasets to Mozilla Data Collective
   in Oct 2025:
   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
   Verify the current access path, license, schema (does client_id still exist? accent?),
   and whether legacy HF versions remain usable. Do not assume.

QUESTIONS TO ANSWER
A. Candidate datasets for the language axis and the accent axis. For each:
   dataset ID, version/URL, config names, split names, clip counts, UNIQUE SPEAKER
   counts (not rows), hours, the exact speaker-ID field, the accent/label field,
   license, whether gated, and a working load command.
   Seed candidates: FLEURS, Common Voice (current + legacy), AfriSpeech / AfriSpeech-200,
   VoxPopuli, MLS, ALFFA, YODAS, and any better 2025-2026 option you find.
B. For Yoruba, Hausa, Swahili (and Igbo only if whisper supports it), which datasets
   can supply at least ~200 test clips from at least 10 distinct speakers, with a
   speaker ID? Report exact numbers. If 10 speakers is impossible, say the maximum
   available and what that does to the interval.
C. For English accents: which 2-3 accent labels have enough clips AND enough distinct
   speakers (report both), and where do they live now?
D. Verify whisper-small language support from the model/tokenizer itself (list of
   language codes), not from memory. Give exact codes for the recommended languages.
   Confirm whether forcing the language token is supported for each.
E. Recommend ONE primary data plan and ONE fallback that satisfy the plan above within
   48 hours, with a time/access estimate. If no option satisfies speaker-clustered
   intervals on both axes, propose the smallest honest deviation and the exact line
   to add to the deviations log. Do not force a bad fit.

DELIVERABLES
1. A dataset matrix (markdown table) with every field in A.
2. A recommended plan: exact dataset IDs/configs for each group, split names, and
   expected speaker counts.
3. Exact Python code using `datasets` that:
   - loads each chosen config,
   - saves 16 kHz WAVs under data/,
   - writes JSONL with exactly these fields:
     {"utt_id", "audio", "speaker", "group", "ref", "language"}
     where group is a free-form label like "yoruba_read" or "english_ng",
     speaker is the dataset's speaker/client ID, and ref is the transcription,
   - assigns speaker-disjoint train/validation/test.
4. Access checklist: accounts, terms, CLI commands, tokens, expected download size.
5. A "blockers and unknowns" section.
6. Citations with URLs and an access date of 2026-11-14.

RULES
- Never invent a field name, row count, or speaker count. Mark anything unverified.
- Prefer datasets loadable through HF `datasets` or a simple documented download.
- Public data and licenses only; no scraping personal audio.
- Do not propose speaker diarization/clustering as a fix for missing speaker IDs.
- Do not propose Rust, TypeScript, or a rewrite.
- Keep the recommendation inside the 48-hour scope.

END WITH a 10-line "handoff back" summary containing: chosen star language, exact
dataset IDs/configs, exact speaker-ID fields, total speakers per group, access steps,
and the top three risks.

===== END PROMPT =====
```

---

## Why this research is needed (verified today)

1. **FLEURS has no speaker ID field.** The HF schema is
   `id, num_samples, path, audio, transcription, raw_transcription, gender, lang_id,
   language, lang_group_id`. `gender` is not a speaker ID. The analysis plan's
   speaker-clustered bootstrap and speaker-disjoint splits cannot be built from FLEURS
   as exposed. Verify whether any official release changes this.
2. **Common Voice 17.0 on Hugging Face is empty.** The card states that since October
   2025 the datasets are exclusively available through Mozilla Data Collective. The
   accent axis in the plan depends on Common Voice, so its access path and schema must
   be re-verified.
3. **The star language is still undecided.** Candidates in the project doc: Yoruba,
   Hausa, Swahili; Igbo only if whisper supports it. The research should recommend one
   based on verified data availability.

---

## Current pipeline (for context)

```
manifest.jsonl  ->  speechgap/eval.py  ->  results/cache/<run>.jsonl
                    (whisper-small +/- merged LoRA adapter)

results/cache/<run>.jsonl  ->  speechgap/report.py  ->  per-group markdown/json
                               (normalize + metrics)

metrics.py: Levenshtein alignment, WER/CER, substitution/deletion/insertion mix,
speaker-clustered bootstrap, paired bootstrap, hallucination check. Standard library
only; 10 tests pass.
```

The manifest contract is the seam the researcher must satisfy:

```
{"utt_id": "...", "audio": "data/....wav", "speaker": "...",
 "group": "yoruba_read", "ref": "...", "language": "yoruba"}
```

`speaker` must be a real dataset speaker identifier. If it is absent or fabricated,
the intervals in `metrics.py` are invalid.

---

## What to return

1. The dataset matrix and the one recommended plan (plus fallback).
2. Exact config names and speaker-ID fields, not prose.
3. Runnable loader code that emits the JSONL schema above with speaker-disjoint splits.
4. Access steps (account, terms, CLI) and download sizes.
5. Whisper language codes for the chosen languages.
6. Blockers, plus the exact deviations-log line if the plan must change.

---

# Appendices: verbatim repository files

Everything below is copied exactly from the repository as of 2026-11-14.


---

## `ANALYSIS_PLAN.md`

````text
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

````

---

## `langnostic project doc.md`

````text
# langnostic: project doc

veritas ai international high school hackathon 2026 | track: ai + languages | solo | working doc for milestones, coordination and memory (not the submission write-up)

## 1. one-line summary

langnostic measures how well a speech recognizer (whisper) hears different languages and different accents, works out where the gaps come from, and tests whether a tiny lora adapter trained on 15 to 30 minutes of audio narrows them, and what that costs on everything else. it is shown through a small web app: a results explorer plus a live comparison of baseline and adapted transcripts.

main axis: language. second axis: accent. the study is the engine and the web app is the front door.

## 2. status and milestones

status as of sat nov 14: track and project chosen, nothing built yet. the star language, the name and the accents are still open (section 3). times below are targets in et, so adjust them to the real clock. submissions close sun nov 15, 11:59 pm et (mon 5:59 am west africa time).

- [ ] reply 'got it' to the kickoff email from daniel marion (the veritas ai team asked for it)
- [ ] decide the star language, the accents and the name (section 3)
- [ ] first-hour checks from section 9: whisper supports the language, fleurs loads, common voice access, gpu works. if data will not load in 30 minutes, switch
- [ ] write the hypotheses and analysis plan into the repo before running any experiment
- [ ] sat: data loading, the evaluation harness and per-language normalization rules written down
- [ ] sat evening: zero-shot baselines per language and accent (wer, cer, intervals) and the 5 worst examples side by side
- [ ] sat evening: record about 40 spontaneous clips (stretch: 100) and start the native-speaker transcripts
- [ ] sat night: first working gradio space with a results tab and a try-it tab, then sleep at least 7 hours
- [ ] sun morning: lora on the star language with a validation split, then the forgetting check on one or two other groups
- [ ] sun midday: read vs spontaneous comparison, and label about 30 errors as harmless, meaning-changing or unintelligible
- [ ] sun 6:00 pm et: freeze experiments (the guide says to stop adding features about 6 hours before the deadline)
- [ ] sun 6:00 to 9:00 pm et: write-up, 2-minute video, notebook with cached results, links checked in a private window
- [ ] sun by 10:00 pm et: first full submission, then resubmit as needed before 11:59 pm et

## 3. decisions log

| decision | current choice | status |
| --- | --- | --- |
| track | ai + languages | confirmed |
| project | whisper study across languages and accents | confirmed |
| main axis | language, with accent kept as a full second axis | confirmed |
| scope | trimmed to the core in section 5 | proposed |
| presentation | gradio space with tabs; a custom typescript site only if everything else is done early | proposed (the user wants a website as the main surface) |
| star language | not chosen | open, blocks the data |
| name | langnostic (recommended) or lingnostic | open |
| accents | 2 or 3 english accents plus our own voice | open |
| second high-resource language | not chosen | open |

## 4. context and constraints

- event: veritas ai international high school hackathon 2026. hacking started sat nov 14, 12:00 am et. submissions close sun nov 15, 11:59 pm et. registration closed nov 13, and the kickoff email arrived, so we are registered.
- one person, working solo. can build anything, so scope is limited by the 48 hours, not by skill. wants the project very technical, research and engineering based, with a website as the main surface.
- the organizers give no technical help. they only help with submission problems (contact@veritasai.com).
- scoring (from the hackathon site): round 1 is blind, by the veritas team, 100 points: technical execution 30, originality 20, impact and track fit 20, testing and honesty 15, communication 15. finals are a live demo, 100 points: demo and clarity 30, technical depth 30, impact 20, q&a 20. final rank is finals 70% plus round 1 30%. the briefs pdf lists four criteria (technical quality, impact, creativity, clarity); plan to the site's table.
- prizes: the gold team gets an internship at arecibo ai (a health ai startup), scholarships go to veritas ai programs, there are category awards (best beginner build, best ai for good, judges' choice) and merit awards, and every submitter gets a certificate. no track is easier to win.
- the track challenge: help people understand each other, learn a language or be understood, especially in a language technology serves poorly. our project is the suggested speech-recognition-across-accents project plus our own additions.
- the sample submission (a scam text filter) is small and tidy: one dataset, two models, 40 hand-written test messages, one clear finding, an honest list of what it still misses, and a notebook that runs top to bottom in under a minute. our bar is the same honesty and clarity, not more features.
- the guide's advice: a simple project that works beats an ambitious one that doesn't, and a basic version should work end to end by saturday night.
- handoff: when starting a new chat or handing this to another assistant, share this doc and update the checklist and decisions log first.

## 5. scope: what we build and what we cut

the base plan (whisper, per-group error rates, lora) is what every team gets from the brief, so what we add beyond it is the spontaneous test set, the read vs spontaneous check and the error-cost labels. the guide says a simple project that works beats an ambitious one that doesn't, so the core is kept small.

core, in order:

1. write the hypotheses and analysis plan into the repo before running experiments (cheap, do first).
2. zero-shot baselines per language and accent, with intervals and the 5 worst examples side by side.
3. our own spontaneous-speech test set: about 40 clips minimum (stretch: 100), in the style of phone voice notes in the star language, in our own voice or people who agree to be recorded, on neutral topics and never anyone's real messages. a native speaker checks every transcript. this is the slow part, so record first and transcribe while the baselines run.
4. read vs spontaneous check: compare error rates on fleurs-style read speech with our spontaneous clips, and test whether a lora adapter trained on read speech helps spontaneous speech.
5. lora on the star language, then the forgetting check.
6. error cost: a native speaker labels about 30 errors as harmless, meaning-changing or unintelligible. if the language has tone marks or diacritics, also report error rates with and without them.
7. a gradio space (results explorer plus a live baseline-vs-adapted comparison) and a notebook that runs from cached results.

the named user: someone who receives long voice notes in the star language and cannot play audio (hard of hearing, or in a meeting). the moment: a two-minute voice note arrives and they need the text now.

cut or stretch: the tiny and base size ablation, the mixed-effects model, extra seeds, the reusable command-line tool, a custom typescript site, and a text-side fix (word-list biasing) compared with lora. these happen only if the core is done and checked.

## 6. research questions and hypotheses

- q1, language gap: how large is the error gap between a high-resource language and lower-resource ones, with intervals?
- q2, accent gap: within english, how large is the gap between accents?
- q3, language vs accent: is the accent gap small or large next to the language gap? one plot answers it.
- q4, lora: how much of the gap does a small adapter close with 15 and 30 minutes of audio (5 if time)?
- q5, cost: what does the adapter do to other languages and accents?
- q6, benchmark validity: does read-speech error predict spontaneous-speech error?
- q7, error anatomy: substitutions vs deletions vs insertions, and how often does the model hallucinate?
- what predicts the language gap (training-data share, script, tokenizer fertility) is descriptive only, since 3 or 4 languages cannot support a regression.

hypotheses, each falsifiable:

- h1: error rises as the language's share of whisper training data falls.
- h2: the accent gap exists but is smaller than the gap to a low-resource language.
- h3: a small lora adapter closes a large share of the gap for the star language.
- h4: that gain costs something on at least one other group.
- h5: for low-resource languages, much of the error is insertions and hallucination.
- h6: spontaneous speech has higher error than read speech, and an adapter trained on read speech helps it less.

if a hypothesis fails, we report that. judges score honest, modest results above unexplained high ones.

## 7. design

- language axis: english as the reference, one more high-resource language, one or two lower-resource languages whisper supports (candidates: swahili, yoruba, hausa), and the star language. fleurs is the source, since it is read speech of the same sentences across languages, as far as i know. verify.
- accent axis: 2 or 3 english accent labels from common voice with enough clips and enough different speakers (check the counts), plus our own voice reading the same sentences. optional: a dialect slice in the star language.
- data: about 200 clips per group, counting speakers and not just clips. speaker-disjoint splits. a validation split for lora early stopping and model choice, and a test set touched once.
- models: whisper-small as the main model, with forced vs auto-detected language. whisper-base only as a fallback if memory or time runs out.
- metrics: wer and cer, because wer is not comparable across languages. the text normalization for each language is fixed and written down. substitution, deletion and insertion split, hallucination rate (repeated loops, very long outputs), 95% bootstrap intervals resampled by speaker, and a paired bootstrap for before and after.
- error analysis: most common substituted words per group, the highest-error clips listened to, and real examples shown with the model output next to the correct text.
- lora (hugging face peft): rank 8 on the attention layers of both encoder and decoder as the starting default, with encoder-only vs decoder-only if time allows. training data of 15 and 30 minutes. early stopping on the validation split. turning the adapter off gives the exact baseline back.
- forgetting: after training for one group, re-evaluate on every other language and accent and report the change with intervals.
- comparing axes: the language gap and the accent gap on one plot with intervals.

## 8. engineering plan

- compute: the m2 (torch with mps, plus ffmpeg) for zero-shot eval; a rented gpu or colab only if lora training is too slow locally. small batches, and save adapters and raw transcripts after every run.
- code layout: data loading, a config-driven evaluation harness (fixed seeds and normalization, and it saves every raw transcript so any number can be recomputed), lora training, an analysis notebook, the gradio app and a figures folder.
- cached results: the repo ships precomputed transcripts and metrics as json, and the notebook loads them by default so 'run all' finishes fast (the sample submission runs in under a minute). the heavy whisper and lora runs sit behind a flag.
- web app: a gradio space on hugging face (free). tabs: results explorer (from the json), try it (record or upload a short clip and see the baseline and adapted transcripts side by side, with the adapter merged into the weights so the endpoint stays simple), and method and limits. keep 5 to 10 precomputed example clips as a fallback if the space is slow or asleep, and label what is live and what is precomputed. audio is processed in memory and never stored or logged.
- notebook outline, modeled on the sample: pitch and how to run, load data, look at the data, normalization rules, zero-shot baselines, where it fails, our addition (spontaneous clips and the read vs spontaneous check), the improvement (lora and forgetting), demo, results table, limitations and sources.

## 9. verify first (first hour)

- does whisper support the star language? as far as i know it covers yoruba, hausa and swahili but not igbo. verify, because an unsupported language has no language token to force.
- which target languages exist in fleurs, and how the data loads.
- common voice: gated or not, and which accents have enough clips and speakers.
- local setup works on the m2 (torch with mps, ffmpeg), and a hugging face account exists for the space.
- text normalization decisions per language (diacritics, numerals, punctuation).
- sharing works: open every link in a private window.
- the 30-minute rule: if a dataset will not load in about 30 minutes, switch.

## 10. risks and mitigations

- speaker leakage inflates results: split by speaker, never by clip.
- common voice accent labels are self-reported and noisy: say so, and use only labels with enough support.
- whisper may have seen similar audio in training: cannot be ruled out, so state it as a limitation.
- our own recordings are mostly one speaker: do not generalize from them alone, and test on other speakers too.
- wer is not comparable across languages: report cer too and fix the normalization.
- lora overfits with very little data: validation split, early stopping and a learning curve.
- wer above 100% from insertions on low-resource languages: report insertions separately and check for hallucination loops.
- picking a model with the test set (the sample admits doing this): keep a clean test set and touch it once.
- local runs can be killed or interrupted: checkpoint often, and keep code plus cached transcripts on github (never audio).
- scope creep: freeze experiments at 6:00 pm et on sunday.
- the gap may not close: report it honestly.

## 11. rules

- build during the weekend only, with nothing extended from earlier work.
- ai must be central: whisper plus lora training and evaluation. a regular website built with ai coding tools does not count on its own.
- public datasets and our own voice only. nobody else is recorded without permission.
- cite every dataset, pretrained model, tutorial and ai tool with links, including the ai assistants used. be able to explain every part.
- report results as they came out, and all numbers come from test data the model never saw in training.

## 12. judging alignment and finals prep

- round 1 (blind, the veritas team): technical execution 30 (the study, the harness, a notebook that runs), originality 20 (two axes, spontaneous clips, the validity check), impact and track fit 20 (a named user, a language technology serves poorly), testing and honesty 15 (baselines, held-out speakers, intervals, failure examples), communication 15 (write-up and video).
- finals (live, 5 minutes plus q&a, 70% of the final score): demo and clarity 30, technical depth 30, impact 20, q&a 20. run the demo live if it is reliable and keep a recorded backup open. use the same structure as the video, with more time on the demo and results.
- q&a prep: how do you know it works? what would cause it to fail? who would use this? how did you rule out speaker leakage? what do the intervals mean? what does lora change and what does it cost? what would you distrust in your own results? be able to explain lora, the bootstrap and the limits in plain words.
- the panel so far includes a computational rock physics researcher and two microsoft researchers, with more to be announced. no tailoring to any one judge: rigor and honest uncertainty are what a methods-minded judge probes.

## 13. submission checklist

- code: a github repo viewable by anyone with the link, with a notebook whose outputs are visible. one person submits for the team. the study has to stand on its own here, since the form has no website field.
- write-up, about one page: problem, what we built, what we added, how it works, results (baseline vs model on held-out data, and where it fails), what we would do next, sources. link the web app here.
- demo video, up to 2 minutes, optional: problem about 15 seconds, demo about 60, how it works about 30, next steps about 15. a screen recording of the web app with a voice-over is enough.
- form fields (from the sample): title, solo or team, track, starting point, one-sentence pitch, code link, write-up link, video link, what we added, datasets/models/ai tools used, new to ai.
- credits: every dataset, model, tutorial and ai tool, with links.
- check every link in a private window, submit a first version a few hours early, then resubmit as it improves.

## 14. story (honest, short)

- the opening line of the demo and write-up is one real moment of a person not being understood by a speech tool. it has to be true: write it after the results, from our real experience, and do not invent a persona.
- emotion goes in the framing. the numbers stay plain, with no overclaiming.
- close with one concrete next step.

## 15. glossary

- wer / cer: word / character error rate versus the reference text.
- lora: low-rank adaptation. freezes the model and trains small added matrices.
- peft: the hugging face library that implements lora.
- fleurs: multilingual read-speech dataset built on the same sentences across languages.
- common voice: crowd-sourced speech dataset with some accent labels.
- bootstrap interval: a confidence interval from resampling the test data, here by speaker.
- spontaneous clips: our own phone-voice-note-style test set.

````

---

## `README.md`

````text
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

````

---

## `speechgap/normalize.py`

````text
"""Text normalization. Fix these settings per language BEFORE running experiments and
write them in ANALYSIS_PLAN.md. Report results with and without diacritics when the
language uses them to carry meaning (for example tone marks)."""
import re
import unicodedata


def normalize(text, lowercase=True, strip_punct=True, strip_diacritics=False):
    text = unicodedata.normalize("NFC", text)
    if lowercase:
        text = text.lower()
    if strip_diacritics:
        decomposed = unicodedata.normalize("NFD", text)
        text = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
        text = unicodedata.normalize("NFC", text)
    if strip_punct:
        text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()

````

---

## `speechgap/metrics.py`

````text
"""Error rates with speaker-clustered bootstrap intervals. Pure python, no dependencies.

Why cluster by speaker: clips from one speaker are not independent, so resampling clips
makes the intervals too narrow. We resample whole speakers instead.
"""
import random
import re
from collections import Counter, defaultdict


def _tokens(text, unit):
    if unit == "word":
        return text.split()
    if unit == "char":
        return list(re.sub(r"\s+", "", text))  # character error rate ignores whitespace
    raise ValueError("unit must be 'word' or 'char'")


def _align_counts(ref, hyp):
    """Levenshtein alignment. Returns (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j, s, d, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                s += 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def utt_counts(ref, hyp, unit="word"):
    r, h = _tokens(ref, unit), _tokens(hyp, unit)
    s, d, i = _align_counts(r, h)
    return {"s": s, "d": d, "i": i, "n": len(r)}


def rate(counts):
    n = sum(c["n"] for c in counts)
    if n == 0:
        return float("nan")
    return sum(c["s"] + c["d"] + c["i"] for c in counts) / n


def breakdown(counts):
    """Share of all errors that are substitutions, deletions and insertions."""
    s, d, i = (sum(c[k] for c in counts) for k in ("s", "d", "i"))
    total = s + d + i
    if total == 0:
        return {"sub": 0.0, "del": 0.0, "ins": 0.0}
    return {"sub": s / total, "del": d / total, "ins": i / total}


def _by_speaker(speakers):
    idx = defaultdict(list)
    for k, sp in enumerate(speakers):
        idx[sp].append(k)
    return list(idx.values())


def _interval(stats, alpha):
    stats = sorted(x for x in stats if x == x)
    lo = stats[int((alpha / 2) * len(stats))]
    hi = stats[min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))]
    return lo, hi


def bootstrap_rate(counts, speakers, n_boot=2000, alpha=0.05, seed=0):
    """Point estimate and (1 - alpha) interval for the corpus error rate."""
    groups = _by_speaker(speakers)
    point = rate(counts)
    if len(groups) < 2:  # one speaker: no valid interval, report the point only
        return {"rate": point, "lo": point, "hi": point, "n_speakers": len(groups)}
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        picked = [counts[k] for g in (rng.choice(groups) for _ in groups) for k in g]
        stats.append(rate(picked))
    lo, hi = _interval(stats, alpha)
    return {"rate": point, "lo": lo, "hi": hi, "n_speakers": len(groups)}


def paired_bootstrap_diff(counts_a, counts_b, speakers, n_boot=2000, alpha=0.05, seed=0):
    """rate(a) - rate(b) on the same utterances, resampling speakers. Lists must align."""
    assert len(counts_a) == len(counts_b) == len(speakers)
    groups = _by_speaker(speakers)
    point = rate(counts_a) - rate(counts_b)
    if len(groups) < 2:
        return {"diff": point, "lo": point, "hi": point, "n_speakers": len(groups)}
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        ks = [k for g in (rng.choice(groups) for _ in groups) for k in g]
        stats.append(rate([counts_a[k] for k in ks]) - rate([counts_b[k] for k in ks]))
    lo, hi = _interval(stats, alpha)
    return {"diff": point, "lo": lo, "hi": hi, "n_speakers": len(groups)}


def looks_like_loop(text, n=3, min_repeats=4):
    """Crude hallucination check: the same n-gram repeated many times."""
    toks = text.split()
    if len(toks) < n * min_repeats:
        return False
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return max(grams.values()) >= min_repeats


def is_hallucination(ref, hyp, ratio=2.0):
    """Flag outputs much longer than the reference, or stuck in a repeated loop."""
    n_hyp = len(hyp.split())
    too_long = n_hyp > 5 and n_hyp > ratio * max(len(ref.split()), 1)
    return too_long or looks_like_loop(hyp)

````

---

## `speechgap/report.py`

````text
"""Score cached transcripts per group.

Input: a jsonl file, one record per clip:
  {"utt_id": "...", "speaker": "...", "group": "yoruba_read", "ref": "...", "hyp": "..."}
Output: per-group wer / cer with speaker-bootstrap intervals, error mix, hallucination rate.

Usage:
  python -m speechgap.report results/cache/zero_shot.jsonl --out results/zero_shot_report.json
"""
import argparse
import json

from .metrics import bootstrap_rate, breakdown, is_hallucination, utt_counts
from .normalize import normalize


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(records, strip_diacritics=False, n_boot=2000, seed=0):
    by_group = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r)
    out = {}
    for group, rows in sorted(by_group.items()):
        refs = [normalize(r["ref"], strip_diacritics=strip_diacritics) for r in rows]
        hyps = [normalize(r["hyp"], strip_diacritics=strip_diacritics) for r in rows]
        speakers = [r["speaker"] for r in rows]
        wc = [utt_counts(a, b, "word") for a, b in zip(refs, hyps)]
        cc = [utt_counts(a, b, "char") for a, b in zip(refs, hyps)]
        out[group] = {
            "n_clips": len(rows),
            "wer": bootstrap_rate(wc, speakers, n_boot=n_boot, seed=seed),
            "cer": bootstrap_rate(cc, speakers, n_boot=n_boot, seed=seed),
            "wer_error_mix": breakdown(wc),
            "hallucination_rate": sum(is_hallucination(a, b) for a, b in zip(refs, hyps)) / len(rows),
        }
    return out


def to_markdown(report):
    lines = ["| group | clips | speakers | wer [95% ci] | cer [95% ci] | insertions share | halluc. |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for g, m in report.items():
        w, c = m["wer"], m["cer"]
        lines.append(
            f"| {g} | {m['n_clips']} | {w['n_speakers']} | "
            f"{w['rate']:.1%} [{w['lo']:.1%}, {w['hi']:.1%}] | "
            f"{c['rate']:.1%} [{c['lo']:.1%}, {c['hi']:.1%}] | "
            f"{m['wer_error_mix']['ins']:.0%} | {m['hallucination_rate']:.0%} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--strip-diacritics", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    rep = score(load_jsonl(args.path), strip_diacritics=args.strip_diacritics)
    print(to_markdown(rep))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)

````

---

## `speechgap/eval.py`

````text
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

````

---

## `tests/test_metrics.py`

````text
import math

from speechgap.metrics import (bootstrap_rate, breakdown, is_hallucination,
                               paired_bootstrap_diff, rate, utt_counts)
from speechgap.normalize import normalize
from speechgap.report import score


def test_wer_known_case():
    c = utt_counts("a b c d", "a x c", "word")   # b->x substitution, d deleted
    assert (c["s"], c["d"], c["i"], c["n"]) == (1, 1, 0, 4)
    assert rate([c]) == 0.5


def test_insertions_counted():
    c = utt_counts("a b", "a b b b", "word")
    assert (c["s"], c["d"], c["i"]) == (0, 0, 2)
    assert breakdown([c])["ins"] == 1.0


def test_cer_ignores_whitespace():
    assert rate([utt_counts("ab cd", "abcd", "char")]) == 0.0
    assert rate([utt_counts("abcd", "abxd", "char")]) == 0.25


def test_perfect_and_empty():
    assert rate([utt_counts("same words", "same words", "word")]) == 0.0
    assert math.isnan(rate([utt_counts("", "", "word")]))


def test_normalize():
    assert normalize("Hello,  World!") == "hello world"
    assert normalize("Café", strip_diacritics=True) == "cafe"
    assert normalize("Café") == "café"


def test_bootstrap_interval_contains_point_and_is_deterministic():
    counts, speakers = [], []
    for sp in range(10):
        for k in range(5):
            counts.append({"s": 1 if (sp + k) % 4 == 0 else 0, "d": 0, "i": 0, "n": 5})
            speakers.append(f"s{sp}")
    a = bootstrap_rate(counts, speakers, n_boot=500, seed=1)
    b = bootstrap_rate(counts, speakers, n_boot=500, seed=1)
    assert a == b
    assert a["lo"] <= a["rate"] <= a["hi"]
    assert a["n_speakers"] == 10


def test_single_speaker_has_no_interval():
    counts = [{"s": 1, "d": 0, "i": 0, "n": 4}] * 3
    r = bootstrap_rate(counts, ["only"] * 3)
    assert r["lo"] == r["hi"] == r["rate"] and r["n_speakers"] == 1


def test_paired_diff_zero_for_identical_systems():
    counts = [{"s": 1, "d": 0, "i": 0, "n": 4}, {"s": 0, "d": 0, "i": 0, "n": 4}] * 4
    speakers = ["a", "a", "b", "b", "c", "c", "d", "d"]
    d = paired_bootstrap_diff(counts, counts, speakers, n_boot=200)
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_hallucination_flags():
    assert is_hallucination("short ref", "no no no no no no no no no no no no no no no")
    assert not is_hallucination("a normal sentence here", "a normal sentence here")


def test_report_scores_groups():
    recs = [
        {"utt_id": str(i), "speaker": f"s{i % 3}", "group": g, "ref": "one two three", "hyp": h}
        for i, (g, h) in enumerate([("good", "one two three")] * 6 + [("bad", "one two")] * 6)
    ]
    rep = score(recs, n_boot=100)
    assert rep["good"]["wer"]["rate"] == 0.0
    assert abs(rep["bad"]["wer"]["rate"] - 1 / 3) < 1e-9


if __name__ == "__main__":  # run without pytest: python tests/test_metrics.py
    import sys
    sys.path.insert(0, ".")
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)

````

---

## `tests/fixtures/tiny_report.jsonl`

````text
{"utt_id": "en1", "speaker": "spk_a", "group": "english_read", "ref": "the quick brown fox", "hyp": "the quick brown fox"}
{"utt_id": "en2", "speaker": "spk_a", "group": "english_read", "ref": "jumps over the lazy dog", "hyp": "jumps over a lazy dog"}
{"utt_id": "en3", "speaker": "spk_b", "group": "english_read", "ref": "hello world", "hyp": "hello world"}
{"utt_id": "yo1", "speaker": "spk_c", "group": "yoruba_read", "ref": "bawo ni o se wa", "hyp": "bawo ni se wa"}
{"utt_id": "yo2", "speaker": "spk_c", "group": "yoruba_read", "ref": "mo n ka iwe", "hyp": "mo n ka iwe"}
{"utt_id": "yo3", "speaker": "spk_d", "group": "yoruba_read", "ref": "ese pupa", "hyp": "ese pupa pupa"}

````

---

## `requirements.txt`

````text
# apple silicon: install torch (pip install torch) and ffmpeg (brew install ffmpeg) first,
# then: pip install -r requirements.txt
transformers
datasets
accelerate
peft
jiwer
soundfile
librosa
pandas
matplotlib
gradio
huggingface_hub
pytest

````
