superseded: written early on saturday before the experiments. the current state is in README.md and the site (https://langnostic.vercel.app). kept as a record of the plan.

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
