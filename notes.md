# notes

three findings that affect how the numbers should be read. they are recorded
here rather than in the report tables because they are caveats, not scores.

## val and test do not use the same decoder path

the pilot gate reads the trainer's own validation pass. `speechgap.train` calls
`trainer.evaluate()` with `predict_with_generate=True` and
`generation_max_length=225`, and the trainer does not chunk audio. the test
numbers come from `speechgap.eval`, which calls whisper through the asr pipeline
with `chunk_length_s=30`. the two paths also batch differently (trainer batch 4
during eval, eval batch 8 by default).

so the val gate and the test score are not a clean before/after. the 30-minute
adapter passed the val gate at 97.13% wer and scored 131.6% wer [110.8, 154.6]
on the 300-clip `yoruba_read` test set. part of that gap can be the decode
difference, not the adapter. the clean comparison is to run the val manifest
through `speechgap.eval` for both the zero-shot and adapter arms, which was
skipped for now. until that runs, treat the val-to-test gap as descriptive.

`val_predictions.jsonl` now lands in the adapter directory so the val hypotheses
can be inspected directly, but it is still the trainer decode, not the eval
decode.

## the same group name means different corpora

the committed fleurs baseline (`c46a73f`) labels its clips `yoruba_read` and
`english_read`. those clips are fleurs `yo_ng` and `en_us`, with ids
`yo_ng_0..99` and `en_us_0..99`.

the manifests use `yoruba_read` and `english_read` for different corpora:
naijavoices for yoruba and voxpopuli for english. the new fleurs cache uses the
group name `fleurs_yo`.

joining the baseline and the new caches on `group` would therefore mix fleurs
with naijavoices and voxpopuli. the baseline is matched to the new fleurs cache
by position and by case/punctuation-normalized reference text, never by group
name. the baseline has no committed `.meta.json`; its conditions are the
`ANALYSIS_PLAN.md` rows from 2026-11-14.

## the adapter's gain is mostly loop suppression

on the 300-clip naijavoices test set the adapter cut wer from 940.0% to 131.6%,
but most of that is fewer repetition loops, not better transcription.

| arm | n | median per-clip wer | share <=100% | wer excluding looped clips | loops excluded | loop rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adapter | 300 | 100.0% | 83.7% | 92.1% | 31 | 10.3% |
| zero-shot | 300 | 897.8% | 37.0% | 104.9% | 231 | 77.0% |

115 of the 300 adapter clips sit exactly at 100% wer. the same pattern shows on
fleurs: non-looped wer 95.6% for the adapter vs 108.4% for the baseline, while
loops fall from 64 to 26. on l2-arctic the adapter makes non-looped output
worse: 45.7% vs 8.7%.

loop exclusion is post-hoc and changes the denominator, so the non-looped wer
column is not comparable to the headline wer. the honest summary is that the
adapter suppresses loops and still does not recover yoruba orthography.
