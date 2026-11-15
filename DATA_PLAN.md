superseded: written early on saturday before the experiments. the current state is in README.md and the site (https://langnostic.vercel.app). kept as a record of the plan.

# langnostic data plan — verified findings and decision

**Research cut-off:** 2026-11-14. **Project window:** 48 hours. This report synthesizes the supplied verified findings and the project brief/Appendix A. It does not infer fields, speaker counts, audio availability, or license permissions beyond those findings. “Rows/clips” below means the source’s reported utterance, clip, or asset rows; it does not mean people unless explicitly labeled speakers.

## Executive decision

**No fully verified plan satisfies all of the draft analysis requirements.** The strongest prospective star-language source is NaijaVoices: it includes Yoruba, Hausa, and Igbo audio with anonymized `speaker_id`, but access is gated, no per-language official test counts are verified, and the reported 20 unseen speakers are for the combined NV Test—not for each language. It is therefore not yet defensible to promise a 200-clip/10-speaker test group in any selected language. The strongest verified current Common Voice accent source is MDC Common Voice Scripted Speech 27.0 English, but it requires MDC registration/API access and a very large 88.43-GB archive; the accent counts are published release-wide, not for a speaker-disjoint test partition. No load or split audit was executed.

**Provisional primary path:** pending confirmation that a team member actually speaks Yoruba, request NaijaVoices access and audit its official partition CSVs before selecting a star language. Yoruba is a *data-first provisional candidate*, not a claim about the team’s language. Pair it with current Common Voice English accents only if access/download and a `client_id`-based re-split can be completed and audited. If the team does not speak Yoruba, select the star language only after the team confirms it; the research cannot decide that fact. Do not start model evaluation or LoRA until the partition audit resolves the unknowns below.

**Narrower fallback:** use VoxPopuli’s pinned English/German test resources plus its English L2 accent test resources, reassigning every included row by actual `speaker_id` before any training/evaluation and removing cross-config speaker collisions. This supports English/German and several *L2 English accent groups*, not Yoruba/Hausa/Swahili/Igbo-language evaluation. Released VoxPopuli default split IDs demonstrably overlap, so the packaged split names alone are not a valid speaker-disjoint plan. This fallback is a reduced-scope study, not a substitute for a star-language result.

**Speaker-clustered intervals:** possible only for groups for which the actual audio rows expose a reliable speaker/client ID and the final split audit leaves at least two independent speakers (and preferably the intended 10+). It is not possible from current FLEURS rows; not established for ALFFA Swahili manifest IDs; not possible for YODAS; and not established for a NaijaVoices test group before approval and reading its split files. Common Voice, AfriSpeech, VoxPopuli, MLS, and NaijaVoices expose or document speaker grouping IDs, but their access, split, or group-size caveats remain dataset-specific. Never fabricate a speaker key or bootstrap clips as if independent speakers.

## 1. Comparison matrix

The brief requests ID/version/URL, configs and splits, clip counts, unique-speaker counts, hours, exact speaker field, accent field, license/gating, and a working load command. “Not verified” is intentionally retained. Commands marked documented/card-verified are not represented as an end-to-end audio smoke test unless explicitly said so. Sizes are included in the access section.

| Candidate / role | ID, version and URL | Configs and split names; clip/row counts | Unique-speaker counts and hours | Actual speaker-ID field | Accent/label field | License and gating | Load route / status |
|---|---|---|---|---|---|---|---|
| **FLEURS** — multilingual read speech | `google/fleurs`; current Hub revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`; [card](https://huggingface.co/datasets/google/fleurs) | `en_us`, `yo_ng`, `ha_ng`, `sw_ke`, `ig_ng`; each `train/validation/test`. Rows (train/val/test): en_us 2,602/394/647; yo_ng 2,339/378/831; ha_ng 3,259/296/621; sw_ke 3,070/211/487; ig_ng 2,839/413/969. 102 language configs plus `all`; all-config rows 271,798/34,452/77,810. | Current unique-speaker counts unavailable. Approx. 12 h/language in paper; card says about 10 h of training supervision. Exact per-config/split hours not verified. | **None in current schema.** A pre-removal official `google/xtreme_s` loader did expose `speaker_id`; it was removed in v2.0.0. Historical coercion can create collisions or `-1`; no current legacy end-to-end load verified. | **None.** `language`/`lang_id` and locale config are not accent labels; `gender` is not identity or accent. | CC BY 4.0; current Hub public, ungated. | `load_dataset("google/fleurs", "yo_ng", split="test")` is current-card documented/public. Loads clips but cannot meet speaker-cluster requirement. Legacy documented pattern `google/xtreme_s`, `fleurs.<code>` is not a verified current working speaker-ID route. |
| **Common Voice Scripted Speech 27 English** — accent candidate | MDC ID `cmu5jplf300nwmh07iqvk9leo`, release 27.0, 2026-09-17; locale `en`; [MDC page](https://mozilladatacollective.com/datasets/cmu5jplf300nwmh07iqvk9leo) | Single archive/locale (not HF-style configs). v27 English: 2,591,456 clip rows; validated 1,929,854, invalidated 314,596, other 347,006; supplied train/dev/test 1,147,829/16,403/16,403. | 100,546 unique speakers/users and 3,795.9 total / 2,826.8 validated hours (published). No published train/dev/test speaker counts or proof that those splits are disjoint. Per-label counts are release-wide, not held-out test counts. | `client_id` (hashed UUID). | **`accents`** (plural, self-reported when supplied). It is a release-specific v27 taxonomy; not a config. Codes and published clips/speakers: `us` 574,821/11,023; `england` 204,881/3,466; `indian` 153,048/3,045; `canada` 101,734/1,225; `australia` 69,393/953; `scotland` 69,286/272; `african` 60,591/452; `newzealand` 20,768/230; `ireland` 11,204/266; `philippines` 7,503/210; `hongkong` 7,004/190; `singapore` 4,722/112; `malaysia` 4,260/158; `wales` 3,037/120; `bermuda` 1,236/75; `southatlandtic` 332/9 (exact unusual spelling); `other` 201,005/1,706. Labels can be multiple per speaker; do not sum their speakers. Empty-accent/null versus `other` equivalence is unverified. | Page says CC0-1.0; dataset terms prohibit identifying speakers or re-hosting/re-sharing. MDC account, terms acceptance and API key required; free tier currently limited to 7 downloads/day. | Documented SDK route after credentials: `pip install datacollective`, set `MDC_API_KEY`, then `download_dataset("cmu5jplf300nwmh07iqvk9leo")`. **Not executed**. This is not a verified `datasets.load_dataset` route. |
| **Common Voice 17.0** — historical accent option | Official HF ID `mozilla-foundation/common_voice_17_0` is empty. Official older snapshot via Mozilla request; unofficial mirror `fsicoli/common_voice_17_0`. [Mozilla policy](https://community.mozilladatacollective.com/were-changing-access-to-older-versions-of-common-voice-datasets/) | No usable official HF split. Historical v17 English published totals: 2,413,408 clips; validated 1,799,288, invalidated 292,773, other 321,347; train/dev/test 1,101,170/16,393/16,393. | 92,325 unique users/speakers, 3,507.21 total and 2,614.76 validated hours. No verified v17 accent breakdown. | Current Common Voice field concept is `client_id`; no v17 archive field/count audit here. | Historical taxonomy/counts not verified for v17. Do not substitute v27 figures. | Mozilla older release requires verified email/request and research-only/no-reshare terms. Empty official HF repo is public metadata, not downloadable data. Unofficial mirror is not an approved substitute. | No working official v17 command until Mozilla approval. Mirror advertises a `load_dataset` command, but HF Dataset Server returns 501 for its script repo; Datasets 4 removed script loading. Not verified. |
| **AfriSpeech-200** — English African-accent groups, not African-language ASR | `intronhealth/afrispeech-200`, pinned HF revision `b538c6e111914a812af28ff677f8cffc9b404b7d`; [card](https://huggingface.co/datasets/intronhealth/afrispeech-200), [official repo](https://github.com/intron-innovation/AfriSpeech-200) | Clean official manifests: 58,000 train, 3,231 dev, 6,346 test rows. Test label rows/speakers/hours from pinned CSV: Yoruba 650/172/1.7952; Hausa 196/53/0.6365; Swahili 523/61/1.3371; Igbo 357/94/0.9137. These are **English speech** by those accent labels. Overall card: 120 accents, 13 countries. | Total 2,463 speakers / 200.91 h (paper). Test has 750 distinct `user_ids`; per-label test counts above. Hausa has 53 speakers but only 196 clips, four below 200. Test overall 18.77 h. Train/dev/test are reported as speaker-split, but audio access was not tested. | CSV uses **`user_ids`**; HF API metadata says `user_id`; card says `speaker_id`. Pseudonymous. Check actual loaded key. | **`accent`**; Yoruba/Hausa/Swahili/Igbo are English-accent labels, not utterance language. | CC BY-NC-SA 4.0; HF manually gated; GitHub clean manifests public but do not provide audio. S3 credentials requested from Intron. | Card documents `load_dataset('tobiolatunji/afrispeech-200','all')`, but actual audio load requires HF approval and was not verified. Public manifest can be downloaded by pinned raw URL; audio requires S3 credentials. |
| **VoxPopuli** — European language and L2 English-accent fallback | `facebook/voxpopuli`, pinned revision `42f01879c780b4a2e90ec0b4f616c2ece526e4f1`; [card](https://huggingface.co/datasets/facebook/voxpopuli) | Configs `cs,de,en,en_accented,es,et,fi,fr,hr,hu,it,lt,multilang,nl,pl,ro,sk,sl`; splits train/dev/test except `en_accented` test-only. Exact current rows for selected configs: en 182,482/1,753/1,842; de 108,473/2,109/1,968; `en_accented` test 8,387. Accent test label rows/speaker IDs/card hours: en_nl 1,151/45/3.52; en_de 1,074/84/3.52; en_cs 974/26/3.30; en_pl 852/33/3.23; en_fr 737/27/2.56; en_hu 676/23/2.33; en_fi 590/20/2.18; en_ro 485/27/1.85; en_es 419/18/1.42; en_sk 409/17/1.46; en_it 360/15/1.11; en_et 289/6/1.08; en_lt 178/7/0.65; en_hr 127/9/0.42; en_sl 66/7/0.25. Eleven labels reach 200 clips and 10 IDs. `multilang` overlaps language configs; do not sum. | Exact speaker-ID unions in current valid rows: English train/dev/test 639/71/286 (pair overlaps 38/109/17; union 841); German 250/67/123 (overlaps 58/72/20; union 309); French 243/54/144 (overlaps 36/77/17; union 327). Paper totals differ: English 1,313, German 531, French 534; discrepancy unexplained. Accent numbers above are distinct IDs in the accent manifest. Language-level source hours/speakers are published but not per split. | **`speaker_id`** (string). Actual ID; same ID overlaps released split labels and English accent/test versus standard English train/dev. | **`accent`**, L2 code groups, not native regional English accents; label-creation method not documented. | Data CC0-1.0 per card/repo; public, ungated. Check European Parliament item-specific terms; repo code/model license CC BY-NC 4.0 is not the data license. | Card/API-documented: `load_dataset("facebook/voxpopuli","en",split="test",revision="42f01879c780b4a2e90ec0b4f616c2ece526e4f1")`; analogous `de` and `en_accented`. Commands were not run end-to-end here. |
| **Multilingual LibriSpeech (MLS)** — read-speech comparison | Official OpenSLR SLR94; HF `facebook/multilingual_librispeech`; [OpenSLR](https://www.openslr.org/94/), [HF card](https://huggingface.co/datasets/facebook/multilingual_librispeech) | Current HF configs only `dutch,french,german,italian,polish,portuguese,spanish` (not English); splits train/dev/test plus limited `1_hours`,`9_hours`. German rows 469,942/3,469/3,394; Dutch 374,287/3,095/3,075; French 258,213/2,416/2,426; Spanish 220,701/2,408/2,385; Italian 59,623/1,248/1,262; Portuguese 37,533/826/871; Polish 25,043/512/520. | Published paper split speaker counts/hours (not row-derived): German 176/30/30 speakers, 1,966.51/14.28/14.29 h; Dutch 40/6/6, 1,554.24/12.76/12.76 h; French 142/18/18, 1,076.58/10.07/10.07 h; Spanish 86/20/20, 917.68/9.99/10.00 h; Italian 65/10/10, 247.38/5.18/5.27 h; Portuguese 42/10/10, 160.96/3.64/3.74 h; Polish 11/4/4, 103.65/2.08/2.14 h. English paper totals 5,490/42/42 speakers, 44,659.74/15.75/15.55 h, but no current HF English config or exact clip rows. | **`speaker_id`**. Paper states speaker-disjoint splits. | None. | CC BY 4.0, public direct resources/no gate. Large archives. | German streaming documented: `load_dataset("facebook/multilingual_librispeech","german",split="test",streaming=True)`. Dataset object created but first shard retrieval errored in sandbox; end-to-end audio not confirmed. English only via very large official OpenSLR route, not current HF config. |
| **ALFFA** — limited public Swahili / purchase-gated Hausa | OpenSLR SLR25 and [ALFFA repo](https://github.com/getalp/ALFFA_PUBLIC); no formal tag/version verified. | No official HF config. Swahili Kaldi train/test 10,180/1,991 transcript rows; Hausa GlobalPhone train/dev/test 5,863/1,021/1,011. Fongbe 8,234 train/2,168 test manifest rows; Fongbe is not Igbo. | Swahili manifest `utt2spk` has 30 train and 6 test IDs; these are documented audio/recording IDs, not proven human speakers; approx. 10/1.8 h in README. Hausa IDs 83/10/10, split IDs disjoint; approx. 7/1/1 h. Fongbe 23/4 IDs; hours unknown. Amharic approx. 20/2 train/test hours; human-speaker total unknown. | `utt2spk` second column is described as audio ID for Swahili; Hausa ID values listed but no accent metadata. **Do not treat Swahili IDs as verified people.** | None found in inspected official schemas. | Swahili public archive has component-specific CC0/public-domain waiver; OpenSLR resource/root repo separately display MIT. Hausa corpus/dictionary must be purchased from ELDA/ELRA; price/terms unknown. | Public Swahili archive: `wget -c https://openslr.trmal.net/resources/25/data_broadcastnews_sw.tar.bz2`; Kaldi `text`,`utt2spk`,`wav.scp` after extraction/path correction. No official `datasets.load_dataset` route. Hausa has no public audio command. |
| **YODAS v3 / v1** — web-caption speech, no speaker evaluation | Current `espnet/yodas3`; prior `espnet/yodas` / `espnet/yodas2`; [v3 card](https://huggingface.co/datasets/espnet/yodas3), [v1 card](https://huggingface.co/datasets/espnet/yodas). | v3 only `train`; metadata rows: English 1,146,215, Hausa 10, Swahili 2,605; no Yoruba/Igbo config. Hausa metadata has 6 transcript segments and 1.2489 h audio (5/10 caption-bearing files); Swahili 811,701 segments, 1,259.9858 h (2,603/2,605 caption-bearing). V1 configs include en000–en005 manual and en100–en127 automatic; one train split. v1 reported English 169,852.7 h; ha000 0.102931 h, ig000 0.086875 h, sw000 1.93409 h, yo000 0.322206 h. | Unique-speaker counts unavailable for both versions; file rows/segments are not speaker counts. No exact v1 row counts. | None. `id`/`utt_id` identify assets/utterances, not people. | None. `locale_lang` is uploader-locale metadata, not speaker accent. | CC BY 3.0, public/ungated; very large assets; YouTube-sourced content may have other rights. | Verified v3 metadata load: `load_dataset('espnet/yodas3','sw_metadata',split='train')` (2,605 rows). Audio streaming documented, not executed. v1 legacy loader script fails current viewer; no current working command verified. |
| **NaijaVoices** — best prospective true star-language source | `naijavoices/naijavoices-dataset`, 2025 release; pinned `e96de66e20528cf7ee1a88561a24d0b6eac70a4a`; [HF](https://huggingface.co/datasets/naijavoices/naijavoices-dataset), [paper](https://arxiv.org/html/2505.20564v3). | Configs `hausa-batch-0..2`, `igbo-batch-0..2`, `yoruba-batch-0..2`; API exposes one `train` split per batch. Audio-example rows: Hausa 236,200/236,200/236,199 (708,599 total); Igbo 195,644/195,644/195,643 (586,931); Yoruba 207,386/207,386/207,384 (622,156). Official paper partition files are `split/train.csv`, `split/dev.csv`, `split/test.csv`; partition clip counts by language unavailable. | Paper total per language: Hausa 1,879 speakers/605.27 h; Igbo 1,808/626.33 h; Yoruba 1,768/606.94 h. Combined NV Test contains 20 unseen speakers, but per-language test counts and test clips unknown. Aggregate split durations train/dev/NV-test 1,832.73/3.02/3.34 h; published split-hour total differs from language-hour sum by 0.55 h, unexplained. | `speaker_id` (anonymized, verified in schema/card). | None. | CC BY-NC-SA 4.0; public but automatically gated. Requires registration/approval; terms restrict identifying donors, surveillance, cloning; commercial use needs separate membership/waiver. | Documented command after access: `load_dataset('naijavoices/naijavoices-dataset','yoruba-batch-0',split='train',revision='e96de66e20528cf7ee1a88561a24d0b6eac70a4a')`. **Not runtime-verified**; gate returned 401. This loads batch train rows, not the paper’s train/dev/test assignment. |
| **Whisper-small tokenizer** — support check, not a dataset | `openai/whisper-small`, pinned `973afd24965f72e36ca33b3055d56a652f456b4d`; [generation config](https://huggingface.co/openai/whisper-small/blob/973afd24965f72e36ca33b3055d56a652f456b4d/generation_config.json) | Not applicable: model artifact. Pinned generation map contains 99 language tokens. | Not applicable. | Not applicable. | Not applicable. | Apache-2.0, public/ungated. | Tokenizer/config retrieval code in source list. No audio inference run. |

## 2. Language and English-accent counts

### Yoruba, Hausa, Swahili, Igbo

| Evidence source | What it can support | Verified counts relevant to a held-out group | Interpretation / blocker |
|---|---|---|---|
| FLEURS current `google/fleurs` | Read-speech rows in all four target configs | Test rows: Yoruba `yo_ng` 831; Hausa `ha_ng` 621; Swahili `sw_ke` 487; Igbo `ig_ng` 969. Unique speakers per config/split: **unknown**. | Current dataset has no speaker ID. Cannot use these counts for speaker-disjoint splitting or speaker-cluster bootstrap. Historical `google/xtreme_s` once exposed IDs, but quality and end-to-end access are unverified. |
| NaijaVoices | Actual Hausa, Igbo, Yoruba speech with anonymized speaker ID | Overall speakers/hours: Hausa 1,879/605.27; Igbo 1,808/626.33; Yoruba 1,768/606.94. **Test clips and unique test speakers by language are unknown.** Combined NV Test has 20 unseen speakers; language allocation unknown. | Best source candidate pending access and exact split-file audit; not yet proof of ≥200 test clips and ≥10 speakers per chosen language. Swahili not included. |
| AfriSpeech-200 | English accent groups, not the named African languages | In the public test CSV: Yoruba-label 650 clips/172 speakers; Hausa-label 196/53; Swahili-label 523/61; Igbo-label 357/94. | These are English utterances spoken with those accent labels. Three pass 200 clips/10 speakers; Hausa misses 200 clips by 4. Audio is gated/S3 credentials unavailable. Must not describe them as Yoruba/Hausa/Swahili/Igbo language ASR. |
| ALFFA | Potential Hausa/Swahili, not Yoruba/Igbo | Hausa test 1,011 rows/10 manifest IDs; Swahili test 1,991/6 manifest IDs. Human-speaker counts are not verified. | Hausa audio requires purchase; Swahili IDs are recording/audio codes and do not validate 10 human speakers. Does not currently establish a qualifying group. |
| YODAS | Web-audio/captions | v3 Hausa 10 metadata rows, 5 caption-bearing; Swahili 2,605 file rows. v1 has 0.103 h Hausa, 1.934 h Swahili, 0.322 h Yoruba, 0.087 h Igbo. | No speaker IDs/counts, partitions, or accent labels. Cannot satisfy speaker intervals. |
| Whisper-small | Decoder token support | Supported: `yo`, `ha`, `sw`; Igbo `ig` absent in pinned model/tokenizer. | Token support does not establish recognition accuracy. |

**Recommendation:** do not use FLEURS to make inferential language comparisons with speaker-clustered intervals. Obtain NaijaVoices approval, read its official train/dev/test partition files, and calculate per-language split counts before locking the protocol. The per-language total speaker figures cannot be substituted for test-speaker counts. If that audit cannot be completed promptly, omit a language-axis result rather than report invalid intervals.

### English accents

| Source / label | Clips and speakers | Threshold result and caveat |
|---|---:|---|
| Common Voice v27 `us` | 574,821 / 11,023 | Plenty of release-wide data; test-split speaker count unknown. |
| Common Voice v27 `england` | 204,881 / 3,466 | Plenty of release-wide data; test-split speaker count unknown. |
| Common Voice v27 `indian` | 153,048 / 3,045 | Plenty of release-wide data; test-split speaker count unknown. |
| Common Voice v27 `canada` | 101,734 / 1,225 | Plenty of release-wide data; test-split speaker count unknown. |
| Common Voice v27 `australia` | 69,393 / 953 | Plenty of release-wide data; test-split speaker count unknown. |
| Common Voice v27 `scotland` | 69,286 / 272 | Plenty of release-wide data; test-split speaker count unknown. |
| Common Voice v27 `african` | 60,591 / 452 | Plenty of release-wide data; test-split speaker count unknown. |
| Common Voice v27 other labels | See matrix | All listed labels through `bermuda` exceed 200/10 at release-wide level; `southatlandtic` has only 9 speakers and does not. Multi-label speaker totals must not be summed. |
| AfriSpeech `yoruba` / `swahili` / `igbo` accent values | 650/172; 523/61; 357/94 in test | Exact public test-manifest counts; English speech; audio access remains gated. Hausa is 196/53 and misses clip threshold. |
| VoxPopuli `en_nl` / `en_de` / `en_cs` | 1,151/45; 1,074/84; 974/26 in `en_accented` test | L2 English groups, not native regional accents. Both clip and ID counts exceed 200/10. Same IDs overlap standard English partitions; reassign/filter by speaker before use. |

For the 48-hour project, prefer **Common Voice v27** if MDC access can be completed and local storage permits. Its labels are contributor self-reports and the supplied partitions do not promise speaker disjointness. As a fallback for accent-only comparisons, three eligible AfriSpeech English accent groups have exact test-manifest speaker counts, but no audio was verified. VoxPopuli offers exact accent-test speaker counts and public access, but only for L2 English and its published splits leak speakers. None of these facts supplies clean test counts for a new Common Voice speaker split; these must be computed from the downloaded archive before fixing group sizes.

## 3. Whisper-small language support

The pinned checkpoint’s `generation_config.json` and `tokenizer_config.json` agree on 99 language-token entries. Exact relevant token IDs: English `en` = 50259; Yoruba `yo` = 50325; Hausa `ha` = 50354; Swahili `sw` = 50318; French `fr` = 50265. **Igbo `ig` is absent** from both files. Transformers v5.22.2 supports forcing the supported languages with `model.generate(..., language=CODE, task="transcribe")`, and pipeline `generate_kwargs` forwards the language argument. `language="ig"` is unsupported by this checkpoint. This confirms decoder interface support, not ASR quality. The upstream OpenAI `main` tokenizer list has changed and is not a substitute for the pinned checkpoint map.

## 4. Recommended plans and expected speakers

### Provisional primary — gated language source + current accent source; conditional, not yet cleared to run

1. **Star language candidate:** Yoruba, only if a team member confirms it is their language. Use `naijavoices/naijavoices-dataset`, configs `yoruba-batch-0`, `yoruba-batch-1`, `yoruba-batch-2`; do not use HF `train` as if it were the paper partition. Obtain access, then use and audit official `split/train.csv`, `split/dev.csv`, `split/test.csv` against `speaker_id`. The expected *total corpus* is 1,768 speakers; expected Yoruba test speakers/clips are **unknown**, not 1,768 and not 20. Do not include Yoruba in final evaluation unless the actual test partition has ≥200 clips and ≥10 distinct speakers and is disjoint from train/validation.
2. **Other target languages:** add Hausa or Igbo only if the team’s objective warrants it and the same audit establishes split counts. Their total corpus speakers are 1,879 and 1,808 respectively; test group counts unknown. Whisper-small supports Hausa and Yoruba; it does not expose an Igbo token. Swahili would need another source and is not in NaijaVoices.
3. **English accents:** acquire Common Voice v27 English via MDC and consider three labels with large published counts, e.g. `us`, `england`, `indian`. The release-wide speaker figures are 11,023, 3,466, and 3,045 respectively; **speaker-disjoint test-group counts after re-splitting are unknown until archive audit**. Use actual `client_id`; retain only speakers whose labels meet the predetermined inclusion policy. The archive is 88.43 GB.
4. **English/high-resource language:** VoxPopuli `en` and `de` are a possible public high-resource comparison, but adding them incurs large downloads and the split overlap means the source rows must be re-partitioned globally by actual `speaker_id`. They do not replace the project’s star language. For a reduced 48-hour workload, prefer one non-English language only after NaijaVoices audit rather than adding multiple large corpora.

**Expected speakers per group (what can be responsibly stated now):** Yoruba corpus total 1,768; Hausa 1,879; Igbo 1,808; Common Voice `us` 11,023 / `england` 3,466 / `indian` 3,045 across the release; none of those is a verified post-repartition test count. AfriSpeech test counts (if audio is obtained) are Yoruba label 172, Hausa label 53, Swahili label 61, Igbo label 94; again these are English accent groups, not language groups. Therefore no primary plan currently has verified test speaker totals for all its proposed groups.

### Fallback — reduced-scope public VoxPopuli comparison

Use `facebook/voxpopuli` pinned to `42f01879c780b4a2e90ec0b4f616c2ece526e4f1`, configs `en`, `de`, and `en_accented`, initially inspecting their `test` splits. Published rows are en 1,842; de 1,968; en_accented 8,387. Candidate L2-English test groups `en_nl`, `en_de`, and `en_cs` have 45, 84, and 26 distinct speaker IDs respectively and 1,151, 1,074, and 974 clip rows. English/German default test IDs are 286/123 in the valid-row manifests, but overlap their train/dev IDs. **These are source/test-manifest counts before project-specific reassignment, not guaranteed final partition counts.** A full-speaker re-assignment may reduce per-group held-out clips, so recalculate and apply the 200-clip/10-speaker rule before evaluation. Include no claim about a star-language test group. This fallback can support speaker-clustered intervals only after the explicit ID leakage audit and re-partition.

### Minimal deviation if blockers remain

The formal draft requires a team-spoken star language, 2–3 English accent groups, ≥200 clips/group, and speaker-clustered 95% intervals on speaker-disjoint partitions. Given the unknown team language, gated NaijaVoices, unverified English-accent audio acquisition, and unverified NaijaVoices split counts, do not assert that the primary plan meets it. If access/audit fails, narrow to groups with verified audio and IDs, and omit unsupported groups/intervals. Add this exact line to the repository deviations log (date format follows the log):

```text
| 2026-11-14 | Defer the star-language and English-accent inferential comparisons unless accessible audio and audited speaker-disjoint partitions meet the pre-set group thresholds; report only groups with verified speaker IDs, and do not calculate speaker-clustered intervals for unsupported groups. | NaijaVoices is gated and has no verified per-language test counts in this review; Common Voice v27 audio/split audit was not executed; current FLEURS exposes no speaker_id. | 
```

## 5. Loader and manifest code: safe contract, explicit source adapters

### Important execution boundary

No single loader can honestly be presented as verified end-to-end for the proposed primary plan: NaijaVoices returned 401 in the research session; Common Voice SDK download was not executed without an MDC API key; and the supplied evidence does not give the exact NaijaVoices transcript column name or verify audio decoding. The following is a **runnable fail-closed adapter** for use *after the source account/key is authorized and the required columns are inspected*. It uses the documented NaijaVoices batch `datasets` route, reads the real verified `speaker_id`, requires the operator to explicitly specify the actual transcript field (thereby not guessing it), resamples and saves audio at 16 kHz, writes exactly six manifest fields, and makes a deterministic speaker-level split. It refuses duplicate IDs, missing IDs/references/audio, unknown transcript columns, and an empty partition. It does **not** claim to reproduce the paper’s official split CSVs: use those files instead for any final analysis after access; a project-created split from batch `train` is not a verified equivalent of the paper partitions.

For the final study, the preferred safer variant is: load the **official** partition assignments after approval, join them to rows on verified utterance IDs, and validate no `speaker_id` crosses partitions; then write the six-key manifests. The present evidence does not establish a working `datasets` method to load those gated split CSVs or the audio’s transcript field, so the code intentionally cannot silently claim that step succeeded. **Do not run Whisper/LoRA or publish results from the batch-only project split as if it were the official held-out NV Test.**

Install after access: `python -m pip install datasets soundfile scipy numpy`. Save as `build_manifest.py`. Invoke only after selecting the verified string-valued transcript field printed by the dataset card/loaded features, e.g. `python build_manifest.py --text-field THE_VERIFIED_FIELD`. The placeholder is deliberate; no field name is invented in this report. If schema is ambiguous, stop and resolve it against the gated source rather than guessing.

```python
#!/usr/bin/env python3
"""Fail-closed audio manifest builder for authorized NaijaVoices HF access.

Writes data/manifests/{train,validation,test}.jsonl. Each JSON object has exactly:
utt_id, audio, speaker, group, ref, language.
The deterministic SHA-256 split is a project-generated split, NOT the published
NaijaVoices partition. Do not use it as a substitute for split/*.csv in final work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from datasets import load_dataset

REV = "e96de66e20528cf7ee1a88561a24d0b6eac70a4a"
REPO = "naijavoices/naijavoices-dataset"
# The batch API names are verified; each API config exposes a single `train` split.
BATCHES = {
    "yoruba": ["yoruba-batch-0", "yoruba-batch-1", "yoruba-batch-2"],
    "hausa": ["hausa-batch-0", "hausa-batch-1", "hausa-batch-2"],
    "igbo": ["igbo-batch-0", "igbo-batch-1", "igbo-batch-2"],
}
PARTS = ((70, "train"), (15, "validation"), (15, "test"))
TARGET_SR = 16_000


def partition_for(speaker: str, seed: str) -> str:
    # One deterministic destination for every occurrence of the same speaker ID.
    value = int(hashlib.sha256(f"{seed}:{speaker}".encode()).hexdigest()[:16], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "validation"
    return "test"


def audio_to_16k(audio_obj):
    if not isinstance(audio_obj, dict) or audio_obj.get("array") is None:
        raise ValueError("Audio is not decoded; verify authorized dataset audio access/decoder.")
    arr = np.asarray(audio_obj["array"])
    if arr.ndim == 2:  # average channels to mono
        arr = arr.mean(axis=0 if arr.shape[0] <= 8 else 1)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    sr = int(audio_obj["sampling_rate"])
    if sr <= 0:
        raise ValueError(f"Invalid sampling rate: {sr}")
    if sr != TARGET_SR:
        # Exact rational resampling; output is written with a 16-kHz WAV header.
        from math import gcd
        g = gcd(TARGET_SR, sr)
        arr = resample_poly(arr, TARGET_SR // g, sr // g).astype(np.float32)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-field", required=True,
                    help="Exact transcript column verified from authorized loaded features; no default guessed.")
    ap.add_argument("--languages", nargs="+", choices=sorted(BATCHES), default=["yoruba"])
    ap.add_argument("--seed", default="langnostic-2026-11-14")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    root = Path(args.out)
    audio_dir = root / "audio"
    manifest_dir = root / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    files = {name: (manifest_dir / f"{name}.jsonl").open("w", encoding="utf-8")
             for _, name in PARTS}
    seen_utt = set()
    speaker_partition = {}
    counts = {name: {"rows": 0, "speakers": set()} for _, name in PARTS}
    try:
        for language in args.languages:
            for config in BATCHES[language]:
                ds = load_dataset(REPO, config, split="train", revision=REV)
                columns = set(ds.column_names)
                # `speaker_id` is the verified field. Audio is represented as audio;
                # transcript column deliberately must be specified after inspection.
                needed = {"speaker_id", "audio", args.text_field}
                if not needed.issubset(columns):
                    raise RuntimeError(
                        f"{config}: required {sorted(needed)}; actual columns={sorted(columns)}. "
                        "Do not substitute a guessed speaker or transcript field."
                    )
                for row in ds:
                    sid = row.get("speaker_id")
                    ref = row.get(args.text_field)
                    if sid is None or not str(sid).strip():
                        raise RuntimeError(f"{config}: missing speaker_id; refusing fabricated IDs")
                    if not isinstance(ref, str) or not ref.strip():
                        raise RuntimeError(f"{config}: missing/invalid reference text")
                    # Use source asset ID only if one is present; otherwise stable row number
                    # is not a verified utterance ID, so fail rather than fabricate one.
                    uid = row.get("utt_id")
                    if uid is None:
                        uid = row.get("id")
                    if uid is None or not str(uid).strip():
                        raise RuntimeError(
                            f"{config}: no verified utterance identifier found among `utt_id`/`id`; "
                            "inspect schema and add a source-specific audited ID mapping."
                        )
                    uid = f"{config}:{uid}"
                    if uid in seen_utt:
                        raise RuntimeError(f"Duplicate utterance identifier: {uid}")
                    seen_utt.add(uid)
                    speaker = str(sid)
                    split = partition_for(speaker, args.seed)
                    previous = speaker_partition.setdefault(speaker, split)
                    if previous != split:
                        raise AssertionError("Internal error: one speaker assigned to two splits")
                    wav_rel = Path("audio") / f"{hashlib.sha256(uid.encode()).hexdigest()}.wav"
                    wav_path = root / wav_rel
                    samples = audio_to_16k(row["audio"])
                    sf.write(wav_path, samples, TARGET_SR, subtype="PCM_16")
                    record = {
                        "utt_id": uid,
                        "audio": wav_rel.as_posix(),
                        "speaker": speaker,
                        "group": f"{language}_read",
                        "ref": ref,
                        "language": language,
                    }
                    # The dict above is deliberately the complete six-key contract.
                    assert set(record) == {"utt_id", "audio", "speaker", "group", "ref", "language"}
                    files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[split]["rows"] += 1
                    counts[split]["speakers"].add(speaker)
    finally:
        for f in files.values():
            f.close()
    # Audit that a real speaker key appears in exactly one destination file.
    owners = {}
    for split, _ in PARTS:
        with (manifest_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if set(rec) != {"utt_id", "audio", "speaker", "group", "ref", "language"}:
                    raise AssertionError("Manifest has keys outside the six-field contract")
                old = owners.setdefault(rec["speaker"], split)
                if old != split:
                    raise AssertionError(f"Speaker leakage: {rec['speaker']} in {old} and {split}")
    for split, _ in PARTS:
        print(f"{split}: {counts[split]['rows']} rows; {len(counts[split]['speakers'])} speakers")
    print("WARNING: this SHA-based split is not the official NaijaVoices partition.")


if __name__ == "__main__":
    main()
```

**Code constraints and next step:** the code does not assume a dataset has a speaker field; it explicitly requires NaijaVoices `speaker_id` and fails otherwise. It also requires the actual transcript key and a verified utterance key (`utt_id` or `id`) to exist; those are schema checks, not claims that NaijaVoices has these exact audio-row keys. The supplied findings verify `speaker_id` and batch names but do not specify the exact transcript/audio utterance-ID key set or verify audio bytes. Consequently this is an executable, fail-closed adapter contract, **not evidence of an end-to-end successful audio load**. For Common Voice, use a separate adapter mapping `client_id`, its release’s verified transcript/audio columns, and `accents`; for AfriSpeech, use `user_ids` in the clean CSV and verify the actual audio-load key spelling (`user_id`/`speaker_id`) after gate approval. Do not reuse the NaijaVoices adapter by guessing other sources’ keys.

## 6. Access checklist and time/size constraints

1. **NaijaVoices (prospective star):** register at NaijaVoices, request/obtain non-commercial Explorer access, accept usage terms, then authenticate HF. Verify that the chosen config loads and that official `split/train.csv`, `split/dev.csv`, `split/test.csv` can be fetched. Card says the uncompressed dataset is 500+ GB; an 84-GB compressed derivative is linked. The exact per-language test audio footprint is unknown. Do not begin a large download until partition counts and terms are confirmed.
2. **Common Voice v27 (accent axis):** create MDC account, accept dataset terms, create API key in Profile > API; `python -m pip install datacollective`; export `MDC_API_KEY`; documented Python call: `from datacollective import download_dataset; print(download_dataset("cmu5jplf300nwmh07iqvk9leo"))`. Archive is 88.43 GB. Current free-account limit is seven downloads/day. SDK download was not run; confirm available disk and extraction needs first. Dataset may not be re-hosted/shared; keep audio local and private.
3. **AfriSpeech fallback accents:** request HF access and accept CC BY-NC-SA 4.0 conditions; for audio, ask Intron for S3 credentials (`intron-open-source/AfriSpeech-200`, region eu-west-2 per README). Published README estimates >2 h, approximately 103 GB train/5.2 GB dev; card estimates approximately 120 GB total. These are source estimates, not verified downloads. Public pinned test CSV is metadata/transcripts only; obtain audio rights/access before use.
4. **VoxPopuli fallback:** public and ungated. Pin revision `42f01879c780b4a2e90ec0b4f616c2ece526e4f1`. Test parquet approximate compressed sizes: en 0.94 GB, de 0.94 GB, en_accented 4.95 GB. Entire Hub repository approximately 673 GB; full English config download is approximately 98.8 GB. Use only selected test configurations initially, audit IDs, and check item-specific European Parliament terms. No token indicated.
5. **FLEURS (inspection only):** public, ungated, CC BY 4.0. Install `datasets` and audio decoder as needed. Config test rows are small relative to above, but current schema lacks speaker ID; do not download it as the basis for speaker intervals.
6. **MLS (optional comparison):** public, CC BY 4.0, but exact download size depends on language; official English FLAC archive is 2.4 TB or 651 GB in Opus; German 115 GB/29 GB. Current HF has no English config. German streaming first-shard retrieval failed in the research sandbox. Not suitable as a quick English route.
7. **ALFFA:** Swahili public 1.2-GB archive; `wget -c https://openslr.trmal.net/resources/25/data_broadcastnews_sw.tar.bz2`. Correct local `wav.scp` paths after extraction. Hausa GlobalPhone audio and dictionary require purchase; cost/order terms unknown. Swahili manifest IDs are not verified people; avoid for valid speaker bootstrap.
8. **YODAS:** public/ungated but impractical at full scale: v1 repository about 35 TB, v3 about 60 TB of OPUS with roughly 8-GB tar shards. v3 metadata-only Swahili load was verified; audio stream was not. No speaker IDs, so not a speaker-evaluation alternative.
9. **Model:** `openai/whisper-small` public/ungated; pin `973afd24965f72e36ca33b3055d56a652f456b4d`. Required codes: `en`, `yo`, `ha`, `sw`, `fr`; no `ig` token. Model load/token access is separate from dataset access.
10. **Before spending the weekend:** first smoke-test data access and first decoded audio for each selected source; count unique speaker IDs and clips after filtering; inspect split intersections; verify source-specific schema/license and local disk. If any source cannot pass these checks promptly, drop that group instead of substituting rows or synthetic IDs.

## 7. Blockers and unknowns

- **Star-language identity:** project documentation leaves this undecided; no evidence states which language the team speaks. Yoruba is only a provisional data-first option, not an authorized assertion that the team speaks it.
- **NaijaVoices access and split cardinalities:** unauthenticated dataset-server and split files returned 401. Per-language held-out test clip/speaker counts, test durations, train-vs-validation speaker overlap, and exact row/audio API behavior remain unknown. The reported 20 unseen speakers is combined NV Test only. The batch `train` command is documented but not verified and is not equivalent to official partition files.
- **Common Voice v27:** no SDK credential or downloaded archive was available. Test-split unique speakers by accent, the speaker intersections between supplied partitions, empty/null accent semantics, and actual current archive field mapping remain unverified. Published label counts cannot be presented as counts for a clean project test set.
- **FLEURS:** current data has no speaker ID; there is no valid way to recover speaker clusters from gender, paths, file names, or row counts. Historical `speaker_id` exists in an official loader but was removed; raw-value coercion, collisions, missing values, cross-language namespaces, and working 2026 load are unverified.
- **AfriSpeech:** actual audio was unavailable due manual HF gate/S3 credentials. Its four African-named test groups are accents of English, not the corresponding languages. The official clean CSV uses `user_ids`, but HF API/card use `user_id`/`speaker_id`; loaded key requires inspection. Published dev/train count discrepancies remain unreconciled.
- **VoxPopuli:** release manifests contradict the paper’s speaker-disjoint split statement. Reassign/remove IDs across standard English and `en_accented` before use; verify resulting sample counts. Accent criteria are not documented as self-reported. Paper-level speaker totals disagree with current-row ID unions.
- **ALFFA:** Swahili `utt2spk` IDs may be recordings, not people; exact human speaker totals and current audio duration unknown. Hausa has paid access. No accent field.
- **MLS:** English absent from current HF configs; English clip-row count unknown. German audio iteration failed locally. Published hours/speakers are paper-level, not current API-computed aggregates. No accent labels.
- **YODAS:** no speaker ID, no speaker counts, no speaker-disjoint design; weak web captions and language IDs. Not eligible for speaker-bootstrap inference.
- **Technical/deadline:** download sizes and gates make it unsafe to promise a complete language-and-accent analysis within the remaining window. M2 LoRA duration is not established by these data-source findings. No accuracy claim follows from Whisper token support.
- **Test handling:** choose and freeze the final speaker-held-out test set before any inference; never use test outcomes to choose data, LoRA settings, or models. Record all filtering and retain local audio only in accordance with each source’s terms.

## 8. Sources

All sources below were accessed/checked **2026-11-14**. Inline findings are summarized above; references are provided for auditability.

1. Hugging Face FLEURS card — https://huggingface.co/datasets/google/fleurs
2. FLEURS live dataset-server info — https://datasets-server.huggingface.co/info?dataset=google/fleurs
3. FLEURS Hub API — https://huggingface.co/api/datasets/google/fleurs
4. Historical FLEURS card revision with `speaker_id` — https://huggingface.co/datasets/google/fleurs/resolve/523337d4e70fc2b73abd71cbd0972f48078c5f2a/README.md
5. Pre-removal XTREME-S FLEURS loader — https://huggingface.co/datasets/google/xtreme_s/resolve/aaa5f581bb8cbf996c19fbef74cec4bfb5045937/xtreme_s.py
6. XTREME-S change removing field — https://huggingface.co/datasets/google/xtreme_s/commit/67c9d395b03d68600af87ff30601f445fcb6db35
7. FLEURS paper — https://arxiv.org/html/2205.12446v1
8. Google XTREME-S overview — https://sites.research.google/xtreme/xtremes
9. XTREME-UP archive README — https://github.com/google-research/xtreme-up
10. CC BY 4.0 deed — https://creativecommons.org/licenses/by/4.0/deed.en
11. MDC Common Voice v27 English dataset page — https://mozilladatacollective.com/datasets/cmu5jplf300nwmh07iqvk9leo
12. Mozilla current Common Voice catalog — https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk
13. Common Voice scripted-speech README — https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech/README.md
14. Common Voice changelog — https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech/CHANGELOG.md
15. Common Voice v27 English stats JSON — https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/cv-corpus-27.0-2026-09-11.json
16. Common Voice v17 English stats JSON — https://raw.githubusercontent.com/common-voice/cv-dataset/main/datasets/scripted-speech/cv-corpus-17.0-2024-03-15.json
17. Mozilla accent demographic source — https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts
18. Mozilla older-version access policy — https://community.mozilladatacollective.com/were-changing-access-to-older-versions-of-common-voice-datasets/
19. Mozilla third-party hosting FAQ — https://community.mozilladatacollective.com/faq-can-i-get-the-common-voice-or-other-mdc-datasets-from-other-platforms-like-github-or-hugging-face/
20. MDC API reference — https://datacollective.mozillafoundation.org/api-reference
21. MDC Python SDK — https://github.com/Mozilla-Data-Collective/datacollective-python
22. MDC download rate notice — https://community.mozilladatacollective.com/rate-limit-changes-on-mozilla-data-collective/
23. Official HF v17 page/API — https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 ; https://huggingface.co/api/datasets/mozilla-foundation/common_voice_17_0
24. Unofficial v17 mirror and split endpoint — https://huggingface.co/datasets/fsicoli/common_voice_17_0 ; https://datasets-server.huggingface.co/splits?dataset=fsicoli%2Fcommon_voice_17_0
25. Hugging Face Datasets 4.0 release — https://github.com/huggingface/datasets/releases/tag/4.0.0
26. AfriSpeech-200 card/API — https://huggingface.co/datasets/intronhealth/afrispeech-200 ; https://huggingface.co/api/datasets/intronhealth/afrispeech-200
27. Official AfriSpeech-200 repository at pinned revision — https://github.com/intron-innovation/AfriSpeech-200/tree/e01d1c368e28a5303a97c5fbfd1227ea1d85cf07
28. AfriSpeech test manifest at pinned revision — https://raw.githubusercontent.com/intron-innovation/AfriSpeech-200/e01d1c368e28a5303a97c5fbfd1227ea1d85cf07/data/intron-test-public-6346-clean.csv
29. AfriSpeech train/dev manifests — https://raw.githubusercontent.com/intron-innovation/AfriSpeech-200/e01d1c368e28a5303a97c5fbfd1227ea1d85cf07/data/intron-train-public-58000-clean.csv ; https://raw.githubusercontent.com/intron-innovation/AfriSpeech-200/e01d1c368e28a5303a97c5fbfd1227ea1d85cf07/data/intron-dev-public-3231-clean.csv
30. AfriSpeech paper — https://arxiv.org/html/2306.02105v5
31. CC BY-NC-SA 4.0 deed — https://creativecommons.org/licenses/by-nc-sa/4.0/
32. VoxPopuli card — https://huggingface.co/datasets/facebook/voxpopuli
33. VoxPopuli dataset-server info/splits and pinned Hub API — https://datasets-server.huggingface.co/info?dataset=facebook/voxpopuli ; https://datasets-server.huggingface.co/splits?dataset=facebook/voxpopuli ; https://huggingface.co/api/datasets/facebook/voxpopuli
34. Official VoxPopuli README and ASR preparation code — https://github.com/facebookresearch/voxpopuli ; https://github.com/facebookresearch/voxpopuli/blob/main/voxpopuli/get_asr_data.py
35. VoxPopuli released annotation manifests — https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_en.tsv.gz ; https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_de.tsv.gz ; https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_fr.tsv.gz ; https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_en_accented.tsv.gz
36. VoxPopuli paper — https://arxiv.org/html/2101.00390v2
37. CC0 deed, European Parliament legal notice, repository code license — https://creativecommons.org/publicdomain/zero/1.0/ ; https://www.europarl.europa.eu/legal-notice/en/ ; https://github.com/facebookresearch/voxpopuli/blob/main/LICENSE
38. OpenSLR MLS SLR94 — https://www.openslr.org/94/
39. MLS Hugging Face card and split/info endpoints — https://huggingface.co/datasets/facebook/multilingual_librispeech ; https://datasets-server.huggingface.co/splits?dataset=facebook/multilingual_librispeech ; https://datasets-server.huggingface.co/info?dataset=facebook/multilingual_librispeech
40. MLS paper — https://arxiv.org/html/2012.03411v2
41. CC BY 4.0 deed — https://creativecommons.org/licenses/by/4.0/
42. ALFFA OpenSLR SLR25 / official repo / README — https://www.openslr.org/25/ ; https://github.com/getalp/ALFFA_PUBLIC ; https://raw.githubusercontent.com/getalp/ALFFA_PUBLIC/master/README.md
43. ALFFA Swahili and Hausa README/manifests — https://raw.githubusercontent.com/getalp/ALFFA_PUBLIC/master/ASR/SWAHILI/README ; https://raw.githubusercontent.com/getalp/ALFFA_PUBLIC/master/ASR/SWAHILI/data/train/utt2spk ; https://raw.githubusercontent.com/getalp/ALFFA_PUBLIC/master/ASR/SWAHILI/data/test/utt2spk ; https://raw.githubusercontent.com/getalp/ALFFA_PUBLIC/master/ASR/HAUSA/README.md
44. ALFFA Swahili paper, Hausa corpus purchase reference — https://www.isca-archive.org/sltu_2012/gelas12_sltu.pdf ; https://aclanthology.org/L16-1611/
45. YODAS v3 card/API/splits/size and release blog — https://huggingface.co/datasets/espnet/yodas3 ; https://huggingface.co/api/datasets/espnet/yodas3 ; https://datasets-server.huggingface.co/splits?dataset=espnet%2Fyodas3 ; https://datasets-server.huggingface.co/size?dataset=espnet%2Fyodas3 ; https://huggingface.co/blog/espnet/yodasv3
46. YODAS v3 Interspeech abstract; v1 card/builder — https://www.isca-archive.org/interspeech_2026/chen26d_interspeech.html ; https://huggingface.co/datasets/espnet/yodas ; https://huggingface.co/datasets/espnet/yodas/resolve/main/yodas.py
47. YODAS paper and CC BY 3.0 deed — https://arxiv.org/html/2406.00899v1 ; https://creativecommons.org/licenses/by/3.0/
48. NaijaVoices HF page/API/DOI — https://huggingface.co/datasets/naijavoices/naijavoices-dataset ; https://huggingface.co/api/datasets/naijavoices/naijavoices-dataset ; https://doi.org/10.57967/hf/3257
49. NaijaVoices paper, terms, membership — https://arxiv.org/html/2505.20564v3 ; https://naijavoices.com/terms-and-conditions ; https://naijavoices.com/membership
50. Whisper-small pinned generation/tokenizer configs and model API — https://huggingface.co/openai/whisper-small/blob/973afd24965f72e36ca33b3055d56a652f456b4d/generation_config.json ; https://huggingface.co/openai/whisper-small/blob/973afd24965f72e36ca33b3055d56a652f456b4d/tokenizer_config.json ; https://huggingface.co/api/models/openai/whisper-small
51. OpenAI Whisper tokenizer, Transformers Whisper docs and v5.22.2 implementation — https://github.com/openai/whisper/blob/main/whisper/tokenizer.py ; https://huggingface.co/docs/transformers/en/model_doc/whisper ; https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/whisper/generation_whisper.py ; https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/pipelines/automatic_speech_recognition.py

## Handoff back

1. Star language: Yoruba is only a provisional data-first choice; confirm a team member speaks it before adopting it.
2. Primary star dataset: `naijavoices/naijavoices-dataset`, configs `yoruba-batch-0`, `yoruba-batch-1`, `yoruba-batch-2`; official split files still need gated access/audit.
3. NaijaVoices speaker field: `speaker_id`; corpus total 1,768 Yoruba speakers, but test speakers/clips per language are unknown.
4. English accent candidate: Common Voice Scripted Speech 27.0, MDC ID `cmu5jplf300nwmh07iqvk9leo`, locale `en`; field `client_id`, label field `accents`.
5. Common Voice candidate labels: `us` 11,023, `england` 3,466, `indian` 3,045 release-wide speakers; clean test counts after re-splitting unknown.
6. Fallback: `facebook/voxpopuli` pinned `42f01879c780b4a2e90ec0b4f616c2ece526e4f1`, configs `en`, `de`, `en_accented`; field `speaker_id`.
7. VoxPopuli candidate L2-English groups: `en_nl` 45, `en_de` 84, `en_cs` 26 test-manifest IDs; all exceed 200 test clips but packaged splits leak speakers.
8. Access: request NaijaVoices Explorer approval; obtain MDC account/terms/API key before downloading its 88.43-GB English archive; neither route was executed in this review.
9. Top risk 1: gated NaijaVoices lacks verified per-language held-out counts; top risk 2: accent counts are release-wide and require speaker re-splitting; top risk 3: VoxPopuli default split leakage / expensive download can break deadline feasibility.
10. Whisper-small pinned tokens support `en`, `yo`, `ha`, `sw`, `fr`; no `ig`; run no speaker-clustered interval for any group until real speaker IDs and non-overlapping partitions are audited.
