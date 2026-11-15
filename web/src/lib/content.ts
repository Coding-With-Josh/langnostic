export const repo = "https://github.com/Coding-With-Josh/langnostic";
export const plan = `${repo}/blob/main/ANALYSIS_PLAN.md`;

export const hypotheses: [string, string][] = [
  ["h1", "i expect error to rise as whisper's training-data share for a language falls. descriptive only."],
  ["h2", "i expect the gap between accents to be smaller than the gap between languages."],
  ["h3", "i expect the adapter to close a large share of the yoruba gap."],
  ["h4", "i expect the gain to cost something on other speech."],
  ["h5", "i expect most low-resource errors to be inserted or repeated text."],
  ["h6", "i expect spontaneous speech to be worse than read speech."],
  ["h7", "i expect whisper to drop or change tone marks and underdots."],
  ["h8", "i expect forcing yoruba to differ from auto-detect on code-switched clips."],
];

export const datasets: string[][] = [
  ["naijavoices yoruba", "train, val, test", "900 (495), 100 (64), 300 (139)", "cc by-nc-sa 4.0, unofficial mirror"],
  ["fleurs yoruba", "test, first 100 scored", "300 (no speaker ids)", "cc by 4.0"],
  ["voxpopuli english", "test, first 100 scored", "300 (153)", "cc0"],
  ["l2-arctic english", "test, first 100 scored", "300 (24)", "cc by-nc 4.0"],
];

export const method: string[] = [
  "model: i ran whisper-small at revision 973afd24965f72e36ca33b3055d56a652f456b4d, greedy decoding, 30 second chunks, language set per clip. letting whisper detect the language is a separate comparison.",
  "adapter: i trained lora through peft, rank 8, alpha 16, dropout 0.05, on q_proj and v_proj only. that trains 884,736 of 242,619,648 parameters (0.36%). the narrow target is my choice.",
  "training: i trained on 15 and 30 minute audio budgets from naijavoices yoruba, with early stopping on validation wer. test clips never reach the trainer.",
  "gates: i set the pilot gates before any run and did not loosen them after seeing results.",
  "compute: i ran on a kaggle t4, fp16 training and fp32 evaluation. transformers 5.20.2, peft 0.21.2, torch 2.11.0, accelerate 1.15.0, datasets 4.8.5.",
];

export const metrics: string[][] = [
  ["wer, cer", "used", "95% bootstrap intervals. paired bootstrap for before and after on the same clips."],
  ["loop rate", "used", "share of clips caught by a repetition test. wer without those clips is post-hoc and changes the denominator."],
  ["error mix, hallucination rate", "used", "insertion share and output that has nothing to do with the audio."],
  ["diacritic retention, adjusted", "used with care", "tone marks and underdots present in the output, loops excluded. it does not check they are in the right place."],
  ["diacritic retention, old", "not trusted", "pooled across clips, so loops inflate it."],
  ["wrong-script rate", "not trusted on looped runs", "skips combining marks and cannot tell yoruba from english, both latin."],
  ["train loss", "not quoted", "the logged value is about 4 times the true mean. eval loss is quoted instead."],
];

export const deviations: [string, string][] = [
  ["i pushed a bad test", "commit 7016051 fixed the adapter wrapper and also held a wrong test. it went out because a shell pipeline hid pytest's exit status from me. 08dfe48 fixed it."],
  ["i abandoned local training", "my laptop took 2,111 seconds per step on mps."],
  ["colab ran out of my gpu quota", "i moved the runs to kaggle t4 and had to uninstall torchao. my first pilot used both gpus by accident; later runs used one."],
  ["the kaggle session reset once", "i rebuilt the data and the manifest hashes matched."],
  ["the 2 minute pilot failed its gates", "i went on because the pilot never leaves learning-rate warmup. it only checks the plumbing."],
  ["the 15 minute run failed twice", "validation wer was 142.4%, then 127.5% on a rerun. the runs differ by seeded but nondeterministic fp16 training, not by config. a flag now saves failed adapters under a .failed name."],
  ["the 30 minute run passed validation and did worse on test", "97.13% on validation, 131.6% on 300 test clips from 139 speakers. validation decodes without chunking and test uses 30 second chunks, so i do not read the two as directly comparable. i changed no gate."],
  ["group names collide", "the committed fleurs baseline calls its clips yoruba_read and english_read, and my manifest uses those names for naijavoices and voxpopuli. i match comparisons by position and normalized text, never by group name."],
];

export const limitations: string[] = [
  "i used one seed per run. one seed moved the 15 minute wer by about 15 points, so i read small differences between runs as noise.",
  "fleurs has no speaker ids, so i give it clip-level intervals, and they are optimistic. groups of 100 clips are descriptive only.",
  "i made the splits from the first rows in file order, not at random.",
  "in the 30 minute run i changed the amount of audio and the learning-rate schedule together, so i cannot separate the effect of the budget.",
  "for adapter runs i checked the language in the generate arguments only. the check at the decoder input never attached under peft.",
  "i make no per-accent claims. l2-arctic is pooled, 24 speakers.",
  "i do not distribute the adapter and there is no live demo, because of the dataset licence.",
  "my naijavoices copy is an unofficial mirror of a gated dataset.",
];

export const sources: string[] = [
  "whisper-small, openai",
  "peft, hugging face",
  "fleurs, google",
  "naijavoices, eyedol mirror",
  "voxpopuli, meta",
  "l2-arctic",
  "kaggle and google colab for compute",
  "ai assistants: claude (anthropic) and a coding agent, for code and analysis",
];
