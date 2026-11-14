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
    # A .part cache is an interrupted transcription, not a result. Refuse it so
    # a partial run can never be scored as if it were complete.
    if str(path).endswith(".part"):
        raise ValueError(f"{path} is an interrupted cache (.part); finish or resume the run")
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
