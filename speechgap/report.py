"""Score cached transcripts per group.

Input: a jsonl file, one record per clip:
  {"utt_id": "...", "speaker": "...", "group": "yoruba_read", "ref": "...", "hyp": "..."}
Output: per-group wer / cer with speaker-bootstrap intervals, error mix, hallucination
rate, wrong-script share, and the five worst clips per group.

Usage:
  python -m speechgap.report results/cache/zero_shot.jsonl --out results/zero_shot_report.json
"""
import argparse
import json
import unicodedata

from .metrics import bootstrap_rate, breakdown, is_hallucination, utt_counts
from .normalize import normalize

# A token loop can produce a hypothesis tens of thousands of characters long.
# Truncate it before it reaches markdown; the marker keeps the cut visible so a
# clipped loop is never mistaken for a complete transcript (Phase 3).
LOOP_CHARS = 200
LOOP_MARKER = " [loop]"


def load_jsonl(path):
    # A .part cache is an interrupted transcription, not a result. Refuse it so
    # a partial run can never be scored as if it were complete.
    if str(path).endswith(".part"):
        raise ValueError(f"{path} is an interrupted cache (.part); finish or resume the run")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def truncate_hyp(text, limit=LOOP_CHARS, marker=LOOP_MARKER):
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + marker


def _cell(text):
    """One markdown table cell: no raw pipes, no embedded newlines.

    Threat (Phase 3): transcripts are untrusted text. A hypothesis containing
    ``|`` or a newline can otherwise forge table rows and misstate a result.
    """
    return " ".join(text.replace("|", "\\|").split())


def per_clip_wer(rec):
    counts = utt_counts(normalize(rec["ref"]), normalize(rec["hyp"]), "word")
    # An empty reference still has to sort and print; keep the ratio finite.
    return (counts["s"] + counts["d"] + counts["i"]) / max(counts["n"], 1)


def wrong_script_rate(text):
    """Share of script-bearing characters outside Latin / Latin-Extended.

    Only letters and marks are counted: spaces, digits and punctuation carry no
    script and would otherwise dilute the rate. A character with no Unicode
    name counts as wrong — unknown output is never silently treated as Latin.
    Latin-based modifier letters (Phonetic Extensions) are not in the named
    Latin ranges and count as wrong too; the rule is deliberately conservative.
    """
    counted = wrong = 0
    for ch in text:
        category = unicodedata.category(ch)
        if not (category.startswith("L") or category.startswith("M")):
            continue
        counted += 1
        if category.startswith("M"):
            continue  # combining marks belong to the base letter's script
        if not unicodedata.name(ch, "").startswith("LATIN"):
            wrong += 1
    return wrong / counted if counted else 0.0


def worst_examples(records, group, k=5, limit=LOOP_CHARS):
    """The k highest per-clip WER rows in one group; ties broken by utt_id.

    Sorting is fully specified so the report is reproducible: a run scored
    twice prints the same rows in the same order.
    """
    scored = [
        (per_clip_wer(r), r["utt_id"], r) for r in records if r["group"] == group
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [
        {"utt_id": uid, "wer": wer, "ref": r["ref"],
         "hyp": truncate_hyp(r["hyp"], limit)}
        for wer, uid, r in scored[:k]
    ]


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
            "wrong_script_rate": wrong_script_rate(" ".join(hyps)),
        }
    return out


def to_markdown(report, worst=None):
    lines = ["| group | clips | speakers | wer [95% ci] | cer [95% ci] | insertions share | halluc. | wrong script |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g, m in report.items():
        w, c = m["wer"], m["cer"]
        lines.append(
            f"| {g} | {m['n_clips']} | {w['n_speakers']} | "
            f"{w['rate']:.1%} [{w['lo']:.1%}, {w['hi']:.1%}] | "
            f"{c['rate']:.1%} [{c['lo']:.1%}, {c['hi']:.1%}] | "
            f"{m['wer_error_mix']['ins']:.0%} | {m['hallucination_rate']:.0%} | "
            f"{m['wrong_script_rate']:.1%} |"
        )
    for g in sorted(worst or {}):
        rows = worst[g]
        if not rows:
            continue
        lines += ["", f"### {g}: {len(rows)} worst clips", "",
                  "| utt_id | wer | ref | hyp |", "| --- | --- | --- | --- |"]
        for e in rows:
            lines.append(
                f"| {_cell(e['utt_id'])} | {e['wer']:.1%} | "
                f"{_cell(e['ref'])} | {_cell(e['hyp'])} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--strip-diacritics", action="store_true")
    ap.add_argument("--worst", type=int, default=5,
                    help="worst clips shown per group (0 disables)")
    ap.add_argument("--loop-chars", type=int, default=LOOP_CHARS)
    ap.add_argument("--out")
    args = ap.parse_args()
    records = load_jsonl(args.path)
    rep = score(records, strip_diacritics=args.strip_diacritics)
    worst = ({g: worst_examples(records, g, k=args.worst, limit=args.loop_chars)
              for g in rep} if args.worst > 0 else None)
    print(to_markdown(rep, worst))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
