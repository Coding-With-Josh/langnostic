"""Score cached transcripts per group.

Input: a jsonl file, one record per clip:
  {"utt_id": "...", "speaker": "...", "group": "yoruba_read", "ref": "...", "hyp": "..."}
Output: per-group wer / cer with speaker-bootstrap intervals, error mix, hallucination
rate, loop rate (reported only), wrong-script share (non-looped and all), fair
per-clip diacritic retention (plus the old pooled value), and the five worst
clips per group.

Usage:
  python -m speechgap.report results/cache/zero_shot.jsonl --out results/zero_shot_report.json
"""
import argparse
import json
import unicodedata
from collections import Counter

from .metrics import (CHAR_LOOP_MIN_REPEATS, CHAR_LOOP_N, LOOP_RATIO,
                      WORD_LOOP_MIN_REPEATS, WORD_LOOP_N, bootstrap_rate,
                      breakdown, is_hallucination, loop_flags, utt_counts)
from .normalize import normalize

# A token loop can produce a hypothesis tens of thousands of characters long.
# Truncate it before it reaches markdown; the marker keeps the cut visible so a
# clipped loop is never mistaken for a complete transcript (Phase 3).
LOOP_CHARS = 200
LOOP_MARKER = " [loop]"

# A hypothesis is only read this many times its reference's character length
# when counting diacritic marks. A mark in the tail of an over-generated output
# is not evidence that the model produced it in the right place (Phase 3).
RETENTION_HYP_CAP = 2

# Printed with every report so a reader can see exactly what "looped" means.
# These are reporting thresholds only; no pilot gate uses them.
THRESHOLDS = {
    "char_ngram": {"n": CHAR_LOOP_N, "min_repeats": CHAR_LOOP_MIN_REPEATS},
    "word_ngram": {"n": WORD_LOOP_N, "min_repeats": WORD_LOOP_MIN_REPEATS},
    "length_ratio": LOOP_RATIO,
    "retention_hyp_cap": RETENTION_HYP_CAP,
}

# Tone marks (grave, acute, circumflex, macron, caron) and the underdot.
# Precomposed Yoruba letters such as ẹ, ọ and ṣ decompose to a Latin base plus
# one of these under NFD, so both spellings are compared on equal terms.
DIACRITIC_MARKS = frozenset({
    "\u0300",  # combining grave accent
    "\u0301",  # combining acute accent
    "\u0302",  # combining circumflex accent
    "\u0304",  # combining macron
    "\u030c",  # combining caron
    "\u0323",  # combining dot below
})


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


def diacritic_marks(text):
    """Count tone marks and underdots attached to Latin base letters.

    NFD splits precomposed letters (ẹ, ọ́, à) into a base plus marks. A mark is
    counted only while the preceding base is Latin (Phase 3): a Greek or Khmer
    loop must not "retain" Yoruba diacritics by reusing U+0301 or U+0323 on a
    non-Latin base. Returns a Counter keyed by the combining character.
    """
    counts = Counter()
    latin_base = False
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.category(ch).startswith("M"):
            if latin_base and ch in DIACRITIC_MARKS:
                counts[ch] += 1
            continue
        latin_base = unicodedata.name(ch, "").startswith("LATIN")
    return counts


def diacritic_retention_all(refs, hyps):
    """Old pooled multiset recall over every hypothesis, loops included.

    Kept only so the report can show what the fair per-clip number replaced: a
    loop repeating the right mark scores 1.0 here, and one clip's marks can
    cover another clip's missing marks. Do not use as the headline value.
    """
    ref_counts, hyp_counts = Counter(), Counter()
    for ref, hyp in zip(refs, hyps):
        ref_counts.update(diacritic_marks(ref))
        hyp_counts.update(diacritic_marks(hyp))
    total = sum(ref_counts.values())
    if total == 0:
        return None
    kept = sum(min(n, hyp_counts.get(mark, 0)) for mark, n in ref_counts.items())
    return kept / total


def diacritic_retention_detail(refs, hyps):
    """Fair per-clip retention, with the old pooled value kept alongside.

    Each clip is scored on its own reference marks, so no clip can cover
    another's missing marks. A looped hypothesis (``loop_flags``) is excluded
    rather than allowed to repeat the right mark to a perfect score. The
    remaining hypothesis is read only up to ``RETENTION_HYP_CAP`` times its
    reference's character length, so an over-long output cannot earn credit for
    a mark that appears only in its tail.

    Returns ``retention`` (None when no scored clip has reference marks),
    ``retention_all`` (the old pooled value) and ``n_looped_excluded`` (looped
    hypotheses left out of the adjusted score).
    """
    kept = total = n_excluded = 0
    for ref, hyp in zip(refs, hyps):
        if loop_flags(ref, hyp)["any"]:
            n_excluded += 1
            continue
        ref_counts = diacritic_marks(ref)
        n_ref = sum(ref_counts.values())
        if n_ref == 0:
            continue
        hyp_counts = diacritic_marks(hyp[:RETENTION_HYP_CAP * len(ref)])
        kept += sum(min(n, hyp_counts.get(mark, 0))
                    for mark, n in ref_counts.items())
        total += n_ref
    return {
        "retention": kept / total if total else None,
        "retention_all": diacritic_retention_all(refs, hyps),
        "n_looped_excluded": n_excluded,
    }


def diacritic_retention(refs, hyps):
    """Adjusted per-clip retention; see ``diacritic_retention_detail``."""
    return diacritic_retention_detail(refs, hyps)["retention"]


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


def loop_rates(refs, hyps):
    """Share of hypotheses flagged by each loop test, plus the union.

    Reported only: no pilot gate reads these. The components are kept apart
    because a space-free character loop and a repeated word phrase fail in
    different ways, and the all-flagged ``loop_rate`` alone cannot say which
    detector fired.
    """
    flags = [loop_flags(ref, hyp) for ref, hyp in zip(refs, hyps)]
    n = len(flags)

    def share(key):
        return sum(f[key] for f in flags) / n if n else 0.0

    return {
        "loop_rate": share("any"),
        "loop_rate_char": share("char"),
        "loop_rate_word": share("word"),
        "loop_rate_length": share("length"),
        "n_looped": sum(f["any"] for f in flags),
    }


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
        # Loop tests and retention use raw text: normalisation can compress
        # whitespace and would change the length ratio the pilot gate uses.
        raw_refs = [r["ref"] for r in rows]
        raw_hyps = [r["hyp"] for r in rows]
        flags = [loop_flags(ref, hyp) for ref, hyp in zip(raw_refs, raw_hyps)]
        nonlooped_hyps = [hyp for hyp, flag in zip(hyps, flags) if not flag["any"]]
        retention = diacritic_retention_detail(raw_refs, raw_hyps)
        out[group] = {
            "n_clips": len(rows),
            "wer": bootstrap_rate(wc, speakers, n_boot=n_boot, seed=seed),
            "cer": bootstrap_rate(cc, speakers, n_boot=n_boot, seed=seed),
            "wer_error_mix": breakdown(wc),
            "hallucination_rate": sum(is_hallucination(a, b) for a, b in zip(refs, hyps)) / len(rows),
            **loop_rates(raw_refs, raw_hyps),
            # Primary wrong-script column excludes looped hypotheses: a
            # Latin-base combining-mark loop scores 0.0 by construction and
            # would dilute the share. The all-hyps value stays visible so the
            # change is auditable; the pilot gate still uses that value.
            "wrong_script_rate": (wrong_script_rate(" ".join(nonlooped_hyps))
                                  if nonlooped_hyps else None),
            "wrong_script_rate_all": wrong_script_rate(" ".join(hyps)),
            "diacritic_retention": retention["retention"],
            "diacritic_retention_all": retention["retention_all"],
            "n_looped_excluded": retention["n_looped_excluded"],
            "thresholds": THRESHOLDS,
        }
    return out


def _pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def thresholds_line():
    t = THRESHOLDS
    return (
        f"loop rule: character {t['char_ngram']['n']}-gram repeated "
        f">={t['char_ngram']['min_repeats']} times, word "
        f"{t['word_ngram']['n']}-gram repeated >={t['word_ngram']['min_repeats']} "
        f"times, or hypothesis longer than {t['length_ratio']:.0f}x its reference. "
        f"retention excludes looped hypotheses and reads each hypothesis only to "
        f"{t['retention_hyp_cap']}x its reference length. these thresholds are "
        "reported only; no pilot gate uses them."
    )


def to_markdown(report, worst=None):
    lines = ["| group | clips | speakers | wer [95% ci] | cer [95% ci] | "
             "insertions share | halluc. | loop rate | wrong script (non-loop) | "
             "wrong script (all) | diacritic retention | retention (all) | "
             "looped excluded |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
             "--- | --- | --- |"]
    for g, m in report.items():
        w, c = m["wer"], m["cer"]
        lines.append(
            f"| {g} | {m['n_clips']} | {w['n_speakers']} | "
            f"{w['rate']:.1%} [{w['lo']:.1%}, {w['hi']:.1%}] | "
            f"{c['rate']:.1%} [{c['lo']:.1%}, {c['hi']:.1%}] | "
            f"{m['wer_error_mix']['ins']:.0%} | {m['hallucination_rate']:.0%} | "
            f"{_pct(m['loop_rate'])} | {_pct(m['wrong_script_rate'])} | "
            f"{_pct(m['wrong_script_rate_all'])} | "
            f"{_pct(m['diacritic_retention'])} | "
            f"{_pct(m['diacritic_retention_all'])} | {m['n_looped_excluded']} |"
        )
    lines += ["", thresholds_line()]
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
