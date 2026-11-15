"""h7: share of word errors that vanish when diacritics are stripped.

Read-only proxy over cached transcripts. Each hypothesis is scored against its
reference twice: as-is, and with diacritics stripped on both sides (NFD, remove
combining marks, NFC). Errors are word substitutions + deletions + insertions.
The share is (errors_with_marks - errors_without_marks) / errors_with_marks.

This is a proxy: it measures only whether errors vanish when marks are stripped.
It does not check whether the marks that are present are correct.

No model, no GPU, no network: the cache JSONL is the only input, and no cache is
rewritten.

Usage:
  python -m speechgap.diacritic_share \\
      results/cache/naija_test_yo30.jsonl \\
      results/cache/fleurs_zero_shot.jsonl \\
      --group-for results/cache/fleurs_zero_shot.jsonl=yoruba_read \\
      --out results/diacritic_share_report.json
"""
import argparse
import json
import os
import random
import re
import tempfile

from .metrics import _by_speaker, _interval, bootstrap_rate, loop_flags, utt_counts
from .normalize import normalize
from .report import _cell, load_jsonl

FIELDS = ("utt_id", "speaker", "group", "ref", "hyp")
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

# One source of truth for the interval, matching the rest of the project.
N_BOOT = 2000
SEED = 0
ALPHA = 0.05


def validate_records(records, source="cache"):
    """Refuse anything that could make a row count twice or lose a field.

    Threat (Phase 3): a duplicate ``utt_id`` doubles one clip in both passes and
    silently changes the speaker resampling; a missing or non-string field turns
    a transcript into an empty string and scores a deletion that never happened.
    Values are not echoed, so untrusted transcript text cannot reach the
    terminal through an error message.
    """
    if not isinstance(records, list) or not records:
        raise ValueError(f"{source}: no records to score")
    seen = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"{source}: row {i} is not an object")
        for key in FIELDS:
            if not isinstance(rec.get(key), str):
                raise ValueError(f"{source}: row {i} field {key!r} is not a string")
        for key in ("utt_id", "speaker", "group"):
            if not rec[key].strip():
                raise ValueError(f"{source}: row {i} field {key!r} is empty")
        if rec["utt_id"] in seen:
            raise ValueError(f"{source}: duplicate utt_id at row {i}")
        seen.add(rec["utt_id"])
    return records


def select_group(records, group):
    """Return one group's rows; an empty or unrequested group is an error.

    Threat (Phase 3): ``fleurs_zero_shot`` carries both FLEURS Yoruba and
    English rows under different group names. Scoring it without a filter would
    mix two corpora into one rate.
    """
    rows = [r for r in records if r["group"] == group]
    if not rows:
        raise ValueError(f"group {group!r} has no rows")
    return rows


def check_reference_agreement(named_records):
    """Same-length caches must hold the same references in the same order.

    ``named_records`` is a list of ``(label, rows)``. References are compared
    after NFC/case/punctuation normalisation, because the committed FLEURS
    baseline differs from later caches only in punctuation and case. Equal
    lengths with different references mean the positional pairing is broken, so
    the run stops before scoring (Phase 2).

    Different-length pairs are not comparable and are skipped; they are
    different corpora, not the same clips.
    """
    for i, (label_a, rows_a) in enumerate(named_records):
        refs_a = [normalize(r["ref"]) for r in rows_a]
        for label_b, rows_b in named_records[i + 1:]:
            if len(rows_a) != len(rows_b):
                continue
            refs_b = [normalize(r["ref"]) for r in rows_b]
            for pos, (a, b) in enumerate(zip(refs_a, refs_b)):
                if a != b:
                    raise ValueError(
                        f"references differ between {label_a} and {label_b} "
                        f"at position {pos}"
                    )


def _counts(records, strip_diacritics):
    return [
        utt_counts(
            normalize(r["ref"], strip_diacritics=strip_diacritics),
            normalize(r["hyp"], strip_diacritics=strip_diacritics),
            "word",
        )
        for r in records
    ]


def _errors(counts):
    return [c["s"] + c["d"] + c["i"] for c in counts]


def _share(errors_marks, errors_stripped):
    total = sum(errors_marks)
    if total == 0:
        return None
    return (total - sum(errors_stripped)) / total


def bootstrap_share(errors_marks, errors_stripped, speakers,
                    n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """Point share and speaker-clustered interval, resampling both passes together.

    Threat (Phase 3): independent resampling of the two passes would compare
    unrelated clip sets and widen or shift the interval. The same resampled
    speaker indices are used for both passes, matching ``paired_bootstrap_diff``.

    A resample with zero with-marks errors has no defined share; it is dropped
    rather than counted as 0 or 1. A single speaker yields a point estimate
    only, as in ``bootstrap_rate``.
    """
    assert len(errors_marks) == len(errors_stripped) == len(speakers)
    groups = _by_speaker(speakers)
    point = _share(errors_marks, errors_stripped)
    if point is None:
        return {"rate": None, "lo": None, "hi": None, "n_speakers": len(groups)}
    if len(groups) < 2:
        return {"rate": point, "lo": point, "hi": point,
                "n_speakers": len(groups)}
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        ks = [k for g in (rng.choice(groups) for _ in groups) for k in g]
        rep = _share([errors_marks[k] for k in ks],
                     [errors_stripped[k] for k in ks])
        if rep is not None:
            stats.append(rep)
    if not stats:
        return {"rate": point, "lo": None, "hi": None, "n_speakers": len(groups)}
    lo, hi = _interval(stats, alpha)
    return {"rate": point, "lo": lo, "hi": hi, "n_speakers": len(groups)}


def variant(records, non_looped=False, n_boot=N_BOOT, seed=SEED):
    """Both WER passes and the vanished-error share for one clip set."""
    validate_records(records)
    rows = records
    if non_looped:
        # Same rule the report uses: loop_flags on the raw text.
        rows = [r for r in records if not loop_flags(r["ref"], r["hyp"])["any"]]
        if not rows:
            raise ValueError("no non-looped clips to score")
    speakers = [r["speaker"] for r in rows]
    with_marks = _counts(rows, strip_diacritics=False)
    stripped = _counts(rows, strip_diacritics=True)
    errors_marks = _errors(with_marks)
    errors_stripped = _errors(stripped)
    # Stripping maps symbols together, so it cannot increase a Levenshtein
    # distance. A violation means the two passes did not see the same rows.
    if sum(errors_stripped) > sum(errors_marks):
        raise ValueError("stripping diacritics increased word errors")
    return {
        "n_clips": len(rows),
        "non_looped": non_looped,
        "reference_words": sum(c["n"] for c in with_marks),
        "errors_with_marks": sum(errors_marks),
        "errors_stripped": sum(errors_stripped),
        "wer_with_marks": bootstrap_rate(with_marks, speakers, n_boot=n_boot, seed=seed),
        "wer_stripped": bootstrap_rate(stripped, speakers, n_boot=n_boot, seed=seed),
        "share_vanished": bootstrap_share(errors_marks, errors_stripped, speakers,
                                          n_boot=n_boot, seed=seed),
    }


def score_cache(records, n_boot=N_BOOT, seed=SEED):
    """All clips and non-looped clips for one cache.

    Threat (Phase 3): a cache is one corpus. More than one group means the
    caller did not select a group, and mixing corpora would change both WER and
    the share without any error.
    """
    validate_records(records)
    groups = {r["group"] for r in records}
    if len(groups) != 1:
        raise ValueError(
            "cache spans several groups; select one with --group-for"
        )
    return {
        "group": next(iter(groups)),
        "all": variant(records, non_looped=False, n_boot=n_boot, seed=seed),
        "non_looped": variant(records, non_looped=True, n_boot=n_boot, seed=seed),
    }


def _pct(value):
    return "n/a" if value is None else f"{value:.1%}"


def _rate_cell(metric):
    if metric["rate"] is None:
        return "n/a"
    if metric["lo"] is None:
        return f"{metric['rate']:.1%} [n/a]"
    return f"{metric['rate']:.1%} [{metric['lo']:.1%}, {metric['hi']:.1%}]"


def to_markdown(results):
    """One row per cache and variant. Labels and groups are escaped (Phase 3)."""
    lines = [
        "| cache | variant | clips | speakers | wer (marks) | wer (stripped) "
        "| share vanished [95% ci] | errors: marks -> stripped |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, entry in results.items():
        name = _cell(label)
        if entry["group"]:
            name += f" ({_cell(entry['group'])})"
        for key, variant_name in (("all", "all clips"),
                                  ("non_looped", "non-looped")):
            m = entry[key]
            share = m["share_vanished"]
            lines.append(
                f"| {name} | {variant_name} | {m['n_clips']} | "
                f"{share['n_speakers']} | {_rate_cell(m['wer_with_marks'])} | "
                f"{_rate_cell(m['wer_stripped'])} | {_rate_cell(share)} | "
                f"{m['errors_with_marks']} -> {m['errors_stripped']} |"
            )
    lines += [
        "",
        "share vanished = (word errors with marks - word errors stripped) / "
        "word errors with marks; errors are substitutions + deletions + "
        "insertions after NFC, case and punctuation normalisation. this is a "
        "proxy: it does not check whether the marks that are present are "
        "correct. intervals resample whole speakers (2000 reps, seed 0); "
        "fleurs has one speaker per clip, so its intervals are clip-level and "
        "optimistic. non-looped uses the report's loop rule "
        "(loop_flags(...)[\"any\"]).",
    ]
    return "\n".join(lines)


def _write_json(path, payload):
    """Write through a temp file and rename, so no partial report is visible."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".diacritic_share.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _label_for(path):
    label = os.path.basename(path)
    if label.endswith(".jsonl"):
        label = label[:-len(".jsonl")]
    if not LABEL_RE.fullmatch(label):
        raise ValueError(f"cache label {label!r} is not allowed")
    return label


def _parse_group_for(values, paths):
    groups = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--group-for expects PATH=GROUP")
        path, group = value.rsplit("=", 1)
        if path not in paths:
            raise ValueError(f"--group-for path {path!r} is not an input cache")
        if not group:
            raise ValueError("--group-for group must not be empty")
        groups[path] = group
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="h7 diacritics-stripped proxy over cached transcripts")
    ap.add_argument("caches", nargs="+", help="cache JSONL paths (read-only)")
    ap.add_argument("--group-for", action="append", default=[],
                    metavar="PATH=GROUP",
                    help="score only GROUP from PATH (repeatable)")
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args(argv)

    groups = _parse_group_for(args.group_for, args.caches)
    labels = [_label_for(p) for p in args.caches]
    if len(set(labels)) != len(labels):
        raise ValueError("cache labels must be unique")

    loaded = []
    for path, label in zip(args.caches, labels):
        records = validate_records(load_jsonl(path), source=label)
        if path in groups:
            records = select_group(records, groups[path])
        loaded.append((label, path, records))
    check_reference_agreement([(label, rows) for label, _, rows in loaded])

    results = {}
    for label, path, records in loaded:
        results[label] = {
            "path": path,
            **score_cache(records),
        }

    print(to_markdown(results))
    if args.out:
        _write_json(args.out, {
            "method": {
                "share": "(errors_with_marks - errors_stripped) / errors_with_marks",
                "errors": "word substitutions + deletions + insertions",
                "strip": "NFD, remove combining marks, NFC, on both sides",
                "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "alpha": ALPHA,
                              "cluster": "speaker"},
                "caveat": "proxy: does not check whether present marks are correct",
            },
            "caches": results,
        })
    return results


if __name__ == "__main__":
    main()
