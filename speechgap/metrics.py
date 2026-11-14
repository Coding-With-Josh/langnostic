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
