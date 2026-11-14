"""Build speaker-disjoint manifests from public speech datasets inside Colab.

The laptop never downloads audio. This module runs where the GPU is, writes
16 kHz mono WAVs plus ``manifest_{train,val,test}.jsonl``, and leaves a
``audit.json`` that records every row that was dropped and why.

Download-then-decode: each source is fetched as pinned parquet files with
``hf_hub_download`` and then read batch by batch, so only rows the allocator
admits are decoded. Per-row streaming was the bottleneck (~10 s/clip).

Design rules baked in (see ANALYSIS_PLAN.md):
  * one row = one clip, but intervals resample speakers, so a speaker may never
    appear in two groups or two splits;
  * ``speaker`` in the manifest is a namespaced key (``voxpopuli:42``), not a raw
    dataset id, so the same id in two corpora cannot silently merge;
  * sources that expose no speaker id are refused unless ``--allow-clip-level``,
    and that refusal is recorded per group;
  * manifests are staged and only published after every source finishes.

Usage (Colab):
  python -m speechgap.data build --out-dir data \
      --naija yoruba:0-0 \
      --voxpopuli en:test \
      --l2arctic \
      --fleurs yo_ng:test --allow-clip-level

  python -m speechgap.data verify --out-dir data

Sources:
  NaijaVoices public mirror  EYEDOL/naija-voices-{language}-split_{shard}
      language in {yoruba, hausa, igbo}; one repo per shard, config "default",
      split "train". Unofficial mirror: cc by-nc-sa 4.0, non-commercial. The
      resolved revision is recorded in audit.json.
  VoxPopuli                 facebook/voxpopuli pinned, configs en, de, en_accented.
  L2-ARCTIC                 chikingsley/l2-arctic-manual-v5.0-16k, cc-by-nc-4.0;
                            all four source splits pooled into one test-only group.
  FLEURS                    google/fleurs; no speaker id (clip-level only).
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
from collections import Counter, defaultdict

from .languages import resolve_language

SIX_KEYS = ("utt_id", "audio", "speaker", "group", "ref", "language")
NAIJA_TEMPLATE = "EYEDOL/naija-voices-{language}-split_{shard}"
NAIJA_LANGUAGES = ("yoruba", "hausa", "igbo")
VOXPOPULI_REPO = "facebook/voxpopuli"
VOXPOPULI_REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
FLEURS_REPO = "google/fleurs"
L2ARCTIC_REPO = "chikingsley/l2-arctic-manual-v5.0-16k"
L2ARCTIC_SPLITS = ("train", "validation", "test", "suitcase")
L2ARCTIC_GROUP = "english_l2_arctic"
L2ARCTIC_LANGUAGE = "en"
TARGET_SR = 16000
# Deterministic speaker-level split. Any change here invalidates old manifests.
SPLIT_RATIOS = (("train", 70), ("val", 15), ("test", 15))
CONFIG_GROUPS = {
    "en": "english_read",
    "de": "german_read",
    "fr": "french_read",
    "es": "spanish_read",
    "it": "italian_read",
    "nl": "dutch_read",
    "pl": "polish_read",
    "ro": "romanian_read",
    "hu": "hungarian_read",
    "fi": "finnish_read",
    "et": "estonian_read",
    "lt": "lithuanian_read",
    "hr": "croatian_read",
    "sk": "slovak_read",
    "sl": "slovenian_read",
    "cs": "czech_read",
}
DEFAULT_ACCENTS = "en_nl,en_de,en_cs"
# Cache of HF repo listings keyed by (repo, revision). Populated on first use.
_REPO_FILE_CACHE = {}


# ---------------------------------------------------------------------------
# pure helpers (stdlib only: these are unit-tested in CI)
# ---------------------------------------------------------------------------

def split_for(speaker_key, ratios=SPLIT_RATIOS):
    """Map a namespaced speaker key to train/val/test, deterministically.

    Threat (Phase 3): row-level splitting leaks a speaker across splits. This
    hashes the speaker key, so every row for that person lands in one split no
    matter which shard or file it came from.
    """
    digest = hashlib.sha256(("split|" + speaker_key).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    edge = 0
    for name, pct in ratios:
        edge += pct
        if bucket < edge:
            return name
    return ratios[-1][0]


def utt_id(source, tag):
    """Filesystem-safe, collision-resistant id derived from a stable source tag."""
    raw = f"{source}-{tag}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")[:80]
    return f"{slug}-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"


def make_record(utt, audio, speaker, group, ref, language):
    """The six-key manifest contract. Raises rather than emitting a bad row."""
    rec = {"utt_id": utt, "audio": audio, "speaker": speaker,
           "group": group, "ref": ref, "language": language}
    if set(rec) != set(SIX_KEYS):
        raise AssertionError("manifest contract changed")
    for k in SIX_KEYS:
        if not isinstance(rec[k], str) or not rec[k].strip():
            raise ValueError(f"manifest field {k!r} is empty: {rec!r}")
    return rec


class Allocator:
    """One speaker -> one group, and per-group/per-split caps.

    Order matters: callers must submit accent groups first so a speaker shared
    with a standard-language config stays in the accent group (Phase 3).
    """

    def __init__(self, caps, min_test_clips=200, min_test_speakers=10):
        self.caps = dict(caps)
        self.min_test_clips = min_test_clips
        self.min_test_speakers = min_test_speakers
        self.owner = {}
        self.counts = defaultdict(Counter)
        self.speakers = defaultdict(set)
        self.dropped = Counter()
        self.order = []

    def _note(self, group):
        if group not in self.order:
            self.order.append(group)

    def admit(self, speaker_key, group, split):
        if self.counts[group][split] >= self.caps[split]:
            self.dropped[f"cap:{group}:{split}"] += 1
            return False
        owner = self.owner.get(speaker_key)
        if owner is not None and owner != group:
            # Same person already assigned to another group: drop this row, keep
            # the interval honest. Recorded in audit.json.
            self.dropped[f"conflict:{owner}->{group}"] += 1
            return False
        self.owner[speaker_key] = group
        self._note(group)
        self.counts[group][split] += 1
        self.speakers[(group, split)].add(speaker_key)
        return True

    def release(self, speaker_key, group, split):
        """Undo an admit whose audio turned out to be unusable."""
        if self.owner.get(speaker_key) == group:
            del self.owner[speaker_key]
        self.counts[group][split] -= 1
        self.speakers[(group, split)].discard(speaker_key)

    def full(self, group, split):
        return self.counts[group][split] >= self.caps[split]

    def summary(self):
        out = defaultdict(lambda: defaultdict(lambda: {"clips": 0, "speakers": 0}))
        for group in self.order:
            for split in ("train", "val", "test"):
                out[group][split] = {
                    "clips": self.counts[group][split],
                    "speakers": len(self.speakers[(group, split)]),
                }
        return out


def _require_columns(features, required, label):
    if features is None:
        print(f"warning: {label}: no feature schema exposed; cannot check columns")
        return
    names = set(features.keys())
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(
            f"{label}: missing columns {missing}; saw {sorted(names)}. "
            "Refusing to build from an unknown schema."
        )


def _stream(repo, config, split, revision):
    from datasets import Audio, load_dataset

    kwargs = {"split": split, "streaming": True}
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset(repo, config, **kwargs) if config else load_dataset(repo, **kwargs)
    try:
        # decode=False keeps raw bytes; we decode only rows that pass the filter.
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as e:  # pragma: no cover - depends on datasets version
        print(f"warning: {repo}: could not defer audio decoding ({e})")
    return ds


def _resolve_revision(repo):
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo, timeout=30).sha
    except Exception as e:  # pragma: no cover - network dependent
        print(f"warning: could not resolve revision for {repo}: {e}")
        return None


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def write_wav(value, dest, target_sr=TARGET_SR):
    """Decode one dataset audio value to a 16 kHz mono PCM_16 wav.

    Accepts the encoded dict (bytes/path) produced with ``Audio(decode=False)``,
    an already-decoded array dict, raw bytes, or a local path.
    """
    import numpy as np
    import soundfile as sf

    if isinstance(value, dict) and value.get("array") is not None:
        y = np.asarray(value["array"], dtype="float32")
        sr = int(value["sampling_rate"])
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != target_sr:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    else:
        if isinstance(value, dict):
            raw = value.get("bytes")
            if raw is None and value.get("path"):
                with open(value["path"], "rb") as f:
                    raw = f.read()
        elif isinstance(value, (bytes, bytearray)):
            raw = bytes(value)
        elif isinstance(value, str):
            with open(value, "rb") as f:
                raw = f.read()
        else:
            raw = None
        if not raw:
            raise ValueError("empty audio payload")
        try:
            import librosa

            y, _ = librosa.load(io.BytesIO(raw), sr=target_sr, mono=True)
        except Exception:
            # Some parquet audio is mp3 that the installed libsndfile cannot read.
            import torch
            import torchaudio

            wav, sr = torchaudio.load(io.BytesIO(raw))
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            y = wav.squeeze(0).numpy()

    y = np.asarray(y, dtype="float32")
    if y.size == 0:
        raise ValueError("decoded audio is empty")
    sf.write(dest, y, target_sr, subtype="PCM_16", format="WAV")
    return len(y) / float(target_sr)


def _repo_files(repo, revision=None):
    """Repo file listing, cached per (repo, revision).

    The listing is the provenance link between a source revision and the exact
    parquet file bytes we download (Phase 1).
    """
    key = (repo, revision)
    if key not in _REPO_FILE_CACHE:
        from huggingface_hub import HfApi

        _REPO_FILE_CACHE[key] = HfApi().list_repo_files(
            repo, repo_type="dataset", revision=revision
        )
    return _REPO_FILE_CACHE[key]


def select_parquet_files(files, config, split, max_files=None):
    """Pure filter over a repo listing; see split_parquet_files for layouts."""
    selected = []
    for path in files:
        parts = path.split("/")
        name = parts[-1]
        if not name.endswith(".parquet"):
            continue
        if config and config != "default" and config not in parts[:-1]:
            continue
        in_split = (
            name.startswith(split + "-")
            or name == split + ".parquet"
            or (len(parts) >= 2 and parts[-2] == split)
        )
        if in_split:
            selected.append(path)
    selected = sorted(set(selected))
    if not selected:
        raise ValueError(
            f"no parquet files for config={config!r} split={split!r}; "
            "refusing to build an empty group"
        )
    return selected[:max_files] if max_files else selected


def split_parquet_files(repo, config, split, revision=None, max_files=None):
    """Pick the parquet files for one config/split from the repo listing.

    Layouts seen in practice:
      data/train-00000-of-00010.parquet              (NaijaVoices, L2-ARCTIC)
      en/test-00000-of-00001.parquet                 (VoxPopuli)
      parquet-data/yo_ng/test-00000-of-00001.parquet (FLEURS)

    A source whose layout is not recognised raises instead of yielding zero
    rows: silence here would look like a complete group with no data.
    """
    return select_parquet_files(_repo_files(repo, revision), config, split, max_files)


def download_parquet(repo, files, revision=None):
    """Download pinned parquet files into the Hugging Face cache.

    The returned paths are local; callers decode batches lazily so a 0.9 GB
    split never has to be materialised as Python objects all at once.
    """
    from huggingface_hub import hf_hub_download

    return [
        hf_hub_download(
            repo_id=repo, filename=filename, repo_type="dataset", revision=revision
        )
        for filename in files
    ]


def _schema_names(paths):
    import pyarrow.parquet as pq

    if not paths:
        raise ValueError("no parquet files to read")
    return pq.ParquetFile(paths[0]).schema_arrow.names


def _require_fields(names, required, label):
    missing = [c for c in required if c not in set(names)]
    if missing:
        raise ValueError(
            f"{label}: missing columns {missing}; saw {sorted(names)}. "
            "Refusing to build from an unknown schema."
        )


def _iter_parquet(paths, columns=None, batch_size=64):
    """Stream rows from downloaded parquet files, one Arrow batch at a time."""
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


# ---------------------------------------------------------------------------
# source iterators: yield candidates, record skips in a Counter
# ---------------------------------------------------------------------------

def iter_naija(language, shard, stats, max_scan=None, revision=None, max_files=1):
    repo = NAIJA_TEMPLATE.format(language=language, shard=shard)
    # One shard file already holds ~20k rows and hundreds of speakers; a second
    # file would only add download time (deviations log, 2026-11-14).
    files = split_parquet_files(repo, None, "train", revision, max_files=max_files)
    paths = download_parquet(repo, files, revision)
    _require_fields(_schema_names(paths), ("audio", "speaker_id", "text"), repo)
    group = f"{language}_read"
    for i, row in enumerate(_iter_parquet(paths)):
        if max_scan is not None and i >= max_scan:
            print(f"{repo}: reached --max-scan {max_scan}")
            break
        stats["rows"] += 1
        spk = _text(row.get("speaker_id"))
        ref = _text(row.get("text"))
        if not spk:
            stats["skip:missing_speaker"] += 1
            continue
        if not ref:
            stats["skip:empty_ref"] += 1
            continue
        lang = _text(row.get("language")) or language
        key = f"naijavoices:{spk}"
        yield {
            "source": "naijavoices",
            "speaker_key": key,
            "group": group,
            "split": split_for(key),
            "language": lang,
            "ref": ref,
            "tag": f"{language}-{shard}-{i}",
            "audio": row.get("audio"),
        }


def iter_voxpopuli(config, split, stats, accents_seen, accents=(), max_scan=None,
                   revision=VOXPOPULI_REVISION):
    files = split_parquet_files(VOXPOPULI_REPO, config, split, revision)
    paths = download_parquet(VOXPOPULI_REPO, files, revision)
    required = ["audio", "speaker_id", "normalized_text", "audio_id"]
    if config == "en_accented":
        required.append("accent")
    _require_fields(_schema_names(paths), required, f"{VOXPOPULI_REPO}/{config}")
    language = "en" if config == "en_accented" else config
    for i, row in enumerate(_iter_parquet(paths)):
        if max_scan is not None and i >= max_scan:
            break
        stats["rows"] += 1
        accent = _text(row.get("accent"))
        if config == "en_accented":
            if accent:
                accents_seen[accent] += 1
            if accents and accent not in accents:
                stats["skip:accent_filter"] += 1
                continue
            group = f"english_l2_{accent.split('_')[-1]}" if accent else ""
        else:
            group = CONFIG_GROUPS.get(config, f"{config}_read")
        if not group:
            stats["skip:missing_accent"] += 1
            continue
        spk = _text(row.get("speaker_id"))
        ref = _text(row.get("normalized_text"))
        audio_id = _text(row.get("audio_id"))
        if not spk:
            stats["skip:missing_speaker"] += 1
            continue
        if not ref or not audio_id:
            stats["skip:empty_ref"] += 1
            continue
        yield {
            "source": "voxpopuli",
            "speaker_key": f"voxpopuli:{spk}",
            "group": group,
            "split": "test",  # evaluation-only source; never trained on
            "language": language,
            "ref": ref,
            "tag": f"{config}-{audio_id}",
            "audio": row.get("audio"),
        }


def iter_fleurs(config, split, stats, clip_level=False, max_scan=None, revision=None):
    if not clip_level:
        raise ValueError(
            "FLEURS exposes no speaker_id; pass --allow-clip-level to include it "
            "and record that its intervals are clip-level, not speaker-clustered"
        )
    files = split_parquet_files(FLEURS_REPO, config, split, revision)
    paths = download_parquet(FLEURS_REPO, files, revision)
    _require_fields(_schema_names(paths), ("audio", "transcription"),
                    f"{FLEURS_REPO}/{config}")
    code = config.split("_")[0]
    resolve_language(code)  # reject unknown language prefixes before streaming
    group = f"fleurs_{code}"
    split_name = {"test": "test", "validation": "val", "train": "train"}[split]
    for i, row in enumerate(_iter_parquet(paths)):
        if max_scan is not None and i >= max_scan:
            break
        stats["rows"] += 1
        ref = _text(row.get("transcription"))
        if not ref:
            stats["skip:empty_ref"] += 1
            continue
        key = f"fleurs:{config}:{split}:{i}"
        yield {
            "source": "fleurs",
            "speaker_key": key,  # clip-level, flagged in audit
            "group": group,
            "split": split_name,
            "language": code,
            "ref": ref,
            "tag": f"{config}-{split}-{i}",
            "audio": row.get("audio"),
        }


def l2arctic_candidates(rows_by_split):
    """Pool L2-ARCTIC's source splits into one test-only, hash-shuffled list.

    Pure and deterministic: the same rows in any file order produce the same
    candidate order. Pooling is deliberate (24 speakers; a 70/15/15 hash split
    would leave ~3 test speakers, below the 10-speaker floor), and every row is
    test-only because L2-ARCTIC is never trained or validated on.
    """
    candidates = []
    for split in L2ARCTIC_SPLITS:
        for i, row in enumerate(rows_by_split.get(split, [])):
            spk = _text(row.get("speaker_id"))
            ref = _text(row.get("transcript"))
            if not spk or not ref:
                continue
            tag = f"{split}-{spk}-{_text(row.get('utterance_id')) or i}"
            candidates.append({
                "source": "l2arctic",
                "speaker_key": f"l2arctic:{spk}",
                "group": L2ARCTIC_GROUP,
                "split": "test",  # evaluation-only; pooled, never trained on
                "language": L2ARCTIC_LANGUAGE,
                "ref": ref,
                "tag": tag,
                "audio": row.get("audio"),
                "_order": hashlib.sha256(
                    f"l2arctic|{tag}".encode("utf-8")
                ).hexdigest(),
            })
    candidates.sort(key=lambda c: (c["_order"], c["tag"]))
    for candidate in candidates:
        candidate.pop("_order")
    return candidates


def iter_l2arctic(stats, max_scan=None, revision=None):
    paths_by_split = {}
    for split in L2ARCTIC_SPLITS:
        files = split_parquet_files(L2ARCTIC_REPO, "default", split, revision)
        paths_by_split[split] = download_parquet(L2ARCTIC_REPO, files, revision)
    names = _schema_names(paths_by_split["test"])
    _require_fields(names, ("audio", "speaker_id", "transcript", "utterance_id"),
                    L2ARCTIC_REPO)
    rows_by_split = {}
    for split, paths in paths_by_split.items():
        rows_by_split[split] = list(_iter_parquet(paths))
        stats[f"rows:{split}"] += len(rows_by_split[split])
    for i, candidate in enumerate(l2arctic_candidates(rows_by_split)):
        if max_scan is not None and i >= max_scan:
            break
        stats["rows"] += 1
        yield candidate


# ---------------------------------------------------------------------------
# verification and build
# ---------------------------------------------------------------------------

def verify_manifests(dir_path, audio_root=None, min_test_clips=200,
                     min_test_speakers=10):
    """Re-read all three manifests and check the invariants the design relies on.

    Returns (summary, problems). Problems are fatal for a build unless the
    operator explicitly allows thin groups.
    """
    audio_root = audio_root or dir_path
    problems = []
    splits = {}
    seen_ids = {}
    for split in ("train", "val", "test"):
        path = os.path.join(dir_path, f"manifest_{split}.jsonl")
        if not os.path.exists(path):
            problems.append(f"missing {path}")
            splits[split] = []
            continue
        rows = []
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(f"{path}:{lineno}: bad json: {e}")
                    continue
                if set(r) != set(SIX_KEYS):
                    problems.append(f"{path}:{lineno}: wrong keys {sorted(r)}")
                    continue
                uid = r["utt_id"]
                if uid in seen_ids:
                    problems.append(f"duplicate utt_id {uid!r} in {split} and {seen_ids[uid]}")
                seen_ids[uid] = split
                p = r["audio"] if os.path.isabs(r["audio"]) else os.path.join(audio_root, r["audio"])
                if not os.path.exists(p):
                    problems.append(f"{split}:{lineno}: audio missing {p}")
                rows.append(r)
        splits[split] = rows

    speaker_split, speaker_group = {}, {}
    groups = defaultdict(lambda: defaultdict(lambda: {"clips": 0, "speakers": set()}))
    for split, rows in splits.items():
        for r in rows:
            spk, grp = r["speaker"], r["group"]
            if spk in speaker_split and speaker_split[spk] != split:
                problems.append(
                    f"speaker {spk!r} in both {speaker_split[spk]} and {split}"
                )
            speaker_split[spk] = split
            if spk in speaker_group and speaker_group[spk] != grp:
                problems.append(f"speaker {spk!r} in both {speaker_group[spk]!r} and {grp!r}")
            speaker_group[spk] = grp
            g = groups[grp][split]
            g["clips"] += 1
            g["speakers"].add(spk)

    for grp, by_split in groups.items():
        t = by_split.get("test")
        if not t or t["clips"] == 0:
            problems.append(f"group {grp!r} has no test clips")
            continue
        if t["clips"] < min_test_clips:
            problems.append(f"group {grp!r}: {t['clips']} test clips < {min_test_clips}")
        if len(t["speakers"]) < min_test_speakers:
            problems.append(
                f"group {grp!r}: {len(t['speakers'])} test speakers < {min_test_speakers}"
            )

    summary = {
        "splits": {s: len(rows) for s, rows in splits.items()},
        "groups": {
            grp: {s: {"clips": v["clips"], "speakers": len(v["speakers"])}
                  for s, v in by_split.items()}
            for grp, by_split in sorted(groups.items())
        },
        "problems": problems,
    }
    return summary, problems


def _atomic_json(path, payload):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _parse_pair(spec, what):
    if ":" not in spec:
        raise ValueError(f"{what} spec must be name:split, got {spec!r}")
    name, split = spec.split(":", 1)
    return name.strip(), split.strip()


def _all_full(alloc, groups, splits=("train", "val", "test")):
    return all(alloc.full(g, s) for g in groups for s in splits)


def build(args):
    caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    want_l2 = bool(getattr(args, "l2arctic", False))
    if not (args.naija or args.voxpopuli or args.fleurs or want_l2):
        raise SystemExit(
            "no sources requested; pass --naija, --voxpopuli, --l2arctic, or --fleurs"
        )

    accents = [a.strip() for a in args.accents.split(",") if a.strip()]
    jobs = []  # (kind, name, split, extra)
    # Accents first: a speaker shared with plain `en` must stay in the accent group.
    for spec in args.voxpopuli:
        name, split = _parse_pair(spec, "--voxpopuli")
        if name == "en_accented":
            jobs.append(("voxpopuli", name, split, accents or None))
    for spec in args.voxpopuli:
        name, split = _parse_pair(spec, "--voxpopuli")
        if name != "en_accented":
            jobs.append(("voxpopuli", name, split, None))
    if want_l2:
        # Pooled: all four source splits, test-only, one group.
        jobs.append(("l2arctic", "pooled", "test", None))
    for spec in args.naija:
        lang, shards = _parse_pair(spec, "--naija")
        if lang not in NAIJA_LANGUAGES:
            raise SystemExit(f"--naija language must be one of {NAIJA_LANGUAGES}, got {lang!r}")
        for shard in [s.strip() for s in shards.split(",") if s.strip()]:
            jobs.append(("naijavoices", lang, shard, None))
    for spec in args.fleurs:
        config, split = _parse_pair(spec, "--fleurs")
        jobs.append(("fleurs", config, split, None))

    out_dir = os.path.abspath(args.out_dir)
    files = [os.path.join(out_dir, f"manifest_{s}.jsonl") for s in ("train", "val", "test")]
    if not args.force and any(os.path.exists(p) for p in files):
        raise SystemExit(
            f"manifests already exist in {out_dir}; pass --force to rebuild"
        )

    if args.dry_run:
        print(json.dumps({"out_dir": out_dir, "caps": caps, "jobs": jobs,
                          "accents": accents, "min_test": [args.min_test_clips,
                                                           args.min_test_speakers]},
                         indent=2))
        return 0

    os.makedirs(out_dir, exist_ok=True)
    staging = os.path.join(out_dir, ".staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)
    audio_dir = os.path.join(out_dir, "audio")
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(audio_dir, split), exist_ok=True)

    alloc = Allocator(caps, min_test_clips=args.min_test_clips,
                      min_test_speakers=args.min_test_speakers)
    stats = defaultdict(Counter)
    accents_seen = Counter()
    revisions = {}
    decode_failures = []
    seen_uids = set()
    handles = {s: open(os.path.join(staging, f"manifest_{s}.jsonl"), "w", encoding="utf-8")
               for s in ("train", "val", "test")}

    try:
        for kind, name, split, extra in jobs:
            key = f"{kind}:{name}:{split}"
            if kind == "naijavoices":
                repo = NAIJA_TEMPLATE.format(language=name, shard=split)
                revisions[repo] = _resolve_revision(repo)
                it = iter_naija(name, split, stats[key], max_scan=args.max_scan,
                                revision=revisions[repo])
                groups = {f"{name}_read"}
                needed = ("train", "val", "test")
            elif kind == "voxpopuli":
                revisions[VOXPOPULI_REPO] = VOXPOPULI_REVISION
                it = iter_voxpopuli(name, split, stats[key], accents_seen,
                                    accents=extra or (), max_scan=args.max_scan,
                                    revision=VOXPOPULI_REVISION)
                if name == "en_accented":
                    chosen = extra or []
                    groups = {f"english_l2_{a.split('_')[-1]}" for a in chosen}
                else:
                    groups = {CONFIG_GROUPS.get(name, f"{name}_read")}
                needed = ("test",)  # evaluation-only source
            elif kind == "l2arctic":
                revisions[L2ARCTIC_REPO] = _resolve_revision(L2ARCTIC_REPO)
                it = iter_l2arctic(stats[key], max_scan=args.max_scan,
                                   revision=revisions[L2ARCTIC_REPO])
                groups = {L2ARCTIC_GROUP}
                needed = ("test",)  # pooled and evaluation-only
            else:
                revisions[FLEURS_REPO] = _resolve_revision(FLEURS_REPO)
                it = iter_fleurs(name, split, stats[key], clip_level=args.allow_clip_level,
                                 max_scan=args.max_scan,
                                 revision=revisions[FLEURS_REPO])
                groups = {f"fleurs_{name.split('_')[0]}"}
                needed = ({"train": "train", "validation": "val", "test": "test"}[split],)

            print(f"building {key} -> groups {sorted(groups)}")
            for cand in it:
                if groups and _all_full(alloc, groups, splits=needed):
                    break
                scanned = stats[key]["rows"]
                if scanned and scanned % 1000 == 0:
                    kept = sum(alloc.counts[g][s] for g in groups
                               for s in ("train", "val", "test"))
                    print(f"  {key}: scanned {scanned}, kept {kept}")
                if not alloc.admit(cand["speaker_key"], cand["group"], cand["split"]):
                    continue
                uid = utt_id(cand["source"], cand["tag"])
                # Threat (Phase 3): two rows mapping to one uid write to one wav
                # path, so the second silently overwrites the first while both
                # stay in the manifest. L2-ARCTIC's utterance_id is only unique
                # within a speaker, which is exactly how this happened. Fail here,
                # before the audio is written, instead of after the whole build.
                if uid in seen_uids:
                    raise RuntimeError(
                        f"utt_id collision {uid!r} from tag {cand['tag']!r}; "
                        "refusing to overwrite audio"
                    )
                seen_uids.add(uid)
                rel = f"audio/{cand['split']}/{uid}.wav"
                dest = os.path.join(out_dir, rel)
                tmp = dest + ".part"
                try:
                    seconds = write_wav(cand["audio"], tmp)
                    if seconds < args.min_seconds:
                        raise ValueError(f"clip is {seconds:.3f}s, below --min-seconds")
                    os.replace(tmp, dest)
                except Exception as e:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    alloc.release(cand["speaker_key"], cand["group"], cand["split"])
                    decode_failures.append({"source": key, "tag": cand["tag"],
                                            "error": f"{type(e).__name__}: {e}"})
                    stats[key]["skip:audio"] += 1
                    if len(decode_failures) > args.max_decode_failures:
                        raise RuntimeError(
                            f"{len(decode_failures)} audio failures (limit "
                            f"{args.max_decode_failures}); last: {decode_failures[-1]}"
                        ) from e
                    continue
                rec = make_record(uid, rel, cand["speaker_key"], cand["group"],
                                  cand["ref"], cand["language"])
                handles[cand["split"]].write(json.dumps(rec, ensure_ascii=False) + "\n")
            if kind == "voxpopuli" and name == "en_accented" and extra:
                # A typo in --accents would otherwise build zero rows silently.
                missing = [a for a in extra if accents_seen.get(a, 0) == 0]
                if missing:
                    raise ValueError(
                        f"accent labels {missing} never appeared; saw "
                        f"{sorted(accents_seen)}"
                    )
        for h in handles.values():
            h.flush()
            os.fsync(h.fileno())
    finally:
        for h in handles.values():
            if not h.closed:
                h.close()

    for s in ("train", "val", "test"):
        os.replace(os.path.join(staging, f"manifest_{s}.jsonl"),
                   os.path.join(out_dir, f"manifest_{s}.jsonl"))
    shutil.rmtree(staging, ignore_errors=True)

    summary, problems = verify_manifests(
        out_dir, min_test_clips=args.min_test_clips,
        min_test_speakers=args.min_test_speakers,
    )
    audit = {
        "out_dir": out_dir,
        "caps": caps,
        "split_policy": (
            "sha256('split|'+speaker_key) % 100: <70 train, <85 val, else test. "
            "Applied to NaijaVoices only. VoxPopuli and L2-ARCTIC are evaluation-only "
            "and stay test. FLEURS follows the source split."
        ),
        "source_order": [f"{k}:{n}:{s}" for k, n, s, _ in jobs],
        "revisions": revisions,
        "l2arctic": {
            "repo": L2ARCTIC_REPO,
            "source_splits": list(L2ARCTIC_SPLITS),
            "pooled_into": "test",
            "group": L2ARCTIC_GROUP,
        },
        "groups": summary["groups"],
        "splits": summary["splits"],
        "dropped": dict(alloc.dropped),
        "stats": {k: dict(v) for k, v in stats.items()},
        "accents_seen": dict(accents_seen),
        "decode_failures": decode_failures[:50],
        "clip_level_groups": sorted({
            g for g in summary["groups"] if g.startswith("fleurs_")
        }),
        "problems": problems,
    }
    _atomic_json(os.path.join(out_dir, "audit.json"), audit)

    print(json.dumps(summary["groups"], indent=2))
    if problems:
        print("\nverification problems:")
        for p in problems:
            print("  -", p)
        if not args.allow_thin_groups:
            print("\nrefusing to call this build usable; pass --allow-thin-groups "
                  "only if you will report the affected groups as descriptive")
            return 2
    print(f"\nwrote manifests + audit.json in {out_dir}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="stream sources and write manifests")
    b.add_argument("--out-dir", default="data")
    b.add_argument("--naija", action="extend", nargs="+", default=[],
                   help="language:shard[,shard], e.g. yoruba:0-0 or hausa:0-0,0-1")
    b.add_argument("--voxpopuli", action="extend", nargs="+", default=[],
                   help="config:split, e.g. en:test or en_accented:test")
    b.add_argument("--l2arctic", action="store_true",
                   help="pool all four L2-ARCTIC source splits into one test-only group")
    b.add_argument("--accents", default=DEFAULT_ACCENTS,
                   help="comma list for en_accented (legacy; L2-ARCTIC is pooled)")
    b.add_argument("--fleurs", action="extend", nargs="+", default=[],
                   help="config:split, e.g. yo_ng:test (requires --allow-clip-level)")
    b.add_argument("--allow-clip-level", action="store_true",
                   help="permit sources without speaker ids; recorded in audit.json")
    b.add_argument("--allow-thin-groups", action="store_true",
                   help="exit 0 even when a test group is below thresholds")
    b.add_argument("--max-train", type=int, default=900)
    b.add_argument("--max-val", type=int, default=100)
    b.add_argument("--max-test", type=int, default=300)
    b.add_argument("--min-test-clips", type=int, default=200)
    b.add_argument("--min-test-speakers", type=int, default=10)
    b.add_argument("--min-seconds", type=float, default=0.10)
    b.add_argument("--max-scan", type=int, default=200000,
                   help="safety cap on rows inspected per source")
    b.add_argument("--max-decode-failures", type=int, default=10)
    b.add_argument("--force", action="store_true")
    b.add_argument("--dry-run", action="store_true")

    v = sub.add_parser("verify", help="re-check existing manifests")
    v.add_argument("--out-dir", default="data")
    v.add_argument("--min-test-clips", type=int, default=200)
    v.add_argument("--min-test-speakers", type=int, default=10)

    a = ap.parse_args(argv)
    if a.cmd == "verify":
        summary, problems = verify_manifests(
            os.path.abspath(a.out_dir),
            min_test_clips=a.min_test_clips, min_test_speakers=a.min_test_speakers,
        )
        print(json.dumps(summary, indent=2))
        if problems:
            for p in problems:
                print("PROBLEM:", p)
            return 1
        return 0
    return build(a)


if __name__ == "__main__":
    raise SystemExit(main())
