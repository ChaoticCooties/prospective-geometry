#!/usr/bin/env python3
"""Read-only corpus-overlap audit against the archived Renegade experiments.

This is one of exactly three permitted forms of archive access, and the only one
that reads archive *content*. It exists to prove that the new study's prompts are
unused. It therefore obeys a hard output contract:

    Raw archive text never leaves this program. Only hashes and counts do.

Strings are hashed three ways so the audit can answer three different questions
without ever retaining text:

  exact       sha256 of the raw string           -> "is this the same prompt?"
  normalized  sha256 of case/punct/space-folded  -> "is this a trivial restyling?"
  shingle     sha256 of each normalized n-gram   -> "is this contained in, or a
                                                     near-duplicate of, an
                                                     archived prompt?"

Subcommands
-----------
harvest   scan the archive, write hash sets + counts
check     score candidate new prompts against a harvested hash set
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

# The audit deliberately does not import pgeom.firewall: it is the sanctioned
# archive reader, and importing the guard would only invite someone to disable it.

SHINGLE_N = 8
MIN_LEN = 15
MAX_LEN = 4000
MAX_JSON_BYTES = 200 * 1024 * 1024
#: Strings longer than this are hashed exactly/normalized but not shingled.
#: Prompts are short; multi-KB strings are generations, and shingling them
#: exhausts memory without improving the containment test for prompts.
SHINGLE_MAX_LEN = 600
TEXT_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt", ".md", ".py", ".yaml", ".yml"}

#: Public corpora whose names, if present in archive paths or text, indicate the
#: archive already drew prompts from that source.
KNOWN_CORPORA = (
    "advbench", "harmbench", "jailbreakbench", "jbb-behaviors", "jbb_behaviors",
    "strongreject", "strong_reject", "maliciousinstruct", "malicious_instruct",
    "alpaca", "dolly", "or-bench", "orbench", "xstest", "sorry-bench", "sorrybench",
    "beavertails", "do-not-answer", "donotanswer", "tinystories", "mmlu", "arc",
    "truthfulqa", "wildguard", "wildjailbreak", "aart", "saladbench", "salad-bench",
    "toxicchat", "hh-rlhf", "anthropic-red-team", "red_team_attempts",
)

_word_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).casefold()
    t = _punct_re.sub(" ", t)
    return _word_re.sub(" ", t).strip()


def h64(data: str) -> int:
    """64-bit truncation of sha256. Truncation only ever creates false overlap
    reports, never false clearances, so it is safe for a fail-closed audit."""
    return int.from_bytes(hashlib.sha256(data.encode("utf-8")).digest()[:8], "big")


def shingles(norm: str, n: int = SHINGLE_N) -> set[int]:
    words = norm.split()
    if len(words) < n:
        return {h64(norm)} if words else set()
    return {h64(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def plausible_prompt(s: str) -> bool:
    """Cheap filter for natural-language strings.

    Over-inclusive by design: hashing extra archive text can only make the audit
    stricter. The filter exists to keep base64 blobs and tensors out of memory.
    """
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    if " " not in s:
        return False
    letters = sum(ch.isalpha() for ch in s)
    if letters < 0.5 * len(s):
        return False
    return True


def iter_strings_json(obj):
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            yield cur
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def extract_from_file(path: Path):
    """Yield candidate natural-language strings from one file."""
    suffix = path.suffix.lower()
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size == 0 or size > MAX_JSON_BYTES:
        return
    try:
        if suffix in {".jsonl", ".ndjson"}:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield from iter_strings_json(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        yield line
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                try:
                    yield from iter_strings_json(json.load(fh))
                except (json.JSONDecodeError, ValueError, RecursionError):
                    return
        elif suffix in {".csv", ".tsv"}:
            delim = "\t" if suffix == ".tsv" else ","
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
                for row in csv.reader(fh, delimiter=delim):
                    yield from row
        elif suffix == ".py":
            src = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                return
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    yield node.value
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            yield text
            for line in text.splitlines():
                yield line
    except (OSError, UnicodeError):
        return



class HashAccumulator:
    """Memory-bounded unique-hash set backed by numpy.

    A Python ``set`` of tens of millions of ints costs gigabytes in object
    overhead. This buffers raw uint64 values and folds them into a sorted,
    deduplicated array whenever the buffer fills, which keeps peak memory at
    8 bytes per unique hash.
    """

    FLUSH_AT = 4_000_000

    def __init__(self) -> None:
        self._sorted = np.empty(0, dtype=np.uint64)
        self._buf: list[int] = []

    def add(self, value: int) -> None:
        self._buf.append(value)
        if len(self._buf) >= self.FLUSH_AT:
            self.flush()

    def update(self, values) -> None:
        self._buf.extend(values)
        if len(self._buf) >= self.FLUSH_AT:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        incoming = np.fromiter(self._buf, dtype=np.uint64, count=len(self._buf))
        self._buf.clear()
        self._sorted = np.unique(np.concatenate([self._sorted, incoming]))

    def __contains__(self, value: int) -> bool:
        self.flush()
        idx = np.searchsorted(self._sorted, np.uint64(value))
        return bool(idx < self._sorted.size and self._sorted[idx] == np.uint64(value))

    def __len__(self) -> int:
        self.flush()
        return int(self._sorted.size)

    def to_array(self) -> "np.ndarray":
        self.flush()
        return self._sorted


def _write_hash_array(path: Path, arr) -> None:
    with gzip.open(path, "wt", encoding="ascii") as fh:
        for v in arr.tolist():
            fh.write(f"{v:016x}\n")


def _read_hash_array(path: Path):
    with gzip.open(path, "rt", encoding="ascii") as fh:
        vals = [int(line, 16) for line in (l.strip() for l in fh) if line]
    return np.array(sorted(vals), dtype=np.uint64)


def _member(arr, value: int) -> bool:
    idx = np.searchsorted(arr, np.uint64(value))
    return bool(idx < arr.size and arr[idx] == np.uint64(value))


def harvest(archive_root: Path, out_dir: Path, *, include_git_history: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    exact = HashAccumulator()
    norm = HashAccumulator()
    shing = HashAccumulator()
    seen_norm: set[int] = set()
    corpora = Counter()
    per_dir = Counter()
    files_scanned = 0
    strings_seen = 0

    def ingest(s: str, bucket: str | None) -> bool:
        """Hash one candidate string into all three views.

        Shingling is done once per distinct normalized string and only for
        prompt-length text: archives are dominated by repeated generations, and
        re-shingling them adds cost without adding coverage of prompts.
        """
        nonlocal strings_seen
        if not plausible_prompt(s):
            return False
        strings_seen += 1
        if bucket:
            per_dir[bucket] += 1
        exact.add(h64(s))
        n = normalize_text(s)
        if not n:
            return True
        nh = h64(n)
        norm.add(nh)
        if nh not in seen_norm:
            seen_norm.add(nh)
            if len(s) <= SHINGLE_MAX_LEN:
                shing.update(shingles(n))
        low_s = s.lower()
        for name in KNOWN_CORPORA:
            if name in low_s:
                corpora[name] += 1
        return True

    for path in sorted(archive_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if ".git/" in str(path):
            continue
        files_scanned += 1
        try:
            rel_top = path.relative_to(archive_root).parts[0]
        except ValueError:
            rel_top = "?"
        low = str(path).lower()
        for name in KNOWN_CORPORA:
            if name in low:
                corpora[name] += 1
        for s in extract_from_file(path):
            ingest(s, rel_top)

    history_strings = 0
    if include_git_history:
        history_strings = _harvest_git_history(archive_root, ingest)

    _write_hash_array(out_dir / "archive_exact.h64.gz", exact.to_array())
    _write_hash_array(out_dir / "archive_normalized.h64.gz", norm.to_array())
    _write_hash_array(out_dir / "archive_shingle.h64.gz", shing.to_array())

    counts = {
        "audit_kind": "archive_corpus_overlap_harvest",
        "shingle_n": SHINGLE_N,
        "shingle_max_len": SHINGLE_MAX_LEN,
        "hash": "sha256-truncated-64bit",
        "files_scanned": files_scanned,
        "candidate_strings": strings_seen,
        "history_strings": history_strings,
        "unique_exact": len(exact),
        "unique_normalized": len(norm),
        "unique_shingles": len(shing),
        "strings_per_archive_dir": dict(sorted(per_dir.items())),
        "known_corpus_markers": dict(sorted(corpora.items())),
        "include_git_history": include_git_history,
        "note": "Hash sets only. No archive text is emitted by this program.",
    }
    (out_dir / "archive_counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return counts


def _harvest_git_history(archive_root: Path, ingest) -> int:
    """Hash text blobs reachable from any ref in each archived repository.

    Working-tree scanning alone would miss a prompt file that was committed and
    later deleted; such a prompt was still used, so it must still be excluded.
    """
    import subprocess

    seen = 0
    for repo in sorted(p.parent for p in archive_root.glob("*/.git")):
        try:
            listing = subprocess.run(
                ["git", "-C", str(repo), "rev-list", "--objects", "--all"],
                capture_output=True, text=True, timeout=900, check=False,
            ).stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            continue
        wanted = []
        for line in listing:
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            sha, name = parts
            if Path(name).suffix.lower() in TEXT_SUFFIXES:
                wanted.append(sha)
        for i in range(0, len(wanted), 500):
            batch = wanted[i:i + 500]
            try:
                proc = subprocess.run(
                    ["git", "-C", str(repo), "cat-file", "--batch"],
                    input="\n".join(batch) + "\n",
                    capture_output=True, text=True, timeout=1800, check=False,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            for chunk in proc.stdout.split("\n"):
                if ingest(chunk, None):
                    seen += 1
    return seen


def check(candidates_path: Path, hash_dir: Path, out_path: Path, containment_frac: float) -> dict:
    exact = _read_hash_array(hash_dir / "archive_exact.h64.gz")
    norm = _read_hash_array(hash_dir / "archive_normalized.h64.gz")
    shing = _read_hash_array(hash_dir / "archive_shingle.h64.gz")

    rows = [
        json.loads(l)
        for l in candidates_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    results = []
    flagged = Counter()
    for row in rows:
        text = row["text"]
        n = normalize_text(text)
        sh = shingles(n)
        if sh and shing.size:
            arr = np.fromiter(sh, dtype=np.uint64, count=len(sh))
            idx = np.clip(np.searchsorted(shing, arr), 0, shing.size - 1)
            overlap = int((shing[idx] == arr).sum()) / len(sh)
        else:
            overlap = 0.0
        verdict = {
            "id": row["id"],
            "role": row.get("role"),
            "exact_hit": _member(exact, h64(text)),
            "normalized_hit": _member(norm, h64(n)),
            "containment_frac": round(overlap, 4),
            "containment_hit": overlap >= containment_frac,
        }
        verdict["clean"] = not (
            verdict["exact_hit"] or verdict["normalized_hit"] or verdict["containment_hit"]
        )
        for k in ("exact_hit", "normalized_hit", "containment_hit"):
            if verdict[k]:
                flagged[k] += 1
        results.append(verdict)

    summary = {
        "audit_kind": "archive_corpus_overlap_check",
        "candidates": len(results),
        "clean": sum(r["clean"] for r in results),
        "flagged": dict(flagged),
        "containment_threshold": containment_frac,
        "shingle_n": SHINGLE_N,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "per_prompt": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest")
    h.add_argument("--archive-root", required=True, type=Path)
    h.add_argument("--out-dir", required=True, type=Path)
    h.add_argument("--include-git-history", action="store_true")

    c = sub.add_parser("check")
    c.add_argument("--candidates", required=True, type=Path)
    c.add_argument("--hash-dir", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--containment-frac", type=float, default=0.5)

    args = ap.parse_args(argv)
    if args.cmd == "harvest":
        counts = harvest(args.archive_root, args.out_dir, include_git_history=args.include_git_history)
        print(json.dumps(counts, indent=2, sort_keys=True))
    else:
        summary = check(args.candidates, args.hash_dir, args.out, args.containment_frac)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
