"""Provenance hashing and manifest records.

Everything that can change a scientific result gets a hash: prompt files, frozen
configuration, copied infrastructure, and run artifacts. Hashes are over bytes
for files and over canonical JSON for structures, so a record is reproducible
from a clean checkout.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .firewall import guard_path

CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], *, allow_archive: bool = False) -> str:
    """SHA-256 of a file's bytes.

    ``allow_archive`` exists only for the corpus-overlap audit and the
    infrastructure-copy recorder, which are the two permitted archive readers.
    Scientific code must leave it False so the firewall applies.
    """
    p = Path(path) if allow_archive else guard_path(path, role="hashed file")
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 text."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def normalize_text(text: str) -> str:
    """Normalization used by the overlap audit's `normalized` comparison.

    Case-folded, Unicode-NFKC, punctuation-stripped, whitespace-collapsed. Two
    prompts that differ only in casing, spacing, or trailing punctuation collide
    here on purpose: near-duplicates of an archived prompt must not pass as new.
    """
    import re
    import unicodedata

    t = unicodedata.normalize("NFKC", text).casefold()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def git_commit(repo: str | os.PathLike[str] = ".", *, allow_archive: bool = False) -> str | None:
    """Current commit of ``repo``, or None if there is no commit yet."""
    p = Path(repo) if allow_archive else guard_path(repo, role="git repo")
    try:
        out = subprocess.run(
            ["git", "-C", str(p), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass(frozen=True)
class ImportedInfrastructure:
    """Record for one file of generic infrastructure copied from the archive.

    The protocol permits copying generic implementation infrastructure only, and
    only with source path, source commit, and file SHA-256 recorded, plus fresh
    tests. Study-specific constants, fitted artifacts, prompt manifests, gates,
    and result readers are never eligible.
    """

    dest_path: str
    source_path: str
    source_commit: str
    source_sha256: str
    dest_sha256: str
    category: str
    modified: bool
    fresh_tests: tuple[str, ...]
    rationale: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


#: Categories of infrastructure the protocol allows importing.
ALLOWED_INFRA_CATEGORIES = frozenset({
    "model_loading",
    "chat_token_layout",
    "canonical_replay_check",
    "generic_gcg_engine",
    "budget_reaper",
    "provenance_hashing",
    "testing_plotting_helpers",
})


def validate_infra_category(category: str) -> str:
    if category not in ALLOWED_INFRA_CATEGORIES:
        raise ValueError(
            f"infrastructure category {category!r} is not importable; allowed: "
            + ", ".join(sorted(ALLOWED_INFRA_CATEGORIES))
        )
    return category


# --- Shared text-identity helpers -------------------------------------------
# Single source of truth for the prompt builder and the role audit. Keeping the
# threshold and the stopword list in one place stops the builder's rule and the
# auditor's rule from silently drifting apart.

#: Content-word Jaccard at or above which two prompts count as the same request.
INTENT_JACCARD_MAX = 0.60

INTENT_STOPWORDS = frozenset("""
a an the and or but if of to in on for with without at by from as is are was were be been being
how what why when where who whom which can could would should will shall may might do does did
i me my we our you your they them their it its this that these those some any not no s t
""".split())


def content_words(text: str) -> frozenset[str]:
    """Content words of a prompt, used for intent/focus identity comparison."""
    return frozenset(
        w for w in normalize_text(text).split()
        if w not in INTENT_STOPWORDS and len(w) > 2
    )


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class IntentIndex:
    """Inverted index for near-duplicate intent lookup during allocation.

    Comparing each candidate against every allocated prompt is quadratic; only
    prompts sharing at least one content word can reach the Jaccard threshold,
    so the index restricts the comparison to those.
    """

    def __init__(self, threshold: float = INTENT_JACCARD_MAX):
        self.threshold = threshold
        self._by_word: dict[str, list[int]] = {}
        self._sets: list[frozenset[str]] = []

    def max_similarity(self, words: frozenset[str]) -> float:
        if not words:
            return 0.0
        seen: set[int] = set()
        for w in words:
            seen.update(self._by_word.get(w, ()))
        best = 0.0
        for i in seen:
            best = max(best, jaccard(words, self._sets[i]))
        return best

    def conflicts(self, words: frozenset[str]) -> bool:
        return self.max_similarity(words) >= self.threshold

    def add(self, words: frozenset[str]) -> None:
        idx = len(self._sets)
        self._sets.append(words)
        for w in words:
            self._by_word.setdefault(w, []).append(idx)
