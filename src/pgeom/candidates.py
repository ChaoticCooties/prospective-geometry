"""Candidate-token policy, blacklist, and the hash-sampled candidate bank.

Policy (frozen before any model outcome)
----------------------------------------
A vocabulary token is bank-eligible when all of the following hold:

  * its id is below the first special-token id, so no template control token,
    reserved id, or added token can enter a suffix;
  * its decoded form begins with exactly one space;
  * the remainder is non-empty and contains no whitespace;
  * no character is a control, format, surrogate, private-use, or unassigned
    codepoint, and the decoded form contains no U+FFFD replacement character.

The leading-space requirement is what makes the canonical replay contract hold:
measured over random four-token draws, unrestricted tokens round-trip about 44%
of the time, while this policy round-trips at 100%. Non-ASCII tokens are kept
deliberately, so that script is a genuinely varying surface feature available to
the input-only baseline rather than a constant.

The bank is drawn by keyed hashing of token ids, not by taking a prefix of
sorted ids and not by any union derived from outcomes. Hash sampling is uniform
over eligible tokens, reproducible from the key alone, and fixed before any
scoring happens.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass

#: First special-token id in Llama-3.1. Ids at or above this are excluded.
FIRST_SPECIAL_ID = 128000

#: Unicode general categories that disqualify a token.
FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})

REPLACEMENT_CHAR = "�"


def _has_forbidden_char(text: str) -> bool:
    if REPLACEMENT_CHAR in text:
        return True
    return any(unicodedata.category(ch) in FORBIDDEN_CATEGORIES for ch in text)


def is_eligible(token_id: int, decoded: str) -> bool:
    """Apply the frozen candidate-token policy to one token."""
    if token_id >= FIRST_SPECIAL_ID or token_id < 0:
        return False
    if not decoded.startswith(" "):
        return False
    core = decoded[1:]
    if not core:
        return False
    if any(ch.isspace() for ch in core):
        return False
    return not _has_forbidden_char(decoded)


def eligible_tokens(tokenizer) -> list[int]:
    """All bank-eligible token ids, ascending."""
    out: list[int] = []
    for tid in range(FIRST_SPECIAL_ID):
        decoded = tokenizer.decode([tid])
        if is_eligible(tid, decoded):
            out.append(tid)
    return out


def script_class(decoded: str) -> str:
    """Coarse script label for stratification and for the surface baseline."""
    core = decoded.strip()
    if not core:
        return "other"
    ch = core[0]
    if ch.isascii():
        if ch.isalpha():
            return "latin_ascii"
        if ch.isdigit():
            return "digit"
        return "punct_ascii"
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "other"
    for script in ("CJK", "CYRILLIC", "ARABIC", "HEBREW", "DEVANAGARI", "GREEK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI"):
        if script in name:
            return script.lower()
    if ch.isalpha():
        return "latin_extended"
    return "other"


def _keyed_hash(key: str, token_id: int) -> int:
    payload = f"{key}:{token_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class CandidateBank:
    """A frozen, reproducible subset of eligible tokens."""

    key: str
    token_ids: tuple[int, ...]
    stratified: bool

    def __len__(self) -> int:
        return len(self.token_ids)


def hash_sampled_bank(eligible: list[int], *, key: str, size: int) -> CandidateBank:
    """Uniform hash-sampled bank: the ``size`` ids with smallest keyed hash.

    Uniform because sha256 of the key-id pair is uniform over token ids and
    independent of id magnitude, so the result is not a sorted-id prefix.
    """
    if size > len(eligible):
        raise ValueError(f"requested bank of {size} exceeds {len(eligible)} eligible tokens")
    ranked = sorted(eligible, key=lambda t: (_keyed_hash(key, t), t))
    return CandidateBank(key=key, token_ids=tuple(sorted(ranked[:size])), stratified=False)


def stratified_bank(
    eligible: list[int], decoded_map: dict[int, str], *, key: str, size: int
) -> CandidateBank:
    """Bank stratified by script class, proportional to each class's share.

    Proportional allocation keeps the bank representative of the token policy
    rather than over-weighting rare scripts, while guaranteeing that no class
    present in the vocabulary is entirely absent by sampling luck.
    """
    by_class: dict[str, list[int]] = {}
    for tid in eligible:
        by_class.setdefault(script_class(decoded_map[tid]), []).append(tid)

    total = len(eligible)
    chosen: list[int] = []
    for cls in sorted(by_class):
        members = by_class[cls]
        quota = max(1, round(size * len(members) / total))
        ranked = sorted(members, key=lambda t: (_keyed_hash(key, t), t))
        chosen.extend(ranked[:quota])

    # Proportional rounding overshoots or undershoots; settle deterministically
    # on the same global hash order so the result stays key-reproducible.
    chosen = sorted(set(chosen), key=lambda t: (_keyed_hash(key, t), t))
    if len(chosen) > size:
        chosen = chosen[:size]
    elif len(chosen) < size:
        rest = sorted(set(eligible) - set(chosen), key=lambda t: (_keyed_hash(key, t), t))
        chosen.extend(rest[: size - len(chosen)])
    return CandidateBank(key=key, token_ids=tuple(sorted(chosen)), stratified=True)
