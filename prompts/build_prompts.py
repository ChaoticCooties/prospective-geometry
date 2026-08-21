#!/usr/bin/env python3
"""Deterministically construct the eight mutually disjoint prompt roles.

Selection is frozen and outcome-blind: within every stratum, prompts are ordered
by a keyed SHA-256 of their source-stable identity and the first N are taken. No
prompt is ever chosen by looking at a model, a score, or a result, and the same
key reproduces the same sets on any machine.

Roles
-----
1 target_fit               fit objective packages A and B
2 target_admission         eight causal-admission blocks, matched harmful/benign
3 engineering              permanently excluded fixture prompts
4 optimizer_train          six search blocks, GCG training prompts
5 graph_discovery          six search blocks, path/orientation choice
6 graph_validation         six search blocks, frozen-choice evaluation
7 benign_control           matched benign negative control
8 behavior_sealed          sealed held-out behavior test

Disjointness is enforced globally: a prompt that has already been consumed by any
earlier role can never be selected again, and the builder asserts this at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgeom.provenance import (  # noqa: E402
    INTENT_JACCARD_MAX,
    IntentIndex,
    content_words,
    normalize_text,
)

#: Frozen selection key. Changing it changes every prompt set, so it is part of
#: the preregistration and is recorded in the manifest.
SELECTION_KEY = "pgeom-prompt-selection-v1"

#: Maximum prompt length in characters. Long prompts inflate sequence length and
#: the attention-routing objective's key span without adding behavioral variety.
MAX_PROMPT_CHARS = 320
MIN_PROMPT_CHARS = 20

SOURCES = {
    "harmfulqa": {
        "hf": ("declare-lab/HarmfulQA", None, "train"),
        "license": "apache-2.0",
        "text_field": "question",
        "stratum_field": "topic",
        "id_field": "id",
        "kind": "harmful",
        "citation": "Bhardwaj & Poria, Red-Teaming Large Language Models using Chain of Utterances (HarmfulQA), 2023",
    },
    "do_not_answer": {
        "hf": ("LibrAI/do-not-answer", None, "train"),
        "license": "apache-2.0",
        "text_field": "question",
        "stratum_field": "risk_area",
        "id_field": "id",
        "kind": "harmful",
        "citation": "Wang et al., Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs, EACL 2024 Findings",
    },
    "orbench_toxic": {
        "hf": ("bench-llms/or-bench", "or-bench-toxic", "train"),
        "license": "cc-by-4.0",
        "text_field": "prompt",
        "stratum_field": "category",
        "id_field": None,
        "kind": "harmful",
        "citation": "Cui et al., OR-Bench: An Over-Refusal Benchmark for Large Language Models, 2024",
    },
    "orbench_hard": {
        "hf": ("bench-llms/or-bench", "or-bench-hard-1k", "train"),
        "license": "cc-by-4.0",
        "text_field": "prompt",
        "stratum_field": "category",
        "id_field": None,
        "kind": "benign_hard",
        "citation": "Cui et al., OR-Bench: An Over-Refusal Benchmark for Large Language Models, 2024",
    },
    "dolly": {
        "hf": ("databricks/databricks-dolly-15k", None, "train"),
        "license": "cc-by-sa-3.0",
        "text_field": "instruction",
        "stratum_field": "category",
        "id_field": None,
        "kind": "benign_utility",
        "citation": "Conover et al., Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM, Databricks 2023",
    },
}

#: Six search blocks. Deliberately spread over two independent corpora and six
#: disjoint semantic strata, so that no single corpus's idiosyncrasies can carry
#: the block-level result on its own.
SEARCH_BLOCKS = [
    ("B1", "harmfulqa", "Science and Technology"),
    ("B2", "harmfulqa", "Business and Economics"),
    ("B3", "harmfulqa", "Health and Medicine"),
    ("B4", "do_not_answer", "Information Hazards"),
    ("B5", "do_not_answer", "Malicious Uses"),
    ("B6", "do_not_answer", "Misinformation Harms"),
]

#: Eight admission blocks, one per OR-Bench category. These categories carry a
#: matched benign counterpart in or-bench-hard-1k, which is what makes a matched
#: harmful/benign contrast possible inside every block.
ADMISSION_BLOCKS = [
    ("A1", "deception"), ("A2", "sexual"), ("A3", "violence"), ("A4", "self-harm"),
    ("A5", "privacy"), ("A6", "unethical"), ("A7", "hate"), ("A8", "illegal"),
]

SIZES = {
    "target_fit_harmful": 128,
    "target_fit_benign": 128,
    "admission_per_arm": 20,       # nomination and validation, harmful and benign
    "engineering_harmful": 12,
    "engineering_benign": 12,
    "search_train": 24,
    "search_discovery": 24,
    "search_validation": 24,
    "benign_control": 24,
    "behavior_sealed_harmful": 16,
    "behavior_sealed_benign": 16,
}

#: Strata reserved for fitting, kept out of every search block so that the
#: objectives are never fitted on the topics they are later searched on.
FIT_TOPICS = [
    "Social Sciences", "Mathematics and Logic", "Literature and Language",
    "Education and Pedagogy", "History and Culture", "Philosophy and Ethics",
    "Geography and Environmental Studies",
]


def keyed_rank(text: str, salt: str) -> str:
    return hashlib.sha256(f"{SELECTION_KEY}:{salt}:{text}".encode("utf-8")).hexdigest()


def tokenizer_round_trips(text: str, tokenizer) -> bool:
    """True when the tokenizer reproduces ``text`` exactly after a round trip.

    Some crowd-sourced prompts contain a space before terminal punctuation. The
    Llama tokenizer encodes " ?" to one token but decodes it without the space,
    so re-encoding yields a different id. A prompt like that can never satisfy
    the canonical replay contract, with any suffix, because the contract is
    exactly `encode(decode(ids)) == ids`. Such prompts are therefore ineligible
    from the start rather than discovered mid-run.
    """
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text


def load_pool(name: str, tokenizer=None):
    from datasets import load_dataset

    if tokenizer is None:
        from pgeom.tokenization import load_tokenizer

        tokenizer = load_tokenizer()

    spec = SOURCES[name]
    repo, cfg, split = spec["hf"]
    ds = load_dataset(repo, cfg, split=split)
    rows = []
    for i, row in enumerate(ds):
        text = (row[spec["text_field"]] or "").strip()
        if not (MIN_PROMPT_CHARS <= len(text) <= MAX_PROMPT_CHARS):
            continue
        if "\n" in text:
            continue
        if not tokenizer_round_trips(text, tokenizer):
            continue
        rows.append({
            "source": name,
            "source_row": i,
            "stratum": str(row[spec["stratum_field"]]),
            "text": text,
        })
    return rows


class Allocator:
    """Hands out prompts, guaranteeing global disjointness across roles."""

    def __init__(self, pools: dict[str, list[dict]]):
        self.pools = pools
        self.used: set[tuple[str, int]] = set()
        self.by_norm: dict[str, tuple[str, int]] = {}
        # Rejects a candidate that restates any already-allocated prompt. Without
        # this, a block's discovery and validation slices can end up holding the
        # same request in different words, which would let a frozen choice be
        # "validated" on the prompt it was made from.
        self.intent = IntentIndex(INTENT_JACCARD_MAX)

    def take(self, source: str, stratum: str | None, n: int, salt: str) -> list[dict]:
        cands = [
            r for r in self.pools[source]
            if (stratum is None or r["stratum"] == stratum)
            and (r["source"], r["source_row"]) not in self.used
        ]
        cands.sort(key=lambda r: keyed_rank(r["text"], salt))
        chosen: list[dict] = []
        for r in cands:
            if len(chosen) == n:
                break
            norm = normalize_text(r["text"])
            if norm in self.by_norm:
                continue  # identical after normalization
            words = content_words(r["text"])
            if self.intent.conflicts(words):
                continue  # same request, different wording
            self.by_norm[norm] = (r["source"], r["source_row"])
            self.intent.add(words)
            self.used.add((r["source"], r["source_row"]))
            chosen.append(r)
        if len(chosen) < n:
            raise RuntimeError(
                f"stratum exhausted: source={source} stratum={stratum!r} "
                f"needed {n}, got {len(chosen)}. The protocol forbids silently "
                f"substituting prompts; widen the source allocation instead."
            )
        return chosen


def emit(rows, role, block, arm, kind, out):
    for r in rows:
        pid = hashlib.sha256(
            f"{role}|{block}|{arm}|{r['source']}|{r['source_row']}".encode()
        ).hexdigest()[:16]
        out.append({
            "id": pid,
            "role": role,
            "block": block,
            "arm": arm,
            "kind": kind,
            "source": r["source"],
            "source_row": r["source_row"],
            "stratum": r["stratum"],
            "license": SOURCES[r["source"]]["license"],
            "text": r["text"],
        })


def build(out_dir: Path) -> dict:
    from pgeom.tokenization import load_tokenizer

    tokenizer = load_tokenizer()
    pools = {name: load_pool(name, tokenizer) for name in SOURCES}
    alloc = Allocator(pools)
    rows: list[dict] = []

    # Role 1: target fit -------------------------------------------------
    fit_h = []
    per_topic = SIZES["target_fit_harmful"] // len(FIT_TOPICS) + 1
    for topic in FIT_TOPICS:
        fit_h += alloc.take("harmfulqa", topic, per_topic, f"fit:{topic}")
    emit(fit_h[: SIZES["target_fit_harmful"]], "target_fit", "-", "harmful", "harmful", rows)
    fit_b = alloc.take("dolly", None, SIZES["target_fit_benign"], "fit:benign")
    emit(fit_b, "target_fit", "-", "benign", "benign_utility", rows)

    # Role 2: target admission, eight blocks -----------------------------
    n = SIZES["admission_per_arm"]
    for blk, cat in ADMISSION_BLOCKS:
        for arm, src, kind in (
            ("nom_harmful", "orbench_toxic", "harmful"),
            ("val_harmful", "orbench_toxic", "harmful"),
            ("nom_benign", "orbench_hard", "benign_hard"),
            ("val_benign", "orbench_hard", "benign_hard"),
        ):
            emit(alloc.take(src, cat, n, f"adm:{blk}:{arm}"),
                 "target_admission", blk, arm, kind, rows)
        emit(alloc.take("dolly", None, n, f"adm:{blk}:util"),
             "target_admission", blk, "benign_utility", "benign_utility", rows)

    # Role 3: engineering only -------------------------------------------
    emit(alloc.take("do_not_answer", "Human-Chatbot Interaction Harms",
                    SIZES["engineering_harmful"], "eng:h"),
         "engineering", "-", "harmful", "harmful", rows)
    emit(alloc.take("dolly", None, SIZES["engineering_benign"], "eng:b"),
         "engineering", "-", "benign", "benign_utility", rows)

    # Roles 4-6 and 8: six search blocks ---------------------------------
    for blk, src, stratum in SEARCH_BLOCKS:
        emit(alloc.take(src, stratum, SIZES["search_train"], f"train:{blk}"),
             "optimizer_train", blk, "harmful", "harmful", rows)
        emit(alloc.take(src, stratum, SIZES["search_discovery"], f"disc:{blk}"),
             "graph_discovery", blk, "harmful", "harmful", rows)
        emit(alloc.take(src, stratum, SIZES["search_validation"], f"val:{blk}"),
             "graph_validation", blk, "harmful", "harmful", rows)
        emit(alloc.take(src, stratum, SIZES["behavior_sealed_harmful"], f"seal:{blk}"),
             "behavior_sealed", blk, "harmful", "harmful", rows)
        emit(alloc.take("orbench_hard", None, SIZES["behavior_sealed_benign"], f"sealb:{blk}"),
             "behavior_sealed", blk, "benign", "benign_hard", rows)
        emit(alloc.take("orbench_hard", None, SIZES["benign_control"], f"bctl:{blk}"),
             "benign_control", blk, "benign", "benign_hard", rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    by_role = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r)
    for role, items in sorted(by_role.items()):
        path = out_dir / f"{role}.jsonl"
        path.write_text(
            "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in items),
            encoding="utf-8",
        )

    ids = [r["id"] for r in rows]
    norms = [normalize_text(r["text"]) for r in rows]
    assert len(set(ids)) == len(ids), "duplicate prompt id"
    assert len(set(norms)) == len(norms), "duplicate normalized prompt across roles"
    for r in rows:
        assert tokenizer_round_trips(r["text"], tokenizer), (
            f"prompt {r['id']} does not survive a tokenizer round trip; it could "
            "never satisfy the canonical replay contract"
        )
    summary_intent = IntentIndex(INTENT_JACCARD_MAX)
    for r in rows:
        w = content_words(r["text"])
        assert not summary_intent.conflicts(w), f"intent-identical prompt survived: {r['id']}"
        summary_intent.add(w)

    summary = {
        "selection_key": SELECTION_KEY,
        "total_prompts": len(rows),
        "per_role": {k: len(v) for k, v in sorted(by_role.items())},
        "per_role_source": {
            k: dict(Counter(r["source"] for r in v)) for k, v in sorted(by_role.items())
        },
        "search_blocks": [list(b) for b in SEARCH_BLOCKS],
        "admission_blocks": [list(b) for b in ADMISSION_BLOCKS],
        "licenses": {k: v["license"] for k, v in SOURCES.items()},
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "intent_jaccard_max": INTENT_JACCARD_MAX,
        "tokenizer_round_trip_required": True,
    }
    (out_dir / "_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    combined = out_dir / "_all_prompts.jsonl"
    combined.write_text(
        "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "roles")
    args = ap.parse_args()
    summary = build(args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
