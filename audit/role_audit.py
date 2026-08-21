#!/usr/bin/env python3
"""Audit the eight new prompt roles against each other.

Checks, in the order the protocol names them:

exact text        identical raw strings anywhere across roles
normalized text   identical after case/punctuation/whitespace folding
containment       shared 8-gram shingles between every pair of roles
intent identity   content-word Jaccard, catching same-request-different-wording
template family   shared leading 4-word skeletons, catching non-independent
                  prompts generated from one template
source overlap    which corpus feeds which role, and reuse of any source row

Roles that are *designed* to share a stratum (a block's train, discovery and
validation prompts) will share vocabulary. That is intended and is reported
rather than failed; what must be zero is duplication, and what must be small is
containment between roles that the design treats as independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pgeom.provenance import (  # noqa: E402
    INTENT_JACCARD_MAX,
    content_words,
    jaccard,
    normalize_text,
)

SHINGLE_N = 8
TEMPLATE_PREFIX_WORDS = 4

def shingles(norm: str, n: int = SHINGLE_N) -> set[str]:
    w = norm.split()
    if len(w) < n:
        return {norm} if w else set()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}



def run(prompts_path: Path, out_path: Path) -> dict:
    rows = [json.loads(l) for l in prompts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["_norm"] = normalize_text(r["text"])
        r["_sh"] = shingles(r["_norm"])
        r["_cw"] = content_words(r["text"])

    findings: dict = {}

    exact = Counter(r["text"] for r in rows)
    norm = Counter(r["_norm"] for r in rows)
    findings["exact_duplicates"] = sum(v - 1 for v in exact.values() if v > 1)
    findings["normalized_duplicates"] = sum(v - 1 for v in norm.values() if v > 1)

    src_rows = Counter((r["source"], r["source_row"]) for r in rows)
    findings["source_row_reuse"] = sum(v - 1 for v in src_rows.values() if v > 1)

    by_role: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r)

    role_shingles = {role: set().union(*[r["_sh"] for r in rs]) for role, rs in by_role.items()}
    containment = {}
    for a, b in combinations(sorted(role_shingles), 2):
        sa, sb = role_shingles[a], role_shingles[b]
        denom = min(len(sa), len(sb))
        containment[f"{a}|{b}"] = round(len(sa & sb) / denom, 4) if denom else 0.0
    findings["role_pair_shingle_containment"] = dict(sorted(containment.items()))
    findings["max_role_pair_containment"] = max(containment.values()) if containment else 0.0

    # Intent identity: only cross-role pairs matter; within a role, near-synonyms
    # are ordinary corpus texture, but the same request appearing in two roles
    # would leak one role into another.
    intent_hits = []
    by_role_list = sorted(by_role)
    for a, b in combinations(by_role_list, 2):
        for ra in by_role[a]:
            for rb in by_role[b]:
                j = jaccard(ra["_cw"], rb["_cw"])
                if j >= INTENT_JACCARD_MAX:
                    intent_hits.append({
                        "role_a": a, "id_a": ra["id"], "role_b": b, "id_b": rb["id"],
                        "jaccard": round(j, 3),
                    })
    findings["intent_identity_threshold"] = INTENT_JACCARD_MAX
    findings["intent_identity_cross_role_hits"] = len(intent_hits)
    findings["intent_identity_examples"] = intent_hits[:10]

    fam = Counter(" ".join(r["_norm"].split()[:TEMPLATE_PREFIX_WORDS]) for r in rows)
    big = {k: v for k, v in fam.items() if v >= 5}
    findings["template_prefix_words"] = TEMPLATE_PREFIX_WORDS
    findings["distinct_template_prefixes"] = len(fam)
    findings["prompts_in_families_ge5"] = sum(big.values())
    findings["largest_template_families"] = dict(Counter(big).most_common(10))

    findings["role_source_matrix"] = {
        role: dict(Counter(r["source"] for r in rs)) for role, rs in sorted(by_role.items())
    }
    findings["role_sizes"] = {role: len(rs) for role, rs in sorted(by_role.items())}
    findings["total_prompts"] = len(rows)

    findings["pass"] = (
        findings["exact_duplicates"] == 0
        and findings["normalized_duplicates"] == 0
        and findings["source_row_reuse"] == 0
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=Path("prompts/roles/_all_prompts.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("audit/artifacts_full/role_audit.json"))
    args = ap.parse_args()
    f = run(args.prompts, args.out)
    slim = {k: v for k, v in f.items() if k not in ("intent_identity_examples", "role_pair_shingle_containment")}
    print(json.dumps(slim, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
