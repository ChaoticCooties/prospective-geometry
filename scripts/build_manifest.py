#!/usr/bin/env python3
"""Build the machine-readable freeze manifest.

The manifest is the object a reviewer checks a later run against: it pins the
model, the frozen parameters, and the SHA-256 of every file that can change a
result. Regenerating it after any edit and diffing is how an unintended change
gets caught.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pgeom import admission, candidates, geometry, nullsim, tokenization  # noqa: E402
from pgeom.provenance import git_commit, sha256_file, sha256_obj  # noqa: E402

TRACKED_GLOBS = (
    "PROTOCOL.md", "CLAIMS.md", "README.md",
    "src/pgeom/*.py", "tests/*.py",
    "prompts/build_prompts.py", "prompts/PROVENANCE.md", "prompts/roles/*.jsonl",
    "audit/*.py", "audit/artifacts_full/*.json", "audit/artifacts_full/*.gz",
    "scripts/*.py", "reports/*.json", "reports/*.md",
)


def tracked_files() -> dict[str, str]:
    out: dict[str, str] = {}
    for pattern in TRACKED_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                out[str(path.relative_to(ROOT))] = sha256_file(path)
    return out


def frozen_parameters() -> dict:
    return {
        "model": {
            "id": tokenization.MODEL_ID,
            "revision": tokenization.MODEL_REVISION,
            "precision": "bfloat16 weights, float32 objective accumulation",
            "attn_implementation": "eager",
            "device": "cuda",
        },
        "template": {
            "pinned_date_string": tokenization.PINNED_DATE_STRING,
            "probe_text": tokenization.TEMPLATE_PROBE_TEXT,
            "probe_sha256": "c19ba073838b8d8e3381838719014dbb1e0fa0cb5ceed1138191788b302e5f5b",
            "suffix_separator": tokenization.SUFFIX_SEPARATOR,
        },
        "suffix": {
            "width": tokenization.SUFFIX_WIDTH,
            "replay_contract": "encode(decode(full_ids)) == full_ids, enforced every phase",
        },
        "token_policy": {
            "first_special_id": candidates.FIRST_SPECIAL_ID,
            "requires_single_leading_space": True,
            "forbidden_unicode_categories": sorted(candidates.FORBIDDEN_CATEGORIES),
            "eligible_token_count": 56547,
            "measured_canonical_rate_policy": 1.000,
            "measured_canonical_rate_unrestricted": 0.445,
        },
        "objectives": {
            "A": {
                "kind": "signed residual-stream refusal direction",
                "layer": 14,
                "site": "residual stream after transformer block 14",
                "position": "assistant-prefill readout, final prompt token",
                "fit": "unit-normalized difference of class-conditional means on target_fit",
                "sign": "larger projection means more refusal; optimizer minimizes",
            },
            "B": {
                "kind": "fixed-k attention-routing objective",
                "k": 8,
                "site": "attention probabilities, readout query row, instruction key span",
                "fit": "top-8 heads by Cohen's d of harmful-minus-benign instruction mass",
                "tie_break": "ascending (layer, head)",
                "aggregation": "mean over the 8 fixed heads",
                "sign": "larger mass means more instruction routing; optimizer minimizes",
            },
        },
        "admission": {
            "blocks_total": admission.BLOCKS_TOTAL,
            "blocks_required": admission.BLOCKS_REQUIRED,
            "min_causal_effect_pp": admission.MIN_CAUSAL_EFFECT_PP,
            "min_placebo_ratio": admission.MIN_PLACEBO_RATIO,
            "max_benign_utility_loss_pp": admission.MAX_BENIGN_UTILITY_LOSS_PP,
            "min_heldout_auroc": admission.MIN_HELDOUT_AUROC,
            "max_abs_spearman": admission.MAX_ABS_SPEARMAN,
            "max_median_grad_cosine": admission.MAX_MEDIAN_GRAD_COSINE,
            "min_diagonal_dominance": admission.MIN_DIAGONAL_DOMINANCE,
            "max_matrix_condition": admission.MAX_MATRIX_CONDITION,
            "hard_cap_usd": 5.0,
            "paid_judge_api": False,
        },
        "search": {
            "blocks": 6, "objectives": 2, "restarts": 2, "total_searches": 24,
            "updates": 100, "early_stopping": False,
            "gradient_topk_per_position": 128,
            "candidates_per_update": 256,
            "optimizer_prompts_per_update": 12,
            "seed_rule": "9000 + 10*block_index + restart_index; 200-205 never used",
            "order_balancing": "A first on even-indexed pairs, B first on odd-indexed pairs",
            "checkpoint": "every update; canonical token IDs saved at each",
            "primary_endpoint": (
                "earliest best canonical own-training-loss suffix through update 99; "
                "tie rule: lowest loss, then earliest update, then lexicographically "
                "smallest token ID tuple"
            ),
        },
        "geometry": {
            "cube_vertices": geometry.N_VERTICES,
            "monotone_paths": 24,
            "blocks_required": geometry.GEOMETRY_BLOCKS_REQUIRED,
            "min_separation": geometry.MIN_SEPARATION,
            "statistic": "mean interior banked-gain minus paid-cost, normalized per objective",
            "block_statistic": "G_b = min(m_cross, m_cross - m_same, m_cross - m_control)",
            "ties": "fail",
            "unit_of_analysis": "block",
            "control_classes": [
                "same-objective restart faces",
                "surface-matched random endpoints",
            ],
        },
        "behavior": {
            "min_effect_pp": 15.0,
            "min_coherent_on_topic": 0.80,
            "path_nodes": 5,
            "reuse_archived_taxonomy": False,
        },
        "nullsim": {
            "max_false_pass": nullsim.MAX_FALSE_PASS,
            "n_control_faces": nullsim.N_CONTROL_FACES,
        },
        "budget": {
            "phase1_hard_cap_usd": 5.0,
            "target_usd": [25.0, 30.0],
            "hard_cap_usd": 30.0,
            "phase2_continue_threshold_p90_usd": 30.0,
        },
    }


def summaries() -> dict:
    def load(rel):
        p = ROOT / rel
        return json.loads(p.read_text()) if p.exists() else None

    overlap = load("audit/artifacts_full/new_prompt_overlap.json")
    role = load("audit/artifacts_full/role_audit.json")
    counts = load("audit/artifacts_full/archive_counts.json")
    nulls = load("reports/nullsim_report.json")
    prompts = load("prompts/roles/_summary.json")
    return {
        "archive_harvest": None if not counts else {
            k: counts[k] for k in (
                "files_scanned", "candidate_strings", "history_strings",
                "unique_exact", "unique_normalized", "unique_shingles",
                "include_git_history",
            )
        },
        "archive_overlap": None if not overlap else overlap["summary"],
        "role_audit": None if not role else {
            k: role[k] for k in (
                "pass", "exact_duplicates", "normalized_duplicates",
                "source_row_reuse", "intent_identity_cross_role_hits",
                "max_role_pair_containment", "total_prompts",
            )
        },
        "prompts": prompts,
        "nullsim": None if not nulls else {
            "worst_false_pass": nulls["null"]["worst_false_pass"],
            "design_accepted": nulls["null"]["design_accepted"],
            "n_sims": nulls["null"]["n_sims"],
            "geometry_power": nulls["power"]["geometry_power"],
        },
    }


def build() -> dict:
    manifest = {
        "manifest_version": 1,
        "study": "prospective-geometry",
        "phase_frozen": 0,
        "gpu_provisioned": False,
        "clean_room": {
            "archive_root": "/Users/chaoticcooties/Desktop/renegade/archive",
            "permitted_access": [
                "read ARCHIVE_MANIFEST.md",
                "standalone corpus-overlap audit emitting hashes and counts only",
                "generic infrastructure copies with source path, commit, and SHA-256",
            ],
            "imported_infrastructure": [],
            "firewall_module": "pgeom.firewall",
        },
        "frozen_parameters": frozen_parameters(),
        "summaries": summaries(),
        "files": tracked_files(),
        "git_commit_at_build": git_commit(ROOT),
    }
    manifest["manifest_sha256"] = sha256_obj(
        {k: v for k, v in manifest.items() if k != "git_commit_at_build"}
    )
    return manifest


def main() -> int:
    out = ROOT / "manifest" / "freeze.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build()
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"  files hashed: {len(manifest['files'])}")
    print(f"  manifest sha256: {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
