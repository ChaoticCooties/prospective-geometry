"""Integrity of the frozen prompt roles.

These assert the properties the protocol requires of the committed prompt sets,
so a later edit that breaks disjointness or reintroduces an archived prompt
fails the suite rather than the study.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from pgeom.provenance import IntentIndex, content_words, normalize_text

ROOT = Path(__file__).resolve().parents[1]
ALL = ROOT / "prompts" / "roles" / "_all_prompts.jsonl"
OVERLAP = ROOT / "audit" / "artifacts_full" / "new_prompt_overlap.json"
ROLE_AUDIT = ROOT / "audit" / "artifacts_full" / "role_audit.json"

EXPECTED_ROLES = {
    "target_fit", "target_admission", "engineering", "optimizer_train",
    "graph_discovery", "graph_validation", "benign_control", "behavior_sealed",
}

pytestmark = pytest.mark.skipif(not ALL.exists(), reason="prompt roles not built yet")


@pytest.fixture(scope="module")
def rows():
    return [json.loads(l) for l in ALL.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_all_eight_roles_present(rows):
    assert {r["role"] for r in rows} == EXPECTED_ROLES


def test_prompt_ids_are_unique(rows):
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids)


def test_roles_are_mutually_disjoint_by_text(rows):
    norms = [normalize_text(r["text"]) for r in rows]
    assert len(set(norms)) == len(norms)


def test_no_source_row_is_reused(rows):
    keys = [(r["source"], r["source_row"]) for r in rows]
    assert len(set(keys)) == len(keys)


def test_no_two_roles_restate_the_same_request(rows):
    idx = IntentIndex()
    for r in rows:
        w = content_words(r["text"])
        assert not idx.conflicts(w), f"intent-identical prompt {r['id']} in role {r['role']}"
        idx.add(w)


def test_every_prompt_carries_provenance_and_license(rows):
    for r in rows:
        assert r["source"] and r["license"] and r["stratum"]
        assert 20 <= len(r["text"]) <= 320


def test_six_search_blocks_span_two_corpora(rows):
    train = [r for r in rows if r["role"] == "optimizer_train"]
    blocks = {r["block"] for r in train}
    assert len(blocks) == 6
    assert len({r["source"] for r in train}) == 2


def test_eight_admission_blocks_have_matched_arms(rows):
    adm = [r for r in rows if r["role"] == "target_admission"]
    blocks = Counter(r["block"] for r in adm)
    assert len(blocks) == 8
    for blk in blocks:
        arms = {r["arm"] for r in adm if r["block"] == blk}
        assert arms == {"nom_harmful", "val_harmful", "nom_benign", "val_benign", "benign_utility"}


def test_engineering_role_is_disjoint_from_every_search_block(rows):
    eng = {r["id"] for r in rows if r["role"] == "engineering"}
    other = {r["id"] for r in rows if r["role"] != "engineering"}
    assert not (eng & other)


@pytest.mark.skipif(not OVERLAP.exists(), reason="overlap audit not run")
def test_no_prompt_overlaps_the_archive():
    d = json.loads(OVERLAP.read_text())
    assert d["summary"]["clean"] == d["summary"]["candidates"]
    assert d["summary"]["flagged"] == {}


@pytest.mark.skipif(not ROLE_AUDIT.exists(), reason="role audit not run")
def test_role_audit_passes():
    d = json.loads(ROLE_AUDIT.read_text())
    assert d["pass"]
    assert d["intent_identity_cross_role_hits"] == 0
    assert d["exact_duplicates"] == 0
    assert d["normalized_duplicates"] == 0
