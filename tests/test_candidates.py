"""Candidate-token policy and hash-sampled bank."""

import pytest

pytest.importorskip("transformers")

from pgeom.candidates import (  # noqa: E402
    FIRST_SPECIAL_ID,
    eligible_tokens,
    hash_sampled_bank,
    is_eligible,
    script_class,
    stratified_bank,
)
from pgeom.tokenization import load_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


@pytest.fixture(scope="module")
def eligible(tok):
    return eligible_tokens(tok)


def test_policy_rejects_special_and_malformed():
    assert not is_eligible(FIRST_SPECIAL_ID, " anything")
    assert not is_eligible(10, "noleadingspace")
    assert not is_eligible(10, " two words")
    assert not is_eligible(10, " ")
    assert not is_eligible(10, " bad\x00char")
    assert is_eligible(10, " alpha")


def test_bank_is_large_enough_to_search(eligible):
    assert len(eligible) > 20000


def test_bank_is_reproducible_and_key_dependent(eligible):
    a = hash_sampled_bank(eligible, key="k1", size=500)
    b = hash_sampled_bank(eligible, key="k1", size=500)
    c = hash_sampled_bank(eligible, key="k2", size=500)
    assert a.token_ids == b.token_ids
    assert a.token_ids != c.token_ids


def test_bank_is_not_a_sorted_id_prefix(eligible):
    """A sorted-id prefix would confound token id with everything else."""
    bank = hash_sampled_bank(eligible, key="k1", size=500)
    assert set(bank.token_ids) != set(sorted(eligible)[:500])
    # spread across the id range rather than clustered at the low end
    assert max(bank.token_ids) > 0.8 * max(eligible)


def test_bank_is_uniform_across_id_range(eligible):
    """Quartile occupancy should be roughly equal under uniform hash sampling."""
    bank = hash_sampled_bank(eligible, key="uniformity", size=4000)
    lo, hi = min(eligible), max(eligible)
    edges = [lo + (hi - lo) * q / 4 for q in range(5)]
    counts = [
        sum(1 for t in bank.token_ids if edges[i] <= t < edges[i + 1]) for i in range(4)
    ]
    expected = [
        sum(1 for t in eligible if edges[i] <= t < edges[i + 1]) * len(bank) / len(eligible)
        for i in range(4)
    ]
    for got, exp in zip(counts, expected):
        assert abs(got - exp) < 0.25 * exp + 30


def test_stratified_bank_covers_every_script_class(eligible, tok):
    decoded = {t: tok.decode([t]) for t in eligible}
    present = {script_class(d) for d in decoded.values()}
    bank = stratified_bank(eligible, decoded, key="strat", size=6000)
    covered = {script_class(decoded[t]) for t in bank.token_ids}
    assert len(bank) == 6000
    assert covered >= {c for c in present if sum(
        1 for d in decoded.values() if script_class(d) == c) > 200}


def test_bank_size_cannot_exceed_eligible(eligible):
    with pytest.raises(ValueError):
        hash_sampled_bank(eligible, key="k", size=len(eligible) + 1)
