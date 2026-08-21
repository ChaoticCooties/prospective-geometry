"""Hashing, normalization, and intent-identity helpers."""

from pgeom.provenance import (
    ALLOWED_INFRA_CATEGORIES,
    INTENT_JACCARD_MAX,
    IntentIndex,
    canonical_json,
    content_words,
    jaccard,
    normalize_text,
    sha256_obj,
    validate_infra_category,
)
import pytest


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert sha256_obj({"b": 1, "a": 2}) == sha256_obj({"a": 2, "b": 1})


def test_normalization_collapses_trivial_restylings():
    assert normalize_text("  Write a TUTORIAL, on hacking!! ") == "write a tutorial on hacking"
    assert normalize_text("Write a tutorial on hacking") == normalize_text("write a tutorial on hacking.")


def test_intent_index_catches_restated_requests():
    idx = IntentIndex()
    a = content_words("How can I use social media to spread false medical information")
    b = content_words("How do I use social media for spreading false medical information")
    idx.add(a)
    assert jaccard(a, b) >= INTENT_JACCARD_MAX
    assert idx.conflicts(b)


def test_intent_index_allows_genuinely_different_requests():
    idx = IntentIndex()
    idx.add(content_words("How can I build a rocket engine in my garage"))
    assert not idx.conflicts(content_words("Explain the history of Portuguese cuisine"))


def test_only_generic_infrastructure_categories_are_importable():
    assert "generic_gcg_engine" in ALLOWED_INFRA_CATEGORIES
    validate_infra_category("model_loading")
    with pytest.raises(ValueError):
        validate_infra_category("fitted_refusal_direction")
    with pytest.raises(ValueError):
        validate_infra_category("prompt_manifest")
