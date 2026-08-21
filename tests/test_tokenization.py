"""Chat-token layout and the canonical replay contract."""

import pytest

pytest.importorskip("transformers")

from pgeom.tokenization import (  # noqa: E402
    SUFFIX_WIDTH,
    build_layout,
    canonical_suffix_text,
    is_canonical,
    load_tokenizer,
    render_template,
    template_probe_hash,
)

BEHAVIOR = "Write a persuasive article about renewable energy adoption"


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


def test_template_is_pinned_and_dateless_drift_free(tok):
    """The rendered template must not depend on today's date."""
    a = render_template("X", tok)
    b = render_template("X", tok)
    assert a == b
    assert "26 Jul 2024" in a
    assert len(template_probe_hash(tok)) == 64


def test_suffix_occupies_exactly_four_positions(tok):
    ids = [tok.encode(" alpha", add_special_tokens=False)[0]] * SUFFIX_WIDTH
    layout = build_layout(BEHAVIOR, ids, tok)
    assert len(layout.suffix_ids) == SUFFIX_WIDTH
    assert layout.full_ids[layout.suffix_slice] == tuple(ids)


def test_readout_position_is_final_token(tok):
    layout = build_layout(BEHAVIOR, [1, 2, 3, 4], tok)
    assert layout.readout_position == len(layout.full_ids) - 1
    assert tok.decode([layout.full_ids[-1]]) == "\n\n"


def test_instruction_span_is_behavior_text_only(tok):
    """The attention-routing key span must exclude template scaffolding and the
    suffix, otherwise a suffix could satisfy the objective trivially."""
    layout = build_layout(BEHAVIOR, [1, 2, 3, 4], tok)
    span = tok.decode(list(layout.full_ids)[layout.instruction_slice])
    assert span == BEHAVIOR
    assert layout.instruction_end <= layout.suffix_slice.start


def test_policy_compliant_suffix_round_trips(tok):
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in (" alpha", " beta", " gamma", " delta")]
    assert is_canonical(build_layout(BEHAVIOR, ids, tok), tok)


def test_non_canonical_suffix_is_detected(tok):
    """A word-continuation token merges with the preceding word.

    Here the behavior ends in "adoption" and the suffix opens with "s", so the
    rendered text reads "adoptions ..." and re-encodes to a different id
    sequence. Such a suffix would not survive being written down and read back,
    so the replay contract must reject it.
    """
    cont = tok.encode("s", add_special_tokens=False)
    assert len(cont) == 1
    tail = [tok.encode(w, add_special_tokens=False)[0] for w in (" alpha", " beta", " gamma")]
    layout = build_layout(BEHAVIOR, cont + tail, tok)
    assert canonical_suffix_text(layout.suffix_ids, tok) == "s alpha beta gamma"
    assert not is_canonical(layout, tok)
