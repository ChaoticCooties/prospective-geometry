"""Intervention operators, exercised on a tiny randomly initialised Llama.

A small local model is enough to prove the mechanics — that the dose lands at
the right site, that the routing bias moves only the heads it targets, and that
every operator restores the model on exit. Those are the properties that would
silently corrupt an experiment if wrong, and none of them need real weights.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from pgeom.interventions import (  # noqa: E402
    attention_route_bias,
    kl_of_rows,
    matched_random_direction,
    matched_random_heads,
    residual_dose,
)

N_LAYERS, N_HEADS, HIDDEN = 4, 4, 64
SEQ = 12
SPAN = (2, 7)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=N_LAYERS, num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS, max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    model.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(1)
    return torch.randint(0, 128, (2, SEQ))


def readout_mass(model, ids, layer, head, span):
    out = model(input_ids=ids, output_attentions=True, use_cache=False)
    return out.attentions[layer][:, head, -1, span[0]:span[1]].sum(-1)


def final_hidden(model, ids, layer):
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer + 1][:, -1, :]


def test_residual_dose_shifts_the_readout_by_exactly_the_dose(tiny, ids):
    layer = 2
    direction = torch.zeros(HIDDEN)
    direction[3] = 1.0
    base = final_hidden(tiny, ids, layer)
    with residual_dose(tiny, layer=layer, direction=direction, alpha=2.5):
        dosed = final_hidden(tiny, ids, layer)
    delta = (dosed - base)
    assert delta[:, 3].allclose(torch.full((ids.shape[0],), 2.5), atol=1e-3)
    other = torch.cat([delta[:, :3], delta[:, 4:]], dim=1)
    assert other.abs().max() < 1e-4


def test_residual_dose_is_removed_on_exit(tiny, ids):
    layer = 1
    direction = torch.randn(HIDDEN)
    before = final_hidden(tiny, ids, layer).clone()
    with residual_dose(tiny, layer=layer, direction=direction, alpha=5.0):
        pass
    assert final_hidden(tiny, ids, layer).allclose(before, atol=1e-5)


def test_route_bias_reduces_instruction_mass_on_the_targeted_head(tiny, ids):
    layer, head = 2, 1
    spans = [SPAN] * ids.shape[0]
    base = readout_mass(tiny, ids, layer, head, SPAN)
    with attention_route_bias(tiny, heads=[(layer, head)], spans=spans, delta=5.0):
        biased = readout_mass(tiny, ids, layer, head, SPAN)
    assert (biased < base).all(), "a positive logit bias must route attention away"


def test_route_bias_leaves_untargeted_heads_and_layers_alone(tiny, ids):
    spans = [SPAN] * ids.shape[0]
    other_head = readout_mass(tiny, ids, 2, 3, SPAN)
    other_layer = readout_mass(tiny, ids, 0, 1, SPAN)
    with attention_route_bias(tiny, heads=[(2, 1)], spans=spans, delta=5.0):
        assert readout_mass(tiny, ids, 2, 3, SPAN).allclose(other_head, atol=1e-5)
        assert readout_mass(tiny, ids, 0, 1, SPAN).allclose(other_layer, atol=1e-5)


def test_route_bias_keeps_rows_a_proper_distribution(tiny, ids):
    """Biasing pre-softmax must leave every attention row summing to one; rescaling
    probabilities afterwards would not."""
    spans = [SPAN] * ids.shape[0]
    with attention_route_bias(tiny, heads=[(2, 1), (3, 0)], spans=spans, delta=4.0):
        out = tiny(input_ids=ids, output_attentions=True, use_cache=False)
    for attn in out.attentions:
        rows = attn[:, :, -1, :].sum(-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-3)


def test_route_bias_restores_the_model_on_exit(tiny, ids):
    spans = [SPAN] * ids.shape[0]
    base = readout_mass(tiny, ids, 2, 1, SPAN).clone()
    with attention_route_bias(tiny, heads=[(2, 1)], spans=spans, delta=6.0):
        pass
    assert readout_mass(tiny, ids, 2, 1, SPAN).allclose(base, atol=1e-6)
    assert not hasattr(tiny.model.layers[2].self_attn, "_pgeom_bias")


def test_route_bias_restores_even_when_the_body_raises(tiny, ids):
    spans = [SPAN] * ids.shape[0]
    with pytest.raises(RuntimeError):
        with attention_route_bias(tiny, heads=[(2, 1)], spans=spans, delta=6.0):
            raise RuntimeError("boom")
    assert not hasattr(tiny.model.layers[2].self_attn, "_pgeom_bias")


def test_matched_placebo_direction_matches_norm():
    ref = torch.randn(HIDDEN) * 3.0
    g = torch.Generator().manual_seed(0)
    placebo = matched_random_direction(ref, generator=g)
    assert placebo.norm().item() == pytest.approx(ref.norm().item(), rel=1e-5)


def test_matched_placebo_heads_avoid_the_fitted_set():
    g = torch.Generator().manual_seed(0)
    fitted = [(0, 0), (1, 1), (2, 2)]
    placebo = matched_random_heads(N_LAYERS, N_HEADS, fitted, 3, generator=g)
    assert len(placebo) == 3
    assert not (set(placebo) & set(fitted))


def test_matched_placebo_heads_fail_rather_than_reuse_fitted_heads():
    g = torch.Generator().manual_seed(0)
    everything = [(l, h) for l in range(2) for h in range(2)]
    with pytest.raises(ValueError, match="not enough heads"):
        matched_random_heads(2, 2, everything, 3, generator=g)


def test_kl_of_rows_is_zero_for_identical_distributions():
    p = torch.tensor([[0.2, 0.3, 0.5]])
    assert float(kl_of_rows(p, p)) == pytest.approx(0.0, abs=1e-9)
    q = torch.tensor([[0.5, 0.3, 0.2]])
    assert float(kl_of_rows(p, q)) > 0
