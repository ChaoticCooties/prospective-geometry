"""Intervention operators for causal admission.

Each package gets an operator that acts on the thing that package is defined by,
plus a sham at a non-target site and a matched placebo. All are context
managers, so a failure cannot leave the model permanently modified.

Package A acts on the layer-14 residual stream, adding a dose along the fitted
direction at the assistant-prefill positions.

Package B acts on attention routing. Attention logits are biased before the
softmax for exactly the eight fitted heads, at the readout query row, over the
instruction key span. The bias is applied pre-softmax rather than by rescaling
probabilities afterwards, so the row stays a proper distribution.

Doses are natural-range: package A's is a multiple of the projection's own
standard deviation on that block's nomination prompts, and package B's is
calibrated to move instruction mass to the level matched benign prompts show.
No extreme steering is used.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F


# --- Package A: residual-stream dose -----------------------------------------

@contextlib.contextmanager
def residual_dose(
    model,
    *,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    positions: slice | None = None,
):
    """Add ``alpha * direction`` to a layer's output residual stream.

    ``positions`` defaults to the final position, which is the readout site. A
    slice covering the assistant prefill can be passed instead.
    """
    block = model.model.layers[layer]
    unit = direction.detach()

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        delta = (alpha * unit).to(hidden.dtype).to(hidden.device)
        if positions is None:
            hidden[:, -1, :] = hidden[:, -1, :] + delta
        else:
            hidden[:, positions, :] = hidden[:, positions, :] + delta
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --- Package B: attention-routing bias ---------------------------------------

_ORIGINAL_EAGER = None


def _pgeom_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention with a pre-softmax bias hook.

    Mirrors the stock implementation exactly, except that a module may carry a
    ``_pgeom_bias`` callable applied to the attention logits before the softmax.
    Biasing pre-softmax keeps every attention row a proper distribution; scaling
    probabilities after the fact would not.
    """
    from transformers.models.llama.modeling_llama import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    bias_fn = getattr(module, "_pgeom_bias", None)
    if bias_fn is not None:
        attn_weights = bias_fn(attn_weights)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _install_patch():
    global _ORIGINAL_EAGER
    from transformers.models.llama import modeling_llama as M

    if _ORIGINAL_EAGER is None:
        _ORIGINAL_EAGER = M.eager_attention_forward
    M.eager_attention_forward = _pgeom_eager_attention


def _remove_patch():
    from transformers.models.llama import modeling_llama as M

    if _ORIGINAL_EAGER is not None:
        M.eager_attention_forward = _ORIGINAL_EAGER


def span_bias(heads_in_layer: Sequence[int], spans: Sequence[tuple[int, int]], delta: float):
    """Bias factory: subtract ``delta`` on the readout row over each row's span."""

    def apply(attn_weights: torch.Tensor) -> torch.Tensor:
        # attn_weights: [batch, heads, q, k]. Only the readout query row moves.
        biased = attn_weights.clone()
        idx = torch.tensor(list(heads_in_layer), device=attn_weights.device, dtype=torch.long)
        for b, (start, end) in enumerate(spans):
            if b >= biased.shape[0]:
                break
            biased[b, idx, -1, start:end] = biased[b, idx, -1, start:end] - delta
        return biased

    return apply


@contextlib.contextmanager
def attention_route_bias(
    model,
    *,
    heads: Iterable[tuple[int, int]],
    spans: Sequence[tuple[int, int]],
    delta: float,
):
    """Bias the readout-row attention logits for the given (layer, head) pairs."""
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    _install_patch()
    touched = []
    try:
        for layer, head_list in by_layer.items():
            module = model.model.layers[layer].self_attn
            module._pgeom_bias = span_bias(head_list, spans, delta)
            touched.append(module)
        yield
    finally:
        for module in touched:
            if hasattr(module, "_pgeom_bias"):
                del module._pgeom_bias
        _remove_patch()


# --- Sham and placebo --------------------------------------------------------

@contextlib.contextmanager
def null_intervention():
    """A no-op, used as the untouched reference condition."""
    yield


def matched_random_direction(
    reference: torch.Tensor, *, generator: torch.Generator
) -> torch.Tensor:
    """A random unit direction matched on norm to ``reference``.

    Matching on norm is what makes the placebo comparable: an unmatched random
    vector would deliver a different dose and its weaker effect would prove
    nothing.
    """
    raw = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    return (raw / raw.norm()) * reference.norm()


def matched_random_heads(
    n_layers: int, n_heads: int, exclude: Iterable[tuple[int, int]], k: int,
    *, generator: torch.Generator,
) -> tuple[tuple[int, int], ...]:
    """``k`` random heads disjoint from ``exclude``, for the package-B placebo."""
    banned = set(exclude)
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads) if (l, h) not in banned]
    if len(pool) < k:
        raise ValueError("not enough heads outside the fitted set for a matched placebo")
    idx = torch.randperm(len(pool), generator=generator)[:k].tolist()
    return tuple(sorted(pool[i] for i in idx))


def kl_of_rows(p: torch.Tensor, q: torch.Tensor, *, eps: float = 1e-9) -> torch.Tensor:
    """KL(p || q) over the last dimension, for dose matching between operators."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p / q).log()).sum(dim=-1)
