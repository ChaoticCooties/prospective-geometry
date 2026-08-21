"""Deterministic fitting of the two candidate objective packages.

Both are fitted from scratch on the `target_fit` role only. Nothing is loaded
or validated against any archived artifact, and no step consults a model
outcome, a generation, or a label beyond the harmful/benign contrast that the
prompt role itself carries.

Every rule here was frozen before any model outcome existed: the site, the sign
convention, the aggregation, the head count, and the tie-break.
"""

from __future__ import annotations

import torch

from .objectives import K_HEADS, PackageA, PackageB
from .provenance import sha256_obj


def fit_package_a(
    resid_harmful: torch.Tensor, resid_benign: torch.Tensor, *, layer: int
) -> PackageA:
    """Unit-normalized difference of class-conditional means.

    The sign is fixed by construction: harmful minus benign, so a larger
    projection means more refusal and the optimizer's job is to drive it down.
    The sign is never flipped after seeing which direction separates better.
    """
    if resid_harmful.ndim != 2 or resid_benign.ndim != 2:
        raise ValueError("expected [n, hidden] residual matrices")
    if resid_harmful.shape[1] != resid_benign.shape[1]:
        raise ValueError("harmful and benign residuals have different widths")

    delta = resid_harmful.float().mean(dim=0) - resid_benign.float().mean(dim=0)
    norm = delta.norm()
    if not torch.isfinite(norm) or norm == 0:
        raise ValueError("refusal direction is degenerate; fitting failed")
    direction = (delta / norm).contiguous()

    fingerprint = sha256_obj({
        "layer": layer,
        "n_harmful": int(resid_harmful.shape[0]),
        "n_benign": int(resid_benign.shape[0]),
        "direction_head": [round(float(x), 6) for x in direction[:16].tolist()],
        "direction_norm": 1.0,
    })
    return PackageA(direction=direction, layer=layer, fit_sha256=fingerprint)


def cohens_d(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Cohen's d of ``a`` minus ``b`` along dim 0, pooled standard deviation."""
    na, nb = a.shape[0], b.shape[0]
    va = a.float().var(dim=0, unbiased=True)
    vb = b.float().var(dim=0, unbiased=True)
    pooled = torch.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1) + eps)
    return (a.float().mean(dim=0) - b.float().mean(dim=0)) / pooled


def fit_package_b(mass_harmful: torch.Tensor, mass_benign: torch.Tensor) -> PackageB:
    """The exactly-eight heads with the largest positive Cohen's d.

    There is deliberately no ``k`` parameter. The protocol requires package B to
    be one exact object rather than a menu, so the head count is not a knob a
    caller can turn.

    Only heads that route *more* to the instruction on harmful prompts are
    eligible, since those are the ones a routing attack would need to move.
    Ties break on ascending (layer, head), so the head set is reproducible and
    never depends on iteration order.
    """
    if mass_harmful.shape[1:] != mass_benign.shape[1:]:
        raise ValueError("harmful and benign attention masses have different shapes")

    k = K_HEADS
    d = cohens_d(mass_harmful, mass_benign)          # [layers, heads]
    n_layers, n_heads = d.shape
    scored = [
        (float(d[layer, head]), layer, head)
        for layer in range(n_layers)
        for head in range(n_heads)
        if torch.isfinite(d[layer, head]) and d[layer, head] > 0
    ]
    if len(scored) < k:
        raise ValueError(
            f"only {len(scored)} heads route more to the instruction on harmful prompts; "
            f"package B needs {k}. Fitting failed; the protocol forbids lowering k."
        )
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    heads = tuple(sorted((layer, head) for _, layer, head in scored[:k]))

    fingerprint = sha256_obj({
        "k": k,
        "heads": [list(h) for h in heads],
        "n_harmful": int(mass_harmful.shape[0]),
        "n_benign": int(mass_benign.shape[0]),
        "top_d": [round(t[0], 6) for t in scored[:k]],
    })
    return PackageB(heads=heads, fit_sha256=fingerprint)
