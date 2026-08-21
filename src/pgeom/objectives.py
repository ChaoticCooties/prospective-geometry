"""The two candidate internal objective packages, as exact frozen formulas.

Both are *minimized* by the optimizer, and both are computed from the readouts
in :mod:`pgeom.model` so that a single forward pass serves both. Neither ever
sees a completion, a generated token, a judge score, or a behavior label.

Package A  signed projection of the layer-14 residual stream at the
           assistant-prefill readout position onto a fitted unit refusal
           direction. Larger means more refusal.

Package B  mean, over exactly eight fitted heads, of the attention probability
           mass the readout query assigns to the instruction key span. Larger
           means more instruction routing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: Exact number of heads in package B. One object, not a menu.
K_HEADS = 8


@dataclass(frozen=True)
class PackageA:
    """A fitted refusal direction. ``direction`` is unit-norm."""

    direction: torch.Tensor          # [hidden]
    layer: int
    fit_sha256: str

    def score(self, resid: torch.Tensor) -> torch.Tensor:
        """Signed projection. ``resid`` is [batch, hidden]."""
        return resid.float() @ self.direction.to(resid.device).float()


@dataclass(frozen=True)
class PackageB:
    """A fitted head set. ``heads`` is exactly K_HEADS (layer, head) pairs."""

    heads: tuple[tuple[int, int], ...]
    fit_sha256: str

    def __post_init__(self) -> None:
        if len(self.heads) != K_HEADS:
            raise ValueError(f"package B must fix exactly {K_HEADS} heads, got {len(self.heads)}")
        if len(set(self.heads)) != len(self.heads):
            raise ValueError("package B head set contains a duplicate")

    def score(self, instr_mass: torch.Tensor) -> torch.Tensor:
        """Mean instruction mass over the fixed heads. ``instr_mass`` is [b, L, H]."""
        idx_l = torch.tensor([h[0] for h in self.heads], device=instr_mass.device)
        idx_h = torch.tensor([h[1] for h in self.heads], device=instr_mass.device)
        return instr_mass[:, idx_l, idx_h].mean(dim=-1)


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation, used for the non-redundancy condition."""
    if x.numel() < 3:
        return float("nan")

    def ranks(v: torch.Tensor) -> torch.Tensor:
        order = v.argsort()
        r = torch.empty_like(order, dtype=torch.float64)
        r[order] = torch.arange(v.numel(), dtype=torch.float64, device=v.device)
        return r

    rx, ry = ranks(x.flatten().double()), ranks(y.flatten().double())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    return float("nan") if denom == 0 else float((rx @ ry) / denom)


def centered_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine between mean-centered vectors.

    Centering matters: raw vocabulary gradients share a large common component,
    so uncentered cosine is high between almost any two objectives and would
    make the non-redundancy condition trivially fail.
    """
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float("nan") if denom == 0 else float((a @ b) / denom)


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC via the rank identity. ``labels`` is 1 for the positive class."""
    scores = scores.flatten().double()
    labels = labels.flatten().double()
    n_pos = int(labels.sum().item())
    n_neg = int(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    # Average ranks within ties, or tied scores would inflate or deflate the area.
    unique = torch.unique(scores)
    if unique.numel() != scores.numel():
        for value in unique:
            mask = scores == value
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
    rank_sum = ranks[labels == 1].sum().item()
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
