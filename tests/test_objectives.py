"""Objective formulas, statistics, and the deterministic fitting rules."""

import pytest

torch = pytest.importorskip("torch")

from pgeom.fitting import cohens_d, fit_package_a, fit_package_b  # noqa: E402
from pgeom.objectives import (  # noqa: E402
    K_HEADS,
    PackageB,
    auroc,
    centered_cosine,
    spearman,
)


def test_package_a_direction_is_unit_norm_and_signed_toward_harmful():
    harmful = torch.tensor([[2.0, 0.0], [2.2, 0.1]])
    benign = torch.tensor([[0.0, 0.0], [-0.2, 0.1]])
    pkg = fit_package_a(harmful, benign, layer=14)
    assert pkg.direction.norm() == pytest.approx(1.0, abs=1e-6)
    # Harmful prompts must score higher than benign under the fitted direction.
    assert pkg.score(harmful).mean() > pkg.score(benign).mean()


def test_package_a_rejects_a_degenerate_direction():
    same = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="degenerate"):
        fit_package_a(same, same.clone(), layer=14)


def test_package_a_fit_is_reproducible():
    h, b = torch.randn(20, 8), torch.randn(20, 8)
    assert fit_package_a(h, b, layer=14).fit_sha256 == fit_package_a(h, b, layer=14).fit_sha256


def test_package_b_selects_exactly_eight_heads_that_route_more_on_harmful():
    torch.manual_seed(0)
    harmful = torch.rand(30, 4, 4) * 0.1
    benign = torch.rand(30, 4, 4) * 0.1
    winners = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 0), (3, 3)]
    for layer, head in winners:
        harmful[:, layer, head] += 1.0
    pkg = fit_package_b(harmful, benign)
    assert len(pkg.heads) == K_HEADS
    assert set(pkg.heads) == set(winners)


def test_package_b_has_no_k_knob():
    """k is not a caller-tunable parameter; B must be one exact object."""
    import inspect
    assert "k" not in inspect.signature(fit_package_b).parameters


def test_package_b_fails_rather_than_lowering_k():
    """The protocol forbids shrinking k, so too few eligible heads is an error."""
    harmful = torch.zeros(10, 2, 2)
    benign = torch.ones(10, 2, 2)
    with pytest.raises(ValueError, match="forbids lowering k"):
        fit_package_b(harmful, benign)


def test_package_b_head_set_is_sorted_and_reproducible():
    torch.manual_seed(1)
    harmful = torch.ones(10, 4, 4) + torch.randn(10, 4, 4) * 1e-6
    benign = torch.zeros(10, 4, 4) + torch.randn(10, 4, 4) * 1e-6
    a, b = fit_package_b(harmful, benign), fit_package_b(harmful, benign)
    assert a.heads == tuple(sorted(a.heads))
    assert a.heads == b.heads and a.fit_sha256 == b.fit_sha256


def test_package_b_rejects_a_wrong_sized_head_set():
    with pytest.raises(ValueError, match="exactly"):
        PackageB(heads=((0, 0), (1, 1)), fit_sha256="x")
    with pytest.raises(ValueError, match="duplicate"):
        PackageB(heads=tuple([(0, 0)] * K_HEADS), fit_sha256="x")


def test_package_b_score_averages_only_the_fixed_heads():
    pkg = PackageB(heads=tuple((0, h) for h in range(K_HEADS)), fit_sha256="x")
    mass = torch.zeros(2, 3, K_HEADS)
    mass[:, 0, :] = 0.5
    mass[:, 1, :] = 99.0          # a layer outside the head set must not contribute
    assert pkg.score(mass).tolist() == pytest.approx([0.5, 0.5])


def test_cohens_d_sign_and_scale():
    a = torch.full((50, 1), 2.0) + torch.randn(50, 1) * 0.01
    b = torch.zeros(50, 1) + torch.randn(50, 1) * 0.01
    assert float(cohens_d(a, b)[0]) > 10


def test_spearman_is_rank_based_not_linear():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_centered_cosine_removes_a_shared_offset():
    """Uncentered cosine is near 1 for any two vectors sharing a large mean, which
    would make the redundancy check fire on almost anything."""
    a = torch.tensor([100.0, 101.0, 99.0])
    b = torch.tensor([100.0, 99.0, 101.0])
    assert centered_cosine(a, b) == pytest.approx(-1.0, abs=1e-9)


def test_auroc_matches_known_values():
    scores = torch.tensor([0.1, 0.4, 0.35, 0.8])
    labels = torch.tensor([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.75)
    assert auroc(torch.tensor([1.0, 2.0]), torch.tensor([0, 1])) == pytest.approx(1.0)


def test_auroc_handles_ties_without_bias():
    scores = torch.tensor([1.0, 1.0, 1.0, 1.0])
    labels = torch.tensor([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.5)
