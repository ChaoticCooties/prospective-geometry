"""The single blockwise admission conjunction."""

import pytest

from pgeom.admission import (
    BLOCKS_REQUIRED,
    BLOCKS_TOTAL,
    BlockMeasurement,
    PackageBlockMeasurement,
    diagonally_selective,
    evaluate_admission,
)


def pkg(**over):
    base = dict(
        effect_pp=25.0, placebo_effect_pp=5.0, sham_effect_pp=4.0,
        benign_utility_loss_pp=2.0, heldout_auroc=0.85,
        reach_own_improvement=1.5, reach_random_control=0.2,
        finite_numerics=True, replay_ok=True,
    )
    base.update(over)
    return PackageBlockMeasurement(**base)


def block(name="A1", a=None, b=None, **over):
    base = dict(
        abs_spearman_ab=0.30, median_grad_cosine=0.30,
        cross_matrix=((3.0, 0.5), (0.5, 3.0)),
    )
    base.update(over)
    return BlockMeasurement(block=name, package_a=a or pkg(), package_b=b or pkg(), **base)


def study(blocks):
    return evaluate_admission(blocks)


def test_all_good_blocks_pass_and_are_called_distinct():
    r = study([block(f"A{i}") for i in range(BLOCKS_TOTAL)])
    assert r.passed and r.n_blocks_passed == BLOCKS_TOTAL
    assert r.causally_distinct


def test_seven_of_eight_is_enough():
    blocks = [block(f"A{i}") for i in range(BLOCKS_TOTAL - 1)]
    blocks.append(block("A7", a=pkg(effect_pp=5.0)))
    r = study(blocks)
    assert r.passed and r.n_blocks_passed == BLOCKS_REQUIRED


def test_six_of_eight_fails():
    blocks = [block(f"A{i}") for i in range(BLOCKS_TOTAL - 2)]
    blocks += [block("A6", a=pkg(effect_pp=5.0)), block("A7", b=pkg(effect_pp=5.0))]
    r = study(blocks)
    assert not r.passed
    assert "NO-GO" in r.target_wording


def test_different_conditions_cannot_be_satisfied_by_different_block_subsets():
    """Four blocks fail on causal effect and four on redundancy. Every condition
    is met by six of eight blocks, but no seven blocks meet all of them, so the
    conjunction must reject."""
    blocks = []
    for i in range(4):
        blocks.append(block(f"C{i}", a=pkg(effect_pp=5.0)))
    for i in range(4):
        blocks.append(block(f"R{i}", abs_spearman_ab=0.95))
    r = study(blocks)
    assert not r.passed and r.n_blocks_passed == 0


def test_effect_must_double_its_matched_placebo():
    r = study([block(f"A{i}", a=pkg(effect_pp=16.0, placebo_effect_pp=15.0))
               for i in range(BLOCKS_TOTAL)])
    assert not r.passed
    assert "A.beats_placebo" in r.failing_blocks["A0"]


def test_benign_utility_loss_blocks_admission():
    r = study([block(f"A{i}", a=pkg(benign_utility_loss_pp=9.0)) for i in range(BLOCKS_TOTAL)])
    assert not r.passed


def test_reachability_must_beat_random_mutation_control():
    r = study([block(f"A{i}", a=pkg(reach_own_improvement=0.1, reach_random_control=0.5))
               for i in range(BLOCKS_TOTAL)])
    assert not r.passed
    assert "A.reach_beats_random" in r.failing_blocks["A0"]


def test_replay_and_numerics_are_part_of_the_same_conjunction():
    r = study([block(f"A{i}", a=pkg(replay_ok=False)) for i in range(BLOCKS_TOTAL)])
    assert not r.passed
    r2 = study([block(f"A{i}", b=pkg(finite_numerics=False)) for i in range(BLOCKS_TOTAL)])
    assert not r2.passed


def test_passing_without_selectivity_yields_distinct_readouts_wording_only():
    r = study([block(f"A{i}", cross_matrix=((3.0, 2.9), (2.9, 3.0))) for i in range(BLOCKS_TOTAL)])
    assert r.passed
    assert not r.causally_distinct
    assert "distinct readouts" in r.target_wording
    assert "causally distinct" not in r.target_wording.split("Cross")[0]


def test_diagonal_selectivity_rejects_near_singular_matrix():
    ok, stats = diagonally_selective(((3.0, 2.99), (2.99, 3.0)))
    assert not ok
    ok2, _ = diagonally_selective(((3.0, 0.2), (0.2, 3.0)))
    assert ok2


def test_wrong_block_count_is_an_error():
    with pytest.raises(ValueError):
        study([block("A1")])
