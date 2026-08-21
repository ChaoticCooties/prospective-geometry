"""Recombination cube, margin statistic, and the block gate."""

import pytest

from pgeom.geometry import (
    GEOMETRY_BLOCKS_REQUIRED,
    N_VERTICES,
    BlockGeometryInput,
    FacePair,
    FacePlan,
    FaceScores,
    choose_plan,
    cube_vertices,
    evaluate_block_geometry,
    evaluate_geometry,
    monotone_paths,
    path_margin,
)


def additive_face(weights_a, weights_b):
    """Face where s_A rises and s_B falls along the cube, with given weights."""
    s_a, s_b = [], []
    for m in range(N_VERTICES):
        inside = [i for i in range(4) if (m >> i) & 1]
        outside = [i for i in range(4) if not (m >> i) & 1]
        s_a.append(float(sum(weights_a[i] for i in inside)))
        s_b.append(float(sum(weights_b[i] for i in outside)))
    return FaceScores(s_a=tuple(s_a), s_b=tuple(s_b))


def test_cube_has_sixteen_vertices_with_correct_endpoints():
    v = cube_vertices([10, 20, 30, 40], [11, 21, 31, 41])
    assert len(v) == N_VERTICES
    assert v[0] == (10, 20, 30, 40)
    assert v[15] == (11, 21, 31, 41)


def test_shared_alleles_keep_duplicate_vertices():
    """Collapsing duplicates would bias margins upward, so they are retained."""
    v = cube_vertices([1, 2, 3, 4], [1, 2, 3, 9])
    assert len(v) == N_VERTICES
    assert len(set(v)) == 2


def test_twenty_four_monotone_paths():
    paths = monotone_paths()
    assert len(paths) == 24
    assert all(p[0] == 0 and p[-1] == 15 and len(p) == 5 for p in paths)


def test_margin_is_exactly_zero_when_objectives_weight_positions_identically():
    """This is the null the statistic must respect, even after maximizing over
    all 24 paths; otherwise path selection alone would manufacture an effect."""
    face = additive_face([1, 2, 3, 4], [1, 2, 3, 4])
    plan = choose_plan(face, source_is_a=True, free_orientation=False)
    assert path_margin(face, plan) == pytest.approx(0.0, abs=1e-12)


def test_margin_is_positive_when_objectives_weight_positions_differently():
    face = additive_face([1, 2, 3, 4], [4, 3, 2, 1])
    plan = choose_plan(face, source_is_a=True, free_orientation=False)
    assert path_margin(face, plan) > 0.1


def test_unusable_face_returns_none_rather_than_a_sign_flipped_margin():
    flat = FaceScores(s_a=(1.0,) * N_VERTICES, s_b=(1.0,) * N_VERTICES)
    plan = FacePlan(source_is_a=True, reversed_=False, path=monotone_paths()[0])
    assert path_margin(flat, plan) is None
    assert choose_plan(flat, source_is_a=True, free_orientation=True) is None


def test_plan_choice_is_deterministic_under_ties():
    face = additive_face([1, 1, 1, 1], [1, 1, 1, 1])
    plans = {choose_plan(face, source_is_a=True, free_orientation=True) for _ in range(5)}
    assert len(plans) == 1


def _block(cross_w, same_w, ctrl_w, name="B1"):
    def pair(wa, wb):
        f = additive_face(wa, wb)
        return FacePair(discovery=f, validation=f)
    return BlockGeometryInput(
        block=name,
        cross_a1_b1=pair([1, 2, 3, 4], cross_w),
        cross_a2_b2=pair([1, 2, 3, 4], cross_w),
        same_a1_a2=pair([1, 2, 3, 4], same_w),
        same_b1_b2=pair([1, 2, 3, 4], same_w),
        controls=(pair([1, 2, 3, 4], ctrl_w), pair([1, 2, 3, 4], ctrl_w)),
    )


def test_block_passes_only_when_cross_beats_both_control_classes():
    good = _block(cross_w=[4, 3, 2, 1], same_w=[1, 2, 3, 4], ctrl_w=[1, 2, 3, 4])
    assert evaluate_block_geometry(good).passes


def test_block_fails_when_same_objective_face_matches_cross():
    """If any two different suffixes front-load equally, there is no
    objective-specific geometry to report."""
    tie = _block(cross_w=[4, 3, 2, 1], same_w=[4, 3, 2, 1], ctrl_w=[1, 2, 3, 4])
    r = evaluate_block_geometry(tie)
    assert not r.passes
    assert any("same-objective" in x for x in r.reasons)


def test_block_fails_when_control_matches_cross():
    tie = _block(cross_w=[4, 3, 2, 1], same_w=[1, 2, 3, 4], ctrl_w=[4, 3, 2, 1])
    r = evaluate_block_geometry(tie)
    assert not r.passes
    assert any("control" in x for x in r.reasons)


def test_exact_tie_fails_rather_than_passes():
    r = evaluate_block_geometry(_block([4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]))
    assert r.g_b is not None and r.g_b <= 0.0
    assert not r.passes


def test_study_requires_all_six_blocks():
    good = [evaluate_block_geometry(_block([4, 3, 2, 1], [1, 2, 3, 4], [1, 2, 3, 4], f"B{i}"))
            for i in range(GEOMETRY_BLOCKS_REQUIRED)]
    assert evaluate_geometry(good)["passed"]
    mixed = good[:-1] + [evaluate_block_geometry(_block([4, 3, 2, 1], [4, 3, 2, 1], [1, 2, 3, 4], "B6"))]
    assert not evaluate_geometry(mixed)["passed"]
    assert not evaluate_geometry(good[:5])["passed"]
