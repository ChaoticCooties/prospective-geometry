"""Phase-4 exact prospective geometry over four-token recombination cubes.

Two four-token endpoints define exactly 2**4 = 16 real-token recombinations: at
each position a vertex carries either the first endpoint's token or the second's.
Shared alleles and duplicate vertices are retained, not collapsed — they are
genuine ties and count as failures, because dropping them would bias every
margin upward.

The margin statistic
--------------------
Walking a monotone path from the source endpoint to the destination endpoint
banks a growing fraction of the achievable gain on the destination objective and
pays a growing fraction of the cost on the source objective. Both reach exactly
one at the far endpoint, so endpoints carry no information; only interior
vertices are scored. The statistic is *front-loading*: positive when the tokens
that buy the destination objective are not the tokens that pay for the source
objective, and identically zero when both objectives weight the four positions
the same way. That zero is what makes it a usable null.

Selection discipline
--------------------
A face's plan — its orientation where free, and its path — is chosen on
graph-discovery prompts and then evaluated once, unchanged, on graph-validation
prompts. Control faces run the identical discovery/validation procedure and are
never handed a path or orientation selected for the optimized faces. Cross
faces have their orientation fixed by design (A -> B) and only their path is
free, so controls have at least as much freedom as the comparison they bound.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

WIDTH = 4
N_VERTICES = 1 << WIDTH
LAST = N_VERTICES - 1

#: Separation below which a face carries no orientation information.
MIN_SEPARATION = 1e-9

#: Every one of the six blocks must pass. Ties fail.
GEOMETRY_BLOCKS_REQUIRED = 6


def cube_vertices(endpoint_u: Sequence[int], endpoint_v: Sequence[int]) -> list[tuple[int, ...]]:
    """All 16 recombinations. Mask bit i selects endpoint_v's token at position i."""
    if len(endpoint_u) != WIDTH or len(endpoint_v) != WIDTH:
        raise ValueError(f"endpoints must have width {WIDTH}")
    return [
        tuple(endpoint_v[i] if (mask >> i) & 1 else endpoint_u[i] for i in range(WIDTH))
        for mask in range(N_VERTICES)
    ]


def monotone_paths() -> list[tuple[int, ...]]:
    """The 24 shortest paths from vertex 0 to vertex 15, as mask sequences."""
    paths = []
    for order in itertools.permutations(range(WIDTH)):
        mask, seq = 0, [0]
        for pos in order:
            mask |= 1 << pos
            seq.append(mask)
        paths.append(tuple(seq))
    return paths


@dataclass(frozen=True)
class FaceScores:
    """Both frozen objective scores at all 16 vertices of one face.

    Scores are minimized, so lower means more suppressed. Index 0 is one
    endpoint and index 15 the other.
    """

    s_a: tuple[float, ...]
    s_b: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.s_a) != N_VERTICES or len(self.s_b) != N_VERTICES:
            raise ValueError(f"expected {N_VERTICES} vertex scores per objective")
        if not all(math.isfinite(x) for x in self.s_a + self.s_b):
            raise ValueError("non-finite vertex score")

    def reversed_face(self) -> "FaceScores":
        """Swap the two endpoint roles by relabelling mask m as its complement."""
        return FaceScores(
            s_a=tuple(self.s_a[LAST ^ m] for m in range(N_VERTICES)),
            s_b=tuple(self.s_b[LAST ^ m] for m in range(N_VERTICES)),
        )

    def diagnostics(self) -> dict[str, float]:
        def var(xs):
            mu = sum(xs) / len(xs)
            return sum((x - mu) ** 2 for x in xs) / len(xs)
        return {
            "sep_a": self.s_a[LAST] - self.s_a[0],
            "sep_b": self.s_b[0] - self.s_b[LAST],
            "range_a": max(self.s_a) - min(self.s_a),
            "range_b": max(self.s_b) - min(self.s_b),
            "var_a": var(self.s_a),
            "var_b": var(self.s_b),
            "distinct_vertex_scores_a": float(len(set(self.s_a))),
        }


@dataclass(frozen=True)
class FacePlan:
    """A frozen choice of orientation and path for one face."""

    source_is_a: bool
    reversed_: bool
    path: tuple[int, ...]


def _oriented(face: FaceScores, plan: FacePlan) -> FaceScores:
    return face.reversed_face() if plan.reversed_ else face


def separations(face: FaceScores, *, source_is_a: bool) -> tuple[float, float]:
    """(source_sep, dest_sep), each positive when its endpoint wins its objective."""
    s_src, s_dst = (face.s_a, face.s_b) if source_is_a else (face.s_b, face.s_a)
    return (s_src[LAST] - s_src[0], s_dst[0] - s_dst[LAST])


def cumulative_margins(face: FaceScores, plan: FacePlan) -> list[float] | None:
    """Interior cumulative dual-objective margins, or None if the face is unusable."""
    oriented = _oriented(face, plan)
    src_sep, dst_sep = separations(oriented, source_is_a=plan.source_is_a)
    if not (math.isfinite(src_sep) and math.isfinite(dst_sep)):
        return None
    if abs(src_sep) < MIN_SEPARATION or abs(dst_sep) < MIN_SEPARATION:
        return None
    s_src, s_dst = (
        (oriented.s_a, oriented.s_b) if plan.source_is_a else (oriented.s_b, oriented.s_a)
    )
    start = plan.path[0]
    out = []
    for vertex in plan.path[1:-1]:
        gain = (s_dst[start] - s_dst[vertex]) / abs(dst_sep)
        cost = (s_src[vertex] - s_src[start]) / abs(src_sep)
        out.append(gain - cost)
    return out


def path_margin(face: FaceScores, plan: FacePlan) -> float | None:
    m = cumulative_margins(face, plan)
    if not m:
        return None
    return sum(m) / len(m)


def choose_plan(
    discovery: FaceScores, *, source_is_a: bool, free_orientation: bool
) -> FacePlan | None:
    """Pick the plan that maximizes the margin on discovery prompts.

    Ties break on the frozen lexicographic order (reversed_, path), so the choice
    never depends on dict ordering or on validation data.
    """
    best: tuple[float, bool, tuple[int, ...]] | None = None
    orientations = (False, True) if free_orientation else (False,)
    for rev in orientations:
        for path in monotone_paths():
            plan = FacePlan(source_is_a=source_is_a, reversed_=rev, path=path)
            m = path_margin(discovery, plan)
            if m is None:
                continue
            key = (-m, rev, path)
            if best is None or key < (-best[0], best[1], best[2]):
                best = (m, rev, path)
    if best is None:
        return None
    return FacePlan(source_is_a=source_is_a, reversed_=best[1], path=best[2])


@dataclass(frozen=True)
class FacePair:
    """Discovery and validation scores for one face."""

    discovery: FaceScores
    validation: FaceScores


@dataclass(frozen=True)
class BlockGeometryInput:
    """One block's faces. Plans are derived here, never supplied from outside."""

    block: str
    cross_a1_b1: FacePair
    cross_a2_b2: FacePair
    same_a1_a2: FacePair
    same_b1_b2: FacePair
    controls: tuple[FacePair, ...]


@dataclass
class BlockGeometryResult:
    block: str
    m_cross: float | None
    m_same_stronger: float | None
    m_control: float | None
    g_b: float | None
    passes: bool
    reasons: list[str]
    diagnostics: dict


def _plan_and_score(pair: FacePair, *, source_is_a: bool, free_orientation: bool):
    plan = choose_plan(pair.discovery, source_is_a=source_is_a, free_orientation=free_orientation)
    if plan is None:
        return None, None
    return plan, path_margin(pair.validation, plan)


def evaluate_block_geometry(g: BlockGeometryInput) -> BlockGeometryResult:
    """Compute the block's joint statistic G_b.

    G_b is the minimum of three quantities, so it is positive only when all
    three hold at once in this same block: the averaged cross-objective margin
    is itself positive, it exceeds the stronger same-objective restart face, and
    it exceeds the strongest surface-matched random endpoint control.
    """
    reasons: list[str] = []

    cross_plans, cross_vals = [], []
    for pair in (g.cross_a1_b1, g.cross_a2_b2):
        plan, val = _plan_and_score(pair, source_is_a=True, free_orientation=False)
        cross_plans.append(plan)
        cross_vals.append(val)
    if any(v is None for v in cross_vals):
        reasons.append("a cross-objective face was unusable (no endpoint separation)")
        m_cross = None
    else:
        m_cross = sum(cross_vals) / len(cross_vals)

    same_vals = []
    for pair, src_a in ((g.same_a1_a2, True), (g.same_b1_b2, False)):
        _, val = _plan_and_score(pair, source_is_a=src_a, free_orientation=True)
        same_vals.append(val)
    usable_same = [v for v in same_vals if v is not None]
    m_same = max(usable_same) if usable_same else None
    if m_same is None:
        reasons.append("no usable same-objective restart face")

    ctrl_vals = []
    for pair in g.controls:
        _, val = _plan_and_score(pair, source_is_a=True, free_orientation=True)
        ctrl_vals.append(val)
    usable_ctrl = [v for v in ctrl_vals if v is not None]
    m_ctrl = max(usable_ctrl) if usable_ctrl else None
    if m_ctrl is None:
        reasons.append("no usable surface-matched control face")

    if m_cross is None or m_same is None or m_ctrl is None:
        g_b, passes = None, False
    else:
        g_b = min(m_cross, m_cross - m_same, m_cross - m_ctrl)
        passes = g_b > 0.0
        if not passes:
            if m_cross <= 0:
                reasons.append("cross-objective margin not positive")
            if m_cross <= m_same:
                reasons.append("did not beat stronger same-objective restart face")
            if m_cross <= m_ctrl:
                reasons.append("did not beat surface-matched control")

    return BlockGeometryResult(
        block=g.block, m_cross=m_cross, m_same_stronger=m_same, m_control=m_ctrl,
        g_b=g_b, passes=passes, reasons=reasons,
        diagnostics={
            "cross_margins": cross_vals,
            "same_margins": same_vals,
            "control_margins": ctrl_vals,
            "cross_1_validation": g.cross_a1_b1.validation.diagnostics(),
            "cross_2_validation": g.cross_a2_b2.validation.diagnostics(),
            "cross_plans": [
                None if p is None else {"reversed": p.reversed_, "path": list(p.path)}
                for p in cross_plans
            ],
        },
    )


def evaluate_geometry(results: Sequence[BlockGeometryResult]) -> dict:
    """Study-level geometry verdict. Blocks are the unit of analysis."""
    n_pass = sum(r.passes for r in results)
    passed = len(results) == GEOMETRY_BLOCKS_REQUIRED and n_pass == GEOMETRY_BLOCKS_REQUIRED
    return {
        "passed": passed,
        "blocks_total": len(results),
        "blocks_passed": n_pass,
        "blocks_required": GEOMETRY_BLOCKS_REQUIRED,
        "per_block": {
            r.block: {
                "g_b": r.g_b, "m_cross": r.m_cross,
                "m_same_stronger": r.m_same_stronger, "m_control": r.m_control,
                "passes": r.passes, "reasons": r.reasons,
            }
            for r in results
        },
    }
