"""Block-correlated null simulation of the complete admission and analysis pipeline.

The simulation drives the *same* decision functions the real study uses, so a
gate that is too permissive shows up here rather than in the results. Synthetic
measurements are generated under nulls, pushed through
:func:`pgeom.admission.evaluate_admission` and
:func:`pgeom.geometry.evaluate_geometry`, and the empirical false-pass rate is
the fraction of simulated studies the pipeline would have called a finding.

Two geometry nulls are simulated, because they fail in different ways.

N1  no geometry at all
    Both objectives weight the four suffix positions identically, so every
    face's true front-loading margin is exactly zero.

N2  geometry exists but is not objective-specific
    Faces have real, nonzero front-loading, but cross-objective and
    same-objective faces are drawn from one exchangeable distribution. This is
    the null that matters scientifically: it asks whether the design can be
    fooled by the ordinary fact that any two different suffixes differ.

Block correlation is applied as a per-block shared scale and offset affecting
every face in that block, so faces within a block are not independent. The unit
of analysis is the block, never the prompt, vertex, edge, or restart.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .admission import (
    BLOCKS_TOTAL,
    BlockMeasurement,
    PackageBlockMeasurement,
    evaluate_admission,
)
from .geometry import (
    GEOMETRY_BLOCKS_REQUIRED,
    BlockGeometryInput,
    FacePair,
    FaceScores,
    N_VERTICES,
    evaluate_block_geometry,
    evaluate_geometry,
)

#: Controls are one-to-one with the optimized cross-objective faces.
N_CONTROL_FACES = 2

#: Rejection threshold on empirical false-pass for the complete pipeline.
MAX_FALSE_PASS = 0.05


@dataclass(frozen=True)
class NullParams:
    """Adversarial-but-plausible null settings."""

    noise_sd: float = 0.15            # per-vertex measurement noise, score units
    disc_val_rho: float = 0.30        # shared discovery/validation noise component
    block_scale_sd: float = 0.25      # per-block multiplicative heterogeneity
    admission_effect_mean: float = 12.0   # pp, just below the 15 pp threshold
    admission_effect_sd: float = 4.0
    admission_block_sd: float = 3.0
    admission_auroc_mean: float = 0.62
    admission_auroc_sd: float = 0.06


def _weights(rng: random.Random) -> list[float]:
    return [rng.expovariate(1.0) + 0.05 for _ in range(4)]


def _true_face(rng: random.Random, *, objective_specific: bool, scale: float):
    """True (noiseless) vertex scores for one face.

    ``objective_specific=False`` ties objective B's position weights to A's, so
    the face has zero true front-loading. ``True`` draws them independently,
    giving real geometry that is nonetheless identical in distribution for cross
    and same faces.
    """
    a = _weights(rng)
    b = [w * (1.0 + rng.gauss(0, 1e-9)) for w in a] if not objective_specific else _weights(rng)
    s_a, s_b = [], []
    for m in range(N_VERTICES):
        inside = [i for i in range(4) if (m >> i) & 1]
        outside = [i for i in range(4) if not (m >> i) & 1]
        s_a.append(scale * sum(a[i] for i in inside))
        s_b.append(scale * sum(b[i] for i in outside))
    return tuple(s_a), tuple(s_b)


def _measure(rng: random.Random, true_scores, p: NullParams, shared) -> tuple[float, ...]:
    """Add measurement noise with a component shared with the paired measurement."""
    out = []
    for value, sh in zip(true_scores, shared):
        indep = rng.gauss(0.0, p.noise_sd)
        out.append(value + math.sqrt(p.disc_val_rho) * sh
                   + math.sqrt(max(0.0, 1.0 - p.disc_val_rho)) * indep)
    return tuple(out)


def _face_pair(rng: random.Random, p: NullParams, *, objective_specific: bool, scale: float) -> FacePair:
    ta, tb = _true_face(rng, objective_specific=objective_specific, scale=scale)
    shared_a = [rng.gauss(0.0, p.noise_sd) for _ in range(N_VERTICES)]
    shared_b = [rng.gauss(0.0, p.noise_sd) for _ in range(N_VERTICES)]
    return FacePair(
        discovery=FaceScores(
            s_a=_measure(rng, ta, p, shared_a), s_b=_measure(rng, tb, p, shared_b)
        ),
        validation=FaceScores(
            s_a=_measure(rng, ta, p, shared_a), s_b=_measure(rng, tb, p, shared_b)
        ),
    )


def simulate_geometry_study(rng: random.Random, p: NullParams, *, objective_specific: bool) -> bool:
    """One simulated six-block geometry study. True when the pipeline would pass it."""
    results = []
    for b in range(GEOMETRY_BLOCKS_REQUIRED):
        scale = math.exp(rng.gauss(0.0, p.block_scale_sd))  # per-block heterogeneity
        kw = {"objective_specific": objective_specific, "scale": scale}
        results.append(evaluate_block_geometry(BlockGeometryInput(
            block=f"B{b + 1}",
            cross_a1_b1=_face_pair(rng, p, **kw),
            cross_a2_b2=_face_pair(rng, p, **kw),
            same_a1_a2=_face_pair(rng, p, **kw),
            same_b1_b2=_face_pair(rng, p, **kw),
            controls=tuple(_face_pair(rng, p, **kw) for _ in range(N_CONTROL_FACES)),
        )))
    return bool(evaluate_geometry(results)["passed"])


def _package(rng: random.Random, p: NullParams, block_offset: float) -> PackageBlockMeasurement:
    """A package with no true advantage over its own matched placebo."""
    effect = max(0.0, rng.gauss(p.admission_effect_mean + block_offset, p.admission_effect_sd))
    placebo = max(0.0, rng.gauss(p.admission_effect_mean + block_offset, p.admission_effect_sd))
    sham = max(0.0, rng.gauss(p.admission_effect_mean + block_offset, p.admission_effect_sd))
    reach = rng.gauss(0.0, 1.0)
    return PackageBlockMeasurement(
        effect_pp=effect,
        placebo_effect_pp=placebo,
        sham_effect_pp=sham,
        benign_utility_loss_pp=max(0.0, rng.gauss(2.0, 2.0)),
        heldout_auroc=min(0.999, max(0.0, rng.gauss(p.admission_auroc_mean, p.admission_auroc_sd))),
        reach_own_improvement=reach,
        reach_random_control=rng.gauss(0.0, 1.0),
        finite_numerics=True,
        replay_ok=True,
    )


def simulate_admission_study(rng: random.Random, p: NullParams) -> bool:
    """One simulated eight-block admission study under the no-effect null."""
    blocks = []
    for i in range(BLOCKS_TOTAL):
        offset = rng.gauss(0.0, p.admission_block_sd)   # block-correlated shift
        blocks.append(BlockMeasurement(
            block=f"A{i + 1}",
            package_a=_package(rng, p, offset),
            package_b=_package(rng, p, offset),
            abs_spearman_ab=abs(rng.gauss(0.0, 0.35)),
            median_grad_cosine=abs(rng.gauss(0.0, 0.35)),
            cross_matrix=((rng.gauss(1, 0.5), rng.gauss(1, 0.5)),
                          (rng.gauss(1, 0.5), rng.gauss(1, 0.5))),
        ))
    return bool(evaluate_admission(blocks).passed)


def run(n_sims: int = 20000, seed: int = 20260821, p: NullParams | None = None) -> dict:
    """Estimate empirical false-pass rates. The complete-pipeline rate is the gate."""
    p = p or NullParams()
    rng = random.Random(seed)

    geo_n1 = sum(simulate_geometry_study(rng, p, objective_specific=False) for _ in range(n_sims))
    geo_n2 = sum(simulate_geometry_study(rng, p, objective_specific=True) for _ in range(n_sims))
    adm = sum(simulate_admission_study(rng, p) for _ in range(n_sims))

    joint = 0
    for _ in range(n_sims):
        if simulate_admission_study(rng, p) and simulate_geometry_study(
            rng, p, objective_specific=True
        ):
            joint += 1

    def ci(k: int) -> list[float]:
        """Wilson 95% interval, which stays sane at k = 0."""
        z, n = 1.959964, n_sims
        ph = k / n
        d = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / d
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
        return [round(max(0.0, c - h), 5), round(min(1.0, c + h), 5)]

    out = {
        "n_sims": n_sims,
        "seed": seed,
        "params": p.__dict__,
        "geometry_null_N1_no_geometry": {"false_pass": geo_n1 / n_sims, "ci95": ci(geo_n1)},
        "geometry_null_N2_not_objective_specific": {"false_pass": geo_n2 / n_sims, "ci95": ci(geo_n2)},
        "admission_null_no_effect": {"false_pass": adm / n_sims, "ci95": ci(adm)},
        "complete_pipeline_null": {"false_pass": joint / n_sims, "ci95": ci(joint)},
        "max_allowed_false_pass": MAX_FALSE_PASS,
    }
    worst = max(
        out["geometry_null_N1_no_geometry"]["false_pass"],
        out["geometry_null_N2_not_objective_specific"]["false_pass"],
        out["admission_null_no_effect"]["false_pass"],
        out["complete_pipeline_null"]["false_pass"],
    )
    out["worst_false_pass"] = worst
    out["design_accepted"] = worst <= MAX_FALSE_PASS
    return out


# --- Power ------------------------------------------------------------------
# A gate with a false-pass of zero is only useful if it can still fire. These
# functions simulate plausible alternatives so the protocol can state honestly
# how often a real effect of a given size would actually clear the gate.


def simulate_geometry_alternative(rng: random.Random, p: NullParams) -> bool:
    """Cross faces carry objective-specific geometry; controls and same faces do not."""
    results = []
    for b in range(GEOMETRY_BLOCKS_REQUIRED):
        scale = math.exp(rng.gauss(0.0, p.block_scale_sd))
        results.append(evaluate_block_geometry(BlockGeometryInput(
            block=f"B{b + 1}",
            cross_a1_b1=_face_pair(rng, p, objective_specific=True, scale=scale),
            cross_a2_b2=_face_pair(rng, p, objective_specific=True, scale=scale),
            same_a1_a2=_face_pair(rng, p, objective_specific=False, scale=scale),
            same_b1_b2=_face_pair(rng, p, objective_specific=False, scale=scale),
            controls=tuple(
                _face_pair(rng, p, objective_specific=False, scale=scale)
                for _ in range(N_CONTROL_FACES)
            ),
        )))
    return bool(evaluate_geometry(results)["passed"])


def simulate_admission_alternative(
    rng: random.Random, p: NullParams, *, effect_pp: float, placebo_pp: float, auroc: float
) -> tuple[bool, bool]:
    """Admission under a real effect. Returns (passed, causally_distinct)."""
    def pkg(offset: float) -> PackageBlockMeasurement:
        return PackageBlockMeasurement(
            effect_pp=max(0.0, rng.gauss(effect_pp + offset, p.admission_effect_sd)),
            placebo_effect_pp=max(0.0, rng.gauss(placebo_pp, 2.0)),
            sham_effect_pp=max(0.0, rng.gauss(placebo_pp, 2.0)),
            benign_utility_loss_pp=max(0.0, rng.gauss(2.0, 1.5)),
            heldout_auroc=min(0.999, rng.gauss(auroc, 0.05)),
            reach_own_improvement=abs(rng.gauss(1.5, 0.6)),
            reach_random_control=rng.gauss(0.0, 0.5),
            finite_numerics=True,
            replay_ok=True,
        )

    blocks = []
    for i in range(BLOCKS_TOTAL):
        offset = rng.gauss(0.0, 2.0)
        blocks.append(BlockMeasurement(
            block=f"A{i + 1}",
            package_a=pkg(offset), package_b=pkg(offset),
            abs_spearman_ab=abs(rng.gauss(0.30, 0.15)),
            median_grad_cosine=abs(rng.gauss(0.30, 0.15)),
            cross_matrix=((rng.gauss(3, 0.5), rng.gauss(0.5, 0.3)),
                          (rng.gauss(0.5, 0.3), rng.gauss(3, 0.5))),
        ))
    r = evaluate_admission(blocks)
    return r.passed, r.causally_distinct


def run_power(n_sims: int = 5000, seed: int = 20260822, p: NullParams | None = None) -> dict:
    p = p or NullParams()
    rng = random.Random(seed)
    geo = sum(simulate_geometry_alternative(rng, p) for _ in range(n_sims)) / n_sims
    grid = {}
    for effect, placebo, auroc in ((22, 8, 0.80), (25, 5, 0.85), (30, 6, 0.90), (35, 4, 0.92)):
        outcomes = [
            simulate_admission_alternative(rng, p, effect_pp=effect, placebo_pp=placebo, auroc=auroc)
            for _ in range(n_sims)
        ]
        grid[f"effect{effect}_placebo{placebo}_auroc{auroc}"] = {
            "power_pass": sum(o[0] for o in outcomes) / n_sims,
            "power_causally_distinct": sum(o[1] for o in outcomes) / n_sims,
        }
    return {"n_sims": n_sims, "seed": seed, "geometry_power": geo, "admission_power": grid}


def main(argv=None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Null and power simulation of the full pipeline.")
    ap.add_argument("--n-sims", type=int, default=20000)
    ap.add_argument("--power-sims", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    report = {"null": run(n_sims=args.n_sims), "power": run_power(n_sims=args.power_sims)}
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        from pathlib import Path
        from .firewall import guard_path

        path = guard_path(args.out, role="nullsim output")
        path.parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
