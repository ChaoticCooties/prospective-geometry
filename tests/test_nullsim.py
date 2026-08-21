"""The null simulation must both reject false passes and retain power."""

import json
import random
from pathlib import Path

import pytest

from pgeom.nullsim import (
    MAX_FALSE_PASS,
    NullParams,
    run,
    simulate_admission_study,
    simulate_geometry_alternative,
    simulate_geometry_study,
)

REPORT = Path(__file__).resolve().parents[1] / "reports" / "nullsim_report.json"


def test_small_null_run_is_far_below_the_ceiling():
    out = run(n_sims=400)
    assert out["design_accepted"]
    assert out["worst_false_pass"] <= MAX_FALSE_PASS


def test_geometry_null_rarely_passes():
    rng = random.Random(1)
    p = NullParams()
    passes = sum(simulate_geometry_study(rng, p, objective_specific=True) for _ in range(400))
    assert passes / 400 <= MAX_FALSE_PASS


def test_admission_null_rarely_passes():
    rng = random.Random(2)
    passes = sum(simulate_admission_study(rng, NullParams()) for _ in range(400))
    assert passes / 400 <= MAX_FALSE_PASS


def test_gate_still_fires_under_a_real_effect():
    """A gate that never passes anything would make the study unfalsifiable in
    the useless direction: guaranteed null regardless of the truth."""
    rng = random.Random(3)
    p = NullParams()
    passes = sum(simulate_geometry_alternative(rng, p) for _ in range(400))
    assert passes / 400 > 0.5


@pytest.mark.skipif(not REPORT.exists(), reason="null simulation report not generated yet")
def test_committed_report_meets_the_preregistered_ceiling():
    d = json.loads(REPORT.read_text())
    assert d["null"]["design_accepted"]
    assert d["null"]["worst_false_pass"] <= MAX_FALSE_PASS
    assert d["null"]["n_sims"] >= 20000
    assert d["power"]["geometry_power"] > 0.5
