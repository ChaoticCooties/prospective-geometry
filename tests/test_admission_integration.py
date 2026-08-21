"""End-to-end integration of the admission runner against a tiny local model.

The admission run is the expensive one, so its structure is exercised here on a
randomly initialised model first. This cannot say anything about whether the
real packages admit; it only proves the pipeline runs, the shapes line up, every
condition is computed, and a block verdict comes out.
"""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "remote"))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import pgeom.model as model_mod  # noqa: E402
from pgeom.tokenization import load_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tiny_lm():
    tok = load_tokenizer()
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, attn_implementation="eager",
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    m = LlamaForCausalLM(cfg).eval()
    m.requires_grad_(False)
    return model_mod.LoadedModel(model=m, tokenizer=tok, device=torch.device("cpu"),
                                 n_layers=4, n_heads=4)


@pytest.fixture(scope="module")
def runner(tiny_lm, monkeypatch_module):
    import fit_and_admit as fa
    monkeypatch_module.setattr(fa, "load", lambda: tiny_lm)
    monkeypatch_module.setattr(fa, "LAYER_A", 2)
    monkeypatch_module.setattr(fa, "MAX_NEW_TOKENS", 8)
    monkeypatch_module.setattr(fa, "BANK_SIZE", 8)
    monkeypatch_module.setattr(fa, "BIAS_STEPS", 3)
    return fa


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def rows_for(role, n_each=3):
    path = ROOT / "prompts" / "roles" / f"{role}.jsonl"
    data = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return data


def test_admission_block_runs_end_to_end_and_produces_every_condition(runner, tiny_lm):
    """A structural check: every gate condition must be computed, whatever its value."""
    import fit_and_admit as fa

    fit = rows_for("target_fit")
    fit_h = [r for r in fit if r["kind"] == "harmful"][:4]
    fit_b = [r for r in fit if r["kind"] != "harmful"][:4]

    engine = fa.Engine.__new__(fa.Engine)
    engine.lm = tiny_lm
    engine.tok = tiny_lm.tokenizer
    engine.pad = tiny_lm.tokenizer.eos_token_id
    engine.suffix = [engine.tok.encode(w, add_special_tokens=False)[0]
                     for w in fa.NEUTRAL_WORDS]

    rh, mh = engine.readouts(engine.layouts(fit_h))
    rb, mb = engine.readouts(engine.layouts(fit_b))
    assert torch.isfinite(rh).all() and torch.isfinite(mh).all()

    pkg_a = fa.fit_package_a(rh, rb, layer=fa.LAYER_A)
    pkg_b = fa.fit_package_b(mh, mb)
    assert len(pkg_b.heads) == 8

    adm = rows_for("target_admission")
    block = sorted({r["block"] for r in adm})[0]
    rows = [r for r in adm if r["block"] == block]
    # Trim each arm so the tiny model finishes quickly; structure is unchanged.
    trimmed = []
    for arm in ("nom_harmful", "val_harmful", "nom_benign", "val_benign", "benign_utility"):
        trimmed.extend([r for r in rows if r["arm"] == arm][:2])

    measurement, diag = fa.run_block(engine, pkg_a, pkg_b, block, trimmed, {})

    expected = {
        "A.causal_effect", "A.beats_placebo", "A.benign_utility", "A.heldout_prediction",
        "A.reach_positive", "A.reach_beats_random", "A.numerics_finite", "A.replay_canonical",
        "B.causal_effect", "B.beats_placebo", "B.benign_utility", "B.heldout_prediction",
        "B.reach_positive", "B.reach_beats_random", "B.numerics_finite", "B.replay_canonical",
        "nonredundant_spearman", "nonredundant_grad_cosine",
    }
    assert set(measurement.conditions()) == expected
    assert isinstance(measurement.passes(), bool)

    for key in ("auroc_a", "auroc_b", "spearman_ab", "median_grad_cosine",
                "dose_alpha", "bias_delta", "cross_matrix", "reachability"):
        assert key in diag, key
    assert len(diag["cross_matrix"]) == 2
    assert diag["reachability"]["mutation_position"] == fa.MUTATION_POSITION
    assert "baseline" in diag["harmful_outcomes"]
    assert {"a", "a_placebo", "a_sham", "b", "b_placebo", "b_sham"} <= set(diag["harmful_outcomes"])
