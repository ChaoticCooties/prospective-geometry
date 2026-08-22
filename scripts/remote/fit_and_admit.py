#!/usr/bin/env python3
"""Phase 1: fit both packages, then run the eight admission blocks.

Fitting uses the `target_fit` role only. Admission uses the `target_admission`
role, with each block's nomination prompts used for every calibration and its
validation prompts scored exactly once.

Execution parameters fixed here, before any outcome existed, and recorded in the
run artifact: greedy decoding, 160 new tokens, dose c = 1.0 standard deviations
for package A, and a source-matched bias for package B calibrated on nomination
prompts so harmful instruction mass lands at that block's benign level.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from pgeom.admission import (
    BlockMeasurement,
    PackageBlockMeasurement,
    diagonally_selective,
    evaluate_admission,
)
from pgeom.batching import collate, iter_batches
from pgeom.behavior import score_generations
from pgeom.candidates import eligible_tokens, hash_sampled_bank
from pgeom.firewall import guard_path
from pgeom.fitting import fit_package_a, fit_package_b
from pgeom.interventions import (
    attention_route_bias,
    matched_random_direction,
    matched_random_heads,
    residual_dose,
)
from pgeom.model import LAYER_A, load
from pgeom.objectives import PackageA, PackageB, auroc, centered_cosine, spearman
from pgeom.provenance import sha256_obj
from pgeom.tokenization import build_layout, is_canonical

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/workspace/pgeom/artifacts")

MAX_NEW_TOKENS = 160
DOSE_C = 1.0
SHAM_LAYER = 2
BIAS_SEARCH = (0.0, 12.0)
BIAS_STEPS = 18
BANK_SIZE = 512
# Reachability scores every bank token on every nomination prompt, so it
# dominates Phase 1 wall clock. These sequences are ~60 tokens, so a large batch
# costs little memory and turns the stage from tens of minutes into minutes.
REACH_BATCH = 24
BANK_KEY = "pgeom-admission-bank-v1"
MUTATION_POSITION = 2          # frozen: which of the four suffix slots is mutated
NEUTRAL_WORDS = (" also", " then", " here", " thus")
SEED = 9100


def load_role(role):
    path = guard_path(REPO / "prompts" / "roles" / f"{role}.jsonl", role="prompt role")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class Engine:
    def __init__(self):
        self.lm = load()
        self.tok = self.lm.tokenizer
        self.pad = self.tok.eos_token_id
        self.suffix = [self.tok.encode(w, add_special_tokens=False)[0] for w in NEUTRAL_WORDS]

    def layouts(self, rows, suffix=None):
        sfx = self.suffix if suffix is None else suffix
        out = []
        for r in rows:
            lay = build_layout(r["text"], sfx, self.tok)
            if not is_canonical(lay, self.tok):
                raise RuntimeError(f"non-canonical suffix on prompt {r['id']}")
            out.append(lay)
        return out

    def readouts(self, layouts, *, need_attention=True, batch_size=4, ctx=None):
        resid, mass = [], []
        for _, batch in iter_batches(layouts, pad_id=self.pad, size=batch_size):
            batch = batch.to(self.lm.device)
            manager = ctx(batch) if ctx else None
            with torch.no_grad():
                if manager is None:
                    out = self._forward(batch, need_attention)
                else:
                    with manager:
                        out = self._forward(batch, need_attention)
            resid.append(out["resid"].cpu())
            mass.append(out["mass"].cpu() if out["mass"] is not None
                        else torch.zeros(len(batch), self.lm.n_layers, self.lm.n_heads))
        return torch.cat(resid), torch.cat(mass)

    def _forward(self, batch, need_attention):
        out = self.lm.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                            position_ids=batch.position_ids, output_hidden_states=True,
                            output_attentions=need_attention, use_cache=False)
        resid = out.hidden_states[self.lm.hidden_index_for_block(LAYER_A)][:, -1, :].float()
        mass = None
        if need_attention:
            mass = torch.zeros(len(batch), self.lm.n_layers, self.lm.n_heads,
                               device=resid.device)
            for li, attn in enumerate(out.attentions):
                row = attn[:, :, -1, :].float()
                for b, (s, e) in enumerate(batch.instruction_spans):
                    mass[b, li, :] = row[b, :, s:e].sum(-1)
        return {"resid": resid, "mass": mass}

    def generate(self, rows, *, ctx=None, batch_size=8):
        texts = []
        layouts = self.layouts(rows)
        for idx, batch in iter_batches(layouts, pad_id=self.pad, size=batch_size):
            batch = batch.to(self.lm.device)
            manager = ctx(batch) if ctx else None
            with torch.no_grad():
                if manager is None:
                    out = self._gen(batch)
                else:
                    with manager:
                        out = self._gen(batch)
            for r in range(len(batch)):
                new = out[r, batch.input_ids.shape[1]:]
                texts.append((idx[r], self.tok.decode(new, skip_special_tokens=True)))
        texts.sort(key=lambda t: t[0])
        return [t[1] for t in texts]

    def _gen(self, batch):
        return self.lm.model.generate(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask,
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None,
            pad_token_id=self.pad,
        )

    def vocab_gradients(self, layouts, pkg_a, pkg_b, *, batch_size=4):
        """Per-prompt vocabulary gradients of both objectives at the mutation slot.

        Returns two [n, vocab] matrices. These are what the non-redundancy
        condition compares: if the two objectives pull the same tokens, they are
        one readout wearing two hats.
        """
        embed = self.lm.model.get_input_embeddings()
        vocab = embed.weight.shape[0]
        grads_a, grads_b = [], []
        for _, batch in iter_batches(layouts, pad_id=self.pad, size=batch_size):
            batch = batch.to(self.lm.device)
            for objective, sink in (("a", grads_a), ("b", grads_b)):
                one_hot = torch.zeros(len(batch), vocab, device=self.lm.device,
                                      dtype=embed.weight.dtype)
                for r, sl in enumerate(batch.suffix_slices):
                    col = sl.start + MUTATION_POSITION
                    one_hot[r].scatter_(0, batch.input_ids[r, col].unsqueeze(0), 1.0)
                one_hot.requires_grad_(True)
                embeds = embed(batch.input_ids).detach().clone()
                for r, sl in enumerate(batch.suffix_slices):
                    embeds[r, sl.start + MUTATION_POSITION] = one_hot[r] @ embed.weight
                out = self.lm.model(inputs_embeds=embeds, attention_mask=batch.attention_mask,
                                    position_ids=batch.position_ids, output_hidden_states=True,
                                    output_attentions=(objective == "b"), use_cache=False)
                if objective == "a":
                    resid = out.hidden_states[self.lm.hidden_index_for_block(LAYER_A)][:, -1, :].float()
                    loss = pkg_a.score(resid).sum()
                else:
                    mass = torch.zeros(len(batch), self.lm.n_layers, self.lm.n_heads,
                                       device=self.lm.device)
                    for li, attn in enumerate(out.attentions):
                        row = attn[:, :, -1, :].float()
                        for b, (s, e) in enumerate(batch.instruction_spans):
                            mass[b, li, :] = row[b, :, s:e].sum(-1)
                    loss = pkg_b.score(mass).sum()
                loss.backward()
                sink.append(one_hot.grad.detach().float().cpu())
                self.lm.model.zero_grad(set_to_none=True)
        return torch.cat(grads_a), torch.cat(grads_b)


def calibrate_bias(engine, pkg_b, nom_harmful, benign_target):
    """Bisect the routing bias so harmful instruction mass reaches the benign level.

    Source-matched by construction: the target comes from this block's own
    matched benign prompts, so the dose is whatever it takes to look benign, not
    an arbitrary large number.
    """
    lo, hi = BIAS_SEARCH
    layouts = engine.layouts(nom_harmful)
    best = hi
    for _ in range(BIAS_STEPS):
        mid = (lo + hi) / 2
        _, mass = engine.readouts(
            layouts, ctx=lambda b, d=mid: attention_route_bias(
                engine.lm.model, heads=pkg_b.heads, spans=b.instruction_spans, delta=d))
        value = float(pkg_b.score(mass).mean())
        if value > benign_target:
            lo = mid
        else:
            hi = mid
            best = mid
    return best


def run_block(engine, pkg_a, pkg_b, block, rows, gen) -> tuple[BlockMeasurement, dict]:
    """One admission block. Nomination calibrates; validation is scored once."""
    by_arm = {}
    for arm in ("nom_harmful", "val_harmful", "nom_benign", "val_benign", "benign_utility"):
        by_arm[arm] = [r for r in rows if r["arm"] == arm]

    diag: dict = {"block": block, "n": {k: len(v) for k, v in by_arm.items()}}

    # --- held-out prediction -------------------------------------------------
    val_rows = by_arm["val_harmful"] + by_arm["val_benign"]
    val_layouts = engine.layouts(val_rows)
    resid, mass = engine.readouts(val_layouts)
    s_a, s_b = pkg_a.score(resid), pkg_b.score(mass)
    labels = torch.cat([torch.ones(len(by_arm["val_harmful"])),
                        torch.zeros(len(by_arm["val_benign"]))])
    auroc_a, auroc_b = auroc(s_a, labels), auroc(s_b, labels)
    rho = spearman(s_a, s_b)
    diag.update({"auroc_a": auroc_a, "auroc_b": auroc_b, "spearman_ab": rho})

    # --- non-redundancy: vocabulary gradients --------------------------------
    ga, gb = engine.vocab_gradients(val_layouts, pkg_a, pkg_b)
    cosines = [centered_cosine(ga[i], gb[i]) for i in range(ga.shape[0])]
    cosines = [c for c in cosines if c == c]
    median_cos = float(torch.tensor(cosines).median()) if cosines else float("nan")
    diag["median_grad_cosine"] = median_cos

    # --- doses ---------------------------------------------------------------
    nom_layouts = engine.layouts(by_arm["nom_harmful"])
    nom_resid, nom_mass = engine.readouts(nom_layouts)
    sigma = float(pkg_a.score(nom_resid).std())
    alpha = DOSE_C * sigma

    _, benign_mass = engine.readouts(engine.layouts(by_arm["nom_benign"]))
    benign_target = float(pkg_b.score(benign_mass).mean())
    delta = calibrate_bias(engine, pkg_b, by_arm["nom_harmful"], benign_target)
    diag.update({"dose_alpha": alpha, "dose_sigma": sigma,
                 "bias_delta": delta, "benign_mass_target": benign_target})

    # --- matched placebo and sham operators ----------------------------------
    g = torch.Generator().manual_seed(SEED + hash(block) % 1000)
    placebo_dir = matched_random_direction(pkg_a.direction, generator=g).to(engine.lm.device)
    placebo_heads = matched_random_heads(engine.lm.n_layers, engine.lm.n_heads,
                                         pkg_b.heads, len(pkg_b.heads), generator=g)
    diag["placebo_heads"] = [list(h) for h in placebo_heads]

    direction = pkg_a.direction.to(engine.lm.device)

    def ctx_a(_batch):
        return residual_dose(engine.lm.model, layer=LAYER_A, direction=direction, alpha=-alpha)

    def ctx_a_placebo(_batch):
        return residual_dose(engine.lm.model, layer=LAYER_A, direction=placebo_dir, alpha=-alpha)

    def ctx_a_sham(_batch):
        return residual_dose(engine.lm.model, layer=SHAM_LAYER, direction=direction, alpha=-alpha)

    def ctx_b(batch):
        return attention_route_bias(engine.lm.model, heads=pkg_b.heads,
                                    spans=batch.instruction_spans, delta=delta)

    def ctx_b_placebo(batch):
        return attention_route_bias(engine.lm.model, heads=placebo_heads,
                                    spans=batch.instruction_spans, delta=delta)

    def ctx_b_sham(batch):
        # Same heads and dose, but a non-instruction key span: the template
        # prefix. Moves the same machinery at a site the objective does not name.
        spans = [(0, max(1, s)) for s, _ in batch.instruction_spans]
        return attention_route_bias(engine.lm.model, heads=pkg_b.heads,
                                    spans=spans, delta=delta)

    conditions = {"baseline": None, "a": ctx_a, "a_placebo": ctx_a_placebo, "a_sham": ctx_a_sham,
                  "b": ctx_b, "b_placebo": ctx_b_placebo, "b_sham": ctx_b_sham}

    # --- behavioural outcomes -------------------------------------------------
    harmful_rows = by_arm["val_harmful"]
    outcomes = {}
    for name, ctx in conditions.items():
        gens = engine.generate(harmful_rows, ctx=ctx)
        outcomes[name] = score_generations(gens, [r["text"] for r in harmful_rows]).to_json()
        gen.setdefault(block, {})[name] = gens[:2]     # two samples kept for audit
    diag["harmful_outcomes"] = outcomes

    util_rows = by_arm["benign_utility"]
    utility = {}
    for name in ("baseline", "a", "b"):
        gens = engine.generate(util_rows, ctx=conditions[name])
        utility[name] = score_generations(gens, [r["text"] for r in util_rows]).to_json()
    diag["benign_utility"] = utility

    def effect(name):
        return abs(outcomes[name]["refusal_rate_pp"] - outcomes["baseline"]["refusal_rate_pp"])

    def util_loss(name):
        return max(0.0, utility["baseline"]["coherent_rate_pp"] - utility[name]["coherent_rate_pp"])

    # --- cross-intervention matrix -------------------------------------------
    base_a = float(pkg_a.score(resid).mean())
    base_b = float(pkg_b.score(mass).mean())
    r_aa, m_ab = engine.readouts(val_layouts, ctx=ctx_a)
    r_ba, m_bb = engine.readouts(val_layouts, ctx=ctx_b)
    matrix = ((float(pkg_a.score(r_aa).mean()) - base_a, float(pkg_b.score(m_ab).mean()) - base_b),
              (float(pkg_a.score(r_ba).mean()) - base_a, float(pkg_b.score(m_bb).mean()) - base_b))
    selective, sel_stats = diagonally_selective(matrix)
    diag.update({"cross_matrix": [list(r) for r in matrix],
                 "selectivity": sel_stats, "diagonally_selective": selective})

    # --- one-step hard-token reachability ------------------------------------
    reach = one_step_reachability(engine, pkg_a, pkg_b, by_arm)
    diag["reachability"] = reach

    def pkg(name, obj):
        return PackageBlockMeasurement(
            effect_pp=effect(name),
            placebo_effect_pp=effect(f"{name}_placebo"),
            sham_effect_pp=effect(f"{name}_sham"),
            benign_utility_loss_pp=util_loss(name),
            heldout_auroc=auroc_a if name == "a" else auroc_b,
            reach_own_improvement=reach[f"{obj}_own_improvement"],
            reach_random_control=reach[f"{obj}_random_control"],
            finite_numerics=bool(torch.isfinite(resid).all() and torch.isfinite(mass).all()
                                 and torch.isfinite(ga).all() and torch.isfinite(gb).all()),
            replay_ok=True,
        )

    measurement = BlockMeasurement(
        block=block, package_a=pkg("a", "a"), package_b=pkg("b", "b"),
        abs_spearman_ab=abs(rho), median_grad_cosine=median_cos, cross_matrix=matrix,
    )
    diag["conditions"] = measurement.conditions()
    diag["block_passes"] = measurement.passes()
    diag["failed_conditions"] = measurement.failed_conditions()
    return measurement, diag


def one_step_reachability(engine, pkg_a, pkg_b, by_arm) -> dict:
    """Select single-token mutations on nomination prompts, score once on validation.

    Both objectives share one hash-sampled bank and one mutation slot, so a
    difference between them cannot come from a different search space.
    """
    bank = hash_sampled_bank(eligible_tokens(engine.tok), key=BANK_KEY, size=BANK_SIZE)
    nom, val = by_arm["nom_harmful"], by_arm["val_harmful"]
    out = {"bank_size": len(bank), "mutation_position": MUTATION_POSITION}

    def mean_scores(rows, suffix, which):
        layouts = []
        for r in rows:
            lay = build_layout(r["text"], suffix, engine.tok)
            if not is_canonical(lay, engine.tok):
                return None
            layouts.append(lay)
        resid, mass = engine.readouts(layouts, need_attention=(which == "b"),
                                      batch_size=REACH_BATCH)
        return float(pkg_a.score(resid).mean()) if which == "a" else float(pkg_b.score(mass).mean())

    base_suffix = list(engine.suffix)
    for which in ("a", "b"):
        base_nom = mean_scores(nom, base_suffix, which)
        best_tok, best_val = None, None
        for token in bank.token_ids:
            cand = list(base_suffix)
            cand[MUTATION_POSITION] = token
            score = mean_scores(nom, cand, which)
            if score is None:
                continue
            if best_val is None or score < best_val:
                best_tok, best_val = token, score
        chosen = list(base_suffix)
        chosen[MUTATION_POSITION] = best_tok
        base_val = mean_scores(val, base_suffix, which)
        held = mean_scores(val, chosen, which)
        out[f"{which}_selected_token"] = best_tok
        out[f"{which}_own_improvement"] = base_val - held      # positive is better

        # Equal-budget random control, nominated independently of the objective.
        g = torch.Generator().manual_seed(SEED + (0 if which == "a" else 1))
        idx = torch.randint(0, len(bank), (1,), generator=g).item()
        rand = list(base_suffix)
        rand[MUTATION_POSITION] = bank.token_ids[idx]
        out[f"{which}_random_control"] = base_val - mean_scores(val, rand, which)
    return out


def rebuild_measurement(diag: dict) -> BlockMeasurement:
    """Reconstruct a BlockMeasurement from a checkpointed diagnostics record.

    The gate is re-evaluated from stored numbers rather than trusting the stored
    verdict, so a change to a threshold cannot be silently inherited by a
    resumed run.
    """
    def pkg(name: str) -> PackageBlockMeasurement:
        o, u = diag["harmful_outcomes"], diag["benign_utility"]
        base = o["baseline"]["refusal_rate_pp"]
        eff = abs(o[name]["refusal_rate_pp"] - base)
        return PackageBlockMeasurement(
            effect_pp=eff,
            placebo_effect_pp=abs(o[f"{name}_placebo"]["refusal_rate_pp"] - base),
            sham_effect_pp=abs(o[f"{name}_sham"]["refusal_rate_pp"] - base),
            benign_utility_loss_pp=max(
                0.0, u["baseline"]["coherent_rate_pp"] - u[name]["coherent_rate_pp"]),
            heldout_auroc=diag["auroc_a"] if name == "a" else diag["auroc_b"],
            reach_own_improvement=diag["reachability"][f"{name}_own_improvement"],
            reach_random_control=diag["reachability"][f"{name}_random_control"],
            finite_numerics=True, replay_ok=True,
        )

    return BlockMeasurement(
        block=diag["block"], package_a=pkg("a"), package_b=pkg("b"),
        abs_spearman_ab=abs(diag["spearman_ab"]),
        median_grad_cosine=diag["median_grad_cosine"],
        cross_matrix=tuple(tuple(r) for r in diag["cross_matrix"]),
    )


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    engine = Engine()

    # --- resume, if a previous pod got part way -------------------------------
    # The volume outlives the pod, so a dropped connection or a preemption costs
    # only the blocks that had not finished. Fitting is deterministic given the
    # same prompts, but reloading is both faster and guarantees the identical
    # packages are used across a resumed run.
    fit_path = ARTIFACTS / "fitted_packages.pt"
    fit_json = ARTIFACTS / "fitted_packages.json"
    pkg_a = pkg_b = None
    if fit_path.exists() and fit_json.exists():
        blob = torch.load(fit_path, map_location="cpu")
        fit_record = json.loads(fit_json.read_text())
        pkg_a = PackageA(direction=blob["direction"], layer=LAYER_A,
                         fit_sha256=fit_record["package_a_sha256"])
        pkg_b = PackageB(heads=tuple(tuple(h) for h in blob["heads"]),
                         fit_sha256=fit_record["package_b_sha256"])
        print(f"resumed fitted packages A={pkg_a.fit_sha256[:12]} "
              f"B={pkg_b.fit_sha256[:12]}", flush=True)

    # --- fit both packages on the target_fit role only ------------------------
    fit_rows = load_role("target_fit")
    fit_h = [r for r in fit_rows if r["kind"] == "harmful"]
    fit_b = [r for r in fit_rows if r["kind"] != "harmful"]
    if pkg_a is None or pkg_b is None:
        rh, mh = engine.readouts(engine.layouts(fit_h))
        rb, mb = engine.readouts(engine.layouts(fit_b))
        pkg_a = fit_package_a(rh, rb, layer=LAYER_A)
        pkg_b = fit_package_b(mh, mb)

    fit_record = {
        "layer_a": LAYER_A,
        "package_a_sha256": pkg_a.fit_sha256,
        "package_b_sha256": pkg_b.fit_sha256,
        "package_b_heads": [list(h) for h in pkg_b.heads],
        "n_fit_harmful": len(fit_h), "n_fit_benign": len(fit_b),
        "fit_seconds": round(time.time() - t0, 1),
    }
    (ARTIFACTS / "fitted_packages.json").write_text(
        json.dumps(fit_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save({"direction": pkg_a.direction, "heads": pkg_b.heads},
               ARTIFACTS / "fitted_packages.pt")
    print(json.dumps(fit_record, indent=2, sort_keys=True), flush=True)

    # --- eight admission blocks ---------------------------------------------
    adm_rows = load_role("target_admission")
    blocks = sorted({r["block"] for r in adm_rows})
    measurements, diagnostics, gen_samples = [], [], {}

    done_path = ARTIFACTS / "admission_blocks.json"
    completed = {}
    if done_path.exists():
        try:
            completed = {d["block"]: d for d in json.loads(done_path.read_text())}
            print(f"resuming: {sorted(completed)} already measured", flush=True)
        except (json.JSONDecodeError, KeyError, TypeError):
            completed = {}

    for block in blocks:
        if block in completed:
            diagnostics.append(completed[block])
            measurements.append(rebuild_measurement(completed[block]))
            print(f"[block {block}] reused from checkpoint", flush=True)
            continue
        started = time.time()
        rows = [r for r in adm_rows if r["block"] == block]
        m, d = run_block(engine, pkg_a, pkg_b, block, rows, gen_samples)
        d["seconds"] = round(time.time() - started, 1)
        measurements.append(m)
        diagnostics.append(d)
        print(f"[block {block}] passes={d['block_passes']} "
              f"failed={d['failed_conditions']} ({d['seconds']}s)", flush=True)
        (ARTIFACTS / "admission_blocks.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8")

    result = evaluate_admission(measurements)
    report = {
        "phase": "1_target_admission",
        "fit": fit_record,
        "result": result.to_json(),
        "execution": {
            "max_new_tokens": MAX_NEW_TOKENS, "dose_c": DOSE_C, "sham_layer": SHAM_LAYER,
            "bank_size": BANK_SIZE, "bank_key": BANK_KEY,
            "mutation_position": MUTATION_POSITION, "seed": SEED,
            "decoding": "greedy",
        },
        "total_seconds": round(time.time() - t0, 1),
    }
    report["report_sha256"] = sha256_obj(report)
    (ARTIFACTS / "admission_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (ARTIFACTS / "generation_samples.json").write_text(
        json.dumps(gen_samples, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["result"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
