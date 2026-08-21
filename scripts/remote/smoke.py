#!/usr/bin/env python3
"""GPU smoke test on the permanently excluded engineering prompts.

Runs before any admission budget is spent, to prove the machinery works end to
end on the real model: the pinned revision loads, the chat template matches its
frozen hash, both readout sites produce finite values, both objectives are
computable, both interventions move what they should, and the canonical replay
contract holds.

It uses only the `engineering` role, which is permanently excluded from every
result, and it fits throwaway packages that are discarded. Nothing here can
enter the study.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from pgeom.batching import collate, iter_batches
from pgeom.candidates import eligible_tokens, hash_sampled_bank
from pgeom.firewall import guard_path
from pgeom.fitting import fit_package_a, fit_package_b
from pgeom.interventions import attention_route_bias, residual_dose
from pgeom.model import LAYER_A, load, readouts
from pgeom.objectives import auroc, centered_cosine, spearman
from pgeom.tokenization import build_layout, is_canonical

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = Path("/workspace/pgeom/artifacts")
NEUTRAL_WORDS = (" also", " then", " here", " thus")


def load_role(role: str) -> list[dict]:
    path = guard_path(REPO / "prompts" / "roles" / f"{role}.jsonl", role="prompt role")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report: dict = {"phase": "gpu_smoke", "role_used": "engineering"}
    t0 = time.time()

    lm = load()                       # verifies the frozen template hash on load
    report["load_seconds"] = round(time.time() - t0, 1)
    report["device"] = str(lm.device)
    report["n_layers"], report["n_heads"] = lm.n_layers, lm.n_heads
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

    rows = load_role("engineering")
    harmful = [r for r in rows if r["kind"] == "harmful"]
    benign = [r for r in rows if r["kind"] != "harmful"]
    report["n_harmful"], report["n_benign"] = len(harmful), len(benign)

    tok = lm.tokenizer
    suffix = [tok.encode(w, add_special_tokens=False)[0] for w in NEUTRAL_WORDS]
    report["neutral_suffix_ids"] = suffix

    def layouts_for(items):
        out = []
        for r in items:
            lay = build_layout(r["text"], suffix, tok)
            if not is_canonical(lay, tok):
                raise RuntimeError(f"neutral suffix is not canonical on prompt {r['id']}")
            out.append(lay)
        return out

    lay_h, lay_b = layouts_for(harmful), layouts_for(benign)
    report["canonical_replay_all_prompts"] = True

    pad = tok.eos_token_id

    def gather(layouts):
        resid, mass = [], []
        for _, batch in iter_batches(layouts, pad_id=pad, size=4):
            batch = batch.to(lm.device)
            with torch.no_grad():
                out = lm.model(input_ids=batch.input_ids,
                               attention_mask=batch.attention_mask,
                               position_ids=batch.position_ids,
                               output_hidden_states=True, output_attentions=True,
                               use_cache=False)
            resid.append(out.hidden_states[lm.hidden_index_for_block(LAYER_A)][:, -1, :].float().cpu())
            m = torch.zeros(len(batch), lm.n_layers, lm.n_heads)
            for li, attn in enumerate(out.attentions):
                row = attn[:, :, -1, :].float()
                for b, (s, e) in enumerate(batch.instruction_spans):
                    m[b, li, :] = row[b, :, s:e].sum(-1).cpu()
            mass.append(m)
            del out
        return torch.cat(resid), torch.cat(mass)

    t1 = time.time()
    rh, mh = gather(lay_h)
    rb, mb = gather(lay_b)
    n = len(lay_h) + len(lay_b)
    report["forward_seconds_per_prompt"] = round((time.time() - t1) / n, 3)
    report["resid_finite"] = bool(torch.isfinite(rh).all() and torch.isfinite(rb).all())
    report["attn_mass_finite"] = bool(torch.isfinite(mh).all() and torch.isfinite(mb).all())
    report["instruction_mass_mean_harmful"] = round(float(mh.mean()), 5)
    report["instruction_mass_mean_benign"] = round(float(mb.mean()), 5)

    # Throwaway packages, fitted on excluded prompts purely to exercise the code.
    pkg_a = fit_package_a(rh, rb, layer=LAYER_A)
    pkg_b = fit_package_b(mh, mb)
    report["throwaway_a_sha"] = pkg_a.fit_sha256[:16]
    report["throwaway_b_heads"] = [list(h) for h in pkg_b.heads]

    sa = torch.cat([pkg_a.score(rh), pkg_a.score(rb)])
    sb = torch.cat([pkg_b.score(mh), pkg_b.score(mb)])
    labels = torch.cat([torch.ones(len(rh)), torch.zeros(len(rb))])
    report["throwaway_auroc_a"] = round(auroc(sa, labels), 4)
    report["throwaway_auroc_b"] = round(auroc(sb, labels), 4)
    report["throwaway_spearman_ab"] = round(spearman(sa, sb), 4)

    # Both interventions must actually move their own objective on real weights.
    batch = collate(lay_h[:4], pad_id=pad).to(lm.device)
    with torch.no_grad():
        base = readouts(lm, batch.input_ids, batch.instruction_spans)
    base_a = float(pkg_a.score(base.resid).mean())
    base_b = float(pkg_b.score(base.instr_mass).mean())

    sigma = float(sa.std())
    with residual_dose(lm.model, layer=LAYER_A, direction=pkg_a.direction.to(lm.device),
                       alpha=-1.0 * sigma):
        with torch.no_grad():
            dosed = readouts(lm, batch.input_ids, batch.instruction_spans, need_attention=False)
    report["intervention_a_delta"] = round(float(pkg_a.score(dosed.resid).mean()) - base_a, 5)
    report["intervention_a_dose_sigma"] = round(sigma, 5)

    with attention_route_bias(lm.model, heads=pkg_b.heads,
                              spans=batch.instruction_spans, delta=2.0):
        with torch.no_grad():
            routed = readouts(lm, batch.input_ids, batch.instruction_spans)
    report["intervention_b_delta"] = round(float(pkg_b.score(routed.instr_mass).mean()) - base_b, 5)

    # Gradient with respect to the suffix one-hots, which is what GCG needs.
    embed = lm.model.get_input_embeddings()
    one_hot = torch.zeros(len(batch), 4, embed.weight.shape[0],
                          device=lm.device, dtype=embed.weight.dtype)
    for r, sl in enumerate(batch.suffix_slices):
        one_hot[r].scatter_(1, batch.input_ids[r, sl].unsqueeze(1), 1.0)
    one_hot.requires_grad_(True)
    embeds = embed(batch.input_ids).detach().clone()
    for r, sl in enumerate(batch.suffix_slices):
        embeds[r, sl] = one_hot[r] @ embed.weight
    out = lm.model(inputs_embeds=embeds, attention_mask=batch.attention_mask,
                   position_ids=batch.position_ids, output_hidden_states=True, use_cache=False)
    loss = (out.hidden_states[lm.hidden_index_for_block(LAYER_A)][:, -1, :].float()
            @ pkg_a.direction.to(lm.device)).mean()
    loss.backward()
    grad = one_hot.grad
    report["suffix_gradient_finite"] = bool(torch.isfinite(grad).all())
    report["suffix_gradient_norm"] = round(float(grad.norm()), 4)

    bank = hash_sampled_bank(eligible_tokens(tok), key="pgeom-smoke", size=256)
    report["bank_size"] = len(bank)
    if torch.cuda.is_available():
        report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    report["total_seconds"] = round(time.time() - t0, 1)

    (ARTIFACTS / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
