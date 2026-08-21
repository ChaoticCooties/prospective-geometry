# Claim boundary and prior work

This study can claim one narrow thing. Almost every neighbouring idea is already
occupied. This file records who occupies what, so the workshop paper does not
overreach.

## What prior work already occupies

### Discrete attacks against internal objectives

**Winninger, Addad, and Kapusta. "Using Mechanistic Interpretability to Craft
Adversarial Attacks against Large Language Models." arXiv:2503.06269** (v1 8 Mar
2025; v2 6 May 2025; v3 3 Jul 2026). Subspace rerouting identifies *acceptance
subspaces* — feature vectors that do not trigger refusal — and uses
gradient-based optimization to reroute embeddings out of refusal subspaces into
them. Reports 80–95% attack success on Gemma2, Llama3.2, and Qwen2.5 in minutes
or seconds.

**Çakar, Guan, and Kehe. "Optimizing Against Safety Representations:
Activation-Guided Adversarial Suffixes and the Geometry of Refusal."
arXiv:2607.08883** (9 Jul 2026). Activation-Guided GCG replaces output-based
objectives with losses that directly target the model's internal refusal
direction. Finds that suppressing refusal globally across all layers and
positions beats targeting a single layer-position pair, and reads that as
safety representations being distributed across the forward pass rather than
causally localized. Also introduces Soft-GCG, a Gumbel-Softmax relaxation with a
33× speedup.

**Srivastava and Panda. "Attention Is Where You Attack." arXiv:2605.00236**
(30 Apr 2026). The Attention Redistribution Attack identifies safety-critical
attention heads and crafts nonsemantic tokens that redirect attention away from
safety-relevant positions, optimizing over the softmax simplex with
Gumbel-softmax. On LLaMA-3-8B-Instruct, Mistral-7B-Instruct-v0.1, and
Gemma-2-9B-it, with as few as 5 tokens and 500 steps: 36% ASR on Mistral-7B, 30%
on LLaMA-3, 1% on Gemma-2. Its central mechanistic finding is a dissociation
between ablation and redistribution — zeroing the top-ranked safety heads flips
at most 1 of 39–50 baseline refusals, while redistribution flips 72/200 and
60/200.

**Consequence.** This study cannot claim the first internal-target attack, the
first attack on a refusal direction, or the first attack on attention routing.
All three exist. Package B's routing objective is deliberately in ARA's family,
and that debt is stated in the paper.

### Optimization placebos, and the precise limit of Mody's null

**Mody, Agarwal, Mittal, and Mahato. "Minimizing Targeted Activations:
Input-Only Suppression of Evaluation-Awareness Latents in Large Language
Models." arXiv:2607.25907** (28 Jul 2026). Optimizes a fluent prompt to drive a
chosen internal latent toward zero, with no inference-time model access, across
five target constructions: a CAA direction, a subspace norm, an SAE feature, a
single MLP neuron, and a behavioral logit, on Llama-3.2-3B and Llama-3.1-8B.

The much-cited negative result is **narrower than it is usually reported**. The
optimization-placebo null applies **only to the CAA direction on
Llama-3.2-3B-Instruct at layer 14**. Two placebo targets — a CAA direction over
a random relabeling of the contrast, and a fixed random unit direction —
suppress about as hard as the real eval direction (|z| = 6.6 and 10.6 versus
8.3) and shift behavioral eval log-odds about as far (0.43 and 0.51 versus 0.44).

The paper's separate SAE experiment uses **Llama Scope `l16r_8x` on
Llama-3.1-8B-Instruct at layer 16**, feature 3602 (Cohen's d = 2.55). That
experiment has **no matched SAE optimization placebo**. The authors say so
directly:

> Non-specificity is shown only for the direction. Our strongest negative is
> established for the linear CAA direction target (3B, layer 14) [...] We did not
> run this placebo for the neuron, logit, or SAE targets—and for the logit (no
> natural window moves it) and the causally-validated, selective SAE feature the
> evidence in fact points toward specificity.

**Consequence.** Mody's result cannot be cited as showing that internal-target
optimization is generically placebo-equivalent. It shows that for one dense
linear direction on one 3B model at one layer. It is a strong reason to run
matched placebos — which this protocol does, at every admission block, for both
packages — but it is not a general null, and this paper will not present it as
one.

### Prompt interactions, landscapes, perturbation, and token importance

**Qin, Wang, Wang, Wei, and Shen. "Evaluating and Explaining Prompt Sensitivity
of LLMs Using Interactions." arXiv:2608.18539** (19 Aug 2026). Decomposes an
LLM's output score into a set of interactions, each a nonlinear relationship
over a set of input variables, and defines an Interaction-based Prompt
Sensitivity metric. Across 50 open-source LLMs, finds that supervised
fine-tuning, scale, dense architectures, and few-shot learning all reduce prompt
sensitivity through the same route: stabilizing low-order interactions.

**Hintze. "Characterizing Fitness Landscape Structures in Prompt Engineering."
arXiv:2509.05375** (4 Sep 2025). Autocorrelation analysis over semantic
embedding spaces on error-detection tasks, comparing systematic enumeration
(1,024 prompts) against novelty-driven diversification (1,000 prompts).
Systematic generation yields smoothly decaying autocorrelation; diversified
generation is non-monotonic with peak correlation at intermediate semantic
distance, indicating rugged, hierarchically structured landscapes.

**Robey, Wong, Hassani, and Pappas. "SmoothLLM: Defending Large Language Models
Against Jailbreaking Attacks." arXiv:2310.03684** (v1 5 Oct 2023; v4 11 Jun
2024). Finds adversarially generated prompts brittle to character-level changes,
and defends by randomly perturbing multiple copies of an input and aggregating
predictions.

**Mu, Ying, Fan, Jing, Zhang, Yu, Zhang, Zou, and Zhang. "Mask-GCG: Are All
Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?"
arXiv:2509.06350** (v1 8 Sep 2025; v2 27 Jan 2026). Learnable token masking
raises update probability at high-impact suffix positions and prunes low-impact
ones. Finds most suffix tokens contribute significantly, and that pruning a
minority of low-impact tokens changes neither loss nor attack success.

**Consequence.** This study cannot claim the first prompt cube, the first exact
Boolean prompt-interaction decomposition, the first prompt landscape, the first
analysis of suffix perturbation, or the first account of per-token importance in
suffixes. Qin et al. occupy interaction decomposition; Hintze occupies prompt
landscape structure; SmoothLLM occupies suffix perturbation sensitivity;
Mask-GCG occupies token importance.

## What this study might contribute

A **fully prospective, matched comparison of real-token recombination geometry
for two independently admitted internal objectives**, with same-objective
restart controls and held-out behavior, on one pinned model.

The elements that are not already occupied, taken together:

1. Two internal objectives that are **independently fitted and separately
   admitted** through a causal gate with matched norm/KL/dynamic-range-matched
   placebos, before any suffix search runs.
2. **One matched output-blind optimizer** for both, sharing initial token IDs,
   prompt batches, RNG stream, token policy, update count, and budget within
   every (block, restart) pair.
3. **Same-objective restart faces as a control class**, which asks whether any
   two different suffixes recombine the same way — the control that separates
   "objective-specific geometry" from "suffixes differ".
4. **Exact enumeration** of all 2⁴ real-token recombinations per endpoint pair,
   with ties and duplicate vertices retained as failures.
5. A **preregistered null simulation** of the complete pipeline, with power
   reported alongside false-pass.

## Words this study has not earned

| Word | What it would require |
| --- | --- |
| mechanism | a separate necessity/sufficiency or mediation study |
| basin | larger connected mutation graphs, restart attraction, measured barriers |
| manifold | smooth low-dimensional structure beyond a finite token graph |
| causal route | the mediation study above |
| global landscape | coverage far beyond 16 vertices per face |
| strategy knob | reliable selectable dose-response on unseen prompts |
| mechanistically distinct targets | passing cross-intervention selectivity, which is a separate gate from admission |
| first internal-target attack | not available; see SSR, Activation-Guided GCG, ARA |
| first prompt cube or interaction decomposition | not available; see Qin et al. |

Until causal selectivity passes, A and B are **candidate internal objective
packages**. If admission passes without selectivity, they are **distinct
readouts**. Neither is a mechanism.
