"""Model loading and the two frozen readout sites.

The model is loaded once at the pinned revision with eager attention, because
the routing objective reads attention probabilities and the fused kernels do not
expose them. Weights stay frozen: nothing here ever needs a weight gradient,
only gradients with respect to the suffix's one-hot inputs, so `requires_grad_`
stays False on parameters and the backward pass keeps its memory small.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tokenization import MODEL_ID, MODEL_REVISION, template_probe_hash

#: Frozen readout site for package A: residual stream after this block.
LAYER_A = 14

#: Expected hash of the rendered template probe. Verified on load so a
#: tokenizer or template change fails loudly instead of silently changing the
#: experiment.
EXPECTED_TEMPLATE_SHA256 = "c19ba073838b8d8e3381838719014dbb1e0fa0cb5ceed1138191788b302e5f5b"


class TemplateDriftError(RuntimeError):
    """The rendered chat template is not the one the study froze."""


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    n_layers: int
    n_heads: int

    def hidden_index_for_block(self, block: int) -> int:
        """Index into ``hidden_states`` for the output of ``block``.

        ``hidden_states[0]`` is the embedding output, so block *l*'s output is
        at *l + 1*. Getting this wrong reads a different site than the protocol
        froze, so it is a named function rather than an inline +1.
        """
        return block + 1


def load(
    *,
    revision: str = MODEL_REVISION,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    verify_template: bool = True,
) -> LoadedModel:
    from transformers import AutoModelForCausalLM

    from .tokenization import load_tokenizer

    tokenizer = load_tokenizer(revision)
    if verify_template:
        got = template_probe_hash(tokenizer)
        if got != EXPECTED_TEMPLATE_SHA256:
            raise TemplateDriftError(
                f"chat template changed: expected {EXPECTED_TEMPLATE_SHA256}, got {got}. "
                "The frozen prompts would render differently; refusing to run."
            )

    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        torch_dtype=dtype,
        attn_implementation="eager",   # required: the routing objective reads attention probs
        low_cpu_mem_usage=True,
    )
    model.to(resolved)
    model.eval()
    model.requires_grad_(False)        # no weight gradients are ever needed

    cfg = model.config
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(resolved),
        n_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_attention_heads,
    )


@dataclass
class Readouts:
    """Raw quantities both objectives are computed from, for one batch.

    Kept as tensors on the model's device. ``resid`` is the layer-14 residual
    stream at the readout position; ``instr_mass`` is, for every (layer, head),
    the attention probability mass the readout query assigns to the instruction
    key span.
    """

    resid: torch.Tensor        # [batch, hidden]
    instr_mass: torch.Tensor   # [batch, layers, heads]


def readouts(
    lm: LoadedModel,
    input_ids: torch.Tensor,
    instruction_spans: list[tuple[int, int]],
    *,
    layer_a: int = LAYER_A,
    need_attention: bool = True,
) -> Readouts:
    """Run one forward pass and extract both readout sites.

    ``instruction_spans`` gives the (start, end) key span per row, since prompts
    differ in length. Rows are assumed left-unpadded and equal length within a
    batch; callers group by length or pad on the left and adjust spans.
    """
    out = lm.model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=need_attention,
        use_cache=False,
    )
    resid = out.hidden_states[lm.hidden_index_for_block(layer_a)][:, -1, :].float()

    if not need_attention:
        empty = torch.zeros(input_ids.shape[0], lm.n_layers, lm.n_heads, device=resid.device)
        return Readouts(resid=resid, instr_mass=empty)

    batch = input_ids.shape[0]
    mass = torch.zeros(batch, lm.n_layers, lm.n_heads, device=resid.device, dtype=torch.float32)
    for layer_idx, attn in enumerate(out.attentions):
        # attn: [batch, heads, q, k]; the readout query is the final position.
        row = attn[:, :, -1, :].float()
        for b, (start, end) in enumerate(instruction_spans):
            mass[b, layer_idx, :] = row[b, :, start:end].sum(dim=-1)
    return Readouts(resid=resid, instr_mass=mass)
