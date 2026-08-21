"""Batching prompts of unequal length without corrupting the readouts.

Both readout sites are positional: package A reads the final token, package B
reads the final token's attention row over the instruction span. Padding
therefore has to be handled explicitly rather than left to defaults.

Left padding is used so the readout stays at index -1 for every row. Two things
follow, and both are easy to get silently wrong:

* instruction spans shift by the pad width, so they are recomputed per row;
* RoPE position ids must be derived from the attention mask. Llama computes
  positions from a cache counter that ignores padding, so a left-padded batch
  would otherwise place every real token at the wrong rotary position and change
  the activations being measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tokenization import PromptLayout


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor          # [b, t]
    attention_mask: torch.Tensor     # [b, t]
    position_ids: torch.Tensor       # [b, t]
    instruction_spans: list[tuple[int, int]]
    suffix_slices: list[slice]

    def to(self, device) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            position_ids=self.position_ids.to(device),
            instruction_spans=self.instruction_spans,
            suffix_slices=self.suffix_slices,
        )

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


def collate(layouts: list[PromptLayout], *, pad_id: int) -> Batch:
    """Left-pad a list of layouts into one batch, adjusting every index."""
    if not layouts:
        raise ValueError("cannot collate an empty batch")
    seqs = [list(layout.full_ids) for layout in layouts]
    width = max(len(s) for s in seqs)

    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), width), dtype=torch.long)
    spans: list[tuple[int, int]] = []
    suffixes: list[slice] = []

    for row, (seq, layout) in enumerate(zip(seqs, layouts)):
        pad = width - len(seq)
        input_ids[row, pad:] = torch.tensor(seq, dtype=torch.long)
        attention_mask[row, pad:] = 1
        start, end = layout.instruction_start, layout.instruction_end
        spans.append((start + pad, end + pad))
        sfx = layout.suffix_slice
        suffixes.append(slice(sfx.start + pad, sfx.stop + pad))

    # Positions must follow the real tokens, not the padded grid.
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)
    return Batch(input_ids, attention_mask, position_ids, spans, suffixes)


def iter_batches(layouts: list[PromptLayout], *, pad_id: int, size: int):
    """Yield batches, longest-first so a length outlier surfaces immediately.

    Sorting by length also keeps padding waste down without changing any result,
    because every index is recomputed per row.
    """
    order = sorted(range(len(layouts)), key=lambda i: -len(layouts[i].full_ids))
    for start in range(0, len(order), size):
        chunk = [layouts[i] for i in order[start:start + size]]
        yield order[start:start + size], collate(chunk, pad_id=pad_id)
