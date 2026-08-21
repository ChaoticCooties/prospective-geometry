"""Frozen chat-token layout, suffix splicing, and the canonical replay contract.

The study optimizes a fixed-width suffix of real vocabulary tokens. Two failure
modes make that harder than it looks, and both are handled here:

1. *Retokenization drift.* Building a prompt by string concatenation and then
   tokenizing it can merge the suffix with its neighbours, so a "four-token"
   suffix silently becomes three or five positions. Prompts are therefore
   assembled as token ID sequences, never as strings.

2. *Non-canonical sequences.* A spliced ID sequence may not be what the tokenizer
   would produce from the corresponding text, so the attack would not survive a
   round trip through a real deployment. The canonical replay contract rejects
   any suffix whose full rendered prompt does not re-encode to itself exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .provenance import sha256_bytes

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"

#: Suffix width. Frozen at four real vocabulary tokens.
SUFFIX_WIDTH = 4

#: The date string is pinned rather than inherited. The Llama-3.1 chat template
#: interpolates a date; letting it default couples every rendered prompt to the
#: transformers version, which would break replay across runtimes.
PINNED_DATE_STRING = "26 Jul 2024"

#: Separator between the behavior text and the suffix, as a literal.
#: Empty on purpose: every bank token carries its own leading space (see
#: :mod:`pgeom.candidates`), so the join is already well formed. Adding a
#: separate space here would double it and break the canonical round trip.
SUFFIX_SEPARATOR = ""

#: Expected SHA-256 of the canonical rendered template probe. Verified at every
#: run so a tokenizer or template change fails loudly instead of silently
#: producing a different experiment.
TEMPLATE_PROBE_TEXT = "USERTEXT"


@dataclass(frozen=True)
class PromptLayout:
    """Token-ID layout of one rendered prompt with a spliced suffix."""

    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    postfix_ids: tuple[int, ...]

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.prefix_ids + self.suffix_ids + self.postfix_ids

    @property
    def suffix_slice(self) -> slice:
        start = len(self.prefix_ids)
        return slice(start, start + len(self.suffix_ids))

    @property
    def readout_position(self) -> int:
        """Index of the assistant-prefill readout token.

        This is the final token of the rendered prompt: the newline pair that
        closes the assistant header, i.e. the position whose residual stream the
        model would use to emit its first content token.
        """
        return len(self.full_ids) - 1

    @property
    def instruction_slice(self) -> slice:
        """Key span for the attention-routing objective: the behavior text only.

        Deliberately excludes template scaffolding and the suffix, so a suffix
        that attracts attention to itself lowers the objective rather than
        trivially satisfying it.
        """
        return slice(self.instruction_start, self.instruction_end)

    instruction_start: int = 0
    instruction_end: int = 0


@lru_cache(maxsize=2)
def load_tokenizer(revision: str = MODEL_REVISION):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)


def render_template(behavior_text: str, tokenizer=None) -> str:
    """Rendered chat string for a behavior with no suffix."""
    tok = tokenizer or load_tokenizer()
    return tok.apply_chat_template(
        [{"role": "user", "content": behavior_text}],
        tokenize=False,
        add_generation_prompt=True,
        date_string=PINNED_DATE_STRING,
    )


def template_probe_hash(tokenizer=None) -> str:
    """Hash of the rendered template for a fixed probe string."""
    return sha256_bytes(render_template(TEMPLATE_PROBE_TEXT, tokenizer).encode("utf-8"))


def build_layout(behavior_text: str, suffix_ids, tokenizer=None) -> PromptLayout:
    """Splice ``suffix_ids`` into the rendered prompt at the token-ID level."""
    tok = tokenizer or load_tokenizer()
    suffix_ids = tuple(int(t) for t in suffix_ids)

    rendered = render_template(behavior_text, tok)
    marker = behavior_text
    idx = rendered.rindex(marker)
    head = rendered[: idx + len(marker)] + SUFFIX_SEPARATOR
    tail = rendered[idx + len(marker) :]

    prefix_ids = tuple(tok.encode(head, add_special_tokens=False))
    postfix_ids = tuple(tok.encode(tail, add_special_tokens=False))

    # Derive the instruction span from the rendered text directly rather than
    # from prefix length, so it stays correct whatever SUFFIX_SEPARATOR is.
    before_behavior = tuple(tok.encode(rendered[:idx], add_special_tokens=False))
    through_behavior = tuple(
        tok.encode(rendered[: idx + len(marker)], add_special_tokens=False)
    )
    return PromptLayout(
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        postfix_ids=postfix_ids,
        instruction_start=len(before_behavior),
        instruction_end=len(through_behavior),
    )


def is_canonical(layout: PromptLayout, tokenizer=None) -> bool:
    """True when the spliced sequence is exactly what the tokenizer would emit.

    A suffix failing this check would not survive being written down and read
    back by a real deployment, so it is not a legitimate attack and is excluded
    from every search and every evaluation.
    """
    tok = tokenizer or load_tokenizer()
    ids = list(layout.full_ids)
    text = tok.decode(ids, skip_special_tokens=False)
    return tok.encode(text, add_special_tokens=False) == ids


def canonical_suffix_text(suffix_ids, tokenizer=None) -> str:
    tok = tokenizer or load_tokenizer()
    return tok.decode(list(suffix_ids), skip_special_tokens=False)
