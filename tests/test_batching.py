"""Left padding must move every positional index consistently."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from pgeom.batching import collate, iter_batches  # noqa: E402
from pgeom.tokenization import build_layout, load_tokenizer  # noqa: E402

PAD = 128009


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


@pytest.fixture(scope="module")
def layouts(tok):
    suffix = [tok.encode(w, add_special_tokens=False)[0]
              for w in (" alpha", " beta", " gamma", " delta")]
    texts = [
        "Explain photosynthesis",
        "Describe in detail how a modern jet engine compressor stage actually works",
        "Summarise the causes of the Peloponnesian War",
    ]
    return [build_layout(t, suffix, tok) for t in texts]


def test_readout_stays_at_the_last_index_for_every_row(layouts):
    batch = collate(layouts, pad_id=PAD)
    assert batch.attention_mask[:, -1].tolist() == [1, 1, 1]


def test_padding_is_on_the_left(layouts):
    batch = collate(layouts, pad_id=PAD)
    shortest = min(range(3), key=lambda i: len(layouts[i].full_ids))
    pad_width = batch.input_ids.shape[1] - len(layouts[shortest].full_ids)
    assert batch.attention_mask[shortest, :pad_width].sum() == 0
    assert batch.attention_mask[shortest, pad_width:].sum() == pad_width * 0 + (
        len(layouts[shortest].full_ids))


def test_instruction_spans_shift_with_padding(layouts, tok):
    batch = collate(layouts, pad_id=PAD)
    for row, layout in enumerate(layouts):
        start, end = batch.instruction_spans[row]
        decoded = tok.decode(batch.input_ids[row, start:end].tolist())
        original = tok.decode(list(layout.full_ids)[layout.instruction_slice])
        assert decoded == original


def test_suffix_slices_shift_with_padding(layouts):
    batch = collate(layouts, pad_id=PAD)
    for row, layout in enumerate(layouts):
        got = batch.input_ids[row, batch.suffix_slices[row]].tolist()
        assert got == list(layout.suffix_ids)


def test_position_ids_ignore_padding(layouts):
    """RoPE positions must count real tokens only, or a padded prompt is measured
    at the wrong rotary positions."""
    batch = collate(layouts, pad_id=PAD)
    for row, layout in enumerate(layouts):
        n = len(layout.full_ids)
        real = batch.position_ids[row][batch.attention_mask[row] == 1]
        assert real.tolist() == list(range(n))


def test_empty_batch_is_an_error():
    with pytest.raises(ValueError):
        collate([], pad_id=PAD)


def test_iter_batches_covers_every_prompt_once(layouts):
    seen = []
    for idx, batch in iter_batches(layouts, pad_id=PAD, size=2):
        seen.extend(idx)
        assert len(batch) == len(idx)
    assert sorted(seen) == [0, 1, 2]
