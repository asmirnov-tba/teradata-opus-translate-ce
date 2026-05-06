"""Unit tests for the pure-Python parity-comparison helpers.

These don't load any model; they only verify the normalisation and
diff logic that the parity report (#6) and converter smoke test rely
on.
"""

from __future__ import annotations

from teradata_opus_translate.parity import (
    compare_token_ids,
    first_divergence,
    strip_padding,
)
from teradata_opus_translate.parity.compare import TokenComparison

PAD = 58100
EOS = 0


def test_strip_padding_strips_trailing_pads() -> None:
    assert strip_padding([1, 2, 3, PAD, PAD, PAD], PAD, EOS) == [1, 2, 3]


def test_strip_padding_strips_one_trailing_eos() -> None:
    # Mirrors the canonical HF output shape: ... <eos> <pad> <pad> ...
    assert strip_padding([5, 6, 7, EOS, PAD, PAD], PAD, EOS) == [5, 6, 7]


def test_strip_padding_does_not_strip_internal_eos() -> None:
    # An EOS in the middle is part of the body and must not be removed.
    assert strip_padding([5, EOS, 6, EOS, PAD], PAD, EOS) == [5, EOS, 6]


def test_strip_padding_no_padding_no_eos() -> None:
    # ONNX BeamSearch already drops the EOS but pads.
    assert strip_padding([5, 6, 7, PAD], PAD, EOS) == [5, 6, 7]


def test_strip_padding_does_not_mutate_input() -> None:
    src = [5, 6, EOS, PAD]
    _ = strip_padding(src, PAD, EOS)
    assert src == [5, 6, EOS, PAD]


def test_strip_padding_empty_sequence() -> None:
    assert strip_padding([], PAD, EOS) == []


def test_strip_padding_only_pads() -> None:
    assert strip_padding([PAD, PAD], PAD, EOS) == []


def test_first_divergence_equal_returns_minus_one() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) == -1


def test_first_divergence_at_index() -> None:
    assert first_divergence([1, 2, 3, 4], [1, 2, 9, 4]) == 2


def test_first_divergence_prefix() -> None:
    # The shorter sequence is a prefix of the longer; divergence is the
    # index where the shorter ends.
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert first_divergence([1, 2, 3], [1, 2]) == 2


def test_compare_token_ids_match() -> None:
    cmp = compare_token_ids([1, 2, 3], [1, 2, 3])
    assert isinstance(cmp, TokenComparison)
    assert cmp.match is True
    assert cmp.divergence_index == -1
    assert cmp.hf_token_at_divergence is None
    assert cmp.onnx_token_at_divergence is None
    assert cmp.hf_length == 3
    assert cmp.onnx_length == 3


def test_compare_token_ids_mismatch_in_middle() -> None:
    cmp = compare_token_ids([1, 2, 3, 4], [1, 2, 9, 4])
    assert cmp.match is False
    assert cmp.divergence_index == 2
    assert cmp.hf_token_at_divergence == 3
    assert cmp.onnx_token_at_divergence == 9


def test_compare_token_ids_hf_shorter() -> None:
    cmp = compare_token_ids([1, 2], [1, 2, 3])
    assert cmp.match is False
    assert cmp.divergence_index == 2
    assert cmp.hf_token_at_divergence is None
    assert cmp.onnx_token_at_divergence == 3


def test_compare_token_ids_onnx_shorter() -> None:
    cmp = compare_token_ids([1, 2, 3], [1, 2])
    assert cmp.match is False
    assert cmp.divergence_index == 2
    assert cmp.hf_token_at_divergence == 3
    assert cmp.onnx_token_at_divergence is None
