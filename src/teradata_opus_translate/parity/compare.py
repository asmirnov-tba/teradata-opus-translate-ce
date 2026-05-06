"""Pure-Python helpers for comparing two token-ID sequences.

These functions have **no** dependency on ``torch``, ``onnxruntime``,
``transformers`` or ``tokenizers``. That keeps the comparison primitives
testable in isolation and cheap to import.
"""

from __future__ import annotations

from dataclasses import dataclass


def strip_padding(
    token_ids: list[int],
    pad_token_id: int,
    eos_token_id: int,
) -> list[int]:
    """Normalise a generated token-ID sequence for token-level parity.

    Strips:

    1. Trailing ``pad_token_id`` tokens (both backends right-pad to
       ``max_length``).
    2. *One* trailing ``eos_token_id`` token, if present.

    The HF and ONNX paths handle the EOS marker differently:

    * ``MarianMTModel.generate()`` *includes* the final EOS token in the
      output sequence and then pads with ``pad_token_id`` out to
      ``max_length``.
    * ``com.microsoft.BeamSearch`` *omits* the EOS token from its
      ``sequences`` output and pads with ``pad_token_id``.

    Both decode to the same text under ``skip_special_tokens=True``.
    For a meaningful token-level comparison we normalise both sequences
    to the canonical form "tokens up to but not including EOS" before
    diffing. This isolates real model-output differences from a purely
    cosmetic boundary-marker convention.

    Parameters
    ----------
    token_ids:
        Raw output IDs from a beam-search call. The decoder-start token
        at index 0 is preserved; only trailing padding/EOS are removed.
    pad_token_id:
        The model's pad token id (``config.pad_token_id``).
    eos_token_id:
        The model's EOS token id (``config.eos_token_id``).

    Returns
    -------
    list[int]
        A new list. The input is not mutated.
    """
    out = list(token_ids)
    while out and out[-1] == pad_token_id:
        out.pop()
    if out and out[-1] == eos_token_id:
        out.pop()
    return out


def first_divergence(a: list[int], b: list[int]) -> int:
    """Return the index of the first differing element.

    If one sequence is a strict prefix of the other, the divergence
    is reported at ``min(len(a), len(b))`` (the index where the
    shorter sequence ends). Returns ``-1`` when the two sequences
    are equal.
    """
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


@dataclass(frozen=True)
class TokenComparison:
    """Outcome of comparing two token-ID sequences.

    Attributes
    ----------
    match:
        True when ``hf_ids == onnx_ids``.
    divergence_index:
        Index of the first mismatch, or ``-1`` when ``match`` is True.
    hf_token_at_divergence:
        Token id on the HF side at ``divergence_index``, or ``None``
        when HF ran out of tokens first.
    onnx_token_at_divergence:
        Token id on the ONNX side at ``divergence_index``, or ``None``
        when ONNX ran out of tokens first.
    hf_length:
        Length of the (normalised) HF sequence.
    onnx_length:
        Length of the (normalised) ONNX sequence.
    """

    match: bool
    divergence_index: int
    hf_token_at_divergence: int | None
    onnx_token_at_divergence: int | None
    hf_length: int
    onnx_length: int


def compare_token_ids(hf_ids: list[int], onnx_ids: list[int]) -> TokenComparison:
    """Diff two normalised token-ID sequences and report where they split."""
    if hf_ids == onnx_ids:
        return TokenComparison(
            match=True,
            divergence_index=-1,
            hf_token_at_divergence=None,
            onnx_token_at_divergence=None,
            hf_length=len(hf_ids),
            onnx_length=len(onnx_ids),
        )
    div = first_divergence(hf_ids, onnx_ids)
    hf_tok = hf_ids[div] if 0 <= div < len(hf_ids) else None
    onnx_tok = onnx_ids[div] if 0 <= div < len(onnx_ids) else None
    return TokenComparison(
        match=False,
        divergence_index=div,
        hf_token_at_divergence=hf_tok,
        onnx_token_at_divergence=onnx_tok,
        hf_length=len(hf_ids),
        onnx_length=len(onnx_ids),
    )
