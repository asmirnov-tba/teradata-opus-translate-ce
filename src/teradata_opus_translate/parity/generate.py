"""Single-sentence beam-search wrappers for the two parity backends.

These helpers exist so the parity report and the converter smoke test
share *exactly* one implementation of each generation path. If only
one of those callers tweaks (say) ``do_sample`` or
``renormalize_logits``, the comparison silently stops being apples-
to-apples — that's a class of bug we'd rather make impossible.

See :data:`teradata_opus_translate.baseline.harness.PARITY_PARAMS` for
the canonical parameter set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import onnxruntime as ort
    from transformers import MarianMTModel, MarianTokenizer


def hf_generate_token_ids(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    sentence: str,
    params: dict[str, Any],
) -> list[int]:
    """Run one sentence through ``MarianMTModel.generate()``.

    The kwargs that the ONNX ``com.microsoft.BeamSearch`` contrib op
    cannot replay (``bad_words_ids``, ``forced_eos_token_id``,
    ``renormalize_logits``) are explicitly disabled here so the HF
    output mirrors what ONNX is *capable* of producing. See issue #17
    for context.

    Parameters
    ----------
    model:
        A ``MarianMTModel`` already in ``eval()`` mode and on CPU.
    tokenizer:
        The matching ``MarianTokenizer``.
    sentence:
        Single source sentence (no batching).
    params:
        Generation parameters; should be ``PARITY_PARAMS``.

    Returns
    -------
    list[int]
        The raw token-id sequence from beam search, *including* the
        decoder-start token, the body, and the trailing EOS + padding
        out to ``max_length``. Apply :func:`strip_padding` before
        comparing to ONNX output.
    """
    import torch

    enc = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=params["num_beams"],
            min_length=params["min_length"],
            max_length=params["max_length"],
            num_return_sequences=params["num_return_sequences"],
            length_penalty=params["length_penalty"],
            repetition_penalty=params["repetition_penalty"],
            no_repeat_ngram_size=params["no_repeat_ngram_size"],
            early_stopping=params["early_stopping"],
            # Disable features ONNX BeamSearch contrib op does not have.
            bad_words_ids=None,
            forced_eos_token_id=None,
            renormalize_logits=False,
            # Force deterministic beam search.
            do_sample=False,
        )
    return out[0].tolist()


def onnx_generate_token_ids(
    sess: ort.InferenceSession,
    tokenizer: MarianTokenizer,
    sentence: str,
    params: dict[str, Any],
) -> list[int]:
    """Run one sentence through the assembled ONNX BeamSearch graph.

    Parameters
    ----------
    sess:
        An ``onnxruntime.InferenceSession`` over the ``model.onnx``
        file emitted by the converter. The session may use any
        execution provider; CPU is the default for the parity report.
    tokenizer:
        The matching ``MarianTokenizer`` (used for input encoding only;
        decoding happens in the caller). Note the ONNX inputs require
        ``int32`` ids, so we cast after tokenisation.
    sentence:
        Single source sentence (no batching).
    params:
        Generation parameters; should be ``PARITY_PARAMS``.

    Returns
    -------
    list[int]
        The first beam's token sequence. ONNX returns a tensor of
        shape ``(batch, num_return_sequences, max_length)``; this
        helper picks ``[0, 0, :]``. The sequence is right-padded with
        ``pad_token_id`` and *omits* the trailing EOS that HF would
        emit.
    """
    import numpy as np

    # ``num_return_sequences`` is intentionally absent from the feeds:
    # as of v1.0.1 it is baked into the graph as a ``Constant(1)`` node
    # and is no longer a top-level input. ``params["num_return_sequences"]``
    # is still consulted on the HF side and must be ``1`` for a valid
    # parity comparison; the ONNX side ignores any other value because
    # the constant is hard-wired. See Issue #82.
    enc = tokenizer(sentence, return_tensors="np")
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int32),
        "attention_mask": enc["attention_mask"].astype(np.int32),
        "num_beams": np.array([params["num_beams"]], dtype=np.int32),
        "min_length": np.array([params["min_length"]], dtype=np.int32),
        "max_length": np.array([params["max_length"]], dtype=np.int32),
        "length_penalty": np.array([params["length_penalty"]], dtype=np.float32),
        "repetition_penalty": np.array([params["repetition_penalty"]], dtype=np.float32),
    }
    out = sess.run(None, feeds)
    sequences = out[0]  # (1, num_return_sequences, max_length)
    return sequences[0, 0].tolist()
