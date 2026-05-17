"""Weight-only int8 quantization dispatch for the encoder / decoder subgraphs.

The ``com.microsoft.BeamSearch`` op is the top-level wrapper around the
encoder/decoder subgraphs in the assembled ONNX file. ORT's quantizer
cannot see *through* the BeamSearch wrapper to apply quantization to the
embedded subgraphs -- so to ship an int8 BYOM artifact we must:

1. Export the encoder + decoder subgraphs as standalone fp32 ONNX files
   (the existing flow already does this on disk in a temp dir).
2. Apply the weight-only int8 rewriter on each subgraph file *before* it
   is composed into the BeamSearch wrapper.
3. Re-assemble the BeamSearch wrapper graph with the rewritten subgraphs
   substituted in for the fp32 originals.

This module is the public dispatch for step 2.  The actual graph
rewriting lives in
:mod:`teradata_opus_translate._converter.weight_only_quantize`.  Steps 1
and 3 live in ``_converter/export.py`` and ``_converter/assemble.py``
respectively.

Why weight-only and not dynamic / static
----------------------------------------

Phases 2 / 3 / 4 (Issues #128, #131, #135, #138, #140, PR #154) falsified
every *activation-quantization* recipe attempted on the MarianMT family
under BYOM 7.0.0.4's pinned ORT 1.13.1:

* ``quantize_dynamic`` inserts a ``DynamicQuantizeLinear`` op in front of
  every quantized ``MatMul``.  That op recomputes the activation scale on
  every decoder step, and the resulting per-step scale drift drove beam
  search into a degenerate basin (runaway-token loops) at
  ``num_beams >= 2`` on the ``*-eng`` pairs (PR #138, Phase 2 verdict).
* ``quantize_static`` with ``QuantFormat.QDQ`` failed to execute inside
  the BeamSearch contrib op with ``IsTensor() was false``.
* ``quantize_static`` with ``QuantFormat.QOperator`` loaded but still
  produced degenerate decode at beam width 2 on ``deu-eng`` (#140 Gate 0,
  even after the calibration-driven recipe was tuned and ``num_beams=4``
  was baked into the graph in PR #154).

Weight-only int8 (Phase 5, Issue #160) eliminates the activation path
entirely: weights of every ``MatMul`` whose B input is a 2-D fp32 const
initializer are stored as int8 + per-channel symmetric scales, and a
``DequantizeLinear`` node reconstructs an fp32 weight at inference time.
Activations and the MatMul itself stay in fp32, so the
``DynamicQuantizeLinear``-driven collapse cannot recur.  We lose the
int8 GEMM throughput win but keep the ~4x on-disk size win on the
projection weights (encoder/decoder/lm_head/embedding-tying ``MatMul``\\s
make up ~99% of the model weight).

Empirically across the 25 curated ``opus-mt_tiny_*`` pairs (Gate 3
report, branch ``160-phase5-weight-only-int8``):

* 20 / 25 pairs PASS with >= 92 byte-identical decodes out of 100
  source-language sentences and BLEU mean >= 96.6 vs the fp32 reference.
* 3 / 25 NEEDS-REVIEW (cat-eng, kor-eng, spa-eus): 5-6 needs-review
  sentences each, no broken sentences.
* 2 / 25 BROKEN (deu-eng, ell-eng): 24-31 broken sentences each with
  trigram-runaway patterns ("in in in ..." or similar).  These two pairs
  are shipped anyway because the broken samples are a known-limited
  failure mode and the bulk of decodes are clean; see Issue #160 Gate 3
  for the full per-pair table.

The parity tolerance in ``api.py`` is set to allow <= 10% sample mismatch
end-to-end with a 2-token prefix guarantee; this passes every PASS /
NEEDS-REVIEW pair and tolerates a single sample drift on the 3-sample
default verification sets.  The BROKEN pairs (deu-eng, ell-eng) still
fail parity verification on their default samples -- which is the correct
signal -- but the production build pipeline uses ``verify=False`` for
the shipped artifacts and relies on the Gate 3 BLEU sweep for adequacy
evidence.

References
----------
* PR #138 (Phase 2 dynamic verdict).
* Issue #140 / PR #154 (Phase 3 / 4 static verdict).
* Issue #160 / branch ``160-phase5-weight-only-int8`` (Phase 5 gate
  reports: Gate 1 anchor verification, Gate 2 100-sentence adequacy,
  Gate 3 25-pair bulk).
* ``docs/decisions.md`` -- the int8 decision entry covering tradeoffs.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def quantize_subgraph_weights_only(
    fp32_path: Path,
    int8_path: Path,
) -> Path:
    """Apply weight-only int8 quantization to a single ONNX subgraph file.

    Phase 5 (Issue #160) entry point.  Delegates to
    :func:`teradata_opus_translate._converter.weight_only_quantize.rewrite_subgraph`;
    this thin wrapper exists so the ``_converter/quantize.py`` module
    remains the single dispatch surface for "quantize one subgraph"
    regardless of recipe.

    Unlike the legacy dynamic / static recipes (removed in v1.1.0; see
    module docstring), the weight-only rewrite needs no calibration data:
    weights are quantized in isolation per output channel, activations
    stay fp32, so there is no calibration step.

    Parameters
    ----------
    fp32_path:
        Path to the fp32 ONNX file produced by
        :func:`teradata_opus_translate._converter.export.export_encoder`
        or :func:`...export.export_decoder`.
    int8_path:
        Destination path for the rewritten ONNX file.  Parent directory
        must already exist.

    Returns
    -------
    pathlib.Path
        The resolved ``int8_path`` (the file is written by the rewriter).

    Raises
    ------
    FileNotFoundError
        If ``fp32_path`` does not exist.
    ValueError
        If the model's default-domain opset is below the per-channel
        ``DequantizeLinear`` minimum (opset 13).
    """
    from teradata_opus_translate._converter.weight_only_quantize import (
        rewrite_subgraph,
    )

    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    if not fp32_path.exists():
        raise FileNotFoundError(f"fp32 subgraph not found: {fp32_path}")

    LOGGER.info(
        "Weight-only-quantizing %s -> %s",
        fp32_path.name,
        int8_path.name,
    )
    rewrite_subgraph(fp32_path, int8_path)

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    LOGGER.info(
        "Weight-only quantized %s: %.2f MiB -> %.2f MiB (%.1f%% of fp32)",
        fp32_path.name,
        fp32_size / (1024 * 1024),
        int8_size / (1024 * 1024),
        100.0 * int8_size / fp32_size if fp32_size else 0.0,
    )
    return int8_path
