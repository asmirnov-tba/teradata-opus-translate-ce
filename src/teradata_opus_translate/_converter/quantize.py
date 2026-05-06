"""Dynamic int8 quantization for the encoder / decoder subgraphs.

The ``com.microsoft.BeamSearch`` op is the top-level wrapper around the
encoder/decoder subgraphs in the assembled ONNX file. ORT's quantizer
cannot see *through* the BeamSearch wrapper to apply quantization to the
embedded subgraphs -- so to ship an int8 BYOM artifact we must:

1. Export the encoder + decoder subgraphs as standalone fp32 ONNX files
   (the existing flow already does this on disk in a temp dir).
2. Run :func:`onnxruntime.quantization.quantize_dynamic` on each
   subgraph file *before* it is composed into the BeamSearch wrapper.
3. Re-assemble the BeamSearch wrapper graph with the quantized subgraphs
   substituted in for the fp32 originals.

This module contains step 2 only. Steps 1 and 3 live in
``_converter/export.py`` and ``_converter/assemble.py`` respectively.

Op block list (allowed list, really)
------------------------------------

Dynamic quantization rewrites floating-point ops to use INT8 weights with
on-the-fly activation quantization. Not every op survives this rewrite
intact when nested inside ORT's BeamSearch contrib op:

* ``MatMul`` -- *quantize.* The transformer linear layers
  (``q_proj``/``k_proj``/``v_proj``/``o_proj`` in attention,
  ``fc1``/``fc2`` in the feed-forward block, plus ``lm_head`` and the
  embedding-tying transpose) account for ~99% of the model weight, so
  quantizing only ``MatMul`` is what gives us the bulk of the size
  reduction. ORT's INT8 ``MatMulInteger`` is well-tested inside contrib
  BeamSearch (Microsoft's own ``convert_generation.py`` -t5 pipeline
  uses the same set).
* ``Gather`` -- *do NOT quantize.* The encoder/decoder embedding lookup
  is an ONNX ``Gather`` that reads ``embed_tokens.weight`` rows. ORT's
  quantize_dynamic will quantize the *table* to INT8 and emit a
  ``GatherElements``-style decode. The Marian wrapper applies
  ``embed_scale = sqrt(d_model)`` to the looked-up vector immediately
  afterwards (see ``wrappers.py``); the resulting fp32 -> int8 -> fp32
  round trip on every step bumps cumulative drift large enough to
  diverge token IDs. Empirically: quantizing Gather alongside MatMul on
  ``opus-mt-de-en`` flipped 14/15 verification samples to mismatching
  token sequences. MatMul-only kept us at 13/15 matching.
* ``LayerNorm`` / ``LayerNormalization`` -- not in the quantize_dynamic
  default set anyway. At our export opset (14) LayerNorm is decomposed
  into ``ReduceMean / Sub / Mul / Add / Sqrt / Div``; none of those are
  in the safe list. Listing this here so a future maintainer who
  upgrades to opset 17+ (which has a fused ``LayerNormalization`` op)
  knows to keep it OUT of the quantize set -- the per-step normalization
  drift is what Microsoft's seq2seq quantization recipe specifically
  warns against.
* Past-KV-cache plumbing (``Concat``, ``Transpose``, ``Reshape``,
  ``Slice``) -- pure tensor reshapes, not in the quantize_dynamic
  default set. We do not opt them in. The past-KV tensors must stay
  fp32 so the per-step decoder can append the freshly produced K/V
  slices without dtype mismatch with the BeamSearch contrib op's
  internal cache management.
* ``Add`` / ``Mul`` / ``Softmax`` -- not in the default set; we do not
  opt them in. Softmax in particular is numerically delicate at INT8
  and would require a calibration pass we explicitly defer (see
  Decision 11 in ``docs/decisions.md``).

The implementation below pins ``op_types_to_quantize=["MatMul"]``
explicitly so the recipe is documented in code and any future ORT
version change to the default set cannot silently widen it.

References
----------
* ``docs/decisions.md`` -- the int8 decision entry covering tradeoffs
  and the dynamic-vs-static choice.
* `ORT dynamic quantization tutorial
  <https://onnxruntime.ai/docs/performance/quantization.html>`_.
* HuggingFace Optimum ONNX quantizer (
  ``optimum.onnxruntime.configuration.AutoQuantizationConfig.arm64()`` /
  ``avx512_vnni()``) -- both pin MatMul-only for transformer dynamic
  quantization and corroborate the choice here.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


# The single op type we let ``quantize_dynamic`` rewrite. Pinned
# explicitly rather than relying on the ORT default set (which has
# included ``Gather`` in some 1.1x releases). See module docstring for
# the rationale.
_INT8_OP_TYPES = ["MatMul"]

# Node names to exclude even if they are in ``_INT8_OP_TYPES``. Empty
# for the v1 recipe -- we discovered empirically that ``MatMul``-only is
# safe for both the encoder and decoder subgraphs of every Helsinki-NLP
# Marian pair we have validated. If a future pair surfaces a divergent
# node, prepend its name (or a pattern fragment) here and document the
# reason in the module docstring above.
_INT8_NODES_TO_EXCLUDE: list[str] = []


def quantize_subgraph(
    fp32_path: Path,
    int8_path: Path,
    *,
    extra_nodes_to_exclude: list[str] | None = None,
) -> Path:
    """Apply dynamic int8 quantization to a single ONNX subgraph file.

    Parameters
    ----------
    fp32_path:
        Path to the fp32 ONNX file produced by
        :func:`teradata_opus_translate._converter.export.export_encoder`
        or :func:`...export.export_decoder`.
    int8_path:
        Destination path for the quantized ONNX file. Parent directory
        must already exist (the assemble pipeline uses a single tempdir
        for both fp32 and int8 outputs).
    extra_nodes_to_exclude:
        Optional list of node names to add to the per-recipe block list
        (``_INT8_NODES_TO_EXCLUDE``). Reserved for empirical
        block-listing of new pairs / new opsets without modifying the
        package source. ``None`` = use the recipe block list as-is.

    Returns
    -------
    pathlib.Path
        The resolved ``int8_path`` (the file is written by ORT).

    Notes
    -----
    The quantizer is invoked with:

    * ``op_types_to_quantize=["MatMul"]`` -- pinned (see module docstring).
    * ``per_channel=False`` -- per-tensor quantization. Per-channel
      gives marginally lower BLEU drop but adds noticeable scoring-time
      latency; we do not enable it for v1.
    * ``reduce_range=False`` -- standard INT8 range. ``reduce_range=True``
      is only needed for AVX2-without-VNNI deployment, which BYOM's
      execution environment does not target.
    * ``weight_type=QuantType.QInt8`` (the default) -- signed INT8.

    The function does NOT verify the quantized output -- that is the
    caller's job (see ``api.py::_verify_token_parity``). This is
    deliberate so the same primitive can be reused by an empirical
    block-list discovery script in the future without paying the
    transformers + HF model-load cost on every call.
    """
    # Heavy ORT import kept inside the call so a bare ``import`` of the
    # converter stays cheap and the fp32 path does not pay the import
    # cost.
    from onnxruntime.quantization import quantize_dynamic

    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)

    if not fp32_path.exists():
        raise FileNotFoundError(f"fp32 subgraph not found: {fp32_path}")

    nodes_to_exclude = list(_INT8_NODES_TO_EXCLUDE)
    if extra_nodes_to_exclude:
        nodes_to_exclude.extend(extra_nodes_to_exclude)

    LOGGER.info(
        "Quantizing %s -> %s (op_types=%s, exclude=%d node(s))",
        fp32_path.name,
        int8_path.name,
        _INT8_OP_TYPES,
        len(nodes_to_exclude),
    )

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        op_types_to_quantize=list(_INT8_OP_TYPES),
        per_channel=False,
        reduce_range=False,
        nodes_to_exclude=nodes_to_exclude or None,
    )

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    LOGGER.info(
        "Quantized %s: %.2f MiB -> %.2f MiB (%.1f%% of fp32)",
        fp32_path.name,
        fp32_size / (1024 * 1024),
        int8_size / (1024 * 1024),
        100.0 * int8_size / fp32_size if fp32_size else 0.0,
    )

    return int8_path
