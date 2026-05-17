"""Weight-only int8 quantization for the encoder / decoder subgraphs.

Phase 5 (Issue #160) shifts the int8 strategy away from
``onnxruntime.quantization.quantize_dynamic`` /
``quantize_static``.  Both of those recipes insert a ``QuantizeLinear``
(or ``DynamicQuantizeLinear``) op in front of every quantized ``MatMul``,
which means activations are also quantized at runtime.  Phase 1-4 falsified
every activation-quantization recipe attempted on the MarianMT family
under pinned ORT 1.13.1 (#128, #131, #138, #140, PR #154):

* Dynamic activation quantization (``DynamicQuantizeLinear`` per call):
  per-step scale drift drove beam search into a degenerate basin at
  ``num_beams >= 2`` (PR #138, Phase 2 verdict).
* Static (calibrated) activation quantization: ``QuantFormat.QDQ`` failed
  to execute inside the BeamSearch contrib op with ``IsTensor() was
  false``; ``QuantFormat.QOperator`` loaded but still produced degenerate
  decode at beam width 2 on ``deu-eng`` (#140 Gate 0).

Weight-only int8 *eliminates* the activation-quantization op set
entirely.  The weights of selected ``MatMul`` nodes are stored as int8
with a per-channel symmetric scale, and at inference time a
``DequantizeLinear`` reconstructs an fp32 weight just-in-time.  Both the
activations and the MatMul itself stay in fp32.  This loses the matmul
throughput win of an int8 GEMM but keeps the on-disk size win (~4x
reduction on the projection weights) without touching the activation path.

The primary unknown for Gate 0 is whether ORT 1.13.1's BeamSearch
contrib op tolerates ``DequantizeLinear``-decorated weight initializers
inside the fused encoder/decoder subgraphs.  This module is the spike
that builds the rewritten ONNX so the smoke decode under ORT 1.13.1 can
falsify or confirm BeamSearch compatibility before any wider
investment.

Algorithm
---------

1. Walk ``graph.node`` looking for ``MatMul`` nodes whose B input is a
   2-D fp32 initializer.
2. For each match (subject to the name allow-list described below):

   * Compute per-output-channel symmetric scale:
     ``scale[c] = max(|W[:, c]|) / 127.0`` (with a small floor to avoid
     divide-by-zero on dead channels).
   * Quantize: ``W_int8[:, c] = round(W[:, c] / scale[c]).clip(-128,
     127).astype(int8)``.
   * Replace the fp32 W initializer with an int8 initializer + an fp32
     scales initializer (shape ``[out_channels]``).  Symmetric quant ->
     no zero_point initializer needed; ``DequantizeLinear`` defaults to
     zero.
   * Insert a ``DequantizeLinear`` node (no zero_point input) that
     produces a fresh fp32 tensor with the same logical shape and dtype
     as the original initializer.
   * Rewire the ``MatMul``'s B input to the ``DequantizeLinear`` output.

3. Save the modified graph.

The rewriter operates on a single ONNX file at a time; the BeamSearch
fusion case is handled by running it once per subgraph (encoder +
decoder) before they are stitched back together by ``assemble.py``.

Layer allow-list
----------------

Phase 3's static-quant recipe was already opted-in to ``MatMul`` only,
with an empty exclusion list.  For the weight-only path we similarly
target every weight-bearing ``MatMul`` whose B input is a constant fp32
initializer -- which catches the same set
(``q_proj``/``k_proj``/``v_proj``/``out_proj``, ``fc1``/``fc2``,
``lm_head``, embedding-tying transpose) without needing to hard-code the
HuggingFace Marian naming convention.  A ``MatMul`` with a non-const B
input (e.g. Q@K^T or attention_weights@V inside attention) has no
initializer to quantize and is naturally skipped.

The ``layer_name_filter`` parameter is reserved for future Gate-1+
narrowing if a specific layer family proves unstable; Gate 0 deliberately
does the simplest thing -- "any MatMul with a 2-D fp32 const B" -- so the
spike can falsify the BeamSearch compatibility question without recipe
fiddling muddying the signal.

Opset compatibility
-------------------

``DequantizeLinear`` is opset 10+ and supports per-channel scales via
the ``axis`` attribute at opset 13+.  The package ships opset 14 by
default (BYOM 7.0.4 ORT 1.16.3 ceiling); the rewriter validates the
opset and raises if it would emit a node the loaded model cannot
represent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LOGGER = logging.getLogger(__name__)

# Minimum opset required for ``DequantizeLinear`` with a per-channel
# ``axis`` attribute.  Below this we'd have to emit per-tensor scales
# (one scalar per layer), which is the very recipe Phase 3 measured as
# breaking ``deu-eng`` decode at every beam width.
_MIN_OPSET_FOR_PER_CHANNEL_DEQUANT = 13

# Floor for the per-channel scale to avoid division-by-zero on dead
# channels.  Picked small enough that a real channel range will dwarf it
# (so it never affects a non-zero column) but large enough to keep the
# subsequent ``round(W/scale)`` finite for an exactly-zero column.
_SCALE_FLOOR = 1e-8


def _opset_for_default_domain(model: onnx.ModelProto) -> int:
    """Return the opset version for the default ONNX domain on ``model``."""
    for entry in model.opset_import:
        # The default domain is represented as either empty string or
        # ``"ai.onnx"``.  Per-channel ``DequantizeLinear`` lives in the
        # default domain.
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    # Default fallback if nothing specifies the default domain (should
    # be impossible for a torch.onnx-exported model).
    return 0


def _build_initializer_index(graph: onnx.GraphProto) -> dict[str, int]:
    """Map initializer name -> index into ``graph.initializer``."""
    return {init.name: i for i, init in enumerate(graph.initializer)}


def _quantize_weight_per_channel(
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric int8 quantization of a 2-D weight.

    Convention: ONNX ``MatMul`` with B shape ``[K, N]`` treats the second
    axis (N) as the output channel.  We compute one scale per output
    channel: ``scale[c] = max(|W[:, c]|) / 127.0``.

    Returns
    -------
    (W_int8, scales):
        ``W_int8`` has the same shape as ``weight``, dtype int8.
        ``scales`` has shape ``[N]``, dtype float32.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"weight_only quant currently supports 2-D MatMul weights only; "
            f"got shape {weight.shape}"
        )
    # Per-channel along axis 1 (output channels).
    abs_max = np.max(np.abs(weight), axis=0)  # shape [N]
    # Floor so dead channels don't divide by zero.  The clip below caps
    # the resulting int8 values to the valid range regardless.
    scales = np.maximum(abs_max / 127.0, _SCALE_FLOOR).astype(np.float32)
    # Broadcast: ``weight / scales`` -> shape [K, N].
    q = np.round(weight / scales[np.newaxis, :]).astype(np.int32)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scales


def _is_quantizable_matmul_b(
    init_idx: dict[str, int],
    graph: onnx.GraphProto,
    b_name: str,
) -> bool:
    """Return True iff ``b_name`` is a 2-D fp32 initializer on ``graph``."""
    if b_name not in init_idx:
        return False
    init = graph.initializer[init_idx[b_name]]
    if init.data_type != TensorProto.FLOAT:
        return False
    return len(init.dims) == 2


def rewrite_subgraph(
    fp32_path: Path,
    int8_path: Path,
    *,
    layer_name_filter: list[str] | None = None,
) -> Path:
    """Rewrite ``fp32_path`` to weight-only int8 and save to ``int8_path``.

    Parameters
    ----------
    fp32_path:
        Existing fp32 ONNX subgraph file.
    int8_path:
        Destination path for the rewritten ONNX.  Parent directory must
        already exist.
    layer_name_filter:
        Optional list of substrings.  When provided, a ``MatMul`` is
        only rewritten if its B initializer name contains at least one
        of the substrings.  When ``None`` (Gate 0 default), every
        ``MatMul`` with a 2-D fp32 const B input is rewritten.

    Returns
    -------
    pathlib.Path
        The resolved ``int8_path`` (file is written to disk).

    Raises
    ------
    FileNotFoundError
        If ``fp32_path`` does not exist.
    ValueError
        If the model's default-domain opset is below the per-channel
        ``DequantizeLinear`` minimum (opset 13).
    """
    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    if not fp32_path.exists():
        raise FileNotFoundError(f"fp32 subgraph not found: {fp32_path}")

    model = onnx.load(str(fp32_path))
    opset = _opset_for_default_domain(model)
    if opset < _MIN_OPSET_FOR_PER_CHANNEL_DEQUANT:
        raise ValueError(
            f"weight-only int8 rewrite needs opset >= "
            f"{_MIN_OPSET_FOR_PER_CHANNEL_DEQUANT} for per-channel "
            f"DequantizeLinear; loaded model is opset {opset}"
        )

    graph = model.graph
    init_idx = _build_initializer_index(graph)

    # The rewriter modifies the graph in place.  Walk a snapshot of the
    # original node list because we'll append nodes (DequantizeLinear)
    # as we go.
    original_nodes = list(graph.node)
    rewritten_count = 0
    skipped_non_const = 0
    fp32_weight_bytes = 0
    int8_weight_bytes = 0

    for node in original_nodes:
        if node.op_type != "MatMul":
            continue
        if len(node.input) < 2:
            continue
        b_name = node.input[1]
        if not _is_quantizable_matmul_b(init_idx, graph, b_name):
            skipped_non_const += 1
            continue
        if layer_name_filter is not None and not any(frag in b_name for frag in layer_name_filter):
            continue

        init = graph.initializer[init_idx[b_name]]
        weight = numpy_helper.to_array(init)  # fp32, shape [K, N]
        fp32_weight_bytes += weight.nbytes

        q, scales = _quantize_weight_per_channel(weight)
        int8_weight_bytes += q.nbytes + scales.nbytes

        # Unique-ify the new tensor names against ``b_name`` so multiple
        # rewrites in the same graph don't collide.
        int8_init_name = f"{b_name}_int8"
        scales_init_name = f"{b_name}_scales"
        dequant_output_name = f"{b_name}_dequant"

        int8_init = numpy_helper.from_array(q, name=int8_init_name)
        scales_init = numpy_helper.from_array(scales, name=scales_init_name)

        # Insert the new initializers; remove the old fp32 weight.
        # ``del graph.initializer[i]`` is safe because we rebuild the
        # index after the loop is finished.
        graph.initializer.append(int8_init)
        graph.initializer.append(scales_init)

        # Build the DequantizeLinear node.  ``axis=1`` selects the
        # output-channel axis (N) of the MatMul B weight.  No zero_point
        # input -> symmetric quant with implicit zero.
        dequant_node = helper.make_node(
            "DequantizeLinear",
            inputs=[int8_init_name, scales_init_name],
            outputs=[dequant_output_name],
            name=f"{node.name or b_name}_DequantizeLinear",
            axis=1,
        )
        graph.node.append(dequant_node)

        # Rewire the MatMul to consume the dequantized weight.
        node.input[1] = dequant_output_name

        # Remove the old fp32 weight initializer.  Do this last so we
        # haven't invalidated the index for the current iteration.
        del graph.initializer[init_idx[b_name]]
        # Rebuild the index because indices shifted after the delete.
        init_idx = _build_initializer_index(graph)

        rewritten_count += 1

    LOGGER.info(
        "weight-only int8 rewrite: %d MatMul(s) rewritten, %d skipped (non-const B); "
        "weight bytes %.2f MiB -> %.2f MiB (%.1f%%)",
        rewritten_count,
        skipped_non_const,
        fp32_weight_bytes / (1024 * 1024),
        int8_weight_bytes / (1024 * 1024),
        100.0 * int8_weight_bytes / fp32_weight_bytes if fp32_weight_bytes else 0.0,
    )

    # Topological sort: the appended DequantizeLinear nodes are at the
    # end of ``graph.node`` but their consumers (the rewired MatMuls)
    # appear earlier.  ONNX requires SSA / topo order for execution.
    # ``onnx.shape_inference.infer_shapes`` will fail on an out-of-order
    # graph; ORT 1.13.1 will reject it on load.  A small in-place topo
    # sort suffices.
    _topo_sort_in_place(graph)

    onnx.save(model, str(int8_path))
    return int8_path


def _topo_sort_in_place(graph: onnx.GraphProto) -> None:
    """In-place topological sort of ``graph.node`` (Kahn's algorithm).

    Required because :func:`rewrite_subgraph` appends ``DequantizeLinear``
    nodes to the tail of ``graph.node`` even though their outputs are
    consumed by earlier ``MatMul`` nodes.  ONNX runtimes require nodes in
    SSA / topological order.
    """
    # Producer map: tensor name -> producing node index in current list.
    nodes = list(graph.node)
    producer: dict[str, int] = {}
    for i, n in enumerate(nodes):
        for out in n.output:
            if out:
                producer[out] = i

    # Initializers and graph inputs are "free" -- they count as already
    # produced.
    free_names: set[str] = set()
    for init in graph.initializer:
        free_names.add(init.name)
    for vi in graph.input:
        free_names.add(vi.name)

    # Compute predecessor counts.
    preds: list[set[int]] = [set() for _ in nodes]
    for i, n in enumerate(nodes):
        for inp in n.input:
            if not inp or inp in free_names:
                continue
            p = producer.get(inp)
            if p is not None and p != i:
                preds[i].add(p)

    in_degree = [len(s) for s in preds]
    # Reverse adjacency: producer -> consumers.
    consumers: list[set[int]] = [set() for _ in nodes]
    for i, ps in enumerate(preds):
        for p in ps:
            consumers[p].add(i)

    ready = [i for i, d in enumerate(in_degree) if d == 0]
    order: list[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for c in consumers[i]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                ready.append(c)

    if len(order) != len(nodes):
        # Cycle in graph (should be impossible for a valid ONNX export).
        # Keep the original order rather than corrupting the graph.
        LOGGER.warning(
            "Topo sort detected a cycle (%d / %d nodes ordered); leaving order untouched.",
            len(order),
            len(nodes),
        )
        return

    sorted_nodes = [nodes[i] for i in order]
    # Replace in place: protobuf repeated fields need explicit clear + extend.
    del graph.node[:]
    graph.node.extend(sorted_nodes)
