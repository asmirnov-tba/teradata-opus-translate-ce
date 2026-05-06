"""Assemble the encoder + decoder + ``com.microsoft.BeamSearch`` op into a
single self-contained ONNX file.

The resulting graph conforms to the BYOM 20.00 ``ONNXSeq2Seq`` I/O
contract (see the ``teradata-byom-onnx-seq2seq`` skill for the full
contract):

* Top-level inputs (always exposed as graph inputs so BYOM ``Const_*``
  USING parameters can override them at scoring time):
    - ``input_ids`` int32 ``(batch_size, sequence_length)``
    - ``attention_mask`` int32 ``(batch_size, sequence_length)``
    - ``num_beams`` int32 ``(1)``
    - ``min_length`` int32 ``(1)``
    - ``max_length`` int32 ``(1)``
    - ``num_return_sequences`` int32 ``(1)``
    - ``length_penalty`` float32 ``(1)``
    - ``repetition_penalty`` float32 ``(1)``

* Outputs:
    - ``sequences`` int32 ``(batch_size, num_return_sequences, max_length)``

* BeamSearch node attributes (baked in at export time, NOT user-tunable
  at scoring time):
    - ``no_repeat_ngram_size``
    - ``early_stopping``
    - ``eos_token_id`` / ``pad_token_id`` / ``decoder_start_token_id`` (from
      ``model.config``)
    - ``model_type=1`` (encoder-decoder)

This split is deliberate: parameters that BYOM exposes via ``Const_*``
USING params (``num_beams``, ``min_length``, ``max_length``,
``num_return_sequences``, ``length_penalty``, ``repetition_penalty``)
remain graph inputs so they can be overridden per-query in SQL. The
remaining n-gram / early-stopping behaviour is fixed at export time
because the BeamSearch contrib op only accepts them as attributes.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

import onnx
from onnx import TensorProto, helper
from transformers import MarianMTModel

from teradata_opus_translate._converter.export import (
    export_decoder,
    export_encoder,
)
from teradata_opus_translate._converter.quantize import quantize_subgraph

LOGGER = logging.getLogger(__name__)

# 2 GiB ONNX file size ceiling: ONNX uses protobuf, which has a hard 2 GiB
# message size limit unless ``external_data`` is used.  The locked v1 API
# does NOT expose an ``external_data`` flag, so we error hard if the
# resulting file would exceed this limit.  See ``docs/decisions.md`` for
# the rationale.
_MAX_ONNX_BYTES = 2 * 1024 * 1024 * 1024


def _make_beamsearch_node(
    *,
    encoder_graph: onnx.GraphProto,
    decoder_graph: onnx.GraphProto,
    eos_token_id: int,
    pad_token_id: int,
    decoder_start_token_id: int,
    no_repeat_ngram_size: int,
    early_stopping: bool,
) -> onnx.NodeProto:
    """Build the ``com.microsoft.BeamSearch`` node referencing the given
    encoder / decoder subgraphs.

    Both ``no_repeat_ngram_size`` and ``early_stopping`` are baked in as
    node attributes at export time -- the BeamSearch contrib op accepts
    them only as attributes, never as runtime inputs.
    """
    node = helper.make_node(
        "BeamSearch",
        inputs=[
            "input_ids",  # F (int32 ok)
            "max_length",
            "min_length",
            "num_beams",
            "num_return_sequences",
            "length_penalty",
            "repetition_penalty",
            "",  # vocab_mask (unused)
            "",  # prefix_vocab_mask (unused)
            "attention_mask",  # custom encoder attention mask
        ],
        outputs=["sequences"],
        name="BeamSearch_marian",
        domain="com.microsoft",
    )
    node.attribute.extend(
        [
            helper.make_attribute("eos_token_id", eos_token_id),
            helper.make_attribute("pad_token_id", pad_token_id),
            helper.make_attribute("decoder_start_token_id", decoder_start_token_id),
            helper.make_attribute("no_repeat_ngram_size", no_repeat_ngram_size),
            helper.make_attribute("early_stopping", 1 if early_stopping else 0),
            # 1 == encoder-decoder (T5-style); 0 == GPT-2.
            helper.make_attribute("model_type", 1),
            helper.make_attribute("encoder", encoder_graph),
            helper.make_attribute("decoder", decoder_graph),
        ]
    )
    return node


def _build_top_level_graph(
    *,
    encoder_proto: onnx.ModelProto,
    decoder_proto: onnx.ModelProto,
    config: Any,
    no_repeat_ngram_size: int,
    early_stopping: bool,
    package_version: str,
) -> onnx.ModelProto:
    """Compose a single top-level ONNX graph that hosts BeamSearch."""
    # Subgraph names must be set explicitly so onnx doesn't reject duplicates.
    encoder_proto.graph.name = "marian_encoder"
    decoder_proto.graph.name = "marian_decoder"

    bs_node = _make_beamsearch_node(
        encoder_graph=encoder_proto.graph,
        decoder_graph=decoder_proto.graph,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        decoder_start_token_id=config.decoder_start_token_id,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
    )

    # BYOM-required top-level inputs.  These remain graph inputs (not
    # initializers / constants) so that BYOM ``Const_*`` USING parameters
    # can override them at scoring time -- see ``docs/decisions.md`` and
    # the ``teradata-byom-onnx-seq2seq`` skill for the full contract.
    input_ids = helper.make_tensor_value_info(
        "input_ids", TensorProto.INT32, ["batch_size", "sequence_length"]
    )
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT32, ["batch_size", "sequence_length"]
    )
    num_beams = helper.make_tensor_value_info("num_beams", TensorProto.INT32, [1])
    min_length = helper.make_tensor_value_info("min_length", TensorProto.INT32, [1])
    max_length = helper.make_tensor_value_info("max_length", TensorProto.INT32, [1])
    num_return_sequences = helper.make_tensor_value_info(
        "num_return_sequences", TensorProto.INT32, [1]
    )
    length_penalty = helper.make_tensor_value_info("length_penalty", TensorProto.FLOAT, [1])
    repetition_penalty = helper.make_tensor_value_info("repetition_penalty", TensorProto.FLOAT, [1])

    sequences = helper.make_tensor_value_info(
        "sequences",
        TensorProto.INT32,
        ["batch_size", "num_return_sequences", "max_length"],
    )

    graph = helper.make_graph(
        nodes=[bs_node],
        name="marian_beamsearch",
        inputs=[
            input_ids,
            attention_mask,
            num_beams,
            min_length,
            max_length,
            num_return_sequences,
            length_penalty,
            repetition_penalty,
        ],
        outputs=[sequences],
        initializer=[],
    )

    # Opset imports: keep the standard onnx opset from the decoder
    # subgraph (whichever was used during export), and add com.microsoft.
    opset_imports = list(decoder_proto.opset_import)
    if not any(o.domain == "com.microsoft" for o in opset_imports):
        opset_imports.append(helper.make_opsetid("com.microsoft", 1))

    model = helper.make_model(
        graph,
        producer_name="teradata-opus-translate",
        producer_version=package_version,
        opset_imports=opset_imports,
    )
    # IR version matches what onnxruntime expects for contrib ops.
    model.ir_version = 9
    return model


def assemble_full_model(
    model: MarianMTModel,
    output_path: Path,
    *,
    opset: int,
    no_repeat_ngram_size: int,
    early_stopping: bool,
    package_version: str,
    precision: Literal["fp32", "int8"] = "fp32",
) -> Path:
    """Build a single-file ONNX model with embedded BeamSearch for the
    given (already-loaded) Marian model.

    Parameters
    ----------
    model:
        Already-loaded ``MarianMTModel`` instance (in eval mode).
    output_path:
        Destination ``.onnx`` path.
    opset:
        ONNX opset for encoder / decoder subgraphs.
    no_repeat_ngram_size:
        Baked into the BeamSearch node attribute at export time.
    early_stopping:
        Baked into the BeamSearch node attribute at export time.
    package_version:
        Version stamped into the model's ``producer_version`` field.
    precision:
        ``"fp32"`` (default) or ``"int8"``. The ``"int8"`` path runs
        :func:`...quantize.quantize_subgraph` on the encoder and decoder
        subgraphs *before* composition into the BeamSearch wrapper --
        the BeamSearch op is opaque to ORT's quantizer so we cannot
        quantize the assembled file in one shot.

    Returns
    -------
    pathlib.Path
        The resolved ``output_path``.

    Raises
    ------
    RuntimeError
        If the resulting ONNX file would exceed 2 GiB. The v1 API does
        not support ``external_data``; ``precision="int8"`` produces a
        roughly 25%-of-fp32 artifact and is the recommended escape hatch
        for larger pairs.
    """
    cfg = model.config
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        enc_fp32 = tmp / "encoder_fp32.onnx"
        dec_fp32 = tmp / "decoder_fp32.onnx"

        LOGGER.info("Exporting encoder subgraph -> %s", enc_fp32)
        encoder_proto = export_encoder(model, enc_fp32, opset=opset)
        LOGGER.info("Exporting decoder subgraph -> %s", dec_fp32)
        decoder_proto = export_decoder(model, dec_fp32, opset=opset)

        if precision == "int8":
            # Quantize the encoder + decoder subgraphs as standalone
            # ONNX files BEFORE composition. The com.microsoft.BeamSearch
            # contrib op is opaque to ORT's quantizer (it cannot recurse
            # through subgraph node attributes), so quantizing the
            # already-assembled top-level file would be a no-op for the
            # weights that matter.
            enc_int8 = tmp / "encoder_int8.onnx"
            dec_int8 = tmp / "decoder_int8.onnx"
            LOGGER.info("Applying int8 dynamic quantization to encoder subgraph")
            quantize_subgraph(enc_fp32, enc_int8)
            LOGGER.info("Applying int8 dynamic quantization to decoder subgraph")
            quantize_subgraph(dec_fp32, dec_int8)
            # Reload the quantized protos and substitute them into the
            # BeamSearch composition.
            encoder_proto = onnx.load(str(enc_int8))
            decoder_proto = onnx.load(str(dec_int8))
        elif precision != "fp32":  # defensive; the public API also validates
            raise ValueError(f"Unsupported precision={precision!r}; expected 'fp32' or 'int8'.")

        LOGGER.info(
            "Composing top-level graph with com.microsoft.BeamSearch (precision=%s)",
            precision,
        )
        full = _build_top_level_graph(
            encoder_proto=encoder_proto,
            decoder_proto=decoder_proto,
            config=cfg,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
            package_version=package_version,
        )

        # Validate before writing: catch shape / type mismatches early.
        # We skip strict full_check because BeamSearch is a contrib op
        # and onnx checker doesn't know its schema.
        try:
            onnx.checker.check_model(full, full_check=False)
        except onnx.checker.ValidationError as exc:
            LOGGER.warning("onnx.checker reported: %s", exc)

        # Pre-flight 2 GiB check using the in-memory protobuf size.  This
        # is the same limit ``onnx.save`` would hit if we tried to write
        # without ``external_data``.  We check before writing so callers
        # get a clean Python error rather than a partially-written file.
        # The check applies to int8 outputs too even though they are
        # roughly half the fp32 size -- the ceiling stays so a future
        # very-large pair cannot silently regress past the protobuf
        # limit.
        serialized = full.SerializeToString()
        size_bytes = len(serialized)
        if size_bytes > _MAX_ONNX_BYTES:
            raise RuntimeError(
                f"Exported ONNX would be {size_bytes / 1e9:.2f} GB, which "
                f"exceeds the 2 GiB protobuf limit. The v1 API does not "
                f"support external_data; consider precision='int8' for a "
                f"roughly half-of-fp32 quantized export, or split the model."
            )

        LOGGER.info("Saving %d bytes to %s", size_bytes, output_path)
        with open(output_path, "wb") as fh:
            fh.write(serialized)

    return output_path
