"""Export the Marian wrapper modules to standalone ONNX subgraphs.

Each subgraph conforms to the I/O contract enforced by ORT 1.16.3 / 1.17.x's
``com.microsoft.BeamSearch`` op (encoder-decoder mode, ``model_type=1``) --
the older "encoder_decoder_init" pattern that BYOM 07.00.00.01 expects.
See ``wrappers.py`` for the wrapper modules themselves and ``assemble.py``
for the top-level graph construction.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
from transformers import MarianMTModel

from teradata_opus_translate._converter.wrappers import (
    MarianDecoderForOnnx,
    MarianEncoderDecoderInitForOnnx,
)


def export_encoder(model: MarianMTModel, output_path: Path, *, opset: int = 14) -> onnx.ModelProto:
    """Export the encoder + seed-decoder-step ("encoder_decoder_init")
    subgraph to ONNX.
    """
    cfg = model.config
    L = cfg.decoder_layers
    wrapper = MarianEncoderDecoderInitForOnnx(model).eval()

    batch_size, seq_len = 2, 7
    encoder_input_ids = torch.randint(
        low=0, high=cfg.vocab_size, size=(batch_size, seq_len), dtype=torch.int32
    )
    encoder_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int32)
    decoder_input_ids = torch.full(
        (batch_size, 1),
        fill_value=cfg.decoder_start_token_id,
        dtype=torch.int32,
    )

    input_names = ["encoder_input_ids", "encoder_attention_mask", "decoder_input_ids"]

    output_names: list[str] = ["logits", "encoder_hidden_states"]
    for i in range(L):
        output_names.append(f"present_key_self_{i}")
        output_names.append(f"present_value_self_{i}")
    for i in range(L):
        output_names.append(f"present_key_cross_{i}")
        output_names.append(f"present_value_cross_{i}")

    dynamic_axes: dict[str, dict[int, str]] = {
        "encoder_input_ids": {0: "batch_size", 1: "encode_sequence_length"},
        "encoder_attention_mask": {0: "batch_size", 1: "encode_sequence_length"},
        "decoder_input_ids": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size", 1: "encode_sequence_length"},
    }
    for i in range(L):
        dynamic_axes[f"present_key_self_{i}"] = {0: "batch_size"}
        dynamic_axes[f"present_value_self_{i}"] = {0: "batch_size"}
        dynamic_axes[f"present_key_cross_{i}"] = {
            0: "batch_size",
            2: "encode_sequence_length",
        }
        dynamic_axes[f"present_value_cross_{i}"] = {
            0: "batch_size",
            2: "encode_sequence_length",
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (encoder_input_ids, encoder_attention_mask, decoder_input_ids),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    return onnx.load(str(output_path))


def export_decoder(model: MarianMTModel, output_path: Path, *, opset: int = 14) -> onnx.ModelProto:
    """Export the decoder step to ONNX (ORT 1.16.3 contract)."""
    cfg = model.config
    L = cfg.decoder_layers
    H = cfg.decoder_attention_heads
    D = cfg.d_model // H

    wrapper = MarianDecoderForOnnx(model).eval()

    batch_size = 2
    enc_seq_len = 7
    past_seq_len = 3  # non-zero so the cat path is exercised during tracing

    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, 1), dtype=torch.int32)
    encoder_attention_mask = torch.ones((batch_size, enc_seq_len), dtype=torch.int32)
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, cfg.d_model, dtype=torch.float32)
    past_self_kvs: list[torch.Tensor] = []
    for _ in range(L):
        past_self_kvs.append(torch.randn(batch_size, H, past_seq_len, D, dtype=torch.float32))
        past_self_kvs.append(torch.randn(batch_size, H, past_seq_len, D, dtype=torch.float32))
    past_cross_kvs: list[torch.Tensor] = []
    for _ in range(L):
        past_cross_kvs.append(torch.randn(batch_size, H, enc_seq_len, D, dtype=torch.float32))
        past_cross_kvs.append(torch.randn(batch_size, H, enc_seq_len, D, dtype=torch.float32))

    input_names: list[str] = [
        "input_ids",
        "encoder_attention_mask",
        "encoder_hidden_states",
    ]
    for i in range(L):
        input_names.append(f"past_key_self_{i}")
        input_names.append(f"past_value_self_{i}")
    for i in range(L):
        input_names.append(f"past_key_cross_{i}")
        input_names.append(f"past_value_cross_{i}")

    output_names: list[str] = ["logits"]
    for i in range(L):
        output_names.append(f"present_key_self_{i}")
        output_names.append(f"present_value_self_{i}")

    dynamic_axes: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch_size"},
        "encoder_attention_mask": {0: "batch_size", 1: "encode_sequence_length"},
        "encoder_hidden_states": {0: "batch_size", 1: "encode_sequence_length"},
        "logits": {0: "batch_size"},
    }
    for i in range(L):
        dynamic_axes[f"past_key_self_{i}"] = {
            0: "batch_size",
            2: "past_decode_sequence_length",
        }
        dynamic_axes[f"past_value_self_{i}"] = {
            0: "batch_size",
            2: "past_decode_sequence_length",
        }
        dynamic_axes[f"past_key_cross_{i}"] = {
            0: "batch_size",
            2: "encode_sequence_length",
        }
        dynamic_axes[f"past_value_cross_{i}"] = {
            0: "batch_size",
            2: "encode_sequence_length",
        }
        dynamic_axes[f"present_key_self_{i}"] = {
            0: "batch_size",
            2: "past_decode_sequence_length_plus_1",
        }
        dynamic_axes[f"present_value_self_{i}"] = {
            0: "batch_size",
            2: "past_decode_sequence_length_plus_1",
        }

    args = (
        input_ids,
        encoder_attention_mask,
        encoder_hidden_states,
        *past_self_kvs,
        *past_cross_kvs,
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    return onnx.load(str(output_path))
