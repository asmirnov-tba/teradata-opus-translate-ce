"""PyTorch wrapper modules that adapt HuggingFace ``MarianMTModel`` to the
input/output tensor layout required by ``com.microsoft.BeamSearch``.

This converter targets the **older T5 "encoder_decoder_init" pattern** that
ORT 1.16.3 / 1.17.x's ``com.microsoft.BeamSearch`` validator enforces (see
``onnxruntime/contrib_ops/cpu/transformers/subgraph_t5_encoder.cc``).
That is also the format that ships inside Teradata BYOM 07.00.00.01's
bundled onnxruntime, so this is the *canonical target* for this project.
See ``docs/converter-architecture.md`` and Issue #28 for the rationale --
do NOT "modernise" this back to the newer split (2-input encoder + separate
decoder-init) format without re-validating against BYOM.

Two subgraphs are produced:

* **encoder subgraph** (``MarianEncoderDecoderInitForOnnx``) -- runs once
  per ``generate()`` call. Takes **3 inputs**:

      * ``encoder_input_ids`` (B, S) int32
      * ``encoder_attention_mask`` (B, S) int32
      * ``decoder_input_ids`` (B, 1) int32 -- BeamSearch fills this with
        the ``decoder_start_token_id``.

  Produces ``2 + 4*L`` outputs (with ``L = num_decoder_layers``):

      * ``logits`` (B, 1, vocab_size)            -- seed-step logits
      * ``encoder_hidden_states`` (B, S, d_model) -- carried into the
        decoder as a constant for every step
      * ``present_key_self_i`` / ``present_value_self_i`` (B, H, 1, D)
        for ``i`` in 0..L-1 -- the seed self-attention KV cache that the
        decoder will grow from on subsequent steps
      * ``present_key_cross_i`` / ``present_value_cross_i`` (B, H, S, D)
        for ``i`` in 0..L-1 -- pre-projected cross KVs, constant for
        every decode step

* **decoder subgraph** (``MarianDecoderForOnnx``) -- runs once per
  generated token. Inputs (in order):

      * ``input_ids`` (B, 1) int32  -- next token to consume
      * ``encoder_attention_mask`` (B, S) int32
      * ``encoder_hidden_states`` (B, S, d_model) f32 -- present in this
        signature to satisfy the ORT 1.16.3 validator; used by the
        scaled-dot cross-attention as a redundant pathway only when the
        cross KV cache is empty (it never is at decode time, so this
        tensor is read but does not influence outputs)
      * ``past_key_self_i`` / ``past_value_self_i`` for i in 0..L-1
      * ``past_key_cross_i`` / ``past_value_cross_i`` for i in 0..L-1

  Outputs:

      * ``logits`` (B, 1, vocab_size)
      * ``present_key_self_i`` / ``present_value_self_i`` for i in 0..L-1

  Cross KVs are NOT echoed back -- they are constant and BeamSearch
  carries them through unchanged.

Marian-specific details handled here:

* Sinusoidal *or* learned positional embeddings -- both expose a
  ``weight`` (or callable producing position vectors) on the
  ``embed_positions`` module; we read the position vectors via a
  manual lookup so the export is identical for both variants.
* ``embed_scale = sqrt(d_model)`` when ``config.scale_embedding`` is
  true; baked into the wrapper.
* SiLU/swish activation in FFN (``config.activation_function``).
* No ``layernorm_embedding`` for opus-mt-de-en, but some Marian
  variants have one -- we read it off the loaded module if present.
* ``final_logits_bias`` added after the LM head matmul.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from transformers import MarianMTModel
from transformers.activations import ACT2FN


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Numerically-stable scaled dot-product attention.

    All tensors are ``(B, H, T, D)``.  ``attn_mask`` if given is broadcast
    additively to the attention scores (i.e. it should already be a
    "large negative for padding" mask, not a 0/1 mask).
    """
    head_dim = q.size(-1)
    # (B, H, T_q, T_k)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    if attn_mask is not None:
        scores = scores + attn_mask
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


class _MarianDecoderCore(nn.Module):
    """Shared per-layer decoder math used by both the seed pass (inside
    the encoder_decoder_init subgraph) and the per-step decoder subgraph.

    Implementing this once means both subgraphs are guaranteed to use the
    same Marian-specific arithmetic (post-norm ordering, embed scale,
    sinusoidal/learned position lookup, SiLU FFN, final_logits_bias).
    """

    def __init__(self, model: MarianMTModel) -> None:
        super().__init__()
        self.model = model  # Keep so weights stay registered.
        decoder = model.get_decoder()
        self.embed_tokens = decoder.embed_tokens
        self.embed_positions = decoder.embed_positions
        self.layernorm_embedding = getattr(decoder, "layernorm_embedding", None)
        self.layers = decoder.layers
        self.lm_head = model.lm_head
        self.register_buffer(
            "final_logits_bias",
            model.final_logits_bias,  # type: ignore[arg-type]  # buffer, not Module
            persistent=False,
        )

        cfg = model.config
        self.num_heads = cfg.decoder_attention_heads
        self.head_dim = cfg.d_model // self.num_heads
        self.num_layers = cfg.decoder_layers
        self.d_model = cfg.d_model
        self.embed_scale = math.sqrt(cfg.d_model) if cfg.scale_embedding else 1.0
        self.activation_fn = ACT2FN[cfg.activation_function]

    def _embed(self, input_ids: torch.Tensor, past_seq_len: torch.Tensor | int) -> torch.Tensor:
        """Token + position embedding for the (B, q_len) decoder step input.

        ``past_seq_len`` is either an int (e.g. 0 for the seed pass) or a
        scalar tensor produced from another tensor's ``shape``; either way
        we add it to the position arange so the export stays dynamic.
        """
        emb = self.embed_tokens(input_ids) * self.embed_scale
        seq_len = input_ids.size(1)
        # ``positions`` is computed on-device.  When ``past_seq_len`` is a
        # 0-d tensor read from another tensor's shape, the resulting graph
        # treats the past length as dynamic rather than baking it in.
        if isinstance(past_seq_len, torch.Tensor):
            base = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            positions = base + past_seq_len
        else:
            positions = torch.arange(
                past_seq_len,
                past_seq_len + seq_len,
                dtype=torch.long,
                device=input_ids.device,
            )
        pos_emb = self.embed_positions.weight[positions]
        hidden = emb + pos_emb
        if self.layernorm_embedding is not None:
            hidden = self.layernorm_embedding(hidden)
        return hidden

    def _self_attn(
        self,
        layer,
        hidden: torch.Tensor,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Causal self-attention.  When ``past_k`` / ``past_v`` are None,
        the q_len=1 (or seed) case is run with no past cache and the
        present KVs are simply the projections of the current step.
        """
        attn = layer.self_attn
        bsz, q_len, _ = hidden.shape
        H, D = self.num_heads, self.head_dim

        q = attn.q_proj(hidden).view(bsz, q_len, H, D).transpose(1, 2).contiguous()
        k_new = attn.k_proj(hidden).view(bsz, q_len, H, D).transpose(1, 2).contiguous()
        v_new = attn.v_proj(hidden).view(bsz, q_len, H, D).transpose(1, 2).contiguous()

        if past_k is None or past_v is None:
            present_k = k_new
            present_v = v_new
        else:
            # past_k / past_v: (B, H, P, D), k_new / v_new: (B, H, q_len, D)
            present_k = torch.cat([past_k, k_new], dim=2)
            present_v = torch.cat([past_v, v_new], dim=2)

        # Decoder is autoregressive with q_len=1 (or the seed q_len=1), so
        # no causal mask is needed -- every query attends to all P+q_len
        # keys, including the new key it just emitted.
        out = _scaled_dot_product_attention(q, present_k, present_v, attn_mask=None)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        out = attn.out_proj(out)
        return out, present_k, present_v

    def _cross_attn(
        self,
        layer,
        hidden: torch.Tensor,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention against the (constant) encoder KVs."""
        attn = layer.encoder_attn
        bsz, q_len, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        src_len = cross_k.size(2)

        q = attn.q_proj(hidden).view(bsz, q_len, H, D).transpose(1, 2).contiguous()
        # cross_k / cross_v already (B, H, S, D)

        # Build additive attention mask: 1 -> 0, 0 -> large-negative.
        mask = encoder_attention_mask.to(q.dtype)
        mask = (1.0 - mask) * torch.finfo(q.dtype).min
        mask = mask.view(bsz, 1, 1, src_len)

        out = _scaled_dot_product_attention(q, cross_k, cross_v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        out = attn.out_proj(out)
        return out

    def _project_cross_kv(
        self,
        layer,
        encoder_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project encoder hidden states through this layer's
        ``encoder_attn`` k_proj / v_proj to produce the cross KV cache
        in the (B, H, S, D) layout BeamSearch expects.
        """
        attn = layer.encoder_attn
        bsz, src_len, _ = encoder_hidden.shape
        H, D = self.num_heads, self.head_dim
        k = attn.k_proj(encoder_hidden).view(bsz, src_len, H, D).transpose(1, 2).contiguous()
        v = attn.v_proj(encoder_hidden).view(bsz, src_len, H, D).transpose(1, 2).contiguous()
        return k, v

    def _decode_step(
        self,
        hidden: torch.Tensor,
        past_self_kvs: list[torch.Tensor] | None,
        cross_kvs: list[torch.Tensor],
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run all decoder layers, returning the final hidden state and
        the per-layer present self-attention KVs (flattened K0,V0,K1,V1,...).
        """
        present_self_kvs: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            past_k = past_self_kvs[2 * i] if past_self_kvs is not None else None
            past_v = past_self_kvs[2 * i + 1] if past_self_kvs is not None else None
            cross_k = cross_kvs[2 * i]
            cross_v = cross_kvs[2 * i + 1]

            # Self-attention sub-block (Marian: post-norm).
            residual = hidden
            sa_out, present_k, present_v = self._self_attn(layer, hidden, past_k, past_v)
            hidden = layer.self_attn_layer_norm(residual + sa_out)
            present_self_kvs.append(present_k)
            present_self_kvs.append(present_v)

            # Cross-attention sub-block.
            residual = hidden
            ca_out = self._cross_attn(layer, hidden, cross_k, cross_v, encoder_attention_mask)
            hidden = layer.encoder_attn_layer_norm(residual + ca_out)

            # FFN sub-block.
            residual = hidden
            ff = self.activation_fn(layer.fc1(hidden))
            ff = layer.fc2(ff)
            hidden = layer.final_layer_norm(residual + ff)
        return hidden, present_self_kvs

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden) + self.final_logits_bias


class MarianEncoderDecoderInitForOnnx(nn.Module):
    """Combined encoder + seed-decoder-step wrapper.

    This is the "encoder_decoder_init" subgraph that ORT 1.16.3's
    BeamSearch contrib op expects as the ``encoder`` attribute.  It runs
    both the encoder and a single decoder forward pass at q_len=1 fed by
    ``decoder_input_ids`` (which BeamSearch fills with
    ``decoder_start_token_id``).

    Forward signature::

        (encoder_input_ids[B, S], encoder_attention_mask[B, S],
         decoder_input_ids[B, 1])
            -> (logits[B, 1, V],
                encoder_hidden_states[B, S, d_model],
                present_key_self_0,  present_value_self_0,
                ...,  present_value_self_{L-1},
                present_key_cross_0, present_value_cross_0,
                ...,  present_value_cross_{L-1})
    """

    def __init__(self, model: MarianMTModel) -> None:
        super().__init__()
        self.encoder = model.get_encoder()
        self.core = _MarianDecoderCore(model)
        self.num_layers = model.config.decoder_layers

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # MarianEncoder expects long input_ids; BeamSearch passes int32 so
        # cast inside the graph.
        enc_ids = encoder_input_ids.to(torch.long)
        enc_mask = encoder_attention_mask.to(torch.long)
        dec_ids = decoder_input_ids.to(torch.long)

        enc = self.encoder(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            return_dict=True,
        )
        encoder_hidden = enc.last_hidden_state  # (B, S, d_model)

        # Pre-project cross KVs once, then run the seed decoder step at
        # q_len=1 with no past self-attention cache.
        cross_kvs: list[torch.Tensor] = []
        for layer in self.core.layers:
            ck, cv = self.core._project_cross_kv(layer, encoder_hidden)
            cross_kvs.append(ck)
            cross_kvs.append(cv)

        hidden = self.core._embed(dec_ids, past_seq_len=0)
        hidden, present_self_kvs = self.core._decode_step(
            hidden,
            past_self_kvs=None,
            cross_kvs=cross_kvs,
            encoder_attention_mask=encoder_attention_mask,
        )
        logits = self.core._logits(hidden)

        # Output order MUST match what subgraph_t5_encoder.cc Validate()
        # expects: logits, encoder_hidden_states, then per-layer
        # present_key_self / present_value_self, then per-layer
        # present_key_cross / present_value_cross.
        outputs: list[torch.Tensor] = [logits, encoder_hidden]
        outputs.extend(present_self_kvs)  # K0,V0,K1,V1,...
        outputs.extend(cross_kvs)  # K0,V0,K1,V1,...
        return tuple(outputs)


class MarianDecoderForOnnx(nn.Module):
    """Decoder wrapper matching the ORT 1.16.3 BeamSearch decoder
    subgraph contract.

    Forward signature::

        (input_ids[B, 1], encoder_attention_mask[B, S],
         encoder_hidden_states[B, S, d_model],
         past_key_self_0, past_value_self_0, ...,
         past_key_cross_0, past_value_cross_0, ...)
            -> (logits[B, 1, V],
                present_key_self_0, present_value_self_0, ...)

    ORT 1.16.3's ``subgraph_t5_decoder.cc`` Validate() requires
    ``encoder_hidden_states`` as input slot 2 -- we accept it here so the
    subgraph signature passes validation.  Cross attention reads the
    pre-projected ``past_key_cross_i`` / ``past_value_cross_i`` tensors
    directly (BeamSearch carries them through unchanged), so
    ``encoder_hidden_states`` is not consumed mathematically; it is wired
    through purely to satisfy the validator.
    """

    def __init__(self, model: MarianMTModel) -> None:
        super().__init__()
        self.core = _MarianDecoderCore(model)
        self.num_heads = self.core.num_heads
        self.head_dim = self.core.head_dim
        self.num_layers = self.core.num_layers
        self.d_model = self.core.d_model

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # args layout (per ORT 1.16.3 T5 decoder contract):
        #   input_ids,
        #   encoder_attention_mask,
        #   encoder_hidden_states,
        #   [past_key_self_i, past_value_self_i] * L,
        #   [past_key_cross_i, past_value_cross_i] * L
        input_ids = args[0]
        encoder_attention_mask = args[1]
        encoder_hidden_states = args[2]
        L = self.num_layers
        past_self = list(args[3 : 3 + 2 * L])
        past_cross = list(args[3 + 2 * L : 3 + 4 * L])

        # Past length comes from the temporal dim of any past_self KV
        # tensor; reading via .shape keeps it dynamic in the trace.
        past_seq_len = past_self[0].shape[2]
        hidden = self.core._embed(input_ids, past_seq_len=past_seq_len)

        hidden, present_self_kvs = self.core._decode_step(
            hidden,
            past_self_kvs=past_self,
            cross_kvs=past_cross,
            encoder_attention_mask=encoder_attention_mask,
        )
        logits = self.core._logits(hidden)

        # ``encoder_hidden_states`` is part of the ORT 1.16.3 decoder
        # subgraph contract (slot 2) but mathematically unused during
        # decode -- the cross KVs are cached.  Add a 0-weighted reference
        # to it so torch.onnx.export keeps the input wired (otherwise
        # constant folding prunes it and the exported graph fails the
        # subgraph_t5_decoder.cc Validate() input-count check).
        zero_ref = encoder_hidden_states.sum() * 0.0
        logits = logits + zero_ref.to(logits.dtype)
        return (logits, *present_self_kvs)
