"""Smoke tests for the Marian -> ONNX-with-BeamSearch converter.

These tests are slow and require:

* the model ``Helsinki-NLP/opus-mt-de-en`` cached locally (or network
  access to download it once),
* enough disk for the ~540 MB exported ONNX file (written to a tmp dir
  and discarded after the test).

They are marked ``slow`` so the default ``make test`` lane stays fast.
Run them with ``pytest -m slow tests/test_converter_smoke.py``.

Validation is intentionally run against an ``onnxruntime`` version that
*strictly* enforces the BYOM-compatible 3-input T5 encoder contract
(see ``pyproject.toml`` -- pinned to 1.21.x via the ``dev`` /
``byom-compat`` extras, the highest line that still requires
``num_subgraph_inputs == 3``). ``test_onnx_loads_in_onnxruntime`` asserts
the assertion is in fact 1.21.x; if a maintainer bumps the pin into the
1.22+ range, this test fails loudly because that range relaxes the
encoder validator and would let the converter regress to a
BYOM-incompatible artifact without notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "de_en_smoke.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def converted_onnx(tmp_path_factory, fixture) -> Path:
    """Convert the model once for all tests in this module.

    Calls the locked v1 public API ``convert_model`` with ``verify=False``:
    the strict in-process token-parity check duplicates what
    ``test_onnx_token_parity_with_transformers`` already does at module
    scope, so running it twice would just double the cost of the slow
    lane.
    """
    from teradata_opus_translate import convert_model

    out_dir = tmp_path_factory.mktemp("onnx")
    out_path = out_dir / "de-en.onnx"
    result = convert_model(fixture["model_id"], output_path=out_path, verify=False)
    return result.output_path


@pytest.mark.slow
def test_converter_writes_onnx(converted_onnx: Path) -> None:
    assert converted_onnx.exists()
    # ~500 MB is the rough size for opus-mt-de-en at fp32.  Anything
    # below 50 MB indicates we lost weights somewhere.
    size_mb = converted_onnx.stat().st_size / (1024 * 1024)
    assert size_mb > 50, f"ONNX file is suspiciously small: {size_mb:.1f} MB"


@pytest.mark.slow
def test_onnx_loads_in_onnxruntime(converted_onnx: Path) -> None:
    import onnxruntime as ort

    # Guardrail: BYOM ships ORT 1.16.3 (no py3.12 wheels on PyPI). 1.17--1.21
    # share the same strict 3-input ``T5EncoderSubgraph::Validate()`` check
    # that BYOM enforces; from 1.22 onward that check is relaxed. If
    # someone bumps the pin past 1.21 without re-checking BYOM
    # compatibility, fail here so we catch the regression locally rather
    # than at deploy time -- which is exactly what bit Issue #28.
    ort_version = tuple(int(x) for x in ort.__version__.split(".")[:2])
    assert ort_version <= (1, 21), (
        f"onnxruntime=={ort.__version__} is past the BYOM-compatible "
        "encoder Validate() boundary (1.22+ relaxes the 3-input check). "
        "Pin onnxruntime<1.22 in pyproject.toml's [dev]/[byom-compat] "
        "extras before bumping. See Issue #28."
    )

    sess = ort.InferenceSession(str(converted_onnx), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    output_names = {o.name for o in sess.get_outputs()}

    # ``num_return_sequences`` is intentionally NOT in this set as of
    # v1.0.1 -- it is baked into the graph as a ``Constant(1)`` node
    # feeding BeamSearch's slot 4 rather than exposed as a top-level
    # input. See Issue #82 / docs/decisions.md Decision 10.
    expected_inputs = {
        "input_ids",
        "attention_mask",
        "num_beams",
        "min_length",
        "max_length",
        "length_penalty",
        "repetition_penalty",
    }
    assert expected_inputs == input_names, (
        f"missing or unexpected inputs: "
        f"missing={expected_inputs - input_names} "
        f"extra={input_names - expected_inputs}"
    )
    assert "sequences" in output_names


@pytest.mark.slow
def test_onnx_translates_smoke_set(converted_onnx: Path, fixture: dict) -> None:
    """Confirm the ONNX file produces sensible English for the smoke set.

    This is a coarse parity check ('does the keyword appear in the
    translation?'), not a strict equivalence test against transformers.
    Strict parity is the job of issue #6.
    """
    import onnxruntime as ort
    from transformers import MarianTokenizer

    tok = MarianTokenizer.from_pretrained(fixture["model_id"])
    sess = ort.InferenceSession(str(converted_onnx), providers=["CPUExecutionProvider"])

    sentences = [pair["de"] for pair in fixture["pairs"]]
    keywords_per_sentence = [pair["en_keywords"] for pair in fixture["pairs"]]

    # ``num_return_sequences`` is intentionally absent from the feeds
    # as of v1.0.1 -- it is baked into the graph as a ``Constant(1)``
    # node and is no longer a top-level input. See Issue #82.
    enc = tok(sentences, return_tensors="np", padding=True)
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int32),
        "attention_mask": enc["attention_mask"].astype(np.int32),
        "num_beams": np.array([4], dtype=np.int32),
        "min_length": np.array([1], dtype=np.int32),
        "max_length": np.array([64], dtype=np.int32),
        "length_penalty": np.array([1.0], dtype=np.float32),
        "repetition_penalty": np.array([1.0], dtype=np.float32),
    }
    out = sess.run(None, feeds)
    sequences = out[0]  # (B, 1, max_length)
    assert sequences.shape[0] == len(sentences)
    assert sequences.shape[1] == 1

    decoded = [
        tok.decode(sequences[i, 0].tolist(), skip_special_tokens=True).lower()
        for i in range(len(sentences))
    ]

    failed: list[str] = []
    for de, en, keywords in zip(sentences, decoded, keywords_per_sentence):
        # At least one of the expected keywords must appear in the
        # decoded translation.  Permissive on purpose -- some models'
        # word choice is OK to drift (e.g. "favourite" vs "favorite").
        if not any(kw in en for kw in keywords):
            failed.append(f"  DE: {de}\n  EN: {en}\n  expected any of: {keywords}")
    assert not failed, "Translations missing all expected keywords:\n" + "\n".join(failed)


# ---------------------------------------------------------------------------
# Strict token-level parity: transformers.generate() vs ONNX runtime
# ---------------------------------------------------------------------------

# Generation parameters used for the parity check.  These are fed both as
# runtime inputs to the ONNX BeamSearch op AND as ``generate()`` kwargs on
# the HF side, so the two paths run with identical configuration.
#
# Values must be consistent with the BeamSearch op's *attributes* baked
# into the ONNX file (see ``converter/assemble.py``):
#
#   * ``no_repeat_ngram_size = 0``
#   * ``early_stopping       = True``
#   * ``eos_token_id         = config.eos_token_id``
#   * ``pad_token_id         = config.pad_token_id``
#   * ``decoder_start_token_id = config.decoder_start_token_id``
#   * ``model_type           = 1`` (encoder-decoder)
#
# The ONNX BeamSearch contrib op does NOT support ``bad_words_ids``,
# ``forced_eos_token_id``, or ``renormalize_logits`` (all of which are
# enabled by default in opus-mt-de-en's ``generation_config.json``).  To
# isolate the converter's correctness from those feature gaps, we
# explicitly DISABLE them on the transformers side via the kwargs below.
# That makes this a true "did the encoder/decoder weights and graph
# survive export" check, not a "does ONNX BeamSearch match HF defaults"
# check.
PARITY_PARAMS = {
    "num_beams": 4,
    "min_length": 1,
    "max_length": 64,
    "num_return_sequences": 1,
    "length_penalty": 1.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "early_stopping": True,
}


def _hf_generate_token_ids(model_id: str, sentence: str, params: dict) -> list[int]:
    """Run a single sentence through ``MarianMTModel.generate()`` with
    suppressions disabled so the result matches what ONNX BeamSearch can
    actually produce.
    """
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    tok = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).eval()

    enc = tok(sentence, return_tensors="pt")
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
            # --- Disable features ONNX BeamSearch contrib op does not have. ---
            bad_words_ids=None,
            forced_eos_token_id=None,
            renormalize_logits=False,
            # Force deterministic beam search.
            do_sample=False,
        )
    return out[0].tolist()


def _onnx_generate_token_ids(sess, tokenizer, sentence: str, params: dict) -> list[int]:
    """Run a single sentence through the ONNX BeamSearch graph."""
    # ``num_return_sequences`` is intentionally absent from the feeds
    # as of v1.0.1 -- it is baked into the graph as a ``Constant(1)``
    # node and is no longer a top-level input. ``params`` still carries
    # ``num_return_sequences`` for the HF side; the ONNX side ignores it.
    # See Issue #82.
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


def _strip_padding(token_ids: list[int], pad_token_id: int, eos_token_id: int) -> list[int]:
    """Strip trailing pad tokens AND a single trailing EOS, if present.

    The HF and ONNX paths handle the EOS marker in returned sequences
    differently:

    * ``MarianMTModel.generate()`` *includes* the final EOS token in the
      output sequence (then pads with ``pad_token_id`` out to
      ``max_length``).
    * ``com.microsoft.BeamSearch`` *omits* the EOS token from its
      ``sequences`` output (and pads with ``pad_token_id``).

    Both decode to the same text under ``skip_special_tokens=True``.
    For a meaningful token-level comparison, we normalize both sequences
    to the canonical form "tokens up to but not including EOS" before
    diffing.  This isolates differences in actual model output from a
    purely cosmetic boundary-marker convention.
    """
    out = list(token_ids)
    while out and out[-1] == pad_token_id:
        out.pop()
    # Strip a single trailing EOS, if present, so the two paths line up.
    if out and out[-1] == eos_token_id:
        out.pop()
    return out


def _first_divergence(a: list[int], b: list[int]) -> int:
    """Return the index of the first differing element, or -1 if the
    sequences are equal up to the shorter length and one is a prefix of
    the other (in which case the divergence is at ``min(len(a), len(b))``).
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


@pytest.mark.slow
def test_onnx_token_parity_with_transformers(converted_onnx: Path, fixture: dict) -> None:
    """Strict token-level parity check.

    For every sentence in the DE->EN fixture, ``MarianMTModel.generate()``
    and the assembled ONNX graph must produce the *exact same token-ID
    sequence*.  Any divergence is reported with the offending sentence,
    both token sequences, and the position of the first mismatch.

    This is the strongest pre-merge guarantee that the converter
    preserved encoder + decoder weights and graph topology.  If this
    test fails, the parity report task in issue #6 is going to fail
    too -- so failures here block merge of #2.
    """
    import onnxruntime as ort
    from transformers import MarianMTModel, MarianTokenizer

    model_id = fixture["model_id"]
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    # Pull pad_token_id off the model config so we don't hardcode 58100.
    cfg = MarianMTModel.from_pretrained(model_id).config
    pad_token_id = cfg.pad_token_id
    eos_token_id = cfg.eos_token_id

    sess = ort.InferenceSession(str(converted_onnx), providers=["CPUExecutionProvider"])

    failures: list[str] = []
    for pair in fixture["pairs"]:
        de = pair["de"]

        hf_ids = _strip_padding(
            _hf_generate_token_ids(model_id, de, PARITY_PARAMS),
            pad_token_id,
            eos_token_id,
        )
        onnx_ids = _strip_padding(
            _onnx_generate_token_ids(sess, tokenizer, de, PARITY_PARAMS),
            pad_token_id,
            eos_token_id,
        )

        if hf_ids != onnx_ids:
            div = _first_divergence(hf_ids, onnx_ids)
            hf_token = (
                tokenizer.convert_ids_to_tokens([hf_ids[div]])[0]
                if 0 <= div < len(hf_ids)
                else "<eos/end>"
            )
            onnx_token = (
                tokenizer.convert_ids_to_tokens([onnx_ids[div]])[0]
                if 0 <= div < len(onnx_ids)
                else "<eos/end>"
            )
            failures.append(
                "\n".join(
                    [
                        f"  DE: {de}",
                        f"  HF   ids ({len(hf_ids)}):   {hf_ids}",
                        f"  ONNX ids ({len(onnx_ids)}): {onnx_ids}",
                        f"  HF   text: {tokenizer.decode(hf_ids, skip_special_tokens=True)!r}",
                        f"  ONNX text: {tokenizer.decode(onnx_ids, skip_special_tokens=True)!r}",
                        f"  first divergence at position {div}: "
                        f"HF={hf_token!r} vs ONNX={onnx_token!r}",
                    ]
                )
            )

    assert not failures, (
        "ONNX BeamSearch output diverges from MarianMTModel.generate() "
        f"on {len(failures)}/{len(fixture['pairs'])} sentence(s):\n\n" + "\n\n".join(failures)
    )
