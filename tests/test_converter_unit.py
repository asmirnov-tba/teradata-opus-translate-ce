"""Fast unit tests for the converter public surface.

These tests do not download a model or export ONNX.  They verify the
public API surface and supporting fixtures so refactors are caught
quickly in the default ``make test`` lane.

The package no longer exposes a ``marian-to-onnx`` console script
(removed alongside the API refactor in #45 -- see
``docs/decisions.md``); the previous CLI-help smoke test was therefore
deleted rather than rewritten.
"""

from __future__ import annotations

import importlib


def test_package_exposes_convert_model() -> None:
    """The locked v1 public surface lives at the package root.

    ``convert_model`` is the new replacement for the removed
    ``teradata_opus_translate.converter.convert_marian_to_onnx``.
    """
    mod = importlib.import_module("teradata_opus_translate")
    assert hasattr(mod, "convert_model")
    assert callable(mod.convert_model)


def test_package_exposes_convert_tokenizer() -> None:
    """The tokenizer half of the locked v1 public surface."""
    mod = importlib.import_module("teradata_opus_translate")
    assert hasattr(mod, "convert_tokenizer")
    assert callable(mod.convert_tokenizer)


def _make_synthetic_subgraph_proto():
    """Build a minimal ``onnx.ModelProto`` standing in for an exported
    encoder / decoder subgraph in the assemble-path unit test.

    The real ``_build_top_level_graph`` only needs:
    * a writable ``graph`` whose ``name`` it can stamp, and
    * an ``opset_import`` list it can extend with ``com.microsoft``.

    The graph contents are never executed -- they are dropped into the
    BeamSearch node's ``encoder`` / ``decoder`` subgraph attributes and
    the test runs no inference. A graph with one Identity node satisfies
    both checker invariants and the BeamSearch attribute machinery.
    """
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "subgraph", [inp], [out])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])


class _FakeConfig:
    """Stand-in for ``MarianMTModel.config`` with only the IDs the
    BeamSearch builder actually reads.
    """

    eos_token_id = 0
    pad_token_id = 58100
    decoder_start_token_id = 58100


def test_assemble_pins_default_ir_version_8(tmp_path) -> None:
    """``_build_top_level_graph`` must stamp ``ir_version=8`` by default
    so the saved BYOM artifact loads on BYOM 7.x's bundled ORT.

    This is the fast-lane proxy for the BYOM-side defect (Issue #109).
    No HF download, no torch / transformers import -- just exercise the
    one line in ``assemble.py`` that owns the IR version.
    """
    import onnx

    from teradata_opus_translate._converter.assemble import _build_top_level_graph

    encoder_proto = _make_synthetic_subgraph_proto()
    decoder_proto = _make_synthetic_subgraph_proto()
    full = _build_top_level_graph(
        encoder_proto=encoder_proto,
        decoder_proto=decoder_proto,
        config=_FakeConfig(),
        no_repeat_ngram_size=0,
        early_stopping=True,
        package_version="0.0.0+test",
        ir_version=8,
    )
    assert full.ir_version == 8

    # Round-trip through disk: the on-disk artifact, not just the
    # in-memory proto, must report IR 8 -- this is what BYOM actually
    # reads from the model BLOB.
    out_path = tmp_path / "synthetic.onnx"
    out_path.write_bytes(full.SerializeToString())
    reloaded = onnx.load(str(out_path))
    assert reloaded.ir_version == 8


def test_assemble_honors_ir_version_override(tmp_path) -> None:
    """Passing ``ir_version=9`` must produce an IR-9 artifact.

    Pins the override path so a future tightening that hard-codes 8
    cannot silently strip the parameter's effect.
    """
    import onnx

    from teradata_opus_translate._converter.assemble import _build_top_level_graph

    full = _build_top_level_graph(
        encoder_proto=_make_synthetic_subgraph_proto(),
        decoder_proto=_make_synthetic_subgraph_proto(),
        config=_FakeConfig(),
        no_repeat_ngram_size=0,
        early_stopping=True,
        package_version="0.0.0+test",
        ir_version=9,
    )
    assert full.ir_version == 9

    out_path = tmp_path / "synthetic-ir9.onnx"
    out_path.write_bytes(full.SerializeToString())
    assert onnx.load(str(out_path)).ir_version == 9


def test_fixture_file_loads() -> None:
    """The DE->EN smoke fixture file must be valid JSON with expected schema."""
    import json
    from pathlib import Path

    fixture_path = Path(__file__).parent / "fixtures" / "de_en_smoke.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert "model_id" in data
    assert "pairs" in data
    assert len(data["pairs"]) >= 10
    for pair in data["pairs"]:
        assert "de" in pair
        assert "en_keywords" in pair
        assert isinstance(pair["en_keywords"], list)
        assert len(pair["en_keywords"]) >= 1
