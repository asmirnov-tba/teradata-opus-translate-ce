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
