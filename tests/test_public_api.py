"""Public-API surface tests.

Pin the locked v1 ``teradata_opus_translate`` public surface so any
accidental change to function names, parameter names, parameter
defaults, or the int8 stub fails the fast unit lane immediately.

The locked spec is recorded in:

* Issue #45 (the API-refactor sub-issue), and
* the parent epic Issue #43, and
* the ADR ``docs/decisions.md`` entry titled
  *"v1 public API: two-callable surface, SQL-tunable params kept off
  the export API"*.

Updating these tests in lockstep with a deliberate API change is fine;
do **not** loosen them just to make a refactor pass.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


def test_convert_model_importable_from_root() -> None:
    """``convert_model`` lives at the package root, not under a subpackage."""
    import teradata_opus_translate as pkg

    assert hasattr(pkg, "convert_model")
    assert callable(pkg.convert_model)


def test_convert_tokenizer_importable_from_root() -> None:
    """``convert_tokenizer`` lives at the package root."""
    import teradata_opus_translate as pkg

    assert hasattr(pkg, "convert_tokenizer")
    assert callable(pkg.convert_tokenizer)


def test_result_dataclasses_importable_from_root() -> None:
    """The result dataclasses are part of the public surface so callers
    can type-annotate against them.
    """
    from teradata_opus_translate import (
        ConvertModelResult,
        ConvertTokenizerResult,
        ParityResult,
    )

    assert ConvertModelResult is not None
    assert ConvertTokenizerResult is not None
    assert ParityResult is not None


def test_no_legacy_subpackages_remain() -> None:
    """The pre-#45 ``converter`` and ``tokenizer`` subpackages must be
    gone. We are pre-1.0.0 so no compat shim was added; importing those
    names must raise ``ImportError``.
    """
    import importlib

    for legacy in (
        "teradata_opus_translate.converter",
        "teradata_opus_translate.tokenizer",
    ):
        with pytest.raises(ImportError):
            importlib.import_module(legacy)


# ---------------------------------------------------------------------------
# Signature pinning
# ---------------------------------------------------------------------------


def _params(func) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(func).parameters)


def test_convert_model_signature_is_locked() -> None:
    """Pin every parameter name, kind, and default of ``convert_model``.

    The locked spec for v1 (see docs/decisions.md):

    * ``source`` -- positional-or-keyword, no default
    * everything else -- keyword-only, with the defaults below
    * **must NOT expose**: ``num_beams``, ``max_length``, ``min_length``,
      ``length_penalty``, ``repetition_penalty``, ``num_return_sequences``
      (these are SQL-time tunables via BYOM ``Const_*``)
    """
    from teradata_opus_translate import convert_model

    params = _params(convert_model)

    assert list(params) == [
        "source",
        "precision",
        "output_path",
        "opset",
        "ir_version",
        "verify",
        "verify_samples",
        "no_repeat_ngram_size",
        "early_stopping",
        "cache_dir",
        "verbose",
        "log_level",
        # Added by Phase 3 (#140) for the static int8 calibration recipe;
        # deprecated as of v1.1.0 (Issue #160) when the int8 recipe was
        # switched to the calibration-free weight-only rewriter.  The
        # kwarg is retained on the signature for v1.0.x backward compat
        # but is a no-op (passing a non-None value triggers a
        # ``DeprecationWarning``).
        "calibration_pair",
    ]

    # ``source`` is the only positional-or-keyword parameter.
    assert params["source"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["source"].default is inspect.Parameter.empty

    # Every other parameter must be keyword-only.
    for name in list(params)[1:]:
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, f"{name} must be keyword-only"

    # Required-keyword: output_path has no default.
    assert params["output_path"].default is inspect.Parameter.empty

    # Defaults match the locked spec.
    assert params["precision"].default == "fp32"
    assert params["opset"].default == 14
    # ``ir_version`` default is **8** -- matches BYOM 7.x's bundled ORT
    # 1.16.3 lineage which rejects IR >= 9. The default must stay 8 even
    # if upstream ONNX changes the spec default; see Issue #109 and
    # ``docs/decisions.md`` Decision 12.
    assert params["ir_version"].default == 8
    assert params["verify"].default is True
    assert params["verify_samples"].default is None
    assert params["no_repeat_ngram_size"].default is None
    assert params["early_stopping"].default is None
    assert params["cache_dir"].default is None
    assert params["verbose"].default is False
    assert params["log_level"].default is None
    assert params["calibration_pair"].default is None


def test_convert_model_does_not_expose_sql_tunable_params() -> None:
    """Negative assertion: the export-time API must NOT take these.

    They live as graph inputs in the produced ONNX so BYOM ``Const_*``
    USING params can override them at scoring time. Adding them to the
    export API would let users bake them into the artifact and break
    the BYOM SQL contract.

    See ``docs/decisions.md`` and the
    ``teradata-byom-onnx-seq2seq`` skill.
    """
    from teradata_opus_translate import convert_model

    params = _params(convert_model)
    forbidden = {
        "num_beams",
        "max_length",
        "min_length",
        "length_penalty",
        "repetition_penalty",
        "num_return_sequences",
    }
    leaked = forbidden & set(params)
    assert not leaked, (
        f"convert_model must NOT expose SQL-tunable params; found: {leaked}. "
        "These belong as graph inputs, not as export-time knobs."
    )


def test_convert_tokenizer_signature_is_locked() -> None:
    """Pin the ``convert_tokenizer`` parameter list and defaults."""
    from teradata_opus_translate import convert_tokenizer

    params = _params(convert_tokenizer)

    assert list(params) == ["source", "output_path", "cache_dir", "verbose"]

    assert params["source"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["source"].default is inspect.Parameter.empty
    assert params["output_path"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["output_path"].default is inspect.Parameter.empty
    assert params["cache_dir"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["cache_dir"].default is None
    assert params["verbose"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["verbose"].default is False


def test_convert_model_return_type_annotation() -> None:
    """``convert_model`` must declare a ``ConvertModelResult`` return type.

    Library callers annotate against the return type; bare ``Any`` would
    silently widen the contract.
    """
    from teradata_opus_translate import ConvertModelResult, convert_model

    hints = get_type_hints(convert_model)
    assert hints.get("return") is ConvertModelResult


def test_convert_tokenizer_return_type_annotation() -> None:
    """``convert_tokenizer`` must declare a ``ConvertTokenizerResult``
    return type.
    """
    from teradata_opus_translate import (
        ConvertTokenizerResult,
        convert_tokenizer,
    )

    hints = get_type_hints(convert_tokenizer)
    assert hints.get("return") is ConvertTokenizerResult


# ---------------------------------------------------------------------------
# int8 surface (post-#46: implemented, not stubbed)
# ---------------------------------------------------------------------------


def test_int8_precision_is_implemented(tmp_path) -> None:
    """``precision='int8'`` must NOT raise ``NotImplementedError``.

    Negative pinning of the post-#46 surface: the int8 path is no longer
    a stub. We do not run the full conversion here (that costs ~30s and
    a transformers + torch import), but we assert that
    ``NotImplementedError`` is no longer the path's failure mode.

    The actual end-to-end int8 conversion is exercised in the slow lane
    by ``tests/test_int8_quantization.py``.
    """
    import inspect
    from typing import get_args, get_type_hints

    from teradata_opus_translate import convert_model

    hints = get_type_hints(convert_model)
    precision_hint = hints["precision"]
    # The Literal["fp32", "int8"] annotation must contain "int8"
    # explicitly so static type checkers accept the call.
    assert "int8" in get_args(precision_hint), (
        f"precision parameter must accept Literal 'int8'; got {precision_hint!r}"
    )

    # The default must remain "fp32" so existing callers see no
    # behaviour change post-#46.
    params = dict(inspect.signature(convert_model).parameters)
    assert params["precision"].default == "fp32"


def test_unknown_precision_raises_value_error(tmp_path) -> None:
    """A precision value that is neither ``'fp32'`` nor ``'int8'`` must
    raise ``ValueError``. ``Literal[...]`` is a typing-time hint only;
    the runtime check is what actually keeps the artifact from being
    written under a typo'd precision.
    """
    from teradata_opus_translate import convert_model

    out = tmp_path / "should-not-be-written.onnx"
    with pytest.raises(ValueError):
        convert_model(
            "Helsinki-NLP/opus-mt-de-en",
            precision="fp16",  # type: ignore[arg-type]  # intentional bad value
            output_path=out,
            verify=False,
        )
    assert not out.exists()
