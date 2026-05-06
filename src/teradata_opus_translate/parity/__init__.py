"""Local parity helpers: ``onnxruntime`` vs HuggingFace ``transformers``.

This sub-package exposes the small, reusable building blocks needed to
compare a Marian model's output between two backends:

* HuggingFace ``MarianMTModel.generate()`` (the reference truth).
* The self-contained ONNX file produced by
  :func:`teradata_opus_translate.convert_model`,
  executed via ``onnxruntime``.

The same conventions are used by ``tests/test_converter_smoke.py``
(which has its own inline copies of these helpers, kept in lock-step
with this module by code review). Issue #6 needs the same logic from
``scripts/build_local_parity_report.py`` (62-sentence comparison
report). When you change ``PARITY_PARAMS`` or :func:`strip_padding`
here, audit the smoke test to keep it aligned.

The intentional design points to know about:

* ``PARITY_PARAMS`` is re-exported from
  :mod:`teradata_opus_translate.baseline.harness` so callers have a
  single import surface.
* :func:`strip_padding` strips trailing pads *and a single trailing
  EOS*. ``MarianMTModel.generate()`` includes the final EOS in the
  output; ``com.microsoft.BeamSearch`` omits it. Both decode to the
  same text under ``skip_special_tokens=True``; for token-level
  parity we normalise both to "tokens up to but not including EOS".
* Every public function takes already-loaded objects (a
  ``MarianMTModel``, a ``MarianTokenizer``, an
  ``onnxruntime.InferenceSession``). Module loading is the caller's
  job — that lets a long-running script load each artifact once and
  reuse it across all 62 sentences.
"""

from __future__ import annotations

from teradata_opus_translate.baseline.harness import PARITY_PARAMS
from teradata_opus_translate.parity.compare import (
    TokenComparison,
    compare_token_ids,
    first_divergence,
    strip_padding,
)
from teradata_opus_translate.parity.generate import (
    hf_generate_token_ids,
    onnx_generate_token_ids,
)
from teradata_opus_translate.parity.report import (
    PARITY_REPORT_SCHEMA_VERSION,
    ParityReport,
    ParityRow,
    render_report_markdown,
    write_report_json,
    write_report_markdown,
)

__all__ = [
    "PARITY_PARAMS",
    "PARITY_REPORT_SCHEMA_VERSION",
    "ParityReport",
    "ParityRow",
    "TokenComparison",
    "compare_token_ids",
    "first_divergence",
    "hf_generate_token_ids",
    "onnx_generate_token_ids",
    "render_report_markdown",
    "strip_padding",
    "write_report_json",
    "write_report_markdown",
]
