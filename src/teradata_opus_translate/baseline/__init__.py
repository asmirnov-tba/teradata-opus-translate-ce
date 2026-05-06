"""Reference baseline harness.

Builds the deterministic gold-standard translation set produced by
HuggingFace ``transformers`` for ``Helsinki-NLP/opus-mt-de-en``. The
artifact is the contract that downstream parity tests (ONNX runtime
parity in #6, Teradata in-DB parity in #9) compare against.

The module exposes:

* :data:`PARITY_PARAMS` — the generation kwargs that match what the
  ONNX BeamSearch op can replay. **Must stay in lock-step with the
  ``PARITY_PARAMS`` constant in** ``tests/test_converter_smoke.py``.
* :data:`BASELINE_SCHEMA_VERSION` — version stamped into every artifact.
* :func:`build_baseline` — the function the CLI script wraps.
* :func:`load_test_set` / :func:`load_baseline` — readers for the JSON
  artifacts.
* :func:`verify_baseline` — re-runs ``transformers`` and checks
  byte-for-byte parity with a previously saved artifact.
"""

from teradata_opus_translate.baseline.harness import (
    BASELINE_SCHEMA_VERSION,
    DEFAULT_MODEL_ID,
    PARITY_PARAMS,
    build_baseline,
    load_baseline,
    load_test_set,
    verify_baseline,
)

__all__ = [
    "BASELINE_SCHEMA_VERSION",
    "DEFAULT_MODEL_ID",
    "PARITY_PARAMS",
    "build_baseline",
    "load_baseline",
    "load_test_set",
    "verify_baseline",
]
