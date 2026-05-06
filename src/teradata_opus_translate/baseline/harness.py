"""Implementation of the reference baseline harness.

Produces a JSON artifact holding the gold-standard translations that
``MarianMTModel.generate()`` emits for our curated DE->EN test set
under the *exact* generation configuration the ONNX ``BeamSearch``
contrib op can replay.

Why these particular generation parameters?
-------------------------------------------
The opus-mt-de-en model ships with a ``generation_config.json`` that
enables ``bad_words_ids``, ``forced_eos_token_id``, and
``renormalize_logits`` by default. The ONNX ``com.microsoft.BeamSearch``
op does not implement any of those features, so to make the HF and ONNX
paths comparable we must pin generation to the *intersection* of what
both implementations support. That intersection is captured in
:data:`PARITY_PARAMS`, which intentionally mirrors the
``PARITY_PARAMS`` constant in ``tests/test_converter_smoke.py``. The
two MUST stay in lock-step — see #17 for the surrounding context.

Determinism guarantees
----------------------
* CPU only (``model.to('cpu')``).
* ``torch.manual_seed`` and ``numpy.random.seed`` set before each call.
* ``do_sample=False`` (greedy beam search).
* No threading/parallel ops in PyTorch (``torch.set_num_threads(1)``).

Given the same model revision, ``transformers`` version and Python
runtime, two runs of :func:`build_baseline` produce byte-identical
artifacts.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Version stamped into every baseline artifact's header. Bump when the
# record schema changes in a way that makes old artifacts unusable.
BASELINE_SCHEMA_VERSION = "1.0"

# Generation parameters that MUST match what ONNX BeamSearch can
# faithfully replay. Mirrors ``PARITY_PARAMS`` in
# ``tests/test_converter_smoke.py``. Keep these two in sync.
#
# The ONNX BeamSearch contrib op does NOT support ``bad_words_ids``,
# ``forced_eos_token_id``, or ``renormalize_logits``. We therefore pass
# ``bad_words_ids=None``, ``forced_eos_token_id=None``,
# ``renormalize_logits=False`` to ``model.generate()`` so the
# transformers output reflects what ONNX can actually produce.
PARITY_PARAMS: dict[str, Any] = {
    "num_beams": 4,
    "min_length": 1,
    "max_length": 64,
    "num_return_sequences": 1,
    "length_penalty": 1.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "early_stopping": True,
}

# Default model identifier for phase 1.
DEFAULT_MODEL_ID = "Helsinki-NLP/opus-mt-de-en"

# Determinism seed — fixed so re-running build_baseline produces the
# same bytes.
DETERMINISTIC_SEED = 0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineRecord:
    """One record in the saved baseline artifact."""

    id: str
    source_text: str
    expected_translation: str
    output_token_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_text": self.source_text,
            "expected_translation": self.expected_translation,
            "output_token_ids": list(self.output_token_ids),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_test_set(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the curated DE source sentence file.

    The file must conform to ``tests/data/de_en_test_set.json``: a JSON
    object with a ``sentences`` array of ``{id, text, domain, ...}``
    records.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if "sentences" not in data or not isinstance(data["sentences"], list):
        raise ValueError(f"{p} does not have a 'sentences' array")
    seen: set[str] = set()
    for entry in data["sentences"]:
        if not {"id", "text", "domain"} <= entry.keys():
            raise ValueError(f"sentence entry missing required keys: {entry!r}")
        if entry["id"] in seen:
            raise ValueError(f"duplicate sentence id: {entry['id']!r}")
        seen.add(entry["id"])
    return data


def load_baseline(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a saved baseline artifact and return the parsed dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_model_revision(model_id: str) -> str | None:
    """Best-effort lookup of the local cache snapshot SHA for a model.

    Returns ``None`` if the model isn't cached or the cache layout isn't
    recognisable. The SHA is a useful provenance pin but not required
    for determinism — :func:`build_baseline` does not fail if it's
    unknown.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi()
        info = api.model_info(model_id)
        sha = getattr(info, "sha", None)
        if isinstance(sha, str) and sha:
            return sha
    except Exception:  # provenance is best-effort
        logger.debug("could not resolve model revision via huggingface_hub", exc_info=True)
    return None


def _set_determinism() -> None:
    """Pin all known sources of nondeterminism for CPU beam search."""
    import contextlib
    import random

    import numpy as np
    import torch

    random.seed(DETERMINISTIC_SEED)
    np.random.seed(DETERMINISTIC_SEED)
    torch.manual_seed(DETERMINISTIC_SEED)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # older torches may not support this
            logger.debug("torch.use_deterministic_algorithms not honoured", exc_info=True)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        # set_num_interop_threads can only be called once per process.
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)


def _generate_one(model: Any, tokenizer: Any, sentence: str) -> tuple[list[int], str]:
    """Run one sentence through ``model.generate()`` and return
    ``(token_ids, decoded_text)``.

    See module docstring for why these particular kwargs are passed.
    """
    import torch

    enc = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=PARITY_PARAMS["num_beams"],
            min_length=PARITY_PARAMS["min_length"],
            max_length=PARITY_PARAMS["max_length"],
            num_return_sequences=PARITY_PARAMS["num_return_sequences"],
            length_penalty=PARITY_PARAMS["length_penalty"],
            repetition_penalty=PARITY_PARAMS["repetition_penalty"],
            no_repeat_ngram_size=PARITY_PARAMS["no_repeat_ngram_size"],
            early_stopping=PARITY_PARAMS["early_stopping"],
            # Disable features ONNX BeamSearch contrib op does not have.
            bad_words_ids=None,
            forced_eos_token_id=None,
            renormalize_logits=False,
            do_sample=False,
        )
    token_ids = out[0].tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return token_ids, text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_baseline(
    test_set_path: str | os.PathLike[str],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Generate the baseline artifact in memory and return it as a dict.

    Parameters
    ----------
    test_set_path:
        Path to ``de_en_test_set.json``.
    model_id:
        HuggingFace model identifier; defaults to ``opus-mt-de-en``.
    timestamp:
        ISO-8601 string to embed in the artifact header. Defaults to
        the literal ``"deterministic"`` placeholder so re-running this
        function produces byte-identical output. Override only when you
        explicitly want a wall-clock stamp.

    The returned dict has stable key ordering so :func:`json.dumps`
    with ``sort_keys=False`` produces deterministic bytes.
    """
    import tokenizers as _tk
    import torch
    import transformers as _tx
    from transformers import MarianMTModel, MarianTokenizer

    _set_determinism()

    test_set = load_test_set(test_set_path)
    sentences = test_set["sentences"]

    logger.info("loading model %s", model_id)
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).eval()
    model.to("cpu")

    revision = _resolve_model_revision(model_id)

    records: list[dict[str, Any]] = []
    for entry in sentences:
        _set_determinism()  # reset seed before each call for full reproducibility
        token_ids, decoded = _generate_one(model, tokenizer, entry["text"])
        # Source token length is useful metadata for downstream debugging.
        src_len = len(tokenizer(entry["text"])["input_ids"])
        rec = BaselineRecord(
            id=entry["id"],
            source_text=entry["text"],
            expected_translation=decoded,
            output_token_ids=token_ids,
            metadata={
                "domain": entry["domain"],
                "note": entry.get("note", ""),
                "source_token_length": src_len,
                "output_token_length": len(token_ids),
            },
        )
        records.append(rec.to_dict())
        logger.debug("translated %s -> %s", entry["id"], decoded)

    header = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": revision,
        "language_pair": test_set.get("language_pair", "de-en"),
        "transformers_version": _tx.__version__,
        "tokenizers_version": _tk.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "generation_params": dict(PARITY_PARAMS),
        "deterministic_seed": DETERMINISTIC_SEED,
        "timestamp": timestamp if timestamp is not None else "deterministic",
        "test_set_source": str(Path(test_set_path).name),
        "test_set_description": test_set.get("description", ""),
        "test_set_license": test_set.get("license", ""),
        "record_count": len(records),
    }

    artifact = {
        "header": header,
        "records": records,
    }
    return artifact


def write_baseline(artifact: dict[str, Any], output_path: str | os.PathLike[str]) -> Path:
    """Serialise the artifact to disk in deterministic, pretty-printed JSON.

    The resulting file is UTF-8, indented with two spaces, with non-ASCII
    characters preserved verbatim (``ensure_ascii=False``) and a trailing
    newline. Re-running :func:`build_baseline` followed by
    :func:`write_baseline` yields byte-identical output.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    out.write_text(text, encoding="utf-8")
    return out


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Convenience: SHA-256 of an on-disk file, hex-encoded."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_baseline(
    test_set_path: str | os.PathLike[str],
    baseline_path: str | os.PathLike[str],
    *,
    model_id: str = DEFAULT_MODEL_ID,
) -> tuple[bool, list[str]]:
    """Re-run the harness and confirm parity with a saved artifact.

    Returns ``(ok, errors)``. The check is record-by-record on the
    ``output_token_ids`` field — that is the strict invariant the
    downstream parity tests care about. Header drift (e.g. a newer
    ``transformers_version``) is reported as a warning, not a hard
    failure, so this function can also surface ecosystem drift.
    """
    saved = load_baseline(baseline_path)
    fresh = build_baseline(
        test_set_path, model_id=model_id, timestamp=saved["header"].get("timestamp")
    )

    errors: list[str] = []

    saved_records = {r["id"]: r for r in saved["records"]}
    fresh_records = {r["id"]: r for r in fresh["records"]}

    if saved_records.keys() != fresh_records.keys():
        missing = saved_records.keys() - fresh_records.keys()
        extra = fresh_records.keys() - saved_records.keys()
        if missing:
            errors.append(f"records missing from fresh run: {sorted(missing)}")
        if extra:
            errors.append(f"records added in fresh run: {sorted(extra)}")

    for rid in sorted(saved_records.keys() & fresh_records.keys()):
        s = saved_records[rid]
        f = fresh_records[rid]
        if s["output_token_ids"] != f["output_token_ids"]:
            errors.append(
                f"token id mismatch for id={rid!r}\n"
                f"  saved: {s['output_token_ids']}\n"
                f"  fresh: {f['output_token_ids']}"
            )
        elif s["expected_translation"] != f["expected_translation"]:
            errors.append(
                f"decoded text mismatch for id={rid!r}\n"
                f"  saved: {s['expected_translation']!r}\n"
                f"  fresh: {f['expected_translation']!r}"
            )

    # Warn (not fail) on header drift that indicates the ecosystem
    # changed underneath the saved baseline.
    for k in ("transformers_version", "torch_version", "tokenizers_version"):
        if saved["header"].get(k) != fresh["header"].get(k):
            logger.warning(
                "header drift on %s: saved=%r vs fresh=%r",
                k,
                saved["header"].get(k),
                fresh["header"].get(k),
            )

    return (not errors, errors)


def utc_now_iso() -> str:
    """Wall-clock ISO-8601 timestamp in UTC. Use only when an artifact
    explicitly wants a non-deterministic stamp."""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
