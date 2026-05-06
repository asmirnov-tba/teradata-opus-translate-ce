"""Schema-level tests for the saved transformers baseline artifact.

These tests are FAST: they only inspect the on-disk JSON. No model
loading, no inference. The slow regeneration of the artifact lives in
``scripts/build_baseline.py``; the slow re-run-and-compare check lives
in ``scripts/verify_reference_baseline.py``.

Tests covered here:

* Test set is well-formed (unique ids, multi-domain coverage, in count
  range).
* Baseline artifact header has all required provenance fields and the
  expected schema version.
* Generation parameters in the baseline header match
  ``PARITY_PARAMS`` exactly.
* Each record has the required fields, with a non-empty ``output_token_ids``
  list, a non-empty decoded translation, and ids matching the source
  test set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from teradata_opus_translate.baseline import (
    BASELINE_SCHEMA_VERSION,
    PARITY_PARAMS,
)

DATA_DIR = Path(__file__).parent / "data"
TEST_SET_PATH = DATA_DIR / "de_en_test_set.json"
BASELINE_PATH = DATA_DIR / "baseline_de_en.json"

REQUIRED_HEADER_FIELDS = {
    "schema_version",
    "model_id",
    "model_revision",
    "language_pair",
    "transformers_version",
    "tokenizers_version",
    "torch_version",
    "python_version",
    "generation_params",
    "deterministic_seed",
    "timestamp",
    "test_set_source",
    "test_set_description",
    "test_set_license",
    "record_count",
}

REQUIRED_RECORD_FIELDS = {
    "id",
    "source_text",
    "expected_translation",
    "output_token_ids",
    "metadata",
}

REQUIRED_METADATA_FIELDS = {
    "domain",
    "note",
    "source_token_length",
    "output_token_length",
}


@pytest.fixture(scope="module")
def test_set() -> dict:
    return json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(
            f"baseline artifact not present at {BASELINE_PATH}; "
            "regenerate with scripts/build_baseline.py"
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------


class TestTestSet:
    def test_test_set_file_exists(self) -> None:
        assert TEST_SET_PATH.exists(), TEST_SET_PATH

    def test_sentence_count_in_range(self, test_set: dict) -> None:
        n = len(test_set["sentences"])
        assert 50 <= n <= 200, f"expected 50-200 sentences, got {n}"

    def test_ids_unique(self, test_set: dict) -> None:
        ids = [s["id"] for s in test_set["sentences"]]
        assert len(set(ids)) == len(ids)

    def test_multi_domain_coverage(self, test_set: dict) -> None:
        domains = {s["domain"] for s in test_set["sentences"]}
        # Must cover at least these high-level buckets.
        for required in ("edge", "news", "conversational", "technical", "idiomatic"):
            assert required in domains, f"missing domain: {required}"

    def test_each_sentence_has_required_fields(self, test_set: dict) -> None:
        for s in test_set["sentences"]:
            assert {"id", "text", "domain"} <= set(s.keys()), s
            assert isinstance(s["text"], str) and s["text"].strip(), s

    def test_test_set_documents_license(self, test_set: dict) -> None:
        # Provenance is mandatory — a future contributor must be able to
        # tell whether they can redistribute these sentences.
        assert test_set.get("license", "").strip(), "test set must declare a license"
        assert test_set.get("description", "").strip(), "test set must have a description"


# ---------------------------------------------------------------------------
# Baseline artifact
# ---------------------------------------------------------------------------


class TestBaselineHeader:
    def test_header_present(self, baseline: dict) -> None:
        assert "header" in baseline
        assert "records" in baseline

    def test_required_header_fields(self, baseline: dict) -> None:
        header = baseline["header"]
        missing = REQUIRED_HEADER_FIELDS - set(header.keys())
        assert not missing, f"missing header fields: {missing}"

    def test_schema_version_matches(self, baseline: dict) -> None:
        assert baseline["header"]["schema_version"] == BASELINE_SCHEMA_VERSION

    def test_generation_params_match_parity_params(self, baseline: dict) -> None:
        # The whole point of the baseline is to lock in the generation
        # config. Drift here breaks every downstream parity test.
        assert baseline["header"]["generation_params"] == PARITY_PARAMS

    def test_record_count_matches_records(self, baseline: dict) -> None:
        assert baseline["header"]["record_count"] == len(baseline["records"])

    def test_language_pair(self, baseline: dict) -> None:
        assert baseline["header"]["language_pair"] == "de-en"


class TestBaselineRecords:
    def test_record_count_in_range(self, baseline: dict) -> None:
        n = len(baseline["records"])
        assert 50 <= n <= 200, f"expected 50-200 baseline records, got {n}"

    def test_each_record_has_required_fields(self, baseline: dict) -> None:
        for rec in baseline["records"]:
            missing = REQUIRED_RECORD_FIELDS - set(rec.keys())
            assert not missing, f"record {rec.get('id')!r} missing {missing}"
            meta_missing = REQUIRED_METADATA_FIELDS - set(rec["metadata"].keys())
            assert not meta_missing, f"record {rec.get('id')!r} metadata missing {meta_missing}"

    def test_token_ids_nonempty_and_int(self, baseline: dict) -> None:
        for rec in baseline["records"]:
            ids = rec["output_token_ids"]
            assert isinstance(ids, list) and ids, f"record {rec['id']!r} has empty token ids"
            assert all(isinstance(t, int) for t in ids), rec["id"]

    def test_decoded_text_nonempty(self, baseline: dict) -> None:
        # Even the shortest source ("Hallo.") should yield a non-empty
        # English string after decoding with skip_special_tokens=True.
        for rec in baseline["records"]:
            assert rec["expected_translation"].strip(), rec["id"]

    def test_record_ids_match_test_set(self, baseline: dict, test_set: dict) -> None:
        baseline_ids = {r["id"] for r in baseline["records"]}
        source_ids = {s["id"] for s in test_set["sentences"]}
        assert baseline_ids == source_ids, (
            f"baseline/test-set id drift: "
            f"only-in-baseline={baseline_ids - source_ids}, "
            f"only-in-test-set={source_ids - baseline_ids}"
        )

    def test_record_source_text_matches_test_set(self, baseline: dict, test_set: dict) -> None:
        by_id = {s["id"]: s["text"] for s in test_set["sentences"]}
        for rec in baseline["records"]:
            assert rec["source_text"] == by_id[rec["id"]], rec["id"]
