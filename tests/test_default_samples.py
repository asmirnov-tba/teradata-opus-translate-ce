"""Coverage tests for ``_DEFAULT_SAMPLES_BY_SRC_LANG``.

The default-samples table is what verifies parity for callers who
don't pass ``verify_samples=`` explicitly. Phase 3 bulk conversion
relies on it: a missed source language means the parity check runs
against generic English text and silently rubber-stamps a model that
might not actually round-trip its real source-language inputs.

These tests pin the language coverage so adding or removing a key is
a deliberate change (the assertion message tells you what to update),
and that every language has a consistent, non-trivial sample set.

The expected coverage was derived from:

* The curated ``Helsinki-NLP/opustranslate`` HuggingFace collection
  (3-letter ISO 639-3 codes used by the ``opus-mt_tiny_*`` repos).
* All ISO 639-1 source codes appearing in 5+ ``opus-mt-*`` repos in
  the wider Helsinki-NLP catalogue snapshot
  (``helsinki-nlp-models.csv`` at the repo root).

Update the ``EXPECTED_LANGS`` constant when broadening or narrowing
the default-samples table; do not loosen the equality assertion.
"""

from __future__ import annotations

from teradata_opus_translate._converter.api import (
    _DEFAULT_SAMPLES_BY_SRC_LANG,
    _GENERIC_FALLBACK_SAMPLES,
    _default_samples,
    _infer_source_lang,
)

# ---------------------------------------------------------------------------
# Expected coverage
# ---------------------------------------------------------------------------

# Curated ``opustranslate`` collection source codes (3-letter ISO 639-3).
EXPECTED_COLLECTION_LANGS: frozenset[str] = frozenset(
    {
        "ara",
        "cat",
        "deu",
        "ell",
        "eng",
        "eus",
        "fra",
        "glg",
        "ita",
        "kor",
        "nld",
        "rus",
        "spa",
        "tur",
        "zho",
    }
)

# 2-letter ISO 639-1 codes appearing in 5+ opus-mt-* repos in the
# Helsinki-NLP catalogue snapshot, plus the 3-letter aliases above.
EXPECTED_LANGS: frozenset[str] = EXPECTED_COLLECTION_LANGS | frozenset(
    {
        # Western Europe
        "de",
        "fr",
        "es",
        "it",
        "en",
        "nl",
        "pt",
        "ca",
        "gl",
        "eu",
        "ro",
        # Northern Europe
        "sv",
        "da",
        "no",
        "is",
        "fi",
        "et",
        "lt",
        "lv",
        # Central / Eastern Europe
        "pl",
        "cs",
        "sk",
        "sl",
        "hu",
        "mt",
        "sq",
        "af",
        "eo",
        # Slavic / Cyrillic
        "ru",
        "uk",
        "bg",
        "mk",
        # Greek
        "el",
        # Semitic / RTL
        "ar",
        "he",
        # Turkic
        "tr",
        # East / Southeast Asian
        "ja",
        "ko",
        "zh",
        "vi",
        "id",
        # Caribbean Creole
        "ht",
        # Sub-Saharan Africa
        "ha",
        "yo",
        "ig",
        "sn",
        "rw",
        "rn",
        "lg",
        "sg",
        "st",
        "ts",
        # Pacific / Austronesian
        "tl",
        "to",
        # Other
        "ee",
    }
)


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------


def test_default_samples_coverage_matches_expected() -> None:
    """The ``_DEFAULT_SAMPLES_BY_SRC_LANG`` keys must equal the
    expected language set exactly.

    Adding a language here is fine -- update ``EXPECTED_LANGS`` in
    lockstep. Removing a language is also fine if the catalogue
    changes. What is **not** fine is silent drift: an unintended add
    means a Phase 3 conversion now tests against wrong-language
    samples; an unintended remove means it falls back to generic
    English without warning.
    """
    actual = frozenset(_DEFAULT_SAMPLES_BY_SRC_LANG.keys())
    missing = EXPECTED_LANGS - actual
    extra = actual - EXPECTED_LANGS
    assert not missing and not extra, (
        f"_DEFAULT_SAMPLES_BY_SRC_LANG drift detected.\n"
        f"  missing keys: {sorted(missing)}\n"
        f"  unexpected keys: {sorted(extra)}\n"
        "Update EXPECTED_LANGS in lockstep with the dict."
    )


def test_each_language_has_exactly_three_samples() -> None:
    """Volume parity per language: 3 samples each. Keeping the count
    consistent makes parity-report rows directly comparable across
    languages and prevents a single language from dominating the
    verification budget.
    """
    bad = {k: len(v) for k, v in _DEFAULT_SAMPLES_BY_SRC_LANG.items() if len(v) != 3}
    assert not bad, f"Languages with non-3 sample count: {bad}"


def test_each_sample_is_non_empty_minimum_length() -> None:
    """Sanity: no blank or near-blank samples slipped in. The 5-char
    floor catches the common copy-paste errors (a stray space, a
    single punctuation character) without being so strict that valid
    short greetings in compact scripts would fail.
    """
    bad: list[tuple[str, int, str]] = []
    for lang, samples in _DEFAULT_SAMPLES_BY_SRC_LANG.items():
        for i, s in enumerate(samples):
            if not s or not s.strip() or len(s) < 5:
                bad.append((lang, i, s))
    assert not bad, f"Suspiciously short / empty samples: {bad}"


def test_each_sample_is_str() -> None:
    """The dict values must be lists of strings. A stray bytes literal
    or accidental ``None`` would crash the verification pass mid-run.
    """
    for lang, samples in _DEFAULT_SAMPLES_BY_SRC_LANG.items():
        assert isinstance(samples, list), f"{lang}: not a list"
        for i, s in enumerate(samples):
            assert isinstance(s, str), f"{lang}[{i}]: not a str ({type(s).__name__})"


def test_collection_coverage_is_complete() -> None:
    """Every source language in the curated ``opustranslate``
    collection must be a key. Phase 3 bulk conversion will at minimum
    iterate the collection; missing one of these keys would make the
    conversion silently fall back to generic English.
    """
    actual = set(_DEFAULT_SAMPLES_BY_SRC_LANG.keys())
    missing = EXPECTED_COLLECTION_LANGS - actual
    assert not missing, (
        f"opustranslate collection languages missing from defaults: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Plumbing tests for the helpers that consume the dict
# ---------------------------------------------------------------------------


def test_default_samples_returns_per_language_set_for_known_code() -> None:
    """``_default_samples`` returns the language-specific list for a
    known code, not the generic fallback.
    """
    samples = _default_samples("de")
    assert samples == _DEFAULT_SAMPLES_BY_SRC_LANG["de"]
    assert samples is not _DEFAULT_SAMPLES_BY_SRC_LANG["de"], (
        "must return a copy so callers can mutate freely"
    )


def test_default_samples_returns_fallback_for_unknown_code() -> None:
    """An unknown / unparseable code falls through to the generic
    English fallback. The fallback exists for the rare case; it should
    still return a usable list.
    """
    samples = _default_samples("xx-not-a-real-code")
    assert samples == _GENERIC_FALLBACK_SAMPLES
    assert samples is not _GENERIC_FALLBACK_SAMPLES, "must return a copy"


def test_default_samples_returns_fallback_for_none() -> None:
    """``_infer_source_lang`` returns ``None`` for non-Helsinki ids
    and local paths; ``_default_samples(None)`` is the documented
    fallback path.
    """
    assert _default_samples(None) == _GENERIC_FALLBACK_SAMPLES


def test_infer_source_lang_handles_opus_mt_2letter() -> None:
    """``opus-mt-{src}-{tgt}`` -> 2-letter src."""
    assert _infer_source_lang("Helsinki-NLP/opus-mt-de-en") == "de"
    assert _infer_source_lang("Helsinki-NLP/opus-mt-fr-es") == "fr"


def test_infer_source_lang_handles_opus_mt_tiny_3letter() -> None:
    """``opus-mt_tiny_{src3}-{tgt3}`` -> 3-letter src (curated
    ``opustranslate`` collection format).
    """
    assert _infer_source_lang("Helsinki-NLP/opus-mt_tiny_deu-eng") == "deu"
    assert _infer_source_lang("Helsinki-NLP/opus-mt_tiny_kor-eng") == "kor"
    assert _infer_source_lang("Helsinki-NLP/opus-mt_tiny_eng-rus") == "eng"


def test_infer_source_lang_strips_tc_prefix() -> None:
    """Tatoeba-Challenge models prefix the src with ``tc`` /
    ``tc-big``; the stripped portion is the real source code.
    """
    assert _infer_source_lang("Helsinki-NLP/opus-mt-tc-big-en-fr") == "en"
    assert _infer_source_lang("Helsinki-NLP/opus-mt-tc-en-de") == "en"


def test_infer_source_lang_returns_none_for_local_path() -> None:
    """A path like ``/data/my-marian-dump/`` has no parseable language
    code; ``_infer_source_lang`` returns ``None`` and callers fall
    through to the generic fallback (or pass ``verify_samples=``
    explicitly).
    """
    assert _infer_source_lang("/data/my-marian-dump") is None
    assert _infer_source_lang("./local-checkout") is None


def test_collection_codes_round_trip_through_default_samples() -> None:
    """Spot-check: the curated collection ids run through
    ``_infer_source_lang`` -> ``_default_samples`` and land on a
    language-specific list (not the generic fallback).
    """
    for model_id in (
        "Helsinki-NLP/opus-mt_tiny_deu-eng",
        "Helsinki-NLP/opus-mt_tiny_kor-eng",
        "Helsinki-NLP/opus-mt_tiny_zho-eng",
        "Helsinki-NLP/opus-mt_tiny_ara-eng",
        "Helsinki-NLP/opus-mt_tiny_ell-eng",
    ):
        src = _infer_source_lang(model_id)
        assert src is not None
        samples = _default_samples(src)
        assert samples != _GENERIC_FALLBACK_SAMPLES, (
            f"{model_id}: fell through to generic fallback (src={src!r})"
        )
