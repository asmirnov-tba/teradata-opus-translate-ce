"""Language metadata used to render Hugging Face model cards.

The S3 manifest (``data/s3_manifest.json``) is the canonical source for
which models we publish. Each model's ``short_id`` ends in ``<src>-<tgt>``
where both codes are **ISO 639-3 (3-letter)**. Hugging Face's ``language:``
front-matter on the other hand expects **BCP-47 / ISO 639-1 (2-letter)**
codes, and the human-facing card body wants the full English name plus
(where reasonable) a flag emoji.

This module bundles all three under one keyed-by-ISO-639-3 lookup.

Editing rules
-------------

* Keep the table in sync with every src/tgt code that appears in
  ``data/s3_manifest.json``. The unit-test ``test_lang_info_covers_manifest``
  fails CI if a code drops in without a matching entry.
* For languages where no single flag is unambiguous (e.g. Basque, Galician,
  cross-border languages, constructed languages), set ``flag = None`` —
  *do not* substitute a country flag. The Jinja template gracefully omits
  flag glyphs and the surrounding whitespace when ``flag`` is ``None``.
* For English use ``🇬🇧`` (NOT ``🇺🇸``). This was an explicit user decision
  reaffirmed in Issue #120.
* Spanish is rendered with the Spanish flag ``🇪🇸``. Catalan is also tagged
  with ``🇪🇸`` because the upstream Helsinki-NLP cards do the same.
"""

from __future__ import annotations

from typing import TypedDict


class LangEntry(TypedDict):
    """A single language entry."""

    name_en: str
    bcp47: str
    flag: str | None


# Keyed by ISO 639-3 (3-letter). All codes that currently appear in
# ``data/s3_manifest.json`` MUST be present.
LANG_INFO: dict[str, LangEntry] = {
    "ara": {"name_en": "Arabic", "bcp47": "ar", "flag": "\U0001f1f8\U0001f1e6"},  # 🇸🇦
    "cat": {"name_en": "Catalan", "bcp47": "ca", "flag": "\U0001f1ea\U0001f1f8"},  # 🇪🇸
    "deu": {"name_en": "German", "bcp47": "de", "flag": "\U0001f1e9\U0001f1ea"},  # 🇩🇪
    "ell": {"name_en": "Greek", "bcp47": "el", "flag": "\U0001f1ec\U0001f1f7"},  # 🇬🇷
    "eng": {"name_en": "English", "bcp47": "en", "flag": "\U0001f1ec\U0001f1e7"},  # 🇬🇧
    "eus": {"name_en": "Basque", "bcp47": "eu", "flag": None},
    "fra": {"name_en": "French", "bcp47": "fr", "flag": "\U0001f1eb\U0001f1f7"},  # 🇫🇷
    "glg": {"name_en": "Galician", "bcp47": "gl", "flag": None},
    "ita": {"name_en": "Italian", "bcp47": "it", "flag": "\U0001f1ee\U0001f1f9"},  # 🇮🇹
    "kor": {"name_en": "Korean", "bcp47": "ko", "flag": "\U0001f1f0\U0001f1f7"},  # 🇰🇷
    "nld": {"name_en": "Dutch", "bcp47": "nl", "flag": "\U0001f1f3\U0001f1f1"},  # 🇳🇱
    "rus": {"name_en": "Russian", "bcp47": "ru", "flag": "\U0001f1f7\U0001f1fa"},  # 🇷🇺
    "spa": {"name_en": "Spanish", "bcp47": "es", "flag": "\U0001f1ea\U0001f1f8"},  # 🇪🇸
    "tur": {"name_en": "Turkish", "bcp47": "tr", "flag": "\U0001f1f9\U0001f1f7"},  # 🇹🇷
    "zho": {"name_en": "Chinese", "bcp47": "zh", "flag": "\U0001f1e8\U0001f1f3"},  # 🇨🇳
}


def lookup(code: str) -> LangEntry:
    """Return the ``LangEntry`` for the given ISO 639-3 ``code``.

    Parameters
    ----------
    code:
        Three-letter ISO 639-3 language code, lower-cased.

    Returns
    -------
    LangEntry
        The matching entry.

    Raises
    ------
    KeyError
        If ``code`` has no entry. The error message names the missing
        code and points at the location where it must be added.
    """

    try:
        return LANG_INFO[code]
    except KeyError as exc:
        raise KeyError(
            f"No LANG_INFO entry for language code {code!r}. "
            "Add it to src/teradata_opus_translate/_publish/lang_info.py "
            "(keyed by ISO 639-3) before publishing."
        ) from exc


def split_short_id(short_id: str) -> tuple[str, str]:
    """Split a manifest ``short_id`` into ``(src_code, tgt_code)``.

    The expected layout is ``opus-mt_tiny_<src>-<tgt>``. The trailing
    ``<src>-<tgt>`` segment is what carries the language pair.

    Raises
    ------
    ValueError
        If ``short_id`` does not match the expected layout.
    """

    parts = short_id.rsplit("_", 1)
    if len(parts) != 2 or "-" not in parts[1]:
        raise ValueError(
            f"Cannot parse language pair from short_id {short_id!r}; "
            "expected '...<src>-<tgt>' suffix."
        )
    pair = parts[1]
    src, _, tgt = pair.partition("-")
    if not src or not tgt:
        raise ValueError(f"Empty src or tgt in short_id {short_id!r}.")
    return src, tgt
