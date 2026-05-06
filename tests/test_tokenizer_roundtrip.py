"""Round-trip tests for the extracted ``tokenizer.json``.

These tests are the authoritative correctness gate for the tokenizer
extraction step. They assert that the freshly built fast tokenizer
produces *byte-identical* token IDs to the original
``transformers.MarianTokenizer`` for every sentence in the DE->EN smoke
fixture, and that decoding round-trips to identical strings.

These tests do NOT require the model weights -- only the tokenizer
files (~2.7 MB total: ``source.spm``, ``target.spm``, ``vocab.json``,
``tokenizer_config.json``). They are fast on a warm HuggingFace cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "de_en_smoke.json"
MODEL_ID = "Helsinki-NLP/opus-mt-de-en"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def marian_tokenizer():
    from transformers import MarianTokenizer

    return MarianTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def tokenizer_json_path(tmp_path_factory) -> Path:
    """Build the tokenizer.json once for the whole module via the locked
    v1 public API ``convert_tokenizer``.
    """
    from teradata_opus_translate import convert_tokenizer

    out_dir = tmp_path_factory.mktemp("tokenizer")
    out_path = out_dir / "tokenizer.json"
    result = convert_tokenizer(MODEL_ID, output_path=out_path)
    return result.output_path


@pytest.fixture(scope="module")
def fast_tokenizer(tokenizer_json_path: Path):
    """Re-load the produced file via ``Tokenizer.from_file`` -- this is
    exactly the path BYOM's ``ONNXSeq2Seq`` will use against the BLOB.
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_json_path))


def test_tokenizer_json_is_self_contained(tokenizer_json_path: Path) -> None:
    """The produced file must be a single self-contained JSON.

    BYOM stores the tokenizer as a BLOB; if loading it required
    side-files (``source.spm`` etc.) the deployment would fail.
    """
    from tokenizers import Tokenizer

    assert tokenizer_json_path.is_file()
    # Sanity: the file must be loadable from disk without any sibling
    # files present.  Copy it to a clean directory and reload.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        clean = Path(d) / "tokenizer.json"
        shutil.copy(tokenizer_json_path, clean)
        tok = Tokenizer.from_file(str(clean))
        assert tok.encode("Hallo Welt.").ids, "loaded tokenizer must encode text"


def test_tokenizer_json_size_is_reasonable(tokenizer_json_path: Path) -> None:
    """Catch obvious regressions in file size.

    For ``opus-mt-de-en`` the file should be ~3-4 MiB. We bound it
    very loosely on both sides; a tokenizer.json that is < 100 KiB is
    almost certainly missing the vocabulary, and one > 50 MiB has
    something abnormal embedded.
    """
    size = tokenizer_json_path.stat().st_size
    assert 100 * 1024 < size < 50 * 1024 * 1024, (
        f"tokenizer.json size {size} bytes is outside the sane range"
    )


def test_token_ids_match_marian_on_fixtures(
    fast_tokenizer, marian_tokenizer, fixture: dict
) -> None:
    """For every fixture sentence, fast IDs must equal Marian IDs."""
    failures: list[str] = []
    for pair in fixture["pairs"]:
        sentence: str = pair["de"]
        marian_ids: list[int] = marian_tokenizer(sentence)["input_ids"]
        fast_ids: list[int] = fast_tokenizer.encode(sentence).ids
        if marian_ids != fast_ids:
            failures.append(f"  DE: {sentence!r}\n    Marian: {marian_ids}\n    fast  : {fast_ids}")
    assert not failures, (
        "Token-ID mismatches against MarianTokenizer "
        f"({len(failures)}/{len(fixture['pairs'])} fixtures):\n" + "\n".join(failures)
    )


def test_decode_matches_marian_on_fixtures(fast_tokenizer, marian_tokenizer, fixture: dict) -> None:
    """Decoding fast tokenizer's IDs must match Marian's decode output."""
    failures: list[str] = []
    for pair in fixture["pairs"]:
        sentence: str = pair["de"]
        ids = fast_tokenizer.encode(sentence).ids
        marian_text = marian_tokenizer.decode(ids, skip_special_tokens=True)
        fast_text = fast_tokenizer.decode(ids, skip_special_tokens=True)
        if marian_text != fast_text:
            failures.append(
                f"  DE: {sentence!r}\n"
                f"    Marian decode: {marian_text!r}\n"
                f"    fast   decode: {fast_text!r}"
            )
    assert not failures, "Decoded-text mismatches:\n" + "\n".join(failures)


def test_full_round_trip_preserves_input(fast_tokenizer, fixture: dict) -> None:
    """encode -> decode (skip specials) on the fast tokenizer must
    return the original input string for every fixture sentence.

    This is a stricter property than ID-equality: it catches metaspace /
    detokenization regressions that the previous test cannot.
    """
    failures: list[str] = []
    for pair in fixture["pairs"]:
        sentence: str = pair["de"]
        ids = fast_tokenizer.encode(sentence).ids
        decoded = fast_tokenizer.decode(ids, skip_special_tokens=True)
        if decoded != sentence:
            failures.append(f"  in:  {sentence!r}\n  out: {decoded!r}")
    assert not failures, "Round-trip lost characters:\n" + "\n".join(failures)


def test_eos_is_appended(fast_tokenizer, marian_tokenizer) -> None:
    """Marian appends ``</s>`` (id 0) to source sequences. The fast
    tokenizer's post-processor must do the same.
    """
    eos_id = marian_tokenizer.eos_token_id
    for sentence in ("Hallo.", "Wie geht es dir?", "Test"):
        ids = fast_tokenizer.encode(sentence).ids
        assert ids[-1] == eos_id, f"expected last id == eos ({eos_id}) for {sentence!r}, got {ids}"


def test_special_token_ids_align_with_marian(fast_tokenizer, marian_tokenizer) -> None:
    """The pad / eos / unk IDs must match what the ONNX BeamSearch graph
    has baked in (it pulls them from ``model.config``, which in turn
    matches MarianTokenizer's). A mismatch here would silently corrupt
    inference: BeamSearch would terminate on the wrong token, padding
    would land on the wrong slot, etc.
    """
    assert fast_tokenizer.token_to_id("</s>") == marian_tokenizer.eos_token_id
    assert fast_tokenizer.token_to_id("<pad>") == marian_tokenizer.pad_token_id
    assert fast_tokenizer.token_to_id("<unk>") == marian_tokenizer.unk_token_id


def test_vocab_size_matches(fast_tokenizer, marian_tokenizer) -> None:
    """Vocab sizes must match exactly so logits indexing is identical."""
    assert fast_tokenizer.get_vocab_size() == marian_tokenizer.vocab_size
