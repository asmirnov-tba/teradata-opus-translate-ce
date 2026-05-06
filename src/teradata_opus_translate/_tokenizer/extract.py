"""Build a single ``tokenizer.json`` for a Marian (OPUS-MT) model.

The produced file is the HuggingFace ``tokenizers`` library's native
serialization format and is fully self-contained -- it embeds the
vocabulary, the Unigram piece scores, the metaspace pre/post processors,
and the EOS post-processing template. Loading it back requires only
``tokenizers.Tokenizer.from_file(path)``; no ``.spm`` files, no
``vocab.json``, no Python wrapper.

This is the format Teradata's BYOM ``ONNXSeq2Seq`` function consumes
when it loads a tokenizer from a database BLOB column.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# The synthetic floor score offset applied to vocabulary pieces that do
# not appear in ``source.spm``. We pick "min source-spm score - 10.0"
# so target-side-only pieces remain strictly less likely than any real
# source piece in the Unigram model. They are unreachable when
# tokenizing source-language text, so the exact value does not affect
# round-trip token IDs; this just keeps the model well-formed.
_UNREACHABLE_SCORE_FLOOR_OFFSET = 10.0


def extract_tokenizer(
    resolved_source: str,
    output_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Extract a self-contained ``tokenizer.json`` from a Marian model.

    Parameters
    ----------
    resolved_source:
        Either a HuggingFace model id (e.g. ``"Helsinki-NLP/opus-mt-de-en"``)
        or an absolute local-directory path that
        ``MarianTokenizer.from_pretrained`` accepts. Use
        :func:`teradata_opus_translate._source.resolve_source` to
        normalize a user-supplied ``source`` argument first.
    output_path:
        Where to write the resulting ``tokenizer.json``. Parent
        directories are created if missing. The file is overwritten if
        it already exists.
    cache_dir:
        Optional HuggingFace cache directory passed through to
        ``from_pretrained``.

    Returns
    -------
    pathlib.Path
        The absolute path of the file that was written.

    Raises
    ------
    RuntimeError
        If the round-trip self-check fails. The function tokenizes a
        canary sentence through both the original ``MarianTokenizer``
        and the freshly built fast tokenizer and verifies that the
        token-id sequences match. A failure here means the produced
        ``tokenizer.json`` is not a faithful drop-in replacement and
        must not be deployed.
    """
    # Imports kept inside the function so that just importing the
    # package does not pull in transformers (heavy) for callers that
    # only want the type names.
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.decoders import Metaspace as MetaspaceDecoder
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Metaspace as MetaspacePre
    from tokenizers.processors import TemplateProcessing
    from transformers import MarianTokenizer

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MarianTokenizer for %s", resolved_source)
    from_pretrained_kwargs: dict[str, object] = {}
    if cache_dir is not None:
        from_pretrained_kwargs["cache_dir"] = str(cache_dir)
    marian = MarianTokenizer.from_pretrained(resolved_source, **from_pretrained_kwargs)

    fast = _build_fast_tokenizer(
        marian,
        Tokenizer=Tokenizer,
        Unigram=Unigram,
        MetaspacePre=MetaspacePre,
        MetaspaceDecoder=MetaspaceDecoder,
        TemplateProcessing=TemplateProcessing,
        AddedToken=AddedToken,
    )

    logger.info("Self-checking round-trip on canary sentence")
    _self_check_round_trip(marian, fast)

    logger.info("Writing %s", out)
    fast.save(str(out))

    size = out.stat().st_size
    logger.info("Wrote %s (%.2f MiB)", out, size / (1024 * 1024))
    return out


def _build_fast_tokenizer(
    marian,
    *,
    Tokenizer,
    Unigram,
    MetaspacePre,
    MetaspaceDecoder,
    TemplateProcessing,
    AddedToken,
):
    """Construct a ``tokenizers.Tokenizer`` equivalent to *marian*.

    The piece <-> id mapping is taken from ``marian.encoder`` (the HF
    combined ``vocab.json``), and the Unigram per-piece scores come
    from ``marian.spm_source`` (the source-side ``.spm``).
    """
    hf_vocab: dict[str, int] = marian.encoder
    n_vocab = len(hf_vocab)

    # Order pieces by id so list index == token id.
    vocab_pieces: list[str | None] = [None] * n_vocab
    for piece, idx in hf_vocab.items():
        if not 0 <= idx < n_vocab:
            raise RuntimeError(f"Vocab id {idx} out of range for piece {piece!r}")
        vocab_pieces[idx] = piece
    if any(p is None for p in vocab_pieces):
        missing = [i for i, p in enumerate(vocab_pieces) if p is None]
        raise RuntimeError(f"Vocab has gaps at ids: {missing[:10]}...")

    # Source-side SentencePiece scores keyed by piece string.
    sp = marian.spm_source
    sp_score: dict[str, float] = {
        sp.id_to_piece(i): sp.get_score(i) for i in range(sp.get_piece_size())
    }
    if not sp_score:
        raise RuntimeError("source.spm is empty -- cannot build tokenizer")
    floor_score = min(sp_score.values()) - _UNREACHABLE_SCORE_FLOOR_OFFSET

    # Special tokens get score 0.0 by convention; they are not produced
    # by Unigram's path search anyway because we register them as
    # added/special tokens after model construction.
    special_pieces = {"</s>", "<unk>", "<pad>"}

    vocab_pairs: list[tuple[str, float]] = []
    for piece in vocab_pieces:
        assert piece is not None  # for type checkers
        if piece in special_pieces:
            score = 0.0
        elif piece in sp_score:
            score = sp_score[piece]
        else:
            score = floor_score
        vocab_pairs.append((piece, score))

    unk_id = hf_vocab["<unk>"]
    eos_id = hf_vocab["</s>"]

    model = Unigram(vocab_pairs, unk_id=unk_id, byte_fallback=False)
    fast = Tokenizer(model)
    # Marian uses SentencePiece with a leading "▁" sentinel for word
    # starts; that's exactly what Metaspace implements.
    fast.pre_tokenizer = MetaspacePre(replacement="▁", prepend_scheme="always")
    fast.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")
    # Marian appends </s> to source sequences. Mirror that.
    fast.post_processor = TemplateProcessing(
        single="$A </s>",
        pair="$A </s> $B </s>",
        special_tokens=[("</s>", eos_id)],
    )
    # Mark specials so decode(skip_special_tokens=True) drops them and
    # they round-trip as atomic tokens.
    fast.add_special_tokens(
        [
            AddedToken("</s>", special=True, normalized=False),
            AddedToken("<unk>", special=True, normalized=False),
            AddedToken("<pad>", special=True, normalized=False),
        ]
    )
    return fast


_CANARY = "Hallo Welt."


def _self_check_round_trip(marian, fast) -> None:
    """Sanity-check the freshly built tokenizer against ``MarianTokenizer``.

    This is an in-process guardrail; the full multi-sentence round-trip
    test lives in ``tests/test_tokenizer_roundtrip.py`` and is the
    authoritative correctness gate.
    """
    expected = marian(_CANARY)["input_ids"]
    got = fast.encode(_CANARY).ids
    if expected != got:
        raise RuntimeError(
            "Self-check failed: token IDs differ for canary "
            f"sentence {_CANARY!r}.\n"
            f"  MarianTokenizer: {expected}\n"
            f"  rebuilt fast:    {got}\n"
            "Refusing to write a tokenizer.json that is not a faithful "
            "drop-in for the original."
        )
    expected_text = marian.decode(expected, skip_special_tokens=True)
    got_text = fast.decode(got, skip_special_tokens=True)
    if expected_text != got_text:
        raise RuntimeError(
            "Self-check failed: decoded text differs for canary.\n"
            f"  MarianTokenizer: {expected_text!r}\n"
            f"  rebuilt fast:    {got_text!r}"
        )
