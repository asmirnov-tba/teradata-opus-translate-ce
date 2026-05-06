"""Public-facing :func:`convert_tokenizer` implementation.

The function exposed at ``teradata_opus_translate.convert_tokenizer``
lives here. The thin re-export at the package root pulls it from this
module.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from teradata_opus_translate._source import is_local_source, resolve_source

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvertTokenizerResult:
    """Summary returned by :func:`convert_tokenizer`.

    Attributes
    ----------
    output_path:
        Resolved absolute path of the written ``tokenizer.json`` file.
    size_bytes:
        Size of the written file, in bytes.
    source:
        The original ``source`` argument as resolved for
        ``from_pretrained``.
    source_kind:
        ``"local"`` if ``source`` was a local directory, otherwise
        ``"hf"``.
    """

    output_path: Path
    size_bytes: int
    source: str
    source_kind: Literal["hf", "local"]


def _setup_logging(verbose: bool) -> None:
    pkg_logger = logging.getLogger("teradata_opus_translate")
    pkg_logger.setLevel(logging.INFO if verbose else logging.WARNING)
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg_logger.addHandler(handler)


def convert_tokenizer(
    source: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> ConvertTokenizerResult:
    """Extract a single self-contained ``tokenizer.json`` from a Marian
    tokenizer.

    Parameters
    ----------
    source:
        HuggingFace repo id (e.g. ``"Helsinki-NLP/opus-mt-de-en"``) OR a
        path to a local directory containing a downloaded HF repo.
        Detection is automatic: an existing directory is treated as a
        local source; everything else is passed to ``from_pretrained``
        as an HF id.
    output_path:
        Destination ``tokenizer.json`` path. Parent directories are
        created if missing; the file is overwritten if it already
        exists.
    cache_dir:
        Optional HuggingFace cache directory passed through to
        ``from_pretrained``.
    verbose:
        If ``True``, configure the package logger at INFO level.

    Returns
    -------
    ConvertTokenizerResult
        Dataclass with ``output_path``, ``size_bytes``, resolved
        ``source`` and ``source_kind``.

    Raises
    ------
    RuntimeError
        If the in-process round-trip self-check fails. See
        :func:`teradata_opus_translate._tokenizer.extract.extract_tokenizer`
        for the canary-sentence check.
    """
    _setup_logging(verbose)

    from teradata_opus_translate._tokenizer.extract import extract_tokenizer

    resolved = resolve_source(source)
    src_kind: Literal["hf", "local"] = "local" if is_local_source(source) else "hf"

    out = Path(os.fspath(output_path)).expanduser().resolve()
    LOGGER.info("Building tokenizer.json (kind=%s) -> %s", src_kind, out)
    written = extract_tokenizer(resolved, out, cache_dir=cache_dir)
    size = written.stat().st_size
    LOGGER.info("Wrote %s (%d bytes / %.2f MiB)", written, size, size / (1024 * 1024))

    return ConvertTokenizerResult(
        output_path=written,
        size_bytes=size,
        source=resolved,
        source_kind=src_kind,
    )
