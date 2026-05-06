"""Source resolution: HuggingFace model id vs local path.

Both ``convert_model`` and ``convert_tokenizer`` accept a ``source``
argument that can be either a HuggingFace repo id (e.g.
``"Helsinki-NLP/opus-mt-de-en"``) or a local filesystem path that
contains a downloaded HuggingFace repo. The detection rule must be
unambiguous and must not depend on whether the path happens to contain
a slash (HF ids contain slashes too).

Decision: a value is a *local path* if and only if the candidate
resolves to an existing directory on disk. Anything else is treated as
a HuggingFace repo id and passed straight through to
``from_pretrained``. This deliberately rejects local *file* paths --
HuggingFace repos are directories, never single files.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_source(source: str | os.PathLike[str]) -> str:
    """Resolve ``source`` to a string suitable for ``from_pretrained``.

    Parameters
    ----------
    source:
        Either a HuggingFace repo id (e.g.
        ``"Helsinki-NLP/opus-mt-de-en"``) or a path to a local directory
        containing a downloaded HuggingFace repo (e.g.
        ``"/tmp/opus-mt-de-en"`` or ``Path(...)``).

    Returns
    -------
    str
        A string suitable to pass to ``MarianMTModel.from_pretrained``
        or ``MarianTokenizer.from_pretrained`` as its first positional
        argument. Local paths are returned as resolved absolute strings;
        HuggingFace ids are returned unchanged.

    Notes
    -----
    The detection rule is intentionally simple: if the candidate
    resolves to an existing **directory**, it is a local source;
    otherwise it is a HuggingFace repo id. A path that does not exist
    is treated as a HuggingFace id and the underlying
    ``from_pretrained`` call will raise the usual download error if it
    is not actually a real repo.
    """
    candidate = Path(os.fspath(source)).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())
    return os.fspath(source)


def is_local_source(source: str | os.PathLike[str]) -> bool:
    """Return ``True`` iff ``source`` resolves to an existing directory.

    Useful for log lines and result metadata that distinguish "loaded
    from disk" from "downloaded from HuggingFace".
    """
    candidate = Path(os.fspath(source)).expanduser()
    return candidate.is_dir()
