"""Parse a project SQL file into individual Teradata-safe statements.

The files under ``sql/`` are written so that humans can run them through
BTEQ as-is, *and* the Python deploy module can apply them through
``teradatasql`` without depending on BTEQ. This module is the bridge:
it strips comments / blank lines, splits on ``;``, and yields one
statement at a time.

Database-name substitution
==========================

The DDL files use a ``${BYOM_DATABASE}`` placeholder rather than a literal
database name (e.g. ``BYOM_USER``). The placeholder follows
:class:`string.Template` syntax — i.e. ``${NAME}`` — so callers can
substitute a value at apply time without pulling in a templating library.

Substitution happens *after* comments are stripped, which keeps the
parser oblivious to placeholder syntax inside comments.

To keep the change minimally invasive, the public functions accept an
optional ``substitutions`` mapping. When omitted, statements are returned
unchanged (which is helpful for the existing tests that pass plain SQL
strings without a placeholder).

Limitations (deliberate):

* No support for BTEQ control commands (``.IF``, ``.RUN``, ``.LOGON`` etc.).
  The current ``sql/`` files do not use any.
* No support for embedded semicolons inside string literals or block
  comments. The current ``sql/`` files have neither.
* Substitutions are pre-validated by :class:`string.Template`; an
  unmatched placeholder raises ``KeyError`` (via
  :meth:`Template.substitute`) rather than silently leaving the
  placeholder in the SQL.

These limitations let us keep the parser tiny and easy to review. If a
future SQL file needs richer features, prefer running it through BTEQ
rather than expanding this parser.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from string import Template

# Strip line comments (``-- ...``) and ``/* ... */`` block comments.
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"--[^\n]*")


def _apply_substitutions(text: str, substitutions: Mapping[str, str] | None) -> str:
    """Substitute ``${NAME}`` placeholders using :class:`string.Template`.

    A ``None`` or empty mapping is a no-op — the original text is
    returned unchanged.
    """
    if not substitutions:
        return text
    return Template(text).substitute(substitutions)


def split_statements(
    sql_text: str,
    substitutions: Mapping[str, str] | None = None,
) -> list[str]:
    """Split ``sql_text`` into individual statements.

    Comments are stripped, blank statements are dropped, and surrounding
    whitespace is trimmed. The trailing ``;`` is removed from each
    statement so callers can pass the result directly to
    ``cursor.execute``.

    If ``substitutions`` is provided, every ``${NAME}`` placeholder in
    the (comment-stripped) SQL is substituted using
    :class:`string.Template` semantics. An unknown / unmatched
    placeholder raises :class:`KeyError`.
    """
    cleaned = _BLOCK_COMMENT.sub("", sql_text)
    cleaned = _LINE_COMMENT.sub("", cleaned)
    cleaned = _apply_substitutions(cleaned, substitutions)
    parts = [p.strip() for p in cleaned.split(";")]
    return [p for p in parts if p]


def iter_file_statements(
    path: Path,
    substitutions: Mapping[str, str] | None = None,
) -> Iterator[str]:
    """Yield non-empty SQL statements from the given file.

    See :func:`split_statements` for ``substitutions`` semantics.
    """
    yield from split_statements(path.read_text(encoding="utf-8"), substitutions)
