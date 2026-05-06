"""In-Teradata parity helpers (Issue #8).

This module is the third backend in the parity story:

* HuggingFace ``transformers.MarianMTModel.generate()`` (the gold-standard
  reference baseline, captured offline in
  ``tests/data/baseline_de_en.json``).
* Local ``onnxruntime`` execution of the converted ONNX model (captured in
  ``tests/data/local_parity_report.json``).
* **This module** -- the same ONNX model, executed *inside Teradata* via
  the ``TD_MLDB.ONNXSeq2Seq`` table operator on a BYOM-enabled Teradata 20.00
  instance.

Comparison contract
-------------------
``ONNXSeq2Seq`` returns a **decoded VARCHAR** sequence per input row -- not
token IDs. The native unit of comparison for issue #8 is therefore the
text string. We pass ``SkipSpecialTokens('true')`` (which is the BYOM
default but stating it explicitly avoids a future-default-change foot-gun)
so the Teradata-side text matches what
``MarianTokenizer.decode(..., skip_special_tokens=True)`` produces locally.

Generation parameters
---------------------
The ``Const_*`` clauses passed to ``ONNXSeq2Seq`` are derived from
:data:`teradata_opus_translate.baseline.harness.PARITY_PARAMS`. Only the
six parameters that BYOM's ``Const_*`` interface exposes are propagated:

* ``min_length``
* ``max_length``
* ``num_beams``
* ``length_penalty``
* ``repetition_penalty``
* ``num_return_sequences``

The remaining ``PARITY_PARAMS`` keys (``no_repeat_ngram_size``,
``early_stopping``) are not surfaced by BYOM's ``Const_*`` API; they were
already pinned to "no-op" / default values in the converter so the ONNX
graph that BYOM executes is identical to what local ``onnxruntime`` ran in
issue #6.

Memory and cache options
------------------------
Two additional ``USING`` clauses are emitted by default to make the
operator behave well on test instances with constrained memory and stale
caches:

* ``EnableMemoryCheck('false')`` -- skips BYOM's pre-flight memory check.
  The 743 MB encoder+decoder ONNX trips the check on smaller AMP
  configurations even when the actual inference fits in memory; the check
  itself is a guardrail, not a correctness requirement.
* ``OverwriteCachedModel('*')`` -- forces BYOM to discard any cached copy
  of every model id and reload from the model table. This protects us
  from "stale cache" bugs where a previously-deployed ONNX is silently
  served instead of the freshly-loaded blob (which we hit during the
  encoder/decoder/init refactor in #28/#29).

Both are harmless on a healthy instance and are critical on a
constrained one. They are exposed as :func:`build_onnxseq2seq_sql`
keyword arguments so callers can opt out for performance benchmarking.

Library policy
--------------
Per the project's coding standards we use the raw ``teradatasql`` driver
(DB-API 2.0) -- **never** ``teradataml``. ``teradataml`` is reserved for
the demo notebook (#9).

Idempotency
-----------
:func:`load_input_table` uses **DROP-and-CREATE** so re-running the report
script is safe even if a previous run was interrupted. The drop tolerates
the "table does not exist" Teradata error (3807).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from teradata_opus_translate.baseline.harness import PARITY_PARAMS

if TYPE_CHECKING:  # pragma: no cover - heavy import only used for typing
    import teradatasql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default schema the project's BYOM tables live in. The input table for the
# parity report is created here too so it shares the same access controls
# as the model and tokenizer rows.
#
# This value reflects the schema actually populated by the deploy module
# (#10) on the project's BYOM-enabled Teradata 20.00 instance. The earlier
# placeholder (``BYOM_USER``) only existed on a stale dev VM image; the
# canonical home is ``OPUS_BYOM`` per the README's connection table and
# PR #29's smoke test.
DEFAULT_INPUT_DATABASE: str = "OPUS_BYOM"

# Default name for the inputs table this module populates. Distinct from
# the BYOM doc's ``JulesBelvezeDummyData`` example so we don't collide
# with anything pre-existing on the VM.
DEFAULT_INPUT_TABLE: str = "de_en_parity_inputs"

# Default model_id / tokenizer_id used by the deploy module (#10) when
# loading ``opus-mt-de-en``.
DEFAULT_MODEL_ID: str = "opus-mt-de-en"

# Schema-qualified ONNXSeq2Seq operator. The BYOM 20.00 user guide uses
# both ``mldb.ONNXSeq2Seq`` and ``td_mldb.ONNXSeq2Seq`` in its examples,
# but on this VM image the operator is registered in ``TD_MLDB``
# (confirmed via ``HELP DATABASE TD_MLDB`` during PR #29 verification --
# the ``mldb`` synonym does not exist on this Teradata 20.00 image). If a
# future Teradata image moves the operator, override via
# :data:`ONNXSEQ2SEQ_SCHEMA`.
ONNXSEQ2SEQ_SCHEMA: str = "TD_MLDB"

# Six Const_* keys that ONNXSeq2Seq supports. Order is significant only for
# diagnostic logging -- BYOM accepts any ordering.
_CONST_KEYS: tuple[str, ...] = (
    "min_length",
    "max_length",
    "num_beams",
    "length_penalty",
    "repetition_penalty",
    "num_return_sequences",
)

# Teradata error codes we treat as "ok, the object wasn't there".
_ERR_TABLE_DOES_NOT_EXIST: int = 3807


# ---------------------------------------------------------------------------
# Const_* parameter formatting
# ---------------------------------------------------------------------------


def build_const_clauses(params: dict[str, Any] | None = None) -> list[str]:
    """Return ``Const_<name>(<value>)`` SQL fragments for ONNXSeq2Seq.

    The defaults come from
    :data:`teradata_opus_translate.baseline.harness.PARITY_PARAMS`. Only the
    keys that BYOM's ``Const_*`` interface accepts are emitted; unsupported
    keys (``early_stopping``, ``no_repeat_ngram_size``) are silently
    skipped because they're inert in the ONNX BeamSearch op.

    Floating-point parameters are formatted without scientific notation so
    the generated SQL is readable in logs (e.g. ``1.0`` not ``1e0``).

    Parameters
    ----------
    params:
        Optional override. When ``None`` the canonical ``PARITY_PARAMS``
        are used.

    Returns
    -------
    list[str]
        One ``Const_xxx(value)`` string per supported key, ordered as in
        :data:`_CONST_KEYS`.
    """
    src = params if params is not None else PARITY_PARAMS
    fragments: list[str] = []
    for k in _CONST_KEYS:
        if k not in src:
            raise KeyError(
                f"PARITY_PARAMS is missing required key {k!r}; refusing to "
                "build a partial Const_* clause set"
            )
        v = src[k]
        if isinstance(v, bool):  # bool is a subclass of int; reject early
            raise TypeError(f"PARITY_PARAMS[{k!r}] is bool; ONNXSeq2Seq expects numeric")
        if isinstance(v, float):
            fragments.append(f"Const_{k}({v:.6f})")
        elif isinstance(v, int):
            fragments.append(f"Const_{k}({v})")
        else:
            raise TypeError(
                f"PARITY_PARAMS[{k!r}]={v!r} is not a numeric value; "
                "ONNXSeq2Seq Const_* requires INTEGER or FLOAT"
            )
    return fragments


# ---------------------------------------------------------------------------
# Input table management
# ---------------------------------------------------------------------------


def _qualified(database: str, table: str) -> str:
    return f"{database}.{table}"


def _drop_input_table(
    connection: teradatasql.TeradataConnection,
    database: str,
    table: str,
) -> None:
    """DROP the inputs table, tolerating "does not exist" (3807)."""
    sql = f"DROP TABLE {_qualified(database, table)}"
    try:
        with connection.cursor() as cur:
            cur.execute(sql)
        logger.info("Dropped existing %s", _qualified(database, table))
    except Exception as exc:
        if f"[Error {_ERR_TABLE_DOES_NOT_EXIST}]" in str(exc):
            logger.debug("DROP %s: table did not exist (3807) -- ok", table)
            return
        raise


def _create_input_table(
    connection: teradatasql.TeradataConnection,
    database: str,
    table: str,
) -> None:
    """Create the parity inputs table.

    Schema: ``id VARCHAR(30)``, ``txt VARCHAR(2000) UNICODE``. The column
    name ``txt`` is **mandatory** -- BYOM's ``ONNXSeq2Seq`` looks up its
    input column by that exact name. The 2000-character width matches the
    BYOM doc's ``JulesBelvezeDummyData`` example and is plenty for the
    62-sentence DE test set (longest sentence is well under 200 chars).
    """
    sql = (
        f"CREATE MULTISET TABLE {_qualified(database, table)} ("
        " id VARCHAR(30) CHARACTER SET LATIN NOT CASESPECIFIC, "
        " txt VARCHAR(2000) CHARACTER SET UNICODE NOT CASESPECIFIC"
        ") PRIMARY INDEX (id)"
    )
    with connection.cursor() as cur:
        cur.execute(sql)
    logger.info("Created %s", _qualified(database, table))


def load_input_table(
    connection: teradatasql.TeradataConnection,
    rows: list[tuple[str, str]],
    *,
    database: str = DEFAULT_INPUT_DATABASE,
    table: str = DEFAULT_INPUT_TABLE,
) -> str:
    """Drop, recreate and populate the parity inputs table.

    Parameters
    ----------
    connection:
        Open Teradata connection (autocommit, per the deploy module).
    rows:
        ``(id, txt)`` pairs. ``id`` must fit in ``VARCHAR(30)``; ``txt``
        is the source-language sentence to be translated.
    database:
        Schema to create the table in. Defaults to
        :data:`DEFAULT_INPUT_DATABASE` (``OPUS_BYOM``).
    table:
        Table name. Defaults to :data:`DEFAULT_INPUT_TABLE`.

    Returns
    -------
    str
        The fully qualified ``database.table`` name, useful for embedding
        in subsequent ``ONNXSeq2Seq`` queries.

    Notes
    -----
    Idempotent via DROP-and-CREATE. The previous contents are not
    preserved -- the user's intent when calling this is "make the table
    look exactly like ``rows``".
    """
    for rid, txt in rows:
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"row id must be a non-empty string, got {rid!r}")
        if len(rid) > 30:
            raise ValueError(f"row id {rid!r} is {len(rid)} chars; column is VARCHAR(30)")
        if not isinstance(txt, str):
            raise ValueError(f"row txt must be a string, got {type(txt).__name__}")

    _drop_input_table(connection, database, table)
    _create_input_table(connection, database, table)

    insert_sql = f"INSERT INTO {_qualified(database, table)} (id, txt) VALUES (?, ?)"
    with connection.cursor() as cur:
        # teradatasql supports executemany with parameter binding; use it
        # so each insert is a parameterised round-trip rather than string
        # interpolation (which would also need escaping for embedded quotes).
        cur.executemany(insert_sql, list(rows))
    logger.info("Inserted %d rows into %s", len(rows), _qualified(database, table))

    return _qualified(database, table)


def teardown_input_table(
    connection: teradatasql.TeradataConnection,
    *,
    database: str = DEFAULT_INPUT_DATABASE,
    table: str = DEFAULT_INPUT_TABLE,
) -> None:
    """Drop the parity inputs table. Safe to call multiple times."""
    _drop_input_table(connection, database, table)


# ---------------------------------------------------------------------------
# ONNXSeq2Seq query
# ---------------------------------------------------------------------------


def build_onnxseq2seq_sql(
    *,
    input_database: str = DEFAULT_INPUT_DATABASE,
    input_table: str = DEFAULT_INPUT_TABLE,
    model_database: str = DEFAULT_INPUT_DATABASE,
    model_table: str = "onnx_models",
    tokenizer_table: str = "sequence_tokenizers",
    model_id: str = DEFAULT_MODEL_ID,
    tokenizer_id: str | None = None,
    params: dict[str, Any] | None = None,
    skip_special_tokens: bool = True,
    output_length: int = 1024,
    enable_memory_check: bool = False,
    overwrite_cached_model: str | None = "*",
) -> str:
    """Render the ``TD_MLDB.ONNXSeq2Seq`` SELECT statement we will execute.

    The SQL pattern is taken from the BYOM 20.00 user guide
    (``ONNXSeq2Seq Example: OutputLength VARCHAR (10000)``), adapted for
    our table names and the constant set in
    :data:`teradata_opus_translate.baseline.harness.PARITY_PARAMS`.

    Parameters
    ----------
    skip_special_tokens:
        Mapped to BYOM's ``SkipSpecialTokens('true'/'false')``. Defaults
        to ``True`` because that is the BYOM default *and* the natural
        comparison unit on the local side
        (``MarianTokenizer.decode(..., skip_special_tokens=True)``).
        Pass ``False`` only as a debug aid.
    output_length:
        Mapped to ``OutputLength(<int>)``. The BYOM doc example uses
        ``10000`` which is far more than we need for short DE sentences,
        but we mirror the doc's pattern of being generous because the
        cost is just the ``VARCHAR(N)`` declaration of the output column.
    tokenizer_id:
        Defaults to the ``model_id`` (the deployment convention used by
        ``deploy/__main__.py``).
    enable_memory_check:
        Mapped to ``EnableMemoryCheck('true'/'false')``. Defaults to
        ``False`` so the 743 MB encoder+decoder ONNX clears the operator's
        pre-flight memory check on AMP-constrained test instances. Set to
        ``True`` for production deployments where the memory ceiling is
        ample and the guardrail is genuinely useful.
    overwrite_cached_model:
        Mapped to ``OverwriteCachedModel('<value>')``. Defaults to
        ``'*'`` (overwrite every cached model id), which protects against
        the "stale cache" failure mode where BYOM serves a previously
        deployed model graph instead of the freshly loaded blob. Pass
        ``None`` to omit the clause entirely (e.g. for repeat
        latency-sensitive runs where the cache is desired).

    Returns
    -------
    str
        A single Teradata SQL statement (no trailing semicolon -- the
        ``teradatasql`` driver doesn't want one).
    """
    if tokenizer_id is None:
        tokenizer_id = model_id
    consts = "\n        ".join(build_const_clauses(params))
    skip = "true" if skip_special_tokens else "false"
    mem = "true" if enable_memory_check else "false"

    # Optional clauses are emitted on their own lines for readability.
    overwrite_clause = ""
    if overwrite_cached_model is not None:
        # Defensive escape: BYOM treats the value as a single-quoted
        # token; the only special characters we'd realistically pass are
        # ``*`` and a model id (alphanumerics + dashes), so a simple
        # apostrophe escape is sufficient. We still reject embedded
        # quotes outright because the BYOM grammar does not document an
        # escape sequence.
        if "'" in overwrite_cached_model:
            raise ValueError(
                f"overwrite_cached_model={overwrite_cached_model!r} contains a "
                "single quote; BYOM has no documented escape -- refuse"
            )
        overwrite_clause = f"\n        OverwriteCachedModel('{overwrite_cached_model}')"

    return (
        f"SELECT id, sequences\n"
        f"FROM {ONNXSEQ2SEQ_SCHEMA}.ONNXSeq2Seq(\n"
        f"    ON (SELECT id, txt FROM {input_database}.{input_table}) AS InputTable\n"
        f"    ON (SELECT model_id, model FROM {model_database}.{model_table}"
        f" WHERE model_id = '{model_id}') AS ModelTable DIMENSION\n"
        f"    ON (SELECT tokenizer FROM {model_database}.{tokenizer_table}"
        f" WHERE tokenizer_id = '{tokenizer_id}') AS TokenizerTable DIMENSION\n"
        f"    USING\n"
        f"        Accumulate('id')\n"
        f"        ModelOutputTensor('sequences')\n"
        f"        SkipSpecialTokens('{skip}')\n"
        f"        OutputLength({int(output_length)})\n"
        f"        EnableMemoryCheck('{mem}')"
        f"{overwrite_clause}\n"
        f"        {consts}\n"
        f") AS td"
    )


def fetch_onnx_seq2seq_outputs(
    connection: teradatasql.TeradataConnection,
    *,
    input_database: str = DEFAULT_INPUT_DATABASE,
    input_table: str = DEFAULT_INPUT_TABLE,
    model_database: str = DEFAULT_INPUT_DATABASE,
    model_table: str = "onnx_models",
    tokenizer_table: str = "sequence_tokenizers",
    model_id: str = DEFAULT_MODEL_ID,
    tokenizer_id: str | None = None,
    params: dict[str, Any] | None = None,
    skip_special_tokens: bool = True,
    output_length: int = 1024,
    enable_memory_check: bool = False,
    overwrite_cached_model: str | None = "*",
) -> dict[str, str]:
    """Run ``TD_MLDB.ONNXSeq2Seq`` and return ``{id: decoded_text}``.

    The keys are exactly the ``id`` values present in the inputs table
    (loaded via :func:`load_input_table`); the values are the decoded
    English strings BYOM returned.

    Whitespace, casing and punctuation are *preserved verbatim* -- the
    parity report compares the strings as-is, with no normalisation, so
    the user can spot cosmetic versus substantive differences.

    Raises
    ------
    RuntimeError
        If a duplicate id is returned from the operator (which would
        indicate a misconfigured ``Const_num_return_sequences`` -- it
        must be 1 for a parity comparison).
    """
    sql = build_onnxseq2seq_sql(
        input_database=input_database,
        input_table=input_table,
        model_database=model_database,
        model_table=model_table,
        tokenizer_table=tokenizer_table,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        params=params,
        skip_special_tokens=skip_special_tokens,
        output_length=output_length,
        enable_memory_check=enable_memory_check,
        overwrite_cached_model=overwrite_cached_model,
    )
    logger.info("Executing ONNXSeq2Seq query")
    logger.debug("SQL:\n%s", sql)
    with connection.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    out: dict[str, str] = {}
    for row in rows:
        rid, seq = row[0], row[1]
        rid_str = str(rid)
        if rid_str in out:
            raise RuntimeError(
                f"ONNXSeq2Seq returned duplicate id={rid_str!r}; check "
                "Const_num_return_sequences (must be 1 for parity)"
            )
        # Sequences come back as VARCHAR. teradatasql may yield bytes for
        # very large objects -- coerce defensively.
        if isinstance(seq, bytes):
            seq = seq.decode("utf-8")
        out[rid_str] = seq
    return out
