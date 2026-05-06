"""Connect to Teradata and load ONNX model + tokenizer BLOBs into BYOM tables.

Public surface (re-exported by :mod:`teradata_opus_translate.deploy`):

* :func:`connect`           -- open a connection from a :class:`ConnectionConfig`.
* :func:`apply_ddl`         -- run one or more SQL files (DDL idempotency strategy
                              is "drop-and-recreate"; see ``sql/`` for details).
* :func:`load_model`        -- insert a model BLOB by ``model_id``.
* :func:`load_tokenizer`    -- insert a tokenizer BLOB by ``tokenizer_id``.
* :func:`verify_model`      -- read back ``BYTES(model)`` for a row.
* :func:`verify_tokenizer`  -- read back ``BYTES(tokenizer)`` for a row.

All Teradata access goes through the raw ``teradatasql`` driver, per the
project's library policy.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from teradata_opus_translate.deploy.config import ConnectionConfig
from teradata_opus_translate.deploy.sql_script import iter_file_statements

# Placeholder name used in ``sql/*.sql`` to mark the BYOM-scoped database
# name. Kept as a module-level constant so callers (CLI, tests, future
# scripts) can build a substitutions map without re-typing the literal.
SQL_DATABASE_PLACEHOLDER = "BYOM_DATABASE"

if TYPE_CHECKING:  # pragma: no cover -- heavy import only used at type-check time
    import teradatasql

logger = logging.getLogger(__name__)


# Teradata error codes we sometimes want to ignore. See the BYOM-tables
# DDL files for context.
_ERR_TABLE_DOES_NOT_EXIST = 3807  # "Object 'foo' does not exist."
_ERR_DATABASE_DOES_NOT_EXIST = 3802
# 5612 = "User does not have <DROP> access to ..." — not raised when the
# object doesn't exist, but kept as a sanity reminder.

DDL_TOLERATED_ERRORS: tuple[int, ...] = (
    _ERR_TABLE_DOES_NOT_EXIST,
    _ERR_DATABASE_DOES_NOT_EXIST,
)


def connect(config: ConnectionConfig) -> teradatasql.TeradataConnection:
    """Open a Teradata connection from the given config.

    Returns a connection in *autocommit* mode (the driver's default) so
    that DDL committing happens implicitly. The caller is responsible
    for closing it (use a ``with`` statement).
    """
    import teradatasql  # local import: heavy module, avoid hard import cost

    logger.info("Connecting to Teradata at %s as %s", config.host, config.user)
    return teradatasql.connect(
        host=config.host,
        user=config.user,
        password=config.password,
    )


# ---------------------------------------------------------------------------
# DDL application
# ---------------------------------------------------------------------------


def apply_ddl(
    connection: teradatasql.TeradataConnection,
    sql_files: Iterable[Path],
    *,
    tolerate_errors: tuple[int, ...] = DDL_TOLERATED_ERRORS,
    substitutions: Mapping[str, str] | None = None,
) -> int:
    """Apply each SQL file's statements through the given connection.

    Args:
        connection: An open Teradata connection (typically from :func:`connect`).
        sql_files: Iterable of SQL file paths, applied in the given order.
        tolerate_errors: Teradata error codes to ignore (default: "table /
            database does not exist" which lets DROP-then-CREATE be idempotent
            on a fresh instance).
        substitutions: Optional ``string.Template`` substitutions applied to
            every SQL file before splitting / execution. Used to inject the
            target BYOM database name into the ``${BYOM_DATABASE}``
            placeholder in ``sql/01_create_database.sql`` and
            ``sql/02_create_tables.sql``. See
            :func:`substitutions_for_config` for the typical mapping.

    Returns:
        The number of statements that were executed *and* succeeded.
    """
    executed = 0
    for sql_file in sql_files:
        logger.info("Applying %s", sql_file)
        for statement in iter_file_statements(sql_file, substitutions):
            try:
                with connection.cursor() as cur:
                    cur.execute(statement)
                executed += 1
                logger.debug("OK: %s", _summarize(statement))
            except Exception as exc:
                if _is_tolerated(exc, tolerate_errors):
                    logger.info("Ignoring tolerated error on: %s", _summarize(statement))
                    continue
                raise
    return executed


def substitutions_for_config(config: ConnectionConfig) -> dict[str, str]:
    """Return the ``string.Template`` substitutions for a given config.

    Currently this maps :data:`SQL_DATABASE_PLACEHOLDER`
    (``BYOM_DATABASE``) to ``config.database``. Centralised here so the
    CLI, scripts, and tests share one source of truth for which
    placeholders the deploy module knows about.
    """
    return {SQL_DATABASE_PLACEHOLDER: config.database}


def _is_tolerated(exc: BaseException, codes: tuple[int, ...]) -> bool:
    """Return True if ``exc`` is a Teradata error whose code is in ``codes``."""
    text = str(exc)
    return any(f"[Error {code}]" in text or f"Error {code} " in text for code in codes)


def _summarize(stmt: str) -> str:
    one_line = " ".join(stmt.split())
    return one_line[:80] + ("..." if len(one_line) > 80 else "")


# ---------------------------------------------------------------------------
# Model / tokenizer load
# ---------------------------------------------------------------------------


def _row_count(
    connection: teradatasql.TeradataConnection,
    table: str,
    id_column: str,
    id_value: str,
) -> int:
    sql = f"SELECT COUNT(*) FROM {table} WHERE {id_column} = ?"
    with connection.cursor() as cur:
        cur.execute(sql, [id_value])
        (count,) = cur.fetchone()
    return int(count)


def _delete_row(
    connection: teradatasql.TeradataConnection,
    table: str,
    id_column: str,
    id_value: str,
) -> None:
    sql = f"DELETE FROM {table} WHERE {id_column} = ?"
    with connection.cursor() as cur:
        cur.execute(sql, [id_value])


def _insert_blob(
    connection: teradatasql.TeradataConnection,
    table: str,
    id_column: str,
    blob_column: str,
    id_value: str,
    blob_bytes: bytes,
) -> None:
    sql = f"INSERT INTO {table} ({id_column}, {blob_column}) VALUES (?, ?)"
    with connection.cursor() as cur:
        cur.execute(sql, [id_value, blob_bytes])


def _read_blob_size(
    connection: teradatasql.TeradataConnection,
    table: str,
    id_column: str,
    blob_column: str,
    id_value: str,
) -> int | None:
    # NOTE on BLOB size functions in Teradata 20.00:
    #   * `OCTET_LENGTH(blob_col)` raises 3580 ("Illegal use of CHARACTERS,
    #     MCHARACTERS, or OCTET_LENGTH functions") -- those text functions
    #     are not allowed on LOB columns.
    #   * `LENGTH(blob_col)` raises 9881.
    #   * `CHAR_LENGTH(blob_col)` and `POSITION(... IN blob)` are likewise
    #     rejected (errors 3580 / 5764).
    #   * `BYTES(blob_col)` is the canonical Teradata function for the
    #     stored byte length of a BLOB / VARBYTE column. We use it here.
    sql = f"SELECT BYTES({blob_column}) FROM {table} WHERE {id_column} = ?"
    with connection.cursor() as cur:
        cur.execute(sql, [id_value])
        row = cur.fetchone()
    if row is None:
        return None
    return int(row[0])


def load_model(
    connection: teradatasql.TeradataConnection,
    config: ConnectionConfig,
    *,
    model_id: str,
    onnx_path: Path,
    force: bool = False,
) -> int:
    """Insert the ONNX file at ``onnx_path`` as a BLOB under ``model_id``.

    Args:
        connection: An open Teradata connection.
        config: The :class:`ConnectionConfig` whose
            ``models_table_qualified`` is the target table.
        model_id: The id to write into the ``model_id`` VARCHAR column.
            Must be ≤ 30 characters (BYOM table limit).
        onnx_path: Path to the converted ONNX file.
        force: If True and a row with this ``model_id`` already exists,
            DELETE it before inserting. If False (default), raise
            :class:`FileExistsError`.

    Returns:
        The number of bytes written (== ``onnx_path.stat().st_size``).
    """
    _check_id(model_id)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    blob = onnx_path.read_bytes()
    size = len(blob)
    table = config.models_table_qualified

    if _row_count(connection, table, "model_id", model_id) > 0:
        if not force:
            raise FileExistsError(
                f"model_id={model_id!r} already exists in {table}; "
                "pass force=True (or --force on the CLI) to replace it."
            )
        logger.info("force=True: deleting existing row model_id=%s", model_id)
        _delete_row(connection, table, "model_id", model_id)

    logger.info("Inserting model BLOB (%d bytes) as model_id=%s", size, model_id)
    _insert_blob(connection, table, "model_id", "model", model_id, blob)
    return size


def load_tokenizer(
    connection: teradatasql.TeradataConnection,
    config: ConnectionConfig,
    *,
    tokenizer_id: str,
    tokenizer_path: Path,
    force: bool = False,
) -> int:
    """Insert the tokenizer.json file at ``tokenizer_path`` as a BLOB.

    Same semantics as :func:`load_model` but writes to
    ``config.tokenizers_table_qualified`` keyed by ``tokenizer_id``.
    """
    _check_id(tokenizer_id)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json file not found: {tokenizer_path}")
    blob = tokenizer_path.read_bytes()
    size = len(blob)
    table = config.tokenizers_table_qualified

    if _row_count(connection, table, "tokenizer_id", tokenizer_id) > 0:
        if not force:
            raise FileExistsError(
                f"tokenizer_id={tokenizer_id!r} already exists in {table}; "
                "pass force=True (or --force on the CLI) to replace it."
            )
        logger.info("force=True: deleting existing row tokenizer_id=%s", tokenizer_id)
        _delete_row(connection, table, "tokenizer_id", tokenizer_id)

    logger.info("Inserting tokenizer BLOB (%d bytes) as tokenizer_id=%s", size, tokenizer_id)
    _insert_blob(connection, table, "tokenizer_id", "tokenizer", tokenizer_id, blob)
    return size


def verify_model(
    connection: teradatasql.TeradataConnection,
    config: ConnectionConfig,
    *,
    model_id: str,
) -> int | None:
    """Return ``BYTES(model)`` for the row, or ``None`` if absent."""
    return _read_blob_size(connection, config.models_table_qualified, "model_id", "model", model_id)


def verify_tokenizer(
    connection: teradatasql.TeradataConnection,
    config: ConnectionConfig,
    *,
    tokenizer_id: str,
) -> int | None:
    """Return ``BYTES(tokenizer)`` for the row, or ``None`` if absent."""
    return _read_blob_size(
        connection,
        config.tokenizers_table_qualified,
        "tokenizer_id",
        "tokenizer",
        tokenizer_id,
    )


def _check_id(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("id must be a non-empty string")
    if len(value) > 30:
        raise ValueError(f"id {value!r} is {len(value)} chars; BYOM tables use VARCHAR(30)")
