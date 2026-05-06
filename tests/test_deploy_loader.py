"""Unit tests for the deploy loader (no live VM dependency).

These tests exercise the pure logic in
:mod:`teradata_opus_translate.deploy.loader` using a tiny fake DB-API
connection / cursor pair. The intent is to cover:

* id validation,
* duplicate detection / force semantics,
* SQL composition (qualified table names, parameter ordering),
* tolerated error handling in apply_ddl.

The live-VM happy path is exercised manually (see PR description) and is
out of scope for the unit suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from teradata_opus_translate.deploy.config import ConnectionConfig
from teradata_opus_translate.deploy.loader import (
    SQL_DATABASE_PLACEHOLDER,
    apply_ddl,
    load_model,
    load_tokenizer,
    substitutions_for_config,
    verify_model,
    verify_tokenizer,
)

# ---------------------------------------------------------------------------
# Fake DB-API plumbing
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, conn: FakeConnection):
        self.conn = conn
        self._result: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        self.conn.calls.append((sql, list(params) if params is not None else None))
        # Simulate the response the loader expects for COUNT(*) / OCTET_LENGTH.
        if self.conn.next_results:
            self._result = self.conn.next_results.pop(0)
        else:
            self._result = []
        # Allow tests to register synthetic exceptions for the next execute.
        if self.conn.next_exception is not None:
            exc = self.conn.next_exception
            self.conn.next_exception = None
            raise exc

    def fetchone(self):
        return self._result[0] if self._result else None


class FakeConnection:
    def __init__(self):
        self.calls: list[tuple[str, list[Any] | None]] = []
        self.next_results: list[list[tuple[Any, ...]]] = []
        self.next_exception: BaseException | None = None
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


@pytest.fixture
def conn() -> FakeConnection:
    return FakeConnection()


@pytest.fixture
def config() -> ConnectionConfig:
    return ConnectionConfig(host="h", user="u", password="p", database="BYOM_USER")


# ---------------------------------------------------------------------------
# load_model / load_tokenizer
# ---------------------------------------------------------------------------


def test_load_model_inserts_blob_and_uses_qualified_table(
    tmp_path: Path, conn: FakeConnection, config: ConnectionConfig
):
    onnx = tmp_path / "m.onnx"
    payload = b"\x00\x01\x02\x03ONNX"
    onnx.write_bytes(payload)
    # Pre-load the COUNT(*) result: 0 rows already present.
    conn.next_results = [[(0,)]]

    size = load_model(conn, config, model_id="opus-mt-de-en", onnx_path=onnx)
    assert size == len(payload)

    # First call: COUNT(*) check; second call: INSERT.
    count_sql, count_params = conn.calls[0]
    insert_sql, insert_params = conn.calls[1]
    assert "BYOM_USER.onnx_models" in count_sql
    assert "WHERE model_id = ?" in count_sql
    assert count_params == ["opus-mt-de-en"]
    assert "INSERT INTO BYOM_USER.onnx_models (model_id, model) VALUES (?, ?)" == insert_sql
    assert insert_params == ["opus-mt-de-en", payload]


def test_load_model_raises_on_existing_without_force(
    tmp_path: Path, conn: FakeConnection, config: ConnectionConfig
):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"abc")
    conn.next_results = [[(1,)]]  # already exists
    with pytest.raises(FileExistsError):
        load_model(conn, config, model_id="x", onnx_path=onnx)


def test_load_model_replaces_with_force(
    tmp_path: Path, conn: FakeConnection, config: ConnectionConfig
):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"abc")
    conn.next_results = [[(1,)]]
    size = load_model(conn, config, model_id="x", onnx_path=onnx, force=True)
    assert size == 3
    sqls = [c[0] for c in conn.calls]
    # COUNT(*), DELETE, INSERT (in that order).
    assert sqls[0].startswith("SELECT COUNT(*)")
    assert sqls[1].startswith("DELETE FROM BYOM_USER.onnx_models")
    assert sqls[2].startswith("INSERT INTO BYOM_USER.onnx_models")


def test_load_model_rejects_long_id(tmp_path: Path, conn: FakeConnection, config: ConnectionConfig):
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    long_id = "x" * 31
    with pytest.raises(ValueError, match="VARCHAR\\(30\\)"):
        load_model(conn, config, model_id=long_id, onnx_path=onnx)


def test_load_model_missing_file(tmp_path: Path, conn: FakeConnection, config: ConnectionConfig):
    with pytest.raises(FileNotFoundError):
        load_model(conn, config, model_id="x", onnx_path=tmp_path / "nope.onnx")


def test_load_tokenizer_uses_tokenizer_table(
    tmp_path: Path, conn: FakeConnection, config: ConnectionConfig
):
    tk = tmp_path / "t.json"
    tk.write_bytes(b'{"a": 1}')
    conn.next_results = [[(0,)]]

    load_tokenizer(conn, config, tokenizer_id="x", tokenizer_path=tk)
    insert_sql, _ = conn.calls[1]
    assert "BYOM_USER.sequence_tokenizers" in insert_sql
    assert "tokenizer_id" in insert_sql
    assert "tokenizer)" in insert_sql


# ---------------------------------------------------------------------------
# verify_*
# ---------------------------------------------------------------------------


def test_verify_model_returns_blob_size(conn: FakeConnection, config: ConnectionConfig):
    conn.next_results = [[(123,)]]
    assert verify_model(conn, config, model_id="x") == 123
    sql = conn.calls[0][0]
    # We must use BYTES() on Teradata BLOB columns; OCTET_LENGTH is rejected
    # with error 3580 on Teradata 20.00.
    assert "BYTES(model)" in sql
    assert "BYOM_USER.onnx_models" in sql


def test_verify_tokenizer_missing_returns_none(conn: FakeConnection, config: ConnectionConfig):
    conn.next_results = [[]]
    assert verify_tokenizer(conn, config, tokenizer_id="x") is None


# ---------------------------------------------------------------------------
# apply_ddl
# ---------------------------------------------------------------------------


def test_apply_ddl_runs_each_statement(tmp_path: Path, conn: FakeConnection):
    sql_file = tmp_path / "x.sql"
    sql_file.write_text("DROP TABLE foo;\nCREATE TABLE foo (id INT);")
    n = apply_ddl(conn, [sql_file])
    assert n == 2
    assert conn.calls[0][0].startswith("DROP TABLE")
    assert conn.calls[1][0].startswith("CREATE TABLE")


def test_apply_ddl_tolerates_3807(tmp_path: Path, conn: FakeConnection):
    sql_file = tmp_path / "x.sql"
    sql_file.write_text("DROP TABLE foo;\nCREATE TABLE foo (id INT);")
    # Make the first execute (the DROP) raise a 3807-style error.
    conn.next_exception = RuntimeError(
        "[Teradata Database] [Error 3807] Object 'foo' does not exist."
    )
    n = apply_ddl(conn, [sql_file])
    # Tolerated DROP doesn't count; the CREATE succeeded.
    assert n == 1


def test_apply_ddl_re_raises_unexpected_error(tmp_path: Path, conn: FakeConnection):
    sql_file = tmp_path / "x.sql"
    sql_file.write_text("CREATE TABLE foo (id INT);")
    conn.next_exception = RuntimeError("[Teradata Database] [Error 3704] kaboom")
    with pytest.raises(RuntimeError):
        apply_ddl(conn, [sql_file])


# ---------------------------------------------------------------------------
# apply_ddl + substitutions
# ---------------------------------------------------------------------------


def test_apply_ddl_substitutes_database_placeholder(tmp_path: Path, conn: FakeConnection):
    """The substituted SQL is what gets sent to the connection."""
    sql_file = tmp_path / "x.sql"
    sql_file.write_text(
        "DROP TABLE ${BYOM_DATABASE}.onnx_models;\n"
        "CREATE TABLE ${BYOM_DATABASE}.onnx_models (id INT);"
    )
    n = apply_ddl(conn, [sql_file], substitutions={"BYOM_DATABASE": "OPUS_BYOM"})
    assert n == 2
    drop_sql = conn.calls[0][0]
    create_sql = conn.calls[1][0]
    assert "${BYOM_DATABASE}" not in drop_sql
    assert "${BYOM_DATABASE}" not in create_sql
    assert "OPUS_BYOM.onnx_models" in drop_sql
    assert "OPUS_BYOM.onnx_models" in create_sql


def test_apply_ddl_default_substitution_uses_byom_user(
    tmp_path: Path, conn: FakeConnection, config: ConnectionConfig
):
    """`substitutions_for_config` on a default config maps to BYOM_USER."""
    sql_file = tmp_path / "x.sql"
    sql_file.write_text("DROP TABLE ${BYOM_DATABASE}.onnx_models;")
    apply_ddl(conn, [sql_file], substitutions=substitutions_for_config(config))
    sent_sql = conn.calls[0][0]
    assert sent_sql == "DROP TABLE BYOM_USER.onnx_models"


def test_apply_ddl_passes_real_ddl_files_with_overridden_database(
    tmp_path: Path, conn: FakeConnection
):
    """End-to-end: feed `apply_ddl` the project's real DDL files with an
    override and check that every statement reaching the connection is
    correctly substituted (and that the default `BYOM_USER` does not
    leak through)."""
    repo_sql = Path(__file__).resolve().parents[1] / "sql"
    files = [repo_sql / "01_create_database.sql", repo_sql / "02_create_tables.sql"]

    custom = ConnectionConfig(host="h", user="u", password="p", database="OPUS_BYOM")
    n = apply_ddl(conn, files, substitutions=substitutions_for_config(custom))

    # 3 statements from 01_create_database.sql (DELETE / DROP / CREATE)
    # + 4 statements from 02_create_tables.sql (2 DROP + 2 CREATE) = 7.
    assert n == 7
    for sent_sql, _ in conn.calls:
        assert "${BYOM_DATABASE}" not in sent_sql
        assert "BYOM_USER" not in sent_sql, (
            f"BYOM_USER leaked through despite override: {sent_sql!r}"
        )
        assert "OPUS_BYOM" in sent_sql


def test_apply_ddl_no_substitutions_passes_placeholder_through(
    tmp_path: Path, conn: FakeConnection
):
    """Calling apply_ddl without a substitutions map is a no-op for
    placeholders — the literal `${BYOM_DATABASE}` is sent to Teradata.

    Real Teradata will reject `${BYOM_DATABASE}.onnx_models` as a
    syntax error, so this is "fail at the server" rather than "fail
    locally". The CLI / scripts thread `substitutions_for_config`
    through so this path is not actually reachable in practice; this
    test pins the parser-level contract (substitutions are opt-in to
    keep simple non-templated SQL strings — used elsewhere in the
    test suite — working unchanged).
    """
    repo_sql = Path(__file__).resolve().parents[1] / "sql"
    files = [repo_sql / "02_create_tables.sql"]
    apply_ddl(conn, files)
    sent = [c[0] for c in conn.calls]
    assert any("${BYOM_DATABASE}" in s for s in sent)


def test_substitutions_for_config_uses_database():
    """Sanity: the helper maps the placeholder to `config.database`."""
    cfg = ConnectionConfig(host="h", user="u", password="p", database="OPUS_BYOM")
    subs = substitutions_for_config(cfg)
    assert subs == {SQL_DATABASE_PLACEHOLDER: "OPUS_BYOM"}
