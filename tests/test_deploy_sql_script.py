"""Unit tests for the SQL-file parser used by the deploy module."""

from __future__ import annotations

from pathlib import Path

import pytest

from teradata_opus_translate.deploy.sql_script import (
    iter_file_statements,
    split_statements,
)

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"

# The placeholder name used by the project DDL files. Kept in lock-step
# with `teradata_opus_translate.deploy.loader.SQL_DATABASE_PLACEHOLDER`,
# but defined locally so this test file does not depend on that detail.
_DB_PLACEHOLDER_KEY = "BYOM_DATABASE"
_DEFAULT_SUBS = {_DB_PLACEHOLDER_KEY: "BYOM_USER"}


def test_split_strips_block_and_line_comments():
    text = """
    /* big block
       comment */
    DROP TABLE foo; -- trailing comment
    -- standalone line
    CREATE TABLE foo (id INTEGER);
    """
    stmts = split_statements(text)
    assert stmts == ["DROP TABLE foo", "CREATE TABLE foo (id INTEGER)"]


def test_split_drops_empty_statements():
    text = ";;\n;;DROP DATABASE x;;\n;"
    assert split_statements(text) == ["DROP DATABASE x"]


def test_split_preserves_inline_whitespace():
    text = "CREATE TABLE foo (\n    id INTEGER,\n    val VARCHAR(30)\n);"
    stmts = split_statements(text)
    assert len(stmts) == 1
    assert stmts[0].startswith("CREATE TABLE foo")
    assert "VARCHAR(30)" in stmts[0]


def test_split_substitutes_placeholder_when_provided():
    text = "DROP TABLE ${BYOM_DATABASE}.t;\nCREATE TABLE ${BYOM_DATABASE}.t (id INT);"
    stmts = split_statements(text, {"BYOM_DATABASE": "OPUS_BYOM"})
    assert stmts == [
        "DROP TABLE OPUS_BYOM.t",
        "CREATE TABLE OPUS_BYOM.t (id INT)",
    ]


def test_split_without_substitutions_leaves_placeholder_alone():
    """A None / empty mapping is a no-op — the SQL is returned verbatim.

    This matches the existing behaviour the parser had before
    parameterisation, so SQL strings without placeholders keep working.
    """
    text = "DROP TABLE foo;"
    assert split_statements(text) == ["DROP TABLE foo"]
    assert split_statements(text, None) == ["DROP TABLE foo"]
    assert split_statements(text, {}) == ["DROP TABLE foo"]


def test_split_unknown_placeholder_raises_keyerror():
    """An unknown placeholder is a hard error — silent leak would be worse."""
    text = "DROP TABLE ${UNKNOWN}.t;"
    with pytest.raises(KeyError):
        split_statements(text, {"BYOM_DATABASE": "OPUS_BYOM"})


def test_split_substitution_skips_placeholders_inside_comments():
    """Comments are stripped first, so comment-only placeholders never run."""
    text = (
        "/* example: ${SOME_PLACEHOLDER} (does not need to resolve) */\n"
        "DROP TABLE ${BYOM_DATABASE}.t;"
    )
    stmts = split_statements(text, _DEFAULT_SUBS)
    assert stmts == ["DROP TABLE BYOM_USER.t"]


def test_real_create_tables_file_parses_into_four_statements():
    """The real DDL file should yield 2 DROPs + 2 CREATEs after substitution."""
    path = SQL_DIR / "02_create_tables.sql"
    stmts = list(iter_file_statements(path, _DEFAULT_SUBS))
    drops = [s for s in stmts if s.upper().startswith("DROP TABLE")]
    creates = [s for s in stmts if s.upper().startswith("CREATE SET TABLE")]
    assert len(drops) == 2
    assert len(creates) == 2
    # Substitution actually happened — the placeholder is gone, the
    # default name is present.
    for s in stmts:
        assert "${BYOM_DATABASE}" not in s
        assert "BYOM_USER" in s


def test_real_create_tables_file_honours_override():
    """An explicit substitution drives the qualified table names."""
    path = SQL_DIR / "02_create_tables.sql"
    stmts = list(iter_file_statements(path, {"BYOM_DATABASE": "OPUS_BYOM"}))
    for s in stmts:
        assert "BYOM_USER" not in s
        assert "OPUS_BYOM" in s
    # Specific lines we care about.
    joined = "\n".join(stmts)
    assert "DROP TABLE OPUS_BYOM.onnx_models" in joined
    assert "DROP TABLE OPUS_BYOM.sequence_tokenizers" in joined
    assert "OPUS_BYOM.onnx_models" in joined
    assert "OPUS_BYOM.sequence_tokenizers" in joined


def test_real_create_database_file_parses():
    path = SQL_DIR / "01_create_database.sql"
    stmts = list(iter_file_statements(path, _DEFAULT_SUBS))
    # Order matters: DELETE must precede DROP, otherwise Teradata 3552
    # ("Cannot DROP databases with tables...") fires on a populated database.
    upper = [s.upper() for s in stmts]
    delete_idx = next(i for i, s in enumerate(upper) if s.startswith("DELETE DATABASE"))
    drop_idx = next(i for i, s in enumerate(upper) if s.startswith("DROP DATABASE"))
    create_idx = next(i for i, s in enumerate(upper) if s.startswith("CREATE DATABASE"))
    assert delete_idx < drop_idx < create_idx
    # All three statements end up referring to the substituted name.
    for s in stmts:
        assert "${BYOM_DATABASE}" not in s
    assert any("BYOM_USER" in s for s in stmts)


def test_real_create_database_file_honours_override():
    path = SQL_DIR / "01_create_database.sql"
    stmts = list(iter_file_statements(path, {"BYOM_DATABASE": "OPUS_BYOM"}))
    joined = "\n".join(stmts)
    assert "BYOM_USER" not in joined
    assert "DELETE DATABASE OPUS_BYOM ALL" in joined
    assert "DROP DATABASE OPUS_BYOM" in joined
    assert "CREATE DATABASE OPUS_BYOM" in joined


def test_real_ddl_files_have_no_literal_byom_user():
    """Acceptance-criterion guard: the executable DDL must not hardcode BYOM_USER.

    Comments and other contextual prose are stripped by the parser, so
    re-checking after parsing (without substitutions) is the cleanest
    way to assert "no literal BYOM_USER survives in any executable
    statement". We can't actually call the parser without
    substitutions because the placeholder would leak through; instead
    we read the raw files, strip comments via the parser's own helpers,
    and check the resulting body before substitution.
    """
    for name in ("01_create_database.sql", "02_create_tables.sql"):
        raw = (SQL_DIR / name).read_text(encoding="utf-8")
        # Use split_statements with a sentinel substitution to strip
        # comments while still being able to inspect the placeholder.
        stmts = split_statements(raw, {"BYOM_DATABASE": "${BYOM_DATABASE}"})
        for s in stmts:
            assert "BYOM_USER" not in s, (
                f"{name}: executable statement still contains literal BYOM_USER: {s!r}"
            )
