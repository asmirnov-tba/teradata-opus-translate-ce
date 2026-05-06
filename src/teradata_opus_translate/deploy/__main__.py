"""CLI entry point for deploying ONNX model + tokenizer BLOBs to Teradata.

Examples
--------

Load a single (model, tokenizer) pair into the BYOM tables::

    python -m teradata_opus_translate.deploy \\
        --model-id opus-mt-de-en \\
        --onnx-path /tmp/de-en.onnx \\
        --tokenizer-path /tmp/tokenizer.json

Re-create the database + tables first (drops existing data), then load::

    python -m teradata_opus_translate.deploy \\
        --create-database --apply-tables \\
        --model-id opus-mt-de-en \\
        --onnx-path /tmp/de-en.onnx \\
        --tokenizer-path /tmp/tokenizer.json

Replace a previously loaded model in place::

    python -m teradata_opus_translate.deploy \\
        --model-id opus-mt-de-en \\
        --onnx-path /tmp/de-en.onnx \\
        --tokenizer-path /tmp/tokenizer.json \\
        --force

Apply only the DDL files (no model load)::

    python -m teradata_opus_translate.deploy --create-database --apply-tables

Tear down a previously loaded model id::

    python -m teradata_opus_translate.deploy --teardown --model-id opus-mt-de-en

Connection details default to the project test VM. Override per-run with
``--host`` / ``--user`` / ``--password`` or set ``TD_HOST`` / ``TD_USER``
/ ``TD_PASSWORD`` env vars.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from teradata_opus_translate.deploy.config import ConnectionConfig
from teradata_opus_translate.deploy.loader import (
    apply_ddl,
    connect,
    load_model,
    load_tokenizer,
    substitutions_for_config,
    verify_model,
    verify_tokenizer,
)

logger = logging.getLogger("teradata_opus_translate.deploy")


# Resolve the SQL files relative to the project root so callers don't have
# to pass paths. The ``sql/`` directory sits next to ``src/`` in the repo;
# when installed as a wheel users would set --create-database / --apply-tables
# to false and run their own DDL.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SQL_DIR = _PROJECT_ROOT / "sql"
SQL_CREATE_DATABASE = _SQL_DIR / "01_create_database.sql"
SQL_CREATE_TABLES = _SQL_DIR / "02_create_tables.sql"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m teradata_opus_translate.deploy",
        description=(
            "Load a converted ONNX model and tokenizer.json into the BYOM "
            "tables on a Teradata test instance. Uses teradatasql (raw "
            "DB-API)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----- Connection ------------------------------------------------------
    conn = parser.add_argument_group("connection")
    conn.add_argument("--host", help="Teradata host (env: TD_HOST)")
    conn.add_argument("--user", help="Teradata user (env: TD_USER)")
    conn.add_argument("--password", help="Teradata password (env: TD_PASSWORD)")
    conn.add_argument(
        "--database",
        help=(
            "BYOM database name (env: TD_BYOM_DATABASE). Used both for "
            "table-qualified DML (load / teardown / verify) AND for "
            "substituting ${BYOM_DATABASE} in the DDL files when "
            "--create-database / --apply-tables is passed."
        ),
    )

    # ----- DDL -------------------------------------------------------------
    ddl = parser.add_argument_group("DDL (drop-and-recreate; data-destructive)")
    ddl.add_argument(
        "--create-database",
        action="store_true",
        help=(
            "Drop and re-create the BYOM database before loading. "
            "DESTROYS all rows in the models / tokenizers tables."
        ),
    )
    ddl.add_argument(
        "--apply-tables",
        action="store_true",
        help=(
            "Drop and re-create the models + tokenizers tables. Implied "
            "by --create-database. DESTROYS any previously loaded models."
        ),
    )

    # ----- Load ------------------------------------------------------------
    load = parser.add_argument_group("load")
    load.add_argument(
        "--model-id",
        help=(
            "Identifier under which to store the model + tokenizer. Used "
            "as both model_id and tokenizer_id (must be <= 30 chars)."
        ),
    )
    load.add_argument(
        "--tokenizer-id",
        help=(
            "Override the tokenizer_id (defaults to --model-id). Rarely "
            "needed; the BYOM ONNXSeq2Seq pattern is one tokenizer per model."
        ),
    )
    load.add_argument("--onnx-path", type=Path, help="Path to the .onnx file to load.")
    load.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Path to the tokenizer.json file to load.",
    )
    load.add_argument(
        "--force",
        action="store_true",
        help=(
            "If a row already exists for the given id, DELETE and re-INSERT. "
            "Default behaviour: fail loudly."
        ),
    )

    # ----- Teardown --------------------------------------------------------
    parser.add_argument(
        "--teardown",
        action="store_true",
        help=(
            "Delete the rows for --model-id from the models and tokenizers "
            "tables (does not drop the tables themselves)."
        ),
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the deploy CLI. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = ConnectionConfig.from_env(
        host=args.host,
        user=args.user,
        password=args.password,
        database=args.database,
    )

    # Validate combinations early so we don't connect for nothing.
    do_ddl = args.create_database or args.apply_tables
    do_load = args.onnx_path is not None or args.tokenizer_path is not None
    do_teardown = args.teardown
    if do_load and (args.onnx_path is None or args.tokenizer_path is None):
        print(
            "error: --onnx-path and --tokenizer-path must be supplied together when loading.",
            file=sys.stderr,
        )
        return 2
    if (do_load or do_teardown) and args.model_id is None:
        print("error: --model-id is required when loading or tearing down.", file=sys.stderr)
        return 2
    if not (do_ddl or do_load or do_teardown):
        print(
            "error: nothing to do — pass at least one of --create-database, "
            "--apply-tables, --teardown, or the load triple "
            "(--model-id + --onnx-path + --tokenizer-path).",
            file=sys.stderr,
        )
        return 2

    try:
        connection = connect(config)
    except Exception as exc:
        print(f"error: could not connect to {config.host} as {config.user}: {exc}", file=sys.stderr)
        print(
            "Hint: is the test VM running? Use `bash scripts/td-vm-start.sh` and re-run.",
            file=sys.stderr,
        )
        return 1

    ddl_substitutions = substitutions_for_config(config)

    with connection:
        if args.create_database:
            apply_ddl(connection, [SQL_CREATE_DATABASE], substitutions=ddl_substitutions)
        if args.create_database or args.apply_tables:
            apply_ddl(connection, [SQL_CREATE_TABLES], substitutions=ddl_substitutions)

        tokenizer_id = args.tokenizer_id or args.model_id

        if do_teardown:
            _teardown(connection, config, args.model_id, tokenizer_id)

        if do_load:
            model_size = load_model(
                connection,
                config,
                model_id=args.model_id,
                onnx_path=args.onnx_path,
                force=args.force,
            )
            tokenizer_size = load_tokenizer(
                connection,
                config,
                tokenizer_id=tokenizer_id,
                tokenizer_path=args.tokenizer_path,
                force=args.force,
            )

            verified_model = verify_model(connection, config, model_id=args.model_id)
            verified_tok = verify_tokenizer(connection, config, tokenizer_id=tokenizer_id)
            _print_load_summary(
                config,
                args.model_id,
                tokenizer_id,
                model_size,
                tokenizer_size,
                verified_model,
                verified_tok,
            )
            if verified_model != model_size or verified_tok != tokenizer_size:
                print(
                    "error: BLOB size mismatch after load — server-side BYTES() "
                    "does not match local file size.",
                    file=sys.stderr,
                )
                return 1
    return 0


def _teardown(
    connection,
    config: ConnectionConfig,
    model_id: str,
    tokenizer_id: str,
) -> None:
    with connection.cursor() as cur:
        cur.execute(
            f"DELETE FROM {config.models_table_qualified} WHERE model_id = ?",
            [model_id],
        )
        cur.execute(
            f"DELETE FROM {config.tokenizers_table_qualified} WHERE tokenizer_id = ?",
            [tokenizer_id],
        )
    print(f"Teardown: deleted rows for model_id={model_id!r}, tokenizer_id={tokenizer_id!r}.")


def _print_load_summary(
    config: ConnectionConfig,
    model_id: str,
    tokenizer_id: str,
    model_size: int,
    tokenizer_size: int,
    verified_model: int | None,
    verified_tokenizer: int | None,
) -> None:
    print(f"host:                  {config.host}")
    print(f"database:              {config.database}")
    print(f"models table:          {config.models_table_qualified}")
    print(f"tokenizers table:      {config.tokenizers_table_qualified}")
    print(f"model_id:              {model_id}")
    print(f"  uploaded bytes:      {model_size:,}  ({model_size / (1024 * 1024):.2f} MiB)")
    print(f"  server BYTES():      {verified_model}")
    print(f"tokenizer_id:          {tokenizer_id}")
    print(f"  uploaded bytes:      {tokenizer_size:,}  ({tokenizer_size / (1024 * 1024):.2f} MiB)")
    print(f"  server BYTES():      {verified_tokenizer}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
