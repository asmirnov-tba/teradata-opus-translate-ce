"""Deployment helpers for loading converted ONNX models and tokenizers
into the Teradata BYOM model / tokenizer tables.

Per the project's library policy, all Teradata access in this package goes
through the raw ``teradatasql`` DB-API driver — never ``teradataml``.

The package exposes three public entry points:

* :func:`connect`            -- open a ``teradatasql.Connection`` from env vars or args.
* :func:`apply_ddl`          -- (re-)create the database and tables from ``sql/``.
* :func:`load_model`         -- insert (or replace) a model BLOB by ``model_id``.
* :func:`load_tokenizer`     -- insert (or replace) a tokenizer BLOB by ``tokenizer_id``.
* :func:`verify_blob`        -- ``SELECT`` back the loaded row size for sanity.

The :mod:`teradata_opus_translate.deploy.__main__` CLI wraps these into a
single ``python -m teradata_opus_translate.deploy ...`` command.
"""

from __future__ import annotations

from teradata_opus_translate.deploy.config import (
    DEFAULT_DATABASE,
    DEFAULT_MODELS_TABLE,
    DEFAULT_TOKENIZERS_TABLE,
    ConnectionConfig,
)
from teradata_opus_translate.deploy.loader import (
    SQL_DATABASE_PLACEHOLDER,
    apply_ddl,
    connect,
    load_model,
    load_tokenizer,
    substitutions_for_config,
    verify_model,
    verify_tokenizer,
)

__all__ = [
    "DEFAULT_DATABASE",
    "DEFAULT_MODELS_TABLE",
    "DEFAULT_TOKENIZERS_TABLE",
    "SQL_DATABASE_PLACEHOLDER",
    "ConnectionConfig",
    "apply_ddl",
    "connect",
    "load_model",
    "load_tokenizer",
    "substitutions_for_config",
    "verify_model",
    "verify_tokenizer",
]
