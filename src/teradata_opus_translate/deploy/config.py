"""Configuration helpers for the deploy package.

Connection details are read from environment variables (preferred) or
passed explicitly via :class:`ConnectionConfig`. Hard-coded defaults
match the project's local test instance documented in the README; do
**not** reuse those defaults against any production system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Defaults that match the project test VM (see README "Test instance").
# These are deliberately weak credentials for an isolated VM.
# ---------------------------------------------------------------------------

DEFAULT_HOST: str = "192.168.178.62"
DEFAULT_USER: str = "dbc"
DEFAULT_PASSWORD: str = "dbc"

DEFAULT_DATABASE: str = "BYOM_USER"
DEFAULT_MODELS_TABLE: str = "onnx_models"
DEFAULT_TOKENIZERS_TABLE: str = "sequence_tokenizers"

# Environment variable names. Documented in the README "Deployment" section.
ENV_HOST = "TD_HOST"
ENV_USER = "TD_USER"
ENV_PASSWORD = "TD_PASSWORD"
ENV_DATABASE = "TD_BYOM_DATABASE"


@dataclass(frozen=True)
class ConnectionConfig:
    """All inputs required to talk to the Teradata test instance.

    Use :meth:`from_env` to populate from environment variables with
    project defaults, or construct directly when you have explicit values.
    """

    host: str
    user: str
    password: str
    database: str = DEFAULT_DATABASE
    models_table: str = DEFAULT_MODELS_TABLE
    tokenizers_table: str = DEFAULT_TOKENIZERS_TABLE

    @classmethod
    def from_env(
        cls,
        *,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> ConnectionConfig:
        """Build a :class:`ConnectionConfig` from env vars / overrides.

        Resolution order per field: explicit kwarg > env var > project default.
        """
        return cls(
            host=host or os.environ.get(ENV_HOST, DEFAULT_HOST),
            user=user or os.environ.get(ENV_USER, DEFAULT_USER),
            password=password or os.environ.get(ENV_PASSWORD, DEFAULT_PASSWORD),
            database=database or os.environ.get(ENV_DATABASE, DEFAULT_DATABASE),
        )

    @property
    def models_table_qualified(self) -> str:
        """``database.models_table`` — safe to interpolate into SQL DDL/DML."""
        return f"{self.database}.{self.models_table}"

    @property
    def tokenizers_table_qualified(self) -> str:
        """``database.tokenizers_table`` — safe to interpolate into SQL DDL/DML."""
        return f"{self.database}.{self.tokenizers_table}"
