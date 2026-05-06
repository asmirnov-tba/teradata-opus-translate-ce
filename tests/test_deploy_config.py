"""Unit tests for ConnectionConfig env-var resolution."""

from __future__ import annotations

import pytest

from teradata_opus_translate.deploy.config import (
    DEFAULT_DATABASE,
    DEFAULT_HOST,
    DEFAULT_PASSWORD,
    DEFAULT_USER,
    ConnectionConfig,
)


def test_defaults_match_test_vm(monkeypatch: pytest.MonkeyPatch):
    for var in ("TD_HOST", "TD_USER", "TD_PASSWORD", "TD_BYOM_DATABASE"):
        monkeypatch.delenv(var, raising=False)
    cfg = ConnectionConfig.from_env()
    assert cfg.host == DEFAULT_HOST
    assert cfg.user == DEFAULT_USER
    assert cfg.password == DEFAULT_PASSWORD
    assert cfg.database == DEFAULT_DATABASE


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TD_HOST", "td.example.com")
    monkeypatch.setenv("TD_USER", "alice")
    monkeypatch.setenv("TD_PASSWORD", "secret")
    monkeypatch.setenv("TD_BYOM_DATABASE", "TEST_DB")
    cfg = ConnectionConfig.from_env()
    assert cfg.host == "td.example.com"
    assert cfg.user == "alice"
    assert cfg.password == "secret"
    assert cfg.database == "TEST_DB"


def test_explicit_kwargs_override_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TD_HOST", "td.example.com")
    cfg = ConnectionConfig.from_env(host="other.example.com")
    assert cfg.host == "other.example.com"


def test_qualified_table_names():
    cfg = ConnectionConfig(host="h", user="u", password="p", database="DB")
    assert cfg.models_table_qualified == "DB.onnx_models"
    assert cfg.tokenizers_table_qualified == "DB.sequence_tokenizers"
