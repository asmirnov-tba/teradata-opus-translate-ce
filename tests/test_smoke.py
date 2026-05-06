"""Smoke tests verifying the package is importable and the scaffold is wired up.

These tests intentionally do not require a Teradata connection or any model
artifacts. They exist so that the CI lane has something to run before the
real test suites land in later issues.
"""

from __future__ import annotations

import importlib

import teradata_opus_translate


def test_package_importable() -> None:
    """The top-level package must import cleanly."""
    module = importlib.import_module("teradata_opus_translate")
    assert module is teradata_opus_translate


def test_package_exposes_version() -> None:
    """The package must expose a ``__version__`` string."""
    assert isinstance(teradata_opus_translate.__version__, str)
    assert teradata_opus_translate.__version__
