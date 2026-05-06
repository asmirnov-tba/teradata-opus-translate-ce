"""Private converter internals.

This package is intentionally private (leading underscore). The public
surface lives at the package root: ``teradata_opus_translate.convert_model``.
Importing anything from here directly is unsupported and may break
between versions without notice.
"""

from teradata_opus_translate._converter.api import (
    ConvertModelResult,
    convert_model,
)

__all__ = ["ConvertModelResult", "convert_model"]
