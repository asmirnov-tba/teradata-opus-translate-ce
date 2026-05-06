"""Private tokenizer internals.

This package is intentionally private (leading underscore). The public
surface lives at the package root: ``teradata_opus_translate.convert_tokenizer``.
Importing anything from here directly is unsupported and may break
between versions without notice.
"""

from teradata_opus_translate._tokenizer.api import (
    ConvertTokenizerResult,
    convert_tokenizer,
)

__all__ = ["ConvertTokenizerResult", "convert_tokenizer"]
