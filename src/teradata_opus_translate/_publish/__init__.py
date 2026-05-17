"""Internal helpers for publishing converted ONNX models to Hugging Face.

This package is *not* part of the public ``teradata-opus-translate`` API. It
backs ``scripts/publish_to_huggingface.py`` and is kept inside the source
tree so it is importable from unit tests without any ``sys.path`` munging.

Nothing here is documented or stable: the CLI is the contract.
"""
