"""teradata_opus_translate.

Convert Helsinki-NLP OPUS translation models to self-contained ONNX files
(with embedded ``com.microsoft.BeamSearch``) and deploy them to Teradata
Vantage.

Public API
----------

Two top-level callables, both accepting either a HuggingFace model id
or a local path to a downloaded HF repo:

* :func:`convert_model` -- build a single self-contained ONNX file with
  embedded ``com.microsoft.BeamSearch``, ready to load into Teradata via
  ``TD_MLDB.ONNXSeq2Seq``.
* :func:`convert_tokenizer` -- build a single self-contained
  ``tokenizer.json`` file, ready to load into the BYOM tokenizer table.

See :mod:`teradata_opus_translate._converter.api` and
:mod:`teradata_opus_translate._tokenizer.api` for the full parameter
reference.

Library policy
--------------
All production-style Teradata access in this package uses ``teradatasql``
(the raw DB-API driver). ``teradataml`` is permitted only inside the demo
notebook under ``notebooks/``. See ``README.md`` for the full convention.
"""

__version__ = "1.0.0"

from teradata_opus_translate._converter.api import (
    ConvertModelResult,
    ParityResult,
    convert_model,
)
from teradata_opus_translate._tokenizer.api import (
    ConvertTokenizerResult,
    convert_tokenizer,
)

__all__ = [
    "ConvertModelResult",
    "ConvertTokenizerResult",
    "ParityResult",
    "__version__",
    "convert_model",
    "convert_tokenizer",
]
