---
language:
  - es
  - eu
license: apache-2.0
library_name: transformers
pipeline_tag: translation
tags:
  - translation
  - LiteRT
  - safetensors
  - Helsinki-NLP/tatoeba
  - openlanguagedata/flores_plus
  - marian
  - onnx
  - teradata
base_model: Helsinki-NLP/opus-mt_tiny_spa-eus
---

> ⚠️ **See [Disclaimer](#disclaimer) below before using.**

# opus-mt_tiny_spa-eus

## A Teradata-compatible Translation Model

A sequence-to-sequence translation model that translates text from
**Spanish 🇪🇸** to
**Basque**.
This repository hosts an ONNX-converted version of
[Helsinki-NLP/opus-mt_tiny_spa-eus](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_spa-eus),
packaged for use with the Teradata `mldb.ONNXSeq2Seq` BYOM function.

**This repository does not redistribute the original model weights.** It contains only:

- `onnx/model-fp32.onnx` — full-precision ONNX graph
- `tokenizer.json` — repacked Marian tokenizer suitable for BYOM
- `config.json` — model architecture metadata, copied unchanged from the upstream repo
- `generation_config.json` — generation defaults, copied unchanged from the upstream repo

A weight-only int8-quantized variant is also published as `onnx/model-int8.onnx`. Use `model-fp32.onnx` unless deployment size is a constraint.

For the original PyTorch weights and training details, see the upstream model:
**[Helsinki-NLP/opus-mt_tiny_spa-eus](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_spa-eus)**.

**Specifications**

| | |
|---|---|
| Source language | Spanish 🇪🇸 (`es`) |
| Target language | Basque (`eu`) |
| Architecture | MarianMT (encoder-decoder) |
| Max input tokens | 512 |
| Max output tokens | 512 |
| ONNX file size | 177 MB (fp32) / 94 MB (int8) |
| ONNX opset | 14 |
| ONNX IR version | 8 (BYOM 7.0+ compatible) |
| License | Apache-2.0 (from upstream) |
| Reference | https://huggingface.co/Helsinki-NLP/opus-mt_tiny_spa-eus |

Generation parameters are configurable at SQL time via the
`mldb.ONNXSeq2Seq` USING clause through `Const_*` keys: `Const_min_length`,
`Const_max_length`, `Const_num_beams`, `Const_length_penalty`,
`Const_repetition_penalty`. They are not fixed in the ONNX graph.
(`num_return_sequences` is the exception — it's baked into the graph as 1.)

## Quickstart: Deploying this Model in Teradata

Requires Teradata 17.20+ with **BYOM 7.0.0.4** or newer (the conversion
targets ONNX IR version 8, which BYOM 7.0.x requires).

> **Note on schema name:** the SQL example below uses `mldb.ONNXSeq2Seq`.
> On modern Teradata deployments BYOM is installed in the `td_mldb` database —
> adjust the schema prefix in the SQL accordingly.

```python
import getpass
import teradataml as tdml
from huggingface_hub import hf_hub_download

repo_id  = "sashamartin/opus-mt_tiny_spa-eus"
model_id = "opus-mt_tiny_spa-eus"          # used as BYOM model_id

# 1. Download artifacts from this repo
hf_hub_download(repo_id=repo_id, filename="onnx/model-fp32.onnx", local_dir="./")
hf_hub_download(repo_id=repo_id, filename="tokenizer.json",       local_dir="./")

# 2. Connect to Teradata
tdml.create_context(
    host=input("host: "),
    username=input("user: "),
    password=getpass.getpass("password: "),
)

# 3. Load model + tokenizer into BYOM tables
tdml.save_byom(model_id=model_id, model_file="onnx/model-fp32.onnx",
               table_name="translation_models")
tdml.save_byom(model_id=model_id, model_file="tokenizer.json",
               table_name="translation_tokenizers")

# 4. Translate
query = f"""
SELECT id, sequences
FROM mldb.ONNXSeq2Seq(
    ON (SELECT id, txt FROM your_input_table) AS InputTable
    ON (SELECT model_id, model FROM translation_models
         WHERE model_id = '{model_id}') AS ModelTable DIMENSION
    ON (SELECT model AS tokenizer FROM translation_tokenizers
         WHERE model_id = '{model_id}') AS TokenizerTable DIMENSION
    USING
        Accumulate('id')
        ModelOutputTensor('sequences')
        SkipSpecialTokens('true')
        OutputLength(512)
        OverwriteCachedModel('true')
        Const_min_length(1)
        Const_max_length(64)
        Const_num_beams(4)
        Const_length_penalty(1.0)
        Const_repetition_penalty(1.0)
) AS t
"""
print(tdml.DataFrame.from_query(query))
```

## How this model was converted

This model was produced with the open-source
[`teradata-opus-translate`](https://pypi.org/project/teradata-opus-translate/)
package, which exports the encoder/decoder, stitches in the BeamSearch op,
applies weight-only int8 quantization, and verifies parity against PyTorch on a
small sample set.

> **Note:** the same package can convert *any* Helsinki-NLP MarianMT model
> (including ones not in this collection) to a BYOM-ready ONNX bundle. If
> you have a translation pair that's not published here, install the package
> and run:
>
> ```python
> from teradata_opus_translate import convert_model, convert_tokenizer
>
> convert_model(
>     "Helsinki-NLP/<your-model>",
>     output_path="model-fp32.onnx",
> )
> convert_tokenizer(
>     "Helsinki-NLP/<your-model>",
>     output_path="tokenizer.json",
> )
> ```
>
> The resulting `model-fp32.onnx` and `tokenizer.json` are ready to deploy
> with the Quickstart flow above.

## Disclaimer

DISCLAIMER: The content herein ("Content") is provided "AS IS" and is not covered by any Teradata Operations, Inc. and its affiliates ("Teradata") agreements. Its listing here does not constitute certification or endorsement by Teradata.

To the extent any of the Content contains or is related to any artificial intelligence ("AI") or other language learning models ("Models") that interoperate with the products and services of Teradata, by accessing, bringing, deploying or using such Models, you acknowledge and agree that you are solely responsible for ensuring compliance with all applicable laws, regulations, and restrictions governing the use, deployment, and distribution of AI technologies. This includes, but is not limited to, AI Diffusion Rules, European Union AI Act, AI-related laws and regulations, privacy laws, export controls, and financial or sector-specific regulations.

While Teradata may provide support, guidance, or assistance in the deployment or implementation of Models to interoperate with Teradata's products and/or services, you remain fully responsible for ensuring that your Models, data, and applications comply with all relevant legal and regulatory obligations. Our assistance does not constitute legal or regulatory approval, and Teradata disclaims any liability arising from non-compliance with applicable laws.

You must determine the suitability of the Models for any purpose. Given the probabilistic nature of machine learning and modeling, the use of the Models may in some situations result in incorrect output that does not accurately reflect the action generated. You should evaluate the accuracy of any output as appropriate for your use case, including by using human review of the output.
