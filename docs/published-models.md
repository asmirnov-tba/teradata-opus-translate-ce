# Published OPUS-MT models

This catalog lists the 25 Marian translation models from the
[`Helsinki-NLP/opustranslate`](https://huggingface.co/collections/Helsinki-NLP/opustranslate)
collection that the `teradata-opus-translate` project has converted to
self-contained ONNX (with `com.microsoft.BeamSearch` embedded) and
published to S3 for direct loading into Teradata BYOM. Every row links
to the HuggingFace source, the fp32 / int8 ONNX artifacts on S3, and
the matching `tokenizer.json`.

Each model has been smoke-tested on a real Teradata installation
through `TD_MLDB.ONNXSeq2Seq` -- the `Smoke` column shows whether the
fp32 graph produced clean output on a representative input. Sample
input/output pairs per model live in
[`data/smoke_results.json`](../data/smoke_results.json).

## Quick start

Each model in the catalog ships three files in S3: `model-fp32.onnx`,
`model-int8.onnx` (optional), and `tokenizer.json`. Download the pair
you need and load them into Teradata via BYOM. The example below uses
the German-to-English tiny model.

```python
import urllib.request

# Download fp32 ONNX + tokenizer for opus-mt_tiny_deu-eng
urllib.request.urlretrieve(
    "https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_deu-eng/model-fp32.onnx",
    "model.onnx",
)
urllib.request.urlretrieve(
    "https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_deu-eng/tokenizer.json",
    "tokenizer.json",
)

# Deploy via teradataml (in a notebook or interactive session)
from teradataml import save_byom

save_byom(
    model_id="opus-mt_tiny_deu-eng",
    model_file="model.onnx",
    table_name="onnx_models",
    schema_name="<your-byom-database>",
)
```

See `notebooks/opus_de_en_demo.ipynb` for a full end-to-end example
(BYOM upload, tokenizer registration, and a `TD_MLDB.ONNXSeq2Seq`
translation query).

## Models

| Model | Languages | Params | Context | Max output | fp32 size | int8 size | fp32 | int8 | Tokenizer | Smoke |
|---|---|---|---|---|---|---|---|---|---|---|
| [`opus-mt_tiny_ara-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_ara-eng) | ara → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_ara-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_ara-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_ara-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_cat-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_cat-eng) | cat → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_cat-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_cat-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_cat-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_cat-spa`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_cat-spa) | cat → spa | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_cat-spa/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_cat-spa/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_cat-spa/tokenizer.json) | ✅ |
| [`opus-mt_tiny_deu-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_deu-eng) | deu → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_deu-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_deu-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_deu-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_ell-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_ell-eng) | ell → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_ell-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_ell-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_ell-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_eng-cat`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_eng-cat) | eng → cat | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-cat/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-cat/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-cat/tokenizer.json) | ✅ |
| [`opus-mt_tiny_eng-deu`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_eng-deu) | eng → deu | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-deu/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-deu/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-deu/tokenizer.json) | ✅ |
| [`opus-mt_tiny_eng-ell`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_eng-ell) | eng → ell | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-ell/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-ell/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-ell/tokenizer.json) | ✅ |
| [`opus-mt_tiny_eng-fra`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_eng-fra) | eng → fra | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-fra/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-fra/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-fra/tokenizer.json) | ✅ |
| [`opus-mt_tiny_eng-ita`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_eng-ita) | eng → ita | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-ita/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-ita/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-ita/tokenizer.json) | ✅ |
| [`opus-mt_tiny_eng-nld`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_eng-nld) | eng → nld | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-nld/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-nld/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-nld/tokenizer.json) | ✅ |
| [`opus-mt_tiny_eng-rus`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_eng-rus) | eng → rus | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-rus/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-rus/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-rus/tokenizer.json) | ✅ |
| [`opus-mt_tiny_eng-spa`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_eng-spa) | eng → spa | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-spa/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-spa/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-spa/tokenizer.json) | ✅ |
| [`opus-mt_tiny_eng-tur`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_eng-tur) | eng → tur | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-tur/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-tur/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_eng-tur/tokenizer.json) | ✅ |
| [`opus-mt_tiny_fra-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_fra-eng) | fra → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_fra-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_fra-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_fra-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_ita-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_ita-eng) | ita → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_ita-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_ita-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_ita-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_kor-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_kor-eng) | kor → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_kor-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_kor-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_kor-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_nld-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_nld-eng) | nld → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_nld-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_nld-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_nld-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_rus-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_rus-eng) | rus → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_rus-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_rus-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_rus-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_spa-cat`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_spa-cat) | spa → cat | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-cat/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-cat/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-cat/tokenizer.json) | ✅ |
| [`opus-mt_tiny_spa-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_spa-eng) | spa → eng | 25M | 256 | 256 | 169 MiB | n/a | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-eng/model-fp32.onnx) | n/a | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_spa-eus`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_spa-eus) | spa → eus | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-eus/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-eus/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-eus/tokenizer.json) | ✅ |
| [`opus-mt_tiny_spa-glg`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_spa-glg) | spa → glg | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-glg/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-glg/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_spa-glg/tokenizer.json) | ✅ |
| [`opus-mt_tiny_tur-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_tur-eng) | tur → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_tur-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_tur-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_tur-eng/tokenizer.json) | ✅ |
| [`opus-mt_tiny_zho-eng`](https://huggingface.co/Helsinki-NLP/opus-mt_tiny_zho-eng) | zho → eng | 25M | 256 | 256 | 169 MiB | 90 MiB | [fp32](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_zho-eng/model-fp32.onnx) | [int8](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_zho-eng/model-int8.onnx) | [tokenizer](https://teradata-opus-translate-ce.s3.us-east-1.amazonaws.com/opus-translate/14/Helsinki-NLP/opus-mt_tiny_zho-eng/tokenizer.json) | ✅ |

## Notes

- All models are MarianMT *tiny* variants from the
  [Helsinki-NLP/opustranslate](https://huggingface.co/collections/Helsinki-NLP/opustranslate)
  collection.
- ONNX produced with **opset 14** for BYOM 7.x ORT 1.16.3 compatibility.
- `int8` is dynamic quantization. Per Decision 11 (best-effort), if a
  model's int8 variant fails the parity tolerance it is omitted from
  S3 and the column shows `n/a`; the fp32 artifact is still publishable.
- `Context` is the model's `max_position_embeddings` (max input tokens
  enforced by the encoder).
- `Max output` is the BeamSearch graph default (`max_length` 256) baked
  in at export time. Override per query at SQL time via
  `Const_max_length(N)`, bounded by the model's training-time output
  limit.
- Smoke status: ✅ = passes BYOM `ONNXSeq2Seq`
  scoring on a representative input. ⚠️ = quarantined (e.g. quantization
  failed, decode regression); the row is kept for transparency and the
  fp32 artifact may still be publishable.

## Versioning

Artifacts are published under `s3://teradata-opus-translate-ce/opus-translate/<opset>/<model-id>/...` (region `us-east-1`). Currently:

- **Opset:** 14
- **Package version that produced these artifacts:** 1.0.2
- **Upload date:** 2026-05-07

The full manifest (sizes, exact URLs, upload timestamp) is checked
into [`data/s3_manifest.json`](../data/s3_manifest.json); the
underlying catalog enumeration is in
[`data/catalog.json`](../data/catalog.json).
