# teradata-opus-translate

Convert Helsinki-NLP **OPUS** (Marian) translation models into self-contained
ONNX files for deployment to Teradata Vantage via Bring Your Own Model (BYOM).

[![PyPI version](https://img.shields.io/pypi/v/teradata-opus-translate.svg?style=flat-square)](https://pypi.org/project/teradata-opus-translate/)
[![Python versions](https://img.shields.io/pypi/pyversions/teradata-opus-translate.svg?style=flat-square)](https://pypi.org/project/teradata-opus-translate/)

---

## What it does

`teradata-opus-translate` turns a HuggingFace OPUS / Marian translation model
into a single self-contained ONNX file with `com.microsoft.BeamSearch`
embedded in the graph, plus a single-file fast `tokenizer.json`. The resulting
artifacts load directly into Teradata BYOM tables and are scored at SQL time
through the `TD_MLDB.ONNXSeq2Seq` table operator.

The package handles the parts of the conversion that aren't obvious from the
HuggingFace optimum / `torch.onnx.export` defaults: stripping generation
features the BYOM beam-search op silently ignores, wiring the SQL-tunable
generation parameters as graph inputs (so customers override them per query
rather than baking them in), keeping the ONNX opset on the BYOM ORT 1.16.3
ceiling, and verifying token-level parity against `MarianMTModel.generate()`
before the file is written.

The intended workflow is: convert once, upload the artifacts to BYOM with
`teradataml.save_byom`, then translate at scale through the database with no
Python hop. The two callable surface (`convert_model`, `convert_tokenizer`)
is deliberately small — everything tunable at scoring time stays out of the
export step.

## Install

```bash
pip install teradata-opus-translate
```

Requires **Python 3.12+**. Key dependencies are pulled in automatically:
`transformers`, `torch`, `onnx`, `onnxruntime`, `tokenizers`,
`sentencepiece`, `sacremoses`, `numpy`.

To upload converted models to Teradata BYOM using the `teradataml` example
below, install `teradataml` separately:

```bash
pip install teradataml
```

## Quickstart

```python
from teradata_opus_translate import convert_model, convert_tokenizer

convert_model(
    "Helsinki-NLP/opus-mt-de-en",
    output_path="model.onnx",
)
convert_tokenizer(
    "Helsinki-NLP/opus-mt-de-en",
    output_path="tokenizer.json",
)
```

This downloads the model from the HuggingFace Hub, exports it to ONNX with
`com.microsoft.BeamSearch` embedded, runs token-parity verification against
`MarianMTModel.generate()`, and writes both files to your working directory.
The two files together are everything BYOM needs.

## API reference

### `convert_model`

```python
def convert_model(
    source: str | os.PathLike[str],
    *,
    precision: Literal["fp32", "int8"] = "fp32",
    output_path: str | os.PathLike[str],
    opset: int = 14,
    verify: bool = True,
    verify_samples: list[str] | None = None,
    no_repeat_ngram_size: int | None = None,
    early_stopping: bool | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
    log_level: str | int | None = None,
) -> ConvertModelResult
```

| name | type | default | description |
|---|---|---|---|
| `source` | `str \| PathLike` | _required_ | HuggingFace repo id (e.g. `"Helsinki-NLP/opus-mt-de-en"`) **or** a local directory containing a downloaded HF repo. An existing directory is auto-detected as local; everything else is treated as an HF id. |
| `precision` | `"fp32" \| "int8"` | `"fp32"` | Precision mode. v1 ships dynamic int8 only; static int8 is deferred. |
| `output_path` | `str \| PathLike` | _required_ | Destination `.onnx` path. Parent directories are created if missing; existing files are overwritten. |
| `opset` | `int` | `14` | ONNX opset for the encoder/decoder subgraphs. `14` matches BYOM 7.x's ORT 1.16.3 ceiling. Increase only if your BYOM version is newer and you've verified compatibility. |
| `verify` | `bool` | `True` | Run full token-parity verification against `MarianMTModel.generate()` after export. Always runs when set, regardless of model size. Failure raises `AssertionError` naming the first divergent sample. |
| `verify_samples` | `list[str] \| None` | `None` | Source-language strings used for verification. `None` picks a sensible default set per inferred source language (covers every language in the curated `Helsinki-NLP/opustranslate` collection). Pass an explicit list for custom pairs or local paths. |
| `no_repeat_ngram_size` | `int \| None` | `None` | Baked into the BeamSearch graph **as a node attribute** at export time. Defaults to `model.config.no_repeat_ngram_size`. Cannot be overridden at SQL-scoring time. |
| `early_stopping` | `bool \| None` | `None` | Baked into the BeamSearch graph **as a node attribute** at export time. Defaults to `model.config.early_stopping`. Cannot be overridden at SQL-scoring time. |
| `cache_dir` | `str \| PathLike \| None` | `None` | Optional HuggingFace cache directory, passed through to `from_pretrained`. |
| `verbose` | `bool` | `False` | If true, configure the package logger at INFO. |
| `log_level` | `str \| int \| None` | `None` | Explicit logging level; takes precedence over `verbose`. |

**Returns** a [`ConvertModelResult`](#result-types).

**Raises:** `NotImplementedError` (`precision="int8"` until that path lands),
`ValueError` (unknown precision), `RuntimeError` (output ONNX > 2 GiB; v1
does not support `external_data`), `AssertionError` (parity divergence when
`verify=True`).

> **Note — SQL-tunable parameters are not export arguments.** The six
> generation parameters customers most often want to tune (`num_beams`,
> `max_length`, `min_length`, `length_penalty`, `repetition_penalty`,
> `num_return_sequences`) are deliberately **not** exposed by `convert_model`.
> They remain as inputs of the produced ONNX graph and are overridden per
> query at SQL time using the BYOM `Const_*` USING clause:
>
> ```sql
> SELECT * FROM TD_MLDB.ONNXSeq2Seq (
>   ON inputs PARTITION BY ANY
>   ON onnx_models AS ModelTable DIMENSION
>   ON sequence_tokenizers AS TokenizerTable DIMENSION
>   USING
>     ModelOutputFields('sequences')
>     Const_num_beams(4)
>     Const_max_length(64)
>     Const_min_length(1)
>     Const_length_penalty(1.0)
>     Const_repetition_penalty(1.0)
>     Const_num_return_sequences(1)
> ) AS dt;
> ```
>
> This keeps a single exported artifact tunable across many SQL workloads
> without re-export. `no_repeat_ngram_size` and `early_stopping` are the
> exception — the BeamSearch contrib op only accepts them as node attributes,
> so they're baked in at export.

### `convert_tokenizer`

```python
def convert_tokenizer(
    source: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> ConvertTokenizerResult
```

| name | type | default | description |
|---|---|---|---|
| `source` | `str \| PathLike` | _required_ | HF repo id or local directory; same auto-detection rule as `convert_model`. |
| `output_path` | `str \| PathLike` | _required_ | Destination `tokenizer.json` path. Parent directories are created if missing; existing files are overwritten. |
| `cache_dir` | `str \| PathLike \| None` | `None` | Optional HuggingFace cache directory, passed through to `from_pretrained`. |
| `verbose` | `bool` | `False` | If true, configure the package logger at INFO. |

**Returns** a [`ConvertTokenizerResult`](#result-types).

**Raises** `RuntimeError` if the in-process round-trip self-check fails
(the writer refuses to produce a `tokenizer.json` that disagrees with
`MarianTokenizer` on a canary sentence).

The output is a single-file `tokenizer.json` that loads via
`tokenizers.Tokenizer.from_file(...)` with no external dependencies. Marian's
`MarianTokenizer` is a slow Python tokenizer wrapping `source.spm` /
`target.spm` / `vocab.json`; this function rebuilds an equivalent fast
tokenizer (Unigram + Metaspace + EOS template) directly from the source-side
SentencePiece scores.

### Result types

All result types are frozen dataclasses.

**`ConvertModelResult`** — returned by `convert_model`.

| field | type | description |
|---|---|---|
| `output_path` | `Path` | Resolved absolute path of the written `.onnx` file. |
| `size_bytes` | `int` | Size of the written ONNX file, in bytes. |
| `source` | `str` | Original `source` argument as resolved for `from_pretrained`. |
| `source_kind` | `"hf" \| "local"` | `"local"` if `source` was a local directory, otherwise `"hf"`. |
| `precision` | `"fp32" \| "int8"` | Precision mode used for the export. |
| `parity` | `ParityResult \| None` | Token-parity result; `None` when `verify=False`. |

**`ConvertTokenizerResult`** — returned by `convert_tokenizer`.

| field | type | description |
|---|---|---|
| `output_path` | `Path` | Resolved absolute path of the written `tokenizer.json`. |
| `size_bytes` | `int` | Size of the written file, in bytes. |
| `source` | `str` | Original `source` argument as resolved for `from_pretrained`. |
| `source_kind` | `"hf" \| "local"` | `"local"` or `"hf"`. |

**`ParityResult`** — produced by the verification step inside `convert_model`.

| field | type | description |
|---|---|---|
| `samples` | `list[str]` | Source-language strings used for verification. |
| `hf_token_ids` | `list[list[int]]` | Per-sample token-id sequence from `MarianMTModel.generate()` (after canonicalising trailing EOS / pad). |
| `onnx_token_ids` | `list[list[int]]` | Per-sample token-id sequence from the exported ONNX BeamSearch graph (same canonicalisation). |
| `mismatches` | `int` | Count of samples whose two id-sequences differ. `0` means full token-parity. |

## End-to-end BYOM example

For a runnable walkthrough of the flow below, see the [demo notebook](https://github.com/asmirnov-tba/teradata-opus-translate-ce/blob/main/notebooks/opus_de_en_demo.ipynb).

The full customer workflow is **convert → upload → score**.

### 1. Convert the model and tokenizer locally

```python
from teradata_opus_translate import convert_model, convert_tokenizer

convert_model(
    "Helsinki-NLP/opus-mt-de-en",
    output_path="opus-mt-de-en.onnx",
)
convert_tokenizer(
    "Helsinki-NLP/opus-mt-de-en",
    output_path="opus-mt-de-en.tokenizer.json",
)
```

### 2. Upload to Teradata BYOM with `save_byom`

```python
from teradataml import create_context, save_byom

create_context(host="...", username="...", password="...")

# Model -> onnx_models(model_id, model)
save_byom(
    model_id="opus-mt-de-en",
    model_file="opus-mt-de-en.onnx",
    table_name="onnx_models",
    schema_name="OPUS_BYOM",
)

# Tokenizer goes into a separate table. save_byom always writes a column
# called "model" regardless of what the artifact actually is, so we save the
# tokenizer.json into a (model_id, model)-shaped table and alias the column
# back to "tokenizer" on SELECT in the SQL below.
save_byom(
    model_id="opus-mt-de-en",
    model_file="opus-mt-de-en.tokenizer.json",
    table_name="sequence_tokenizers",
    schema_name="OPUS_BYOM",
)
```

### 3. Score with `TD_MLDB.ONNXSeq2Seq`

```sql
CREATE MULTISET TABLE OPUS_BYOM.de_en_inputs (
    id INTEGER,
    input_text VARCHAR(2000) CHARACTER SET UNICODE
) PRIMARY INDEX (id);

INSERT INTO OPUS_BYOM.de_en_inputs VALUES (1, 'Hallo Welt.');
INSERT INTO OPUS_BYOM.de_en_inputs VALUES (2, 'Das Wetter ist heute schön.');

SELECT id, input_text, output_text
FROM TD_MLDB.ONNXSeq2Seq (
    ON OPUS_BYOM.de_en_inputs PARTITION BY ANY
    ON (SELECT model_id, model FROM OPUS_BYOM.onnx_models
        WHERE model_id = 'opus-mt-de-en') AS ModelTable DIMENSION
    ON (SELECT model_id, model AS tokenizer FROM OPUS_BYOM.sequence_tokenizers
        WHERE model_id = 'opus-mt-de-en') AS TokenizerTable DIMENSION
    USING
        Accumulate('id', 'input_text')
        TextColumn('input_text')
        ModelOutputFields('sequences')
        OutputFormat('FLOAT(1)')
        EnableMemoryCheck('false')
        OverwriteCachedModel('*')
        Const_num_beams(4)
        Const_max_length(64)
        Const_min_length(1)
        Const_length_penalty(1.0)
        Const_repetition_penalty(1.0)
        Const_num_return_sequences(1)
) AS dt;
```

Adjust the `Const_*` cluster per workload — those are exactly the SQL-time
overrides for the parameters `convert_model` deliberately keeps off the
export API.

## Supported models

`teradata-opus-translate` v1 targets the **MarianMT / OPUS** family. It is
tested against `Helsinki-NLP/opus-mt-*` models and the curated
`Helsinki-NLP/opustranslate` collection. Other seq2seq architectures (T5,
BART, NLLB, mBART) are **not supported in v1**.

The default verification samples cover every source language in the curated
`opustranslate` collection plus every ISO 639-1 source code that appears in
five or more `Helsinki-NLP/opus-mt-*` repos, so most production language
pairs land on language-appropriate samples without you needing to pass
`verify_samples=` explicitly.

## Verification (`verify=True`)

When `verify=True` (the default), `convert_model` runs full token-parity
verification immediately after the ONNX file is written:

1. For each sample in `verify_samples`, run `MarianMTModel.generate()` with
   the BeamSearch-compatible parameter intersection (`bad_words_ids=None`,
   `forced_eos_token_id=None`, `renormalize_logits=False`, `do_sample=False`).
2. Run the same sample through the exported ONNX graph in `onnxruntime`.
3. Canonicalise the trailing-EOS / pad-to-max difference (HF emits a trailing
   EOS, the BeamSearch op omits it and pads with `pad_token_id`).
4. Compare token-ID sequences exactly.

Any divergence raises `AssertionError` naming the first divergent sample,
the HF and ONNX id sequences side by side, and the position of the first
differing token. This catches export-time silent regressions before the
artifact ever lands in BYOM.

Pass `verify_samples=[...]` to use your own canary inputs (recommended for
local-path sources where the language can't be inferred from a HuggingFace
id). The default sample set covers every language in the
`Helsinki-NLP/opustranslate` collection.

## Limitations

- **2 GiB ONNX size ceiling.** ONNX uses protobuf for serialization, which
  caps a single message at 2 GiB. Larger models error at export time with a
  clear `RuntimeError`. v1 does not support the `external_data` workaround;
  the path forward for larger pairs is `precision="int8"` (in flight).
- **int8 quantization is dynamic only in v1.** Static (calibration-based)
  quantization is deferred.
- **The BeamSearch contrib op silently ignores three generation features**
  that `MarianMTModel.generate()` accepts: `bad_words_ids`,
  `forced_eos_token_id`, and `renormalize_logits`. The internal verification
  step disables all three on the HF side. If you compare the exported ONNX
  against `MarianMTModel.generate()` outside this package, disable these
  features explicitly or you'll see false-positive divergences.
- **Default opset is 14** because BYOM 7.x ships ORT 1.16.3, which is the
  ceiling for the BeamSearch contrib op signature this package targets.
  Don't change `opset=` unless you know your BYOM version ships a newer ORT
  and you've verified the `T5EncoderSubgraph::Validate()` check still
  accepts our 3-input encoder layout.
- **MarianMT / OPUS only.** No T5, BART, NLLB, or mBART support in v1.

## Acknowledgements

- Microsoft for the `com.microsoft.BeamSearch` contrib op pattern, which
  makes single-graph beam-search inside `onnxruntime` possible.
- The Helsinki-NLP group for the OPUS-MT model family and the curated
  `opustranslate` HuggingFace collection.
