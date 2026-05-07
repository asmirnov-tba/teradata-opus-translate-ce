# Notebooks

This directory holds the customer-facing demo notebooks for the
Teradata BYOM translation pipeline.

- [`opus_de_en_demo.ipynb`](opus_de_en_demo.ipynb) — end-to-end
  demonstration of running `Helsinki-NLP/opus-mt-de-en` inside
  Teradata via `TD_MLDB.ONNXSeq2Seq`. Walks through the **full**
  pipeline: HuggingFace download, ONNX export, BYOM deploy, SQL
  translation, parity check against `transformers`. Use this when
  you want to see how a model gets converted, or when no pre-built
  ONNX is available for the checkpoint you need.
- [`opus_de_en_s3_demo.ipynb`](opus_de_en_s3_demo.ipynb) — same
  destination, shorter route. Downloads a pre-built ONNX and
  tokenizer for `Helsinki-NLP/opus-mt_tiny_deu-eng` from the
  public `teradata-opus-translate-ce` S3 bucket, deploys via
  `teradataml.save_byom`, and translates a set of German
  sentences. Skips the `torch.onnx.export` step entirely. Use
  this when a pre-built artifact exists for the model you want
  (see [`data/s3_manifest.json`](../data/s3_manifest.json)).

Both notebooks are committed with output cells populated so they
can be read top-to-bottom without a database connection.

## Running it against your own Teradata instance

The notebook is parameterised on environment variables and does
not hard-code any host or credential. To re-execute it against a
different instance, set the four `TD_*` variables and run
`jupyter nbconvert --execute`.

### Prerequisites

- Teradata Vantage with the `TD_MLDB.ONNXSeq2Seq` BYOM operator
  available (Teradata 20.00 or later, BYOM package installed).
- A target database (default name `OPUS_BYOM`) containing two
  empty BYOM tables:
  - `onnx_models (model_id VARCHAR(128), model BLOB)`
  - `sequence_tokenizers (tokenizer_id VARCHAR(128), tokenizer BLOB)`
- Python 3.12 with the project venv installed.

### Setup

```bash
make install-notebook

.venv/bin/python -m ipykernel install --user \
    --name teradata-opus-translate \
    --display-name "Python (teradata-opus-translate)"
```

### Connection environment variables

| Variable           | Default                  | Notes                                                      |
| ------------------ | ------------------------ | ---------------------------------------------------------- |
| `TD_HOST`          | `<your-teradata-host>`   | Target Teradata host. No default — must be set explicitly. |
| `TD_USER`          | `<your-user>`            | Database user with privileges on the BYOM database.        |
| `TD_PASSWORD`      | `<your-password>`        | Password for `TD_USER`.                                    |
| `TD_BYOM_DATABASE` | `<your-byom-database>`   | Where the BYOM tables live (e.g. `OPUS_BYOM`).             |

### Execute

```bash
TD_HOST=<your-host> \
TD_USER=<your-user> \
TD_PASSWORD=<your-password> \
TD_BYOM_DATABASE=OPUS_BYOM \
    .venv/bin/jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=900 \
        notebooks/opus_de_en_demo.ipynb
```

The first execution of `opus_de_en_demo.ipynb` is the slow one —
exporting the model to ONNX runs once and caches a 743 MB file
under `~/.cache/teradata-opus-translate/`. Re-runs reuse the
cached artifact instead of re-exporting.

`opus_de_en_s3_demo.ipynb` skips the export step entirely; first
execution downloads ~177 MiB from S3 once into
`~/.cache/teradata-opus-translate/s3/` and reuses it on re-runs.

### Interactive use

For interactive exploration, launch JupyterLab and open the
notebook there. The kernel is pinned to
`Python (teradata-opus-translate)` — register that kernel against
your project venv as shown above. Set the `TD_*` environment
variables in the shell that launches Jupyter so the setup cell
picks them up.
