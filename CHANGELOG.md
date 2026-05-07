# Changelog

All notable changes to `teradata-opus-translate` are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] — 2026-05-07

### Added

- **Phase 3 publication**: 25 MarianMT tiny-variant models from the
  [`Helsinki-NLP/opustranslate`](https://huggingface.co/collections/Helsinki-NLP/opustranslate)
  collection are now published as ready-to-deploy ONNX artifacts on S3 at
  `s3://teradata-opus-translate-ce/opus-translate/14/Helsinki-NLP/<model-id>/`.
  Each model ships fp32 + (when quantization holds) int8 + tokenizer.json,
  verified end-to-end via per-model BYOM smoke tests.
- **Published catalog**: `docs/published-models.md` lists every available
  model with language pair, parameter count, context size, output cap,
  fp32/int8 sizes, S3 URLs, and smoke status.
- **Customer notebook B**: `notebooks/opus_de_en_s3_demo.ipynb` shows the
  download-from-S3 path — pull a pre-built ONNX directly into Teradata
  BYOM without re-converting from HuggingFace.
- **Manifest files**: `data/catalog.json` (collection metadata),
  `data/s3_manifest.json` (per-model S3 URLs), and
  `data/smoke_results.json` (BYOM scoring results) provide a programmatic
  index of what's available.

## [1.0.2] — 2026-05-07

### Changed

- Removed PyPI version + Python version shields.io badges from the README
  header. Cosmetic-only change to address a stuck GitHub Camo cache rendering
  issue. PyPI and Python compatibility info remain visible via PyPI's own UI
  and the package metadata.

## [1.0.1] — 2026-05-07

### Fixed

- **`num_return_sequences` is now locked at 1 in the produced ONNX graph.**
  Previously the converter exposed `num_return_sequences` as a top-level
  graph input, which implied it could be overridden per query via BYOM's
  `Const_num_return_sequences(N)` USING clause. The locked-to-1 design
  simplifies output semantics: each input row always returns exactly one
  translation. As of v1.0.1 the value is baked into the graph as a
  `Constant(1)` node feeding BeamSearch's slot 4, and the
  `Const_num_return_sequences(N)` USING clause has no effect (silently
  ignored by BYOM). The other five SQL-tunable parameters
  (`num_beams`, `max_length`, `min_length`, `length_penalty`,
  `repetition_penalty`) remain overridable via their `Const_*` USING
  clauses. See [Issue #82](https://github.com/alexander-smirnov_teradata/teradata-opus-translate/issues/82)
  and `docs/decisions.md` Decision 10.

## [1.0.0] — 2026-05-06

First public release. `teradata-opus-translate` turns a Helsinki-NLP **OPUS**
(Marian) translation model into a single self-contained ONNX file with beam
search baked into the graph — ready to drop into a Teradata Vantage BYOM
table and score directly from SQL.

If you've ever tried to export a Marian model with stock HuggingFace tooling
you know the pain: a Cache class that won't trace, sinusoidal positional
embeddings that need special handling, a `final_logits_bias` term most
exporters drop on the floor, an `embed_scale` factor that changes outputs if
you forget it, a pad / decoder-start token ID overlap that silently corrupts
generation, and an encoder seed step that has to be hoisted out of the loop
or the graph won't match what BYOM's onnxruntime expects. This package handles
all of that for you and verifies the result against the reference PyTorch
model before handing you the file.

### Added

- **A two-call public API.** `convert_model(...)` and `convert_tokenizer(...)`
  are exposed at the package root. Each accepts a HuggingFace model id (e.g.
  `Helsinki-NLP/opus-mt-de-en`) **or** a local folder. One call in, one
  self-contained ONNX file out.

  ```python
  from teradata_opus_translate import convert_model, convert_tokenizer

  convert_model("Helsinki-NLP/opus-mt-de-en", "opus-de-en.onnx")
  convert_tokenizer("Helsinki-NLP/opus-mt-de-en", "opus-de-en.tokenizer.json")
  ```

- **Two precision targets, fp32 and dynamic int8.** The int8 path produces
  files roughly **half the size** with roughly **2× lower latency**, with a
  documented quality envelope: chrF ≥ 95 vs the fp32 reference on a
  1000-sentence German→English evaluation corpus. Pick fp32 when you want
  bit-exact parity with the PyTorch source; pick int8 when you want speed and
  smaller deployment artifacts and can live inside the documented quality
  envelope.

- **Built-in parity verification.** `verify=True` (the default) runs a
  token-level comparison against `MarianMTModel.generate()` on a built-in
  multi-sentence corpus immediately after export, so you find out *before*
  you upload to Teradata if your converted graph drifts from the reference.
  The verification harness is exercised in CI across **56 OPUS language
  pairs** — covering the full Helsinki-NLP collection this package targets.

- **A customer demo notebook** in the repo's `notebooks/` directory walks
  through the full BYOM workflow end-to-end: `pip install`, convert, load
  into Teradata, run translations from SQL.

- **Designed against Teradata BYOM 7.x with onnxruntime 1.16.3.** The graph
  shape targets the `encoder_decoder_init` pattern that BYOM's bundled ORT
  expects. Local validation runs against `onnxruntime ≤ 1.21`, so what passes
  the package's tests will load and score inside BYOM without surprises.

- **The Marian quirks handled for you.** The Cache class non-exportability,
  sinusoidal positional embeddings, `final_logits_bias`, `embed_scale`, the
  pad / decoder-start ID overlap, the encoder seed-step hoisting, and the
  zero-weighted reference-input trick used to keep the beam search graph
  type-stable — none of which you should ever need to think about. The
  package is the result of figuring those out so you don't have to.

### Notes

- This is the first public release. PyPI is the canonical distribution
  channel — see <https://pypi.org/project/teradata-opus-translate/> for
  installation and version metadata.
- The CHANGELOG is now the canonical release-notes location for this project;
  GitHub releases are not used.
