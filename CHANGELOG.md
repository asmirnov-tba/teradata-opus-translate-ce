# Changelog

All notable changes to `teradata-opus-translate` are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] — 2026-05-17

### Changed

- **`precision="int8"` now uses the weight-only int8 recipe (BREAKING).**
  The v1.0.x int8 path applied
  `onnxruntime.quantization.quantize_static` (Phase 3 / 4) to the
  encoder and decoder subgraphs, which inserted activation-quantization
  ops (`QuantizeLinear` / `DynamicQuantizeLinear`) in front of every
  rewritten `MatMul`.  Phase 2 (PR #138), Phase 3 (Issue #140), and
  Phase 4 (PR #154) showed that recipe drives beam search into a
  degenerate basin (runaway-token loops) on the `*-eng` Marian pairs
  under BYOM 7.0.0.4's pinned ORT 1.13.1, regardless of dynamic vs
  static calibration or whether `num_beams` was baked into the graph.
  v1.1.0 replaces it with the weight-only rewriter introduced in
  Issue #160: every `MatMul` whose B input is a 2-D fp32 const
  initializer is stored as int8 + per-channel symmetric scales with a
  `DequantizeLinear` node inserted at inference time, and activations
  stay fp32.  Empirical evidence from the Phase 5 Gate 3 25-pair
  100-sentence sweep (`analysis/phase5_weight_only_int8/`): 20 / 25
  pairs PASS with ≥ 92 byte-identical decodes and BLEU ≥ 96.6 vs fp32;
  3 NEEDS-REVIEW pairs (no broken decodes); 2 BROKEN pairs (deu-eng,
  ell-eng) on a minority of inputs.  Artifact size is roughly half of
  fp32 (~90 MiB vs ~170 MiB on the curated `opus-mt_tiny_*` collection).
  See [Issue #160](https://github.com/alexander-smirnov_teradata/teradata-opus-translate/issues/160).

- **Int8 parity tolerance tightened from 20% to 10%.** Weight-only
  drift is much narrower than the v1.0.x activation-quantization
  drift; the small-N floor (`max(1, ceil(N * 0.10))`) still permits
  the empirically-observed 1-sample mismatch on the 3-sample German
  default verification set, while the 10% bound flags regressions on
  PASS-pair sweeps that typically produce 0–3 mismatches per 100.

- **`num_beams` is once again a top-level graph input on int8
  artifacts.** Phase 4 (PR #154) baked `num_beams=4` into the
  BeamSearch contrib op for int8 builds to work around
  activation-quantization-induced beam collapse.  The weight-only
  recipe keeps activations in fp32 so that drift cannot recur, and
  the baking machinery was removed.  `Const_num_beams(N)` USING
  clauses on the int8 artifact are now respected the same as on fp32.

### Removed

- **Static / dynamic int8 recipes.** `quantize.py` no longer ships
  `quantize_subgraph`, `_INT8_OP_TYPES`, `_INT8_NODES_TO_EXCLUDE`, or
  the `_ListCalibrationDataReader` helper.  The
  `_converter/calibration.py` module (Tatoeba corpus loader) and its
  `test_calibration.py` were removed entirely — no production code
  path needs a calibration corpus any more.

- **`bake_num_beams` kwarg on `assemble_full_model`.** Internal-only
  knob introduced in PR #154 for the int8 path; removed alongside the
  static-quant recipe it supported.  No public callers existed.

- **`datasets` and `platformdirs` runtime dependencies.** Both were
  pulled in only by the calibration loader; with that module gone the
  package goes back to its pre-#153 minimal core dependency set.

### Deprecated

- **`calibration_pair` kwarg on `convert_model`.** Retained on the
  signature for backward compatibility with v1.0.x callers but is now
  a no-op.  Passing a non-None value triggers a `DeprecationWarning`;
  the kwarg will be removed in a future major release.

### Added

- **Hugging Face publisher tooling.** New `scripts/publish_to_huggingface.py`
  publishes the 25 ONNX-converted MarianMT models listed in
  `data/s3_manifest.json` to private Hugging Face repos under a
  configurable target org, renders a locked-layout model card from
  `scripts/templates/hf_model_card.md.j2`, and adds each repo to a
  curated collection. Idempotent (re-runs skip existing repos and
  collection memberships); per-model failures don't abort the batch.
  See `docs/publishing-to-huggingface.md` for the operator runbook.
  Adds a `publish` optional-dependencies extra for `huggingface_hub`,
  `jinja2`, and `requests`. Tooling only — no package version bump.
  See
  [Issue #120](https://github.com/alexander-smirnov_teradata/teradata-opus-translate/issues/120).

## [1.0.5] — 2026-05-07

### Fixed

- **Default ONNX IR version pinned at 8 to match BYOM 7.x bundled ORT.**
  v1.0.0–1.0.4 produced IR v9 (the upstream `torch.onnx.export` default),
  which BYOM 7.0.4's bundled ORT (1.16.3 lineage) rejects with
  `Unsupported model IR version: 9, max supported IR version: 8` raised
  from `onnxruntime/core/graph/model.cc` at model-load time. The new
  `ir_version` parameter on `convert_model` (default `8`) lets customers
  override only if their downstream runtime supports newer IRs. The
  default is a **hard** pin: even if upstream flips the implicit IR
  again, this package's output stays BYOM-compatible until the default
  is explicitly bumped. Opset stays at `14`. See
  [Issue #109](https://github.com/alexander-smirnov_teradata/teradata-opus-translate/issues/109),
  [Issue #108](https://github.com/alexander-smirnov_teradata/teradata-opus-translate/issues/108)
  (the BYOM 7.0.0.4 dev-VM rebuild that surfaced the defect), and
  `docs/decisions.md` Decision 12.

## [1.0.4] — 2026-05-07

### Fixed

- **Relax `onnxruntime` runtime pin from `<1.22` to `<2.0`**. ORT 1.17–1.21 wheels are no longer on PyPI for current Python versions, making v1.0.3 uninstallable on Python 3.12+. The original pin protected against a theoretical converter regression to a 2-input encoder pattern that ORT 1.22+ accepts but BYOM 1.16.3 rejects; that protection is no longer load-bearing since the converter's wiring (in `_converter/assemble.py`) produces the 3-input encoder regardless of which ORT version validates it locally.

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
