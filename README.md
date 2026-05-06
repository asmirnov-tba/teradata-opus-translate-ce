# teradata-opus-translate-ce

**Community Edition** — Helsinki-NLP OPUS translation models converted to Teradata BYOM-compatible ONNX.

> **v1.0.0 release pending.** Content syncs on tag push from the upstream
> `teradata-opus-translate` development repo. See
> https://pypi.org/project/teradata-opus-translate/ for the package once it
> is published.

## What this repo will contain

Once the first release tag lands:

- `src/teradata_opus_translate/` — the Python library that converts OPUS
  Marian models into ONNX artifacts deployable to Teradata Vantage via BYOM.
- `pyproject.toml` — installable package definition.
- Customer-facing notebooks demonstrating end-to-end usage.
- Public unit tests that don't require a Teradata instance.
- `LICENSE` (MIT).

## What this repo will NOT contain

- Internal development docs, architecture decision records, or design
  scratch.
- Scripts that target our development VMs or CI runners.
- Tests that require a live Teradata connection.
- SQL DDL specific to our internal Teradata environments.

The full include/exclude rules are documented in the upstream repo at
`docs/public-mirror.md`.

## License

MIT — see [`LICENSE`](LICENSE).
