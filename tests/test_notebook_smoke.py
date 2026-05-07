"""Smoke tests for the customer-facing demo notebooks (Issues #9, #61).

These checks are intentionally cheap — they parse the notebook
JSON and assert structural invariants that are easy to silently
regress when editing the notebooks by hand:

* the notebook has cells (and a non-trivial number of code cells),
* the env-var read pattern is present (so a future edit doesn't
  hard-code a connection target),
* ``teradatasql`` is **not** imported anywhere in the notebook —
  notebooks are the project's single ``teradataml`` exception,
  and double-driver setups invite confusing bugs,
* ``teradataml`` *is* imported (otherwise the policy carve-out
  has no purpose),
* the notebook is committed with output cells populated (the
  committed executed copy is itself the deliverable).

The bulk of the file pins notebook A
(``notebooks/opus_de_en_demo.ipynb``), which converts a HuggingFace
checkpoint to ONNX before deploying it. A parallel section at the
bottom pins notebook B (``notebooks/opus_de_en_s3_demo.ipynb``),
which downloads a pre-built ONNX from the public S3 bucket
instead of converting locally.

We do not execute either notebook here — that's an explicit local
operation against a running Teradata VM, see ``notebooks/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "opus_de_en_demo.ipynb"
NB_S3_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "opus_de_en_s3_demo.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    """Load the demo notebook as parsed JSON."""
    if not NB_PATH.exists():
        pytest.fail(f"notebook not found at {NB_PATH}")
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def s3_notebook() -> dict:
    """Load the S3-download demo notebook as parsed JSON."""
    if not NB_S3_PATH.exists():
        pytest.fail(f"notebook not found at {NB_S3_PATH}")
    return json.loads(NB_S3_PATH.read_text(encoding="utf-8"))


def _code_cell_sources(nb: dict) -> list[str]:
    """Return concatenated source for every code cell."""
    out: list[str] = []
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            out.append(src)
    return out


def test_notebook_has_cells(notebook: dict) -> None:
    cells = notebook.get("cells", [])
    assert len(cells) > 0, "notebook has no cells"


def test_notebook_has_meaningful_code_cells(notebook: dict) -> None:
    code_cells = _code_cell_sources(notebook)
    # The customer demo notebook walks through ~9 logical steps; we
    # expect at least eight code cells. If a future refactor
    # consolidates cells aggressively this can be relaxed, but keep
    # the floor high enough that a structural regression is obvious.
    assert len(code_cells) >= 8, f"expected >= 8 code cells, found {len(code_cells)}"


def test_notebook_reads_connection_env_vars(notebook: dict) -> None:
    """The notebook must take its TD connection from env vars.

    This guards against a re-edit accidentally hard-coding a host or
    password — both because secrets do not belong in cells and
    because the same notebook is supposed to be re-runnable against
    a different instance via env vars (see ``notebooks/README.md``).
    """
    joined = "\n".join(_code_cell_sources(notebook))
    required_env_reads = (
        'os.environ.get("TD_HOST"',
        'os.environ.get("TD_USER"',
        'os.environ.get("TD_PASSWORD"',
        'os.environ.get("TD_BYOM_DATABASE"',
    )
    for needle in required_env_reads:
        assert needle in joined, (
            f"expected env-var read pattern {needle!r} not found in any "
            "code cell — was the connection block hard-coded?"
        )


def test_notebook_does_not_import_teradatasql(notebook: dict) -> None:
    """Library policy: the notebook is the teradataml exception.

    The rest of the codebase uses ``teradatasql`` exclusively. The
    notebook intentionally does **not** mix the two drivers; any
    Teradata access in notebook cells goes through ``teradataml``.
    Helper modules imported by the notebook are allowed to use
    ``teradatasql`` — this assertion only forbids a direct import in
    notebook cells.
    """
    for src in _code_cell_sources(notebook):
        assert "import teradatasql" not in src, (
            "notebook cell imports teradatasql directly; per project "
            "library policy the notebook uses teradataml exclusively "
            "for Teradata access (see notebooks/README.md)"
        )
        assert "from teradatasql" not in src, (
            "notebook cell imports from teradatasql directly; per "
            "project library policy the notebook uses teradataml "
            "exclusively for Teradata access"
        )


def test_notebook_imports_teradataml(notebook: dict) -> None:
    """The notebook IS the project's teradataml carve-out.

    If a future refactor pulls all teradataml usage out of the
    notebook, the carve-out has no purpose and this test fires as
    an early signal that the policy comment in the README is now
    stale.
    """
    joined = "\n".join(_code_cell_sources(notebook))
    assert "teradataml" in joined, (
        "notebook does not reference teradataml; per Issue #9 it is "
        "the project's single teradataml exception"
    )


def test_notebook_is_committed_with_outputs(notebook: dict) -> None:
    """The committed executed .ipynb is itself the deliverable.

    ``nbstripout`` would strip outputs; this assertion deliberately
    fires if someone wires the notebook up to nbstripout or commits
    a freshly-cleared copy.
    """
    code_cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    cells_with_outputs = [c for c in code_cells if c.get("outputs")]
    # Every code cell in the demo prints or displays something, so
    # we expect *all* of them to carry outputs in the committed copy.
    # If a future cell legitimately produces no output we can soften
    # this to ">= len(code_cells) - 2" or similar.
    assert len(cells_with_outputs) == len(code_cells), (
        f"committed notebook is missing outputs: "
        f"{len(cells_with_outputs)}/{len(code_cells)} code cells have outputs. "
        "Re-run with `jupyter nbconvert --execute --inplace`."
    )


def test_notebook_inlines_onnxseq2seq_sql(notebook: dict) -> None:
    """The centerpiece SQL must be visible inline.

    The customer-facing demo deliberately writes the
    ``TD_MLDB.ONNXSeq2Seq`` query as a multi-line Python string so
    a reader can see every clause. If a future refactor hides the
    query behind a helper call, this check fires as a warning that
    the demo has lost a key teaching beat.
    """
    joined = "\n".join(_code_cell_sources(notebook))
    assert "TD_MLDB.ONNXSeq2Seq" in joined, (
        "notebook does not contain the literal string "
        "'TD_MLDB.ONNXSeq2Seq' — was the centerpiece SQL hidden "
        "behind a helper?"
    )
    # Spot-check a few of the named parameters whose presence is a
    # demo requirement (memory check off, fresh model load, beam
    # search constants explained in the markdown).
    for clause in (
        "EnableMemoryCheck",
        "OverwriteCachedModel",
        "Const_num_beams",
        "ModelOutputTensor",
    ):
        assert clause in joined, f"notebook is missing expected ONNXSeq2Seq clause {clause!r}"


def test_notebook_uses_dataframe_from_query_for_results(notebook: dict) -> None:
    """The BYOM SQL is executed via teradataml's ``DataFrame.from_query``.

    The previous shape opened a raw cursor on
    ``get_context().raw_connection()`` and called ``fetchall()`` to
    materialise results. That worked but added boilerplate the
    customer-facing demo did not need. The current shape lets
    teradataml handle materialisation via ``DataFrame.from_query``
    so the result renders inline. If a future edit re-introduces
    the raw cursor for the *result* fetch we want to know.
    """
    joined = "\n".join(_code_cell_sources(notebook))
    assert "DataFrame.from_query(" in joined or "from_query(ONNXSEQ2SEQ_SQL" in joined, (
        "notebook does not appear to use DataFrame.from_query for "
        "the BYOM result fetch — was the raw-cursor fetchall pattern "
        "re-introduced?"
    )


def test_notebook_uses_copy_to_sql_for_demo_input(notebook: dict) -> None:
    """The demo input table is loaded via teradataml's ``copy_to_sql``.

    The previous shape used a raw cursor and ``executemany`` to
    INSERT the demo rows. ``copy_to_sql`` collapses CREATE + INSERT
    into a single call with ``if_exists='replace'`` for
    idempotency. This assertion locks in the simpler shape.
    """
    joined = "\n".join(_code_cell_sources(notebook))
    assert "copy_to_sql(" in joined, (
        "notebook does not appear to use copy_to_sql for the demo "
        "input table — was the raw-cursor executemany pattern "
        "re-introduced?"
    )


def test_notebook_tokenizer_uses_canonical_save_byom_schema(notebook: dict) -> None:
    """Tokenizer deployment uses ``save_byom``'s canonical column names.

    A previous iteration inserted into the legacy
    ``sequence_tokenizers`` table via a staging-table dance because
    its column names did not match what ``save_byom`` writes. The
    current shape uses a dedicated table with the canonical
    ``(model_id, model)`` columns and aliases ``model AS tokenizer``
    inline in the BYOM SQL. The ``model AS tokenizer`` alias is the
    customer-visible signal that we are doing this; assert it is
    present.
    """
    joined = "\n".join(_code_cell_sources(notebook))
    assert "model AS tokenizer" in joined, (
        "notebook does not alias 'model AS tokenizer' in the "
        "TokenizerTable SELECT — was the legacy staging-table "
        "tokenizer flow re-introduced?"
    )


def test_notebook_pins_kernelspec(notebook: dict) -> None:
    """Ensure the notebook's kernelspec is the project's named kernel.

    The project README documents how to register the kernel as
    ``teradata-opus-translate``. If the notebook gets resaved with a
    bare ``python3`` kernelspec a reader on a different machine will
    have to fish around for the right venv.
    """
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    assert kernelspec.get("name") == "teradata-opus-translate", (
        f"notebook kernelspec.name is {kernelspec.get('name')!r}; "
        "expected 'teradata-opus-translate' (see notebooks/README.md)"
    )


# ---------------------------------------------------------------------------
# Notebook B: opus_de_en_s3_demo.ipynb (Issue #61).
#
# Same structural invariants as notebook A — kernelspec pin, env-var
# reads, no teradatasql import, teradataml present, outputs committed,
# inline ONNXSeq2Seq SQL — plus a few S3-specific checks: the public
# bucket URL pattern is hard-coded into the notebook, urllib is the
# download client (no boto3, no requests), and the BYOM row id has a
# `-from-s3` suffix so the two notebooks don't collide on a shared
# database.
# ---------------------------------------------------------------------------


def test_s3_notebook_has_meaningful_code_cells(s3_notebook: dict) -> None:
    code_cells = _code_cell_sources(s3_notebook)
    # The S3 notebook is shorter than notebook A (no convert/tokenize
    # step, no HF parity comparison) but still walks through ~7 logical
    # steps. Floor of 7 keeps a structural regression visible.
    assert len(code_cells) >= 7, f"expected >= 7 code cells, found {len(code_cells)}"


def test_s3_notebook_reads_connection_env_vars(s3_notebook: dict) -> None:
    """Same env-var pattern as notebook A; same justification."""
    joined = "\n".join(_code_cell_sources(s3_notebook))
    required_env_reads = (
        'os.environ.get("TD_HOST"',
        'os.environ.get("TD_USER"',
        'os.environ.get("TD_PASSWORD"',
        'os.environ.get("TD_BYOM_DATABASE"',
    )
    for needle in required_env_reads:
        assert needle in joined, (
            f"expected env-var read pattern {needle!r} not found in any "
            "code cell — was the connection block hard-coded?"
        )


def test_s3_notebook_does_not_import_teradatasql(s3_notebook: dict) -> None:
    """Same library policy as notebook A: no direct teradatasql usage."""
    for src in _code_cell_sources(s3_notebook):
        assert "import teradatasql" not in src, (
            "notebook cell imports teradatasql directly; per project "
            "library policy notebooks use teradataml exclusively"
        )
        assert "from teradatasql" not in src, (
            "notebook cell imports from teradatasql directly; per "
            "project library policy notebooks use teradataml exclusively"
        )


def test_s3_notebook_imports_teradataml(s3_notebook: dict) -> None:
    joined = "\n".join(_code_cell_sources(s3_notebook))
    assert "teradataml" in joined, (
        "S3 notebook does not reference teradataml — every BYOM "
        "deploy/query path goes through teradataml"
    )


def test_s3_notebook_is_committed_with_outputs(s3_notebook: dict) -> None:
    """The committed executed .ipynb is itself the deliverable."""
    code_cells = [c for c in s3_notebook["cells"] if c.get("cell_type") == "code"]
    cells_with_outputs = [c for c in code_cells if c.get("outputs")]
    assert len(cells_with_outputs) == len(code_cells), (
        f"committed S3 notebook is missing outputs: "
        f"{len(cells_with_outputs)}/{len(code_cells)} code cells have outputs."
    )


def test_s3_notebook_inlines_onnxseq2seq_sql(s3_notebook: dict) -> None:
    """Centerpiece SQL must be inline, with the same v1.0.1 Const_* cluster."""
    joined = "\n".join(_code_cell_sources(s3_notebook))
    assert "TD_MLDB.ONNXSeq2Seq" in joined, (
        "S3 notebook does not contain the literal string "
        "'TD_MLDB.ONNXSeq2Seq' — was the centerpiece SQL hidden behind a helper?"
    )
    for clause in (
        "EnableMemoryCheck",
        "OverwriteCachedModel",
        "Const_num_beams",
        "Const_min_length",
        "Const_max_length",
        "Const_length_penalty",
        "Const_repetition_penalty",
        "ModelOutputTensor",
    ):
        assert clause in joined, f"S3 notebook is missing expected ONNXSeq2Seq clause {clause!r}"
    # `Const_num_return_sequences` is intentionally NOT present (per
    # v1.0.1: BYOM bakes num_return_sequences into the ONNX graph as a
    # constant and ignores any USING clause for it). Locking this in so
    # a future edit doesn't accidentally re-add the no-op clause.
    assert "Const_num_return_sequences" not in joined, (
        "S3 notebook should not pass Const_num_return_sequences — it is "
        "baked into the ONNX graph as a constant and silently ignored"
    )


def test_s3_notebook_uses_dataframe_from_query_for_results(s3_notebook: dict) -> None:
    joined = "\n".join(_code_cell_sources(s3_notebook))
    assert "DataFrame.from_query(" in joined or "from_query(ONNXSEQ2SEQ_SQL" in joined, (
        "S3 notebook does not appear to use DataFrame.from_query for the BYOM result fetch"
    )


def test_s3_notebook_uses_copy_to_sql_for_demo_input(s3_notebook: dict) -> None:
    joined = "\n".join(_code_cell_sources(s3_notebook))
    assert "copy_to_sql(" in joined, (
        "S3 notebook does not appear to use copy_to_sql for the demo input table"
    )


def test_s3_notebook_uses_save_byom_for_both_artifacts(s3_notebook: dict) -> None:
    """Both model and tokenizer must go through save_byom().

    The whole point of notebook B is to show customers the standard
    teradataml deployment path; if a future edit replaces save_byom
    with a raw cursor + INSERT, the demo loses its key teaching beat.
    """
    joined = "\n".join(_code_cell_sources(s3_notebook))
    assert joined.count("save_byom(") >= 2, (
        f"expected at least 2 save_byom() calls (one per artifact); "
        f"found {joined.count('save_byom(')}"
    )


def test_s3_notebook_tokenizer_uses_canonical_save_byom_schema(s3_notebook: dict) -> None:
    """Tokenizer SELECT aliases ``model AS tokenizer`` — same pattern as notebook A."""
    joined = "\n".join(_code_cell_sources(s3_notebook))
    assert "model AS tokenizer" in joined, (
        "S3 notebook does not alias 'model AS tokenizer' in the TokenizerTable SELECT"
    )


def test_s3_notebook_uses_urllib_for_download(s3_notebook: dict) -> None:
    """Download client is stdlib ``urllib.request``.

    Issue #61 originally specified ``requests``, but we ship with
    ``urllib.request.urlretrieve`` instead so the notebook has zero
    extra runtime dependencies beyond ``teradataml`` and the
    ``teradata-opus-translate`` package — important since the bucket
    is public-read and a heavyweight HTTP client adds nothing here.
    If a future edit re-adds ``requests`` or ``boto3`` we want to know.
    """
    joined = "\n".join(_code_cell_sources(s3_notebook))
    assert "urllib.request" in joined, (
        "S3 notebook does not import urllib.request — was the download "
        "client switched to requests or boto3?"
    )
    assert "urlretrieve" in joined, "S3 notebook does not call urllib.request.urlretrieve"
    # Defensive: stop a future edit from sneaking boto3 in. requests
    # is harder to ban outright (it's a common transitive import) so
    # we only block boto3, which is the AWS-credentials path we
    # explicitly want to avoid for a public bucket.
    for src in _code_cell_sources(s3_notebook):
        assert "import boto3" not in src, (
            "S3 notebook imports boto3; the bucket is public-read so "
            "no AWS credentials should be required"
        )


def test_s3_notebook_references_public_s3_bucket(s3_notebook: dict) -> None:
    """The notebook must point at the public ``teradata-opus-translate-ce``
    bucket using the URL shape published in ``data/s3_manifest.json``.

    The exact pattern (bucket, region, opset prefix, model_id, file
    name) is the customer-visible promise. If a future edit changes
    bucket name or region this test fires before the notebook is
    published.
    """
    joined = "\n".join(_code_cell_sources(s3_notebook))
    assert "teradata-opus-translate-ce" in joined, (
        "S3 notebook does not reference the public 'teradata-opus-translate-ce' bucket"
    )
    # Region is interpolated into the URL via an `S3_REGION = "..."`
    # constant rather than appearing as a literal `s3.us-east-1...` in
    # the source. Check for the constant value plus the AWS endpoint.
    assert "us-east-1" in joined, "S3 notebook does not use the us-east-1 S3 region"
    assert "amazonaws.com" in joined, "S3 notebook does not target the AWS S3 endpoint"
    # The opset directory is the manifest's stable layout key — must
    # match data/s3_manifest.json's `url_template`. Either the assembled
    # URL substring or the constant assignment must be present.
    assert "S3_OPSET = 14" in joined or "opus-translate/14/" in joined, (
        "S3 notebook does not include the opset-14 prefix in the URL "
        "(see data/s3_manifest.json url_template)"
    )
    # Model files referenced in the notebook should cover both the
    # ONNX model and the tokenizer.
    assert "model-fp32.onnx" in joined, "S3 notebook does not download model-fp32.onnx"
    assert "tokenizer.json" in joined, "S3 notebook does not download tokenizer.json"


def test_s3_notebook_uses_distinct_byom_row_id(s3_notebook: dict) -> None:
    """BYOM row id has a ``-from-s3`` suffix so it doesn't collide
    with rows that ``opus_de_en_demo.ipynb`` may have left behind.

    Both notebooks store their BYOM artifacts in the same default
    database (``OPUS_BYOM``) and the same tables (``onnx_models``,
    ``opus_tokenizers``). If both used the same model_id row key, a
    customer running them back-to-back would see the second notebook
    silently overwrite the first's deployment. The suffix keeps them
    independent.
    """
    joined = "\n".join(_code_cell_sources(s3_notebook))
    assert "-from-s3" in joined, (
        "S3 notebook does not use a distinct '-from-s3'-suffixed "
        "BYOM_MODEL_ID; runs of the two notebooks may collide on "
        "shared BYOM tables"
    )


def test_s3_notebook_pins_kernelspec(s3_notebook: dict) -> None:
    kernelspec = s3_notebook.get("metadata", {}).get("kernelspec", {})
    assert kernelspec.get("name") == "teradata-opus-translate", (
        f"S3 notebook kernelspec.name is {kernelspec.get('name')!r}; "
        "expected 'teradata-opus-translate'"
    )
