"""Smoke tests for ``notebooks/opus_de_en_demo.ipynb`` (Issue #9).

These checks are intentionally cheap — they parse the notebook
JSON and assert structural invariants that are easy to silently
regress when editing the notebook by hand:

* the notebook has cells (and a non-trivial number of code cells),
* the env-var read pattern is present (so a future edit doesn't
  hard-code a connection target),
* ``teradatasql`` is **not** imported anywhere in the notebook —
  the notebook is the project's single ``teradataml`` exception,
  and double-driver setups invite confusing bugs,
* ``teradataml`` *is* imported (otherwise the policy carve-out
  has no purpose),
* the notebook is committed with output cells populated (the
  committed executed copy is itself the deliverable).

We do not execute the notebook here — that's an explicit local
operation against a running Teradata VM, see ``notebooks/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "opus_de_en_demo.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    """Load the demo notebook as parsed JSON."""
    if not NB_PATH.exists():
        pytest.fail(f"notebook not found at {NB_PATH}")
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


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
