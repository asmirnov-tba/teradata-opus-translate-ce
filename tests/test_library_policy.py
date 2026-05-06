"""Library-policy guard: ``teradataml`` may only be imported in ``notebooks/``.

Project rule (see README "Teradata client library policy" section and
``notebooks/README.md``): production code — everything under ``src/``,
``tests/``, ``scripts/`` — uses ``teradatasql`` exclusively. The demo
notebook is the project's single ``teradataml`` carve-out.

Until now this rule was enforced by README copy and human review. Issue
#14 made it CI-checkable. This test parses every ``.py`` file under the
in-scope production directories with :mod:`ast` and fails if any of them
contain an ``import teradataml`` or ``from teradataml ...`` statement.

Why AST and not a ``grep``:

* ``teradataml`` legitimately appears as a *string* in plenty of places
  — docstrings (including this one), policy comments, test assertions
  about the notebook, README excerpts copied as comments, etc. A
  substring ``grep`` would either need an ever-growing allowlist or
  would deliver false positives.
* ``ast.Import`` / ``ast.ImportFrom`` nodes are precisely the construct
  the policy is about. Walking the parsed AST gives us a zero-false-
  positive answer.

Symmetry note: ``test_notebook_smoke.py`` already asserts the *positive*
side — the demo notebook **must** import ``teradataml`` and **must not**
import ``teradatasql``. This test is the negative twin for production
code and the two together pin the policy from both directions.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Repository root, computed from this test file's location so the test
# is location-independent (works locally, in CI, in editable installs).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that hold production-style Python and are therefore in
# scope for the policy. ``notebooks/`` is intentionally absent — that
# tree is the project's single ``teradataml`` carve-out and is covered
# by the inverse assertion in ``test_notebook_smoke.py``.
PRODUCTION_DIRS = ("src", "tests", "scripts")


def _python_files() -> list[Path]:
    """Collect every ``.py`` file under the in-scope production dirs.

    Files inside hidden directories (``.venv``, ``.git``, ``.mypy_cache``,
    etc.) and inside ``__pycache__`` are skipped — those either are not
    project source or are build artefacts.
    """
    files: list[Path] = []
    for top in PRODUCTION_DIRS:
        root = REPO_ROOT / top
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            # Skip anything under a hidden directory or a __pycache__.
            if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
                continue
            files.append(path)
    return files


def _teradataml_import_lines(path: Path) -> list[int]:
    """Return the line numbers in ``path`` that import ``teradataml``.

    Both ``import teradataml[.x]`` and ``from teradataml[.x] import y``
    forms are detected. Returns an empty list if the file does not
    import ``teradataml`` at all.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "teradataml" or alias.name.startswith("teradataml."):
                    hits.append(node.lineno)
                    break
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "teradataml" or mod.startswith("teradataml."):
                hits.append(node.lineno)
    return hits


def test_no_teradataml_imports_in_production_code() -> None:
    """``teradataml`` must not be imported under ``src/``, ``tests/``, or ``scripts/``.

    If this test fires, the failure message lists every offending file
    and line so the contributor can either move the import into
    ``notebooks/`` or open a project-policy discussion. Do **not**
    silence this test by adding a per-file allowlist — if production
    code legitimately needs ``teradataml`` that is a policy change, not
    a test exception.
    """
    files = _python_files()
    # Sanity-check the walker actually found something. If it returns
    # an empty list the assertion below would pass vacuously and a
    # future refactor that moves the production tree could silently
    # disable this guard. Keep the floor low — the test suite alone
    # already has well over a dozen files.
    assert len(files) >= 5, (
        f"library-policy guard scanned only {len(files)} Python files under "
        f"{PRODUCTION_DIRS!r}; expected several. Has the project layout moved?"
    )

    offenders: list[str] = []
    for path in files:
        for lineno in _teradataml_import_lines(path):
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"{rel}:{lineno}")

    if offenders:
        joined = "\n  ".join(offenders)
        pytest.fail(
            "teradataml is imported outside notebooks/, which violates the "
            "project's library policy (see README, 'Teradata client library "
            f"policy'). Offending location(s):\n  {joined}\n"
            "Fix: use teradatasql for production code, or move the affected "
            "code into notebooks/. Do not add a per-file allowlist to this "
            "test — that's a project-policy discussion."
        )


def test_policy_guard_detects_a_synthetic_violation(tmp_path: Path) -> None:
    """Self-check: the AST detector fires on a deliberate violation.

    This guards against a future refactor of ``_teradataml_import_lines``
    accidentally turning into a no-op (e.g. a bad rename of a node
    class) — without this, the main test could pass vacuously even if
    the detector silently stopped looking.
    """
    bad = tmp_path / "violation.py"
    bad.write_text("import teradataml\n", encoding="utf-8")
    assert _teradataml_import_lines(bad) == [1]

    bad_from = tmp_path / "violation_from.py"
    bad_from.write_text("from teradataml.context import get_context\n", encoding="utf-8")
    assert _teradataml_import_lines(bad_from) == [1]

    # And conversely: a file that merely *mentions* teradataml in
    # strings/comments must NOT trip the detector. This is the whole
    # reason we use the AST instead of a grep.
    clean = tmp_path / "clean.py"
    clean.write_text(
        '"""policy doc: teradataml is allowed only in notebooks/."""\n'
        "# teradataml mentioned here too\n"
        "x = 'teradataml'\n",
        encoding="utf-8",
    )
    assert _teradataml_import_lines(clean) == []
