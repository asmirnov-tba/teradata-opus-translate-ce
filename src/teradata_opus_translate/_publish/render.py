"""Render the Hugging Face model card from the Jinja2 template.

The template is the *single source of truth* for the card layout —
everything Python does here is build a context dict and feed it to Jinja2.
No card prose lives in Python.

The template is loaded from ``scripts/templates/hf_model_card.md.j2``
relative to the repository root. We resolve the path at runtime rather
than baking it into the package, because the template ships only with
the repository (it is not part of the wheel).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

from .lang_info import lookup, split_short_id

# Repo-root-relative path to the template. Resolved lazily so that callers
# can override via ``render_card(template_path=...)`` in tests.
DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "templates" / "hf_model_card.md.j2"
)


def _bytes_to_mb(n: int) -> int:
    """Round bytes to whole megabytes for display in the spec table."""

    return round(n / 1_000_000)


def build_context(
    *,
    short_id: str,
    manifest_entry: dict[str, Any],
    upstream_config: dict[str, Any],
    upstream_generation_config: dict[str, Any] | None,
    hf_repo_id: str,
    has_generation_config: bool | None = None,
) -> dict[str, Any]:
    """Build the Jinja2 context dict for one model.

    Parameters
    ----------
    short_id:
        e.g. ``opus-mt_tiny_ara-eng``.
    manifest_entry:
        The matching dict from ``data/s3_manifest.json``'s ``models`` array.
    upstream_config:
        Parsed contents of ``Helsinki-NLP/<short_id>/config.json``.
    upstream_generation_config:
        Parsed contents of ``generation_config.json`` if present, else ``None``.
    hf_repo_id:
        Full target repo id, e.g. ``sashamartin/opus-mt_tiny_ara-eng`` or
        ``Teradata/opus-mt_tiny_ara-eng``.
    has_generation_config:
        Whether the upstream repo shipped a ``generation_config.json``
        that we are forwarding into the published repo. Controls the
        rendered "This repository contains only:" file list. When left
        as ``None``, defaults to ``upstream_generation_config is not
        None`` — most callers should let it default; the explicit kwarg
        exists for tests that want to render the "no gen-config" layout
        even though they pass a synthetic ``upstream_generation_config``
        dict for the max-length derivation.

    Returns
    -------
    dict
        Keys exactly match the variables referenced in the Jinja template.
    """

    src_code, tgt_code = split_short_id(short_id)
    src = lookup(src_code)
    tgt = lookup(tgt_code)

    # Marian models historically use ``max_position_embeddings`` for input
    # capacity. Generation length comes from ``generation_config.max_length``
    # if available, otherwise from the model config's ``max_length``, else
    # falls back to the same number as the input cap (a safe upper bound
    # given BYOM caps total length anyway).
    max_input_tokens = upstream_config.get(
        "max_position_embeddings",
        upstream_config.get("max_length", 512),
    )
    if upstream_generation_config and "max_length" in upstream_generation_config:
        max_output_tokens = upstream_generation_config["max_length"]
    elif "max_length" in upstream_config:
        max_output_tokens = upstream_config["max_length"]
    else:
        max_output_tokens = max_input_tokens

    files = manifest_entry.get("files", {})
    fp32_size = files.get("model-fp32.onnx", {}).get("size", 0)
    int8_size = files.get("model-int8.onnx", {}).get("size", 0)

    if has_generation_config is None:
        has_generation_config = upstream_generation_config is not None

    return {
        "short_id": short_id,
        "hf_repo_id": hf_repo_id,
        "bcp47_src": src["bcp47"],
        "bcp47_tgt": tgt["bcp47"],
        "src_lang_full": src["name_en"],
        "tgt_lang_full": tgt["name_en"],
        "src_flag": src["flag"],
        "tgt_flag": tgt["flag"],
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "fp32_mb": _bytes_to_mb(fp32_size),
        "int8_mb": _bytes_to_mb(int8_size),
        "has_generation_config": has_generation_config,
    }


def render_card(
    context: dict[str, Any],
    *,
    template_path: Path | None = None,
) -> str:
    """Render the model-card README from ``context``.

    Parameters
    ----------
    context:
        Output of :func:`build_context`.
    template_path:
        Override the default ``scripts/templates/hf_model_card.md.j2``
        path. Used by the unit tests to load the template from a fixture
        location.

    Returns
    -------
    str
        The fully-rendered Markdown card. No trailing whitespace; ends
        with exactly one newline.
    """

    path = template_path or DEFAULT_TEMPLATE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model-card template not found at {path}")

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(path.parent)),
        autoescape=False,
        keep_trailing_newline=True,
        # We deliberately do NOT trim_blocks / lstrip_blocks. The template
        # uses explicit ``{%- ... -%}`` markers where whitespace control
        # matters; global trimming would also collapse desired newlines.
    )
    template = env.get_template(path.name)
    rendered = template.render(**context)

    # Normalise trailing whitespace: strip trailing spaces on each line,
    # collapse 3+ newlines to 2, and end the file with exactly one newline.
    lines = [line.rstrip() for line in rendered.splitlines()]
    out = "\n".join(lines).rstrip() + "\n"
    return out
