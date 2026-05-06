"""Build the side-by-side comparison report produced for issue #6.

The artifact has two physical forms, both checked into the repo:

* ``tests/data/local_parity_report.json`` — machine-readable. One
  record per baseline sentence with the full token-id sequences from
  both backends, the decoded text, the match flag and the diagnosed
  divergence point. Future runs diff this file structurally.
* ``tests/data/local_parity_report.md`` — human-readable. Renders a
  header (provenance, generation params, normalisation caveat,
  excluded features) and a Markdown table with one row per sentence.

The two files are kept in lock-step: the script always writes both
from the same in-memory artifact dict.

Schema versioning
-----------------
Bump :data:`PARITY_REPORT_SCHEMA_VERSION` if you make a
backwards-incompatible change to the JSON record shape.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PARITY_REPORT_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Dataclasses describing one row of the report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityRow:
    """One per-sentence row in the parity report."""

    id: str
    domain: str
    source_text: str
    transformers_text: str
    onnx_text: str
    transformers_token_ids: list[int]
    onnx_token_ids: list[int]
    transformers_token_ids_normalized: list[int]
    onnx_token_ids_normalized: list[int]
    match: bool
    divergence_index: int
    transformers_token_at_divergence: int | None
    transformers_piece_at_divergence: str | None
    onnx_token_at_divergence: int | None
    onnx_piece_at_divergence: str | None
    transformers_length_normalized: int
    onnx_length_normalized: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Ensure list copies so the dataclass remains immutable in spirit.
        d["transformers_token_ids"] = list(self.transformers_token_ids)
        d["onnx_token_ids"] = list(self.onnx_token_ids)
        d["transformers_token_ids_normalized"] = list(self.transformers_token_ids_normalized)
        d["onnx_token_ids_normalized"] = list(self.onnx_token_ids_normalized)
        return d


@dataclass
class ParityReport:
    """Full report artifact.

    Header captures provenance and the comparison contract; ``rows`` is
    the per-sentence detail; ``summary`` captures the headline match
    counts so a reader can scan the top of the JSON and see the
    bottom-line answer without traversing 62 records.
    """

    header: dict[str, Any]
    summary: dict[str, Any]
    rows: list[ParityRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": dict(self.header),
            "summary": dict(self.summary),
            "rows": [r.to_dict() for r in self.rows],
        }


# ---------------------------------------------------------------------------
# JSON / Markdown writers
# ---------------------------------------------------------------------------


def write_report_json(artifact: dict[str, Any], path: str | Path) -> Path:
    """Serialise the report dict to disk in deterministic, pretty JSON.

    Same conventions as the baseline writer: UTF-8, indent=2, no
    ``sort_keys`` (we control ordering ourselves), ``ensure_ascii=False``,
    trailing newline.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    out.write_text(text, encoding="utf-8")
    return out


def render_report_markdown(artifact: dict[str, Any]) -> str:
    """Render a human-readable Markdown report from the artifact dict.

    Output structure:

    1. Title and high-level summary (match rate).
    2. Provenance block (model, generation params, EOS normalisation,
       excluded features).
    3. The 62-row comparison table.
    4. A "Mismatches" appendix listing each non-matching row's full
       token-id sequences so reviewers can see *exactly* where things
       diverged without running anything.
    """
    h = artifact["header"]
    s = artifact["summary"]
    rows = artifact["rows"]

    lines: list[str] = []
    lines.append("# Local ONNX parity report (transformers vs onnxruntime)")
    lines.append("")
    lines.append(
        f"**Match rate:** {s['matches']}/{s['total']} "
        f"({s['match_rate_percent']:.1f}%) sentences produce token-ID-equal "
        "output after EOS normalisation."
    )
    lines.append("")
    lines.append("## What this report is")
    lines.append("")
    lines.append(
        "For every sentence in the curated DE -> EN baseline test set, this "
        "report runs the source through:"
    )
    lines.append("")
    lines.append(
        "1. `transformers.MarianMTModel.generate()` -- the reference truth, "
        "loaded from `tests/data/baseline_de_en.json` (no re-running of "
        "transformers is required at report time)."
    )
    lines.append(
        "2. The self-contained ONNX model emitted by "
        "`teradata_opus_translate.convert_model`, executed with `onnxruntime` "
        "on CPU."
    )
    lines.append("")
    lines.append(
        "Both outputs are decoded back to text via the extracted fast "
        "tokenizer (the same `tokenizer.json` BYOM consumes) and compared "
        "at token-ID level after the EOS normalisation described below."
    )
    lines.append("")
    lines.append(
        "**There is no automatic pass/fail threshold.** This report is for "
        "human review per Issue #6. Use it to decide whether to proceed to "
        "in-Teradata parity (#8), to investigate divergences, or to revisit "
        "the converter / tokenizer."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- **Model:** `{h['model_id']}`")
    if h.get("model_revision"):
        lines.append(f"- **Model revision (HF hub SHA):** `{h['model_revision']}`")
    lines.append(f"- **Test set:** `{h['test_set_source']}` ({s['total']} sentences)")
    lines.append(f"- **Test set licence:** {h.get('test_set_license', '(unknown)')}")
    lines.append(
        f"- **Reference baseline:** `{h['baseline_source']}` (sha256 `{h['baseline_sha256']}`)"
    )
    lines.append(f"- **Report generated:** {h['generated_at']}")
    lines.append(f"- **Schema version:** {h['schema_version']}")
    lines.append(
        f"- **Versions:** transformers {h.get('transformers_version', '?')}, "
        f"tokenizers {h.get('tokenizers_version', '?')}, "
        f"torch {h.get('torch_version', '?')}, "
        f"onnxruntime {h.get('onnxruntime_version', '?')}, "
        f"Python {h.get('python_version', '?')}."
    )
    lines.append("")
    lines.append("### Generation parameters")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(h["generation_params"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### EOS normalisation")
    lines.append("")
    lines.append(
        "`MarianMTModel.generate()` emits a trailing EOS token (id 0) before "
        "right-padding to `max_length`; the ONNX `com.microsoft.BeamSearch` "
        "contrib op omits the trailing EOS and right-pads the same way. "
        "Both decode to identical text under `skip_special_tokens=True`. "
        "The token-ID comparison below normalises both sequences by "
        "stripping trailing pad tokens and a single trailing EOS, so the "
        "comparison reflects real model output -- not a cosmetic boundary "
        "marker convention. This is the same normalisation used by "
        "`tests/test_converter_smoke.py` (issue #2)."
    )
    lines.append("")
    lines.append("### Disabled generation features (per Issue #17)")
    lines.append("")
    lines.append(
        "The `com.microsoft.BeamSearch` op does not implement "
        "`bad_words_ids`, `forced_eos_token_id`, or `renormalize_logits` -- "
        "all of which are enabled by default in the upstream "
        "`generation_config.json` shipped with `opus-mt-de-en`. To make the "
        "two backends comparable, those features are explicitly disabled on "
        "the transformers side when the reference baseline is generated. "
        'This means the comparison answers **"do the encoder + decoder '
        'weights and graph topology survive ONNX export"**, not **"does '
        "ONNX BeamSearch match transformers' default generation behaviour\"**."
    )
    lines.append("")
    lines.append("## Per-sentence comparison")
    lines.append("")
    lines.append(
        "Token-level differences (when `Match` is `no`) are summarised in "
        "the `Diff` column. Full normalised token-ID sequences for any "
        "mismatch are listed in the appendix at the end of this document."
    )
    lines.append("")
    lines.append(
        "| ID | Domain | Source (DE) | transformers (EN) | onnxruntime (EN) | Match | Diff |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        diff_txt = _format_diff_short(r)
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(r["id"]),
                    _md_cell(r["domain"]),
                    _md_cell(r["source_text"]),
                    _md_cell(r["transformers_text"]),
                    _md_cell(r["onnx_text"]),
                    "yes" if r["match"] else "**no**",
                    _md_cell(diff_txt),
                ]
            )
            + " |"
        )
    lines.append("")

    mismatches = [r for r in rows if not r["match"]]
    if mismatches:
        lines.append("## Mismatch appendix")
        lines.append("")
        lines.append(
            f"{len(mismatches)} sentence(s) produced token-ID-different "
            "output. Each entry below shows the full *normalised* sequence "
            "(post EOS-strip) and the first divergence position."
        )
        lines.append("")
        for r in mismatches:
            lines.append(f"### {r['id']} ({r['domain']})")
            lines.append("")
            lines.append(f"- **DE:** {r['source_text']}")
            lines.append(f"- **transformers text:** {r['transformers_text']!r}")
            lines.append(f"- **onnxruntime text:** {r['onnx_text']!r}")
            lines.append(
                f"- **transformers ids ({r['transformers_length_normalized']}):**"
                f" `{r['transformers_token_ids_normalized']}`"
            )
            lines.append(
                f"- **onnxruntime ids ({r['onnx_length_normalized']}):**"
                f" `{r['onnx_token_ids_normalized']}`"
            )
            lines.append(
                f"- **First divergence at position {r['divergence_index']}:** "
                f"transformers="
                f"{_fmt_tok(r['transformers_token_at_divergence'], r['transformers_piece_at_divergence'])}"
                f" vs onnxruntime="
                f"{_fmt_tok(r['onnx_token_at_divergence'], r['onnx_piece_at_divergence'])}"
            )
            lines.append("")
    else:
        lines.append("## Mismatch appendix")
        lines.append("")
        lines.append(
            "No mismatches: every sentence produced token-ID-equal output after EOS normalisation."
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_report_markdown(artifact: dict[str, Any], path: str | Path) -> Path:
    """Render the Markdown report and write it to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report_markdown(artifact), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md_cell(text: str) -> str:
    """Escape a value for safe inclusion in a Markdown table cell.

    Replaces pipes (which would close the cell), collapses newlines
    (which would break the row), and trims runs of whitespace so the
    table renders compactly on GitHub.
    """
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\\", "\\\\")
    s = s.replace("|", "\\|")
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    # Collapse repeated whitespace
    s = " ".join(s.split())
    return s


def _format_diff_short(row: dict[str, Any]) -> str:
    """One-line human description of a row's diff.

    Empty for matching rows; for non-matches captures the divergence
    index, the offending tokens, and any length difference.
    """
    if row["match"]:
        return ""
    div = row["divergence_index"]
    hf = _fmt_tok(row["transformers_token_at_divergence"], row["transformers_piece_at_divergence"])
    onnx = _fmt_tok(row["onnx_token_at_divergence"], row["onnx_piece_at_divergence"])
    bits = [f"pos {div}: tx={hf} vs onnx={onnx}"]
    hf_len = row["transformers_length_normalized"]
    onnx_len = row["onnx_length_normalized"]
    if hf_len != onnx_len:
        bits.append(f"lengths {hf_len} vs {onnx_len}")
    return "; ".join(bits)


def _fmt_tok(tok_id: int | None, piece: str | None) -> str:
    """Format a (possibly absent) divergence token for the report."""
    if tok_id is None:
        return "<end>"
    if piece is None:
        return f"id={tok_id}"
    return f"id={tok_id} ({piece!r})"


def utc_now_iso() -> str:
    """Wall-clock ISO-8601 timestamp in UTC, second precision."""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
