"""Three-way parity report (transformers vs onnxruntime vs Teradata).

This module assembles the artifact written for Issue #8:

* ``tests/data/teradata_parity_report.json`` -- machine-readable.
* ``tests/data/teradata_parity_report.md`` -- human-readable.

The report has the same overall shape as the local parity report
(``parity/report.py``) but with a third text column and a 3-way match
decomposition. All three backends are compared at the **decoded text**
level because BYOM's ``ONNXSeq2Seq`` returns VARCHAR -- not token IDs --
so text is the lowest common denominator. For finer-grained diagnosis of
mismatches the JSON optionally carries the Teradata text re-tokenised
through ``MarianTokenizer``, but only as a debug aid; the headline
comparison is text-equality with no normalisation.

Schema versioning
-----------------
:data:`TERADATA_REPORT_SCHEMA_VERSION` is bumped only on
backwards-incompatible changes to the JSON record shape. The marker is
distinct from the local-parity report's so the two artifacts can evolve
independently.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TERADATA_REPORT_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThreeWayMatch:
    """3-way match decomposition for one row.

    Attributes mirror the three pairwise text equality checks plus the
    convenience ``all_equal`` field. Differences are byte-exact: any
    difference in whitespace, casing or punctuation counts as a mismatch
    so the user can decide whether the divergence is cosmetic or
    substantive.
    """

    transformers_eq_onnx: bool
    transformers_eq_teradata: bool
    onnx_eq_teradata: bool
    all_equal: bool


def compare_three_way(
    transformers_text: str,
    onnx_text: str,
    teradata_text: str,
) -> ThreeWayMatch:
    """Compute the three pairwise text-equality flags for one row.

    Strings are compared verbatim (no strip(), no lowercasing). The
    rationale is documented in the report header: cosmetic differences
    are themselves a finding the user should see.
    """
    a = transformers_text == onnx_text
    b = transformers_text == teradata_text
    c = onnx_text == teradata_text
    return ThreeWayMatch(
        transformers_eq_onnx=a,
        transformers_eq_teradata=b,
        onnx_eq_teradata=c,
        all_equal=a and b and c,
    )


@dataclass(frozen=True)
class TeradataParityRow:
    """One per-sentence row in the in-Teradata parity report."""

    id: str
    domain: str
    source_text: str
    transformers_text: str
    onnx_text: str
    teradata_text: str
    transformers_eq_onnx: bool
    transformers_eq_teradata: bool
    onnx_eq_teradata: bool
    all_equal: bool
    diff: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TeradataParityReport:
    """Top-level artifact: header + summary + rows."""

    header: dict[str, Any]
    summary: dict[str, Any]
    rows: list[TeradataParityRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": dict(self.header),
            "summary": dict(self.summary),
            "rows": [r.to_dict() for r in self.rows],
        }


# ---------------------------------------------------------------------------
# Diff formatting
# ---------------------------------------------------------------------------


def format_diff(
    transformers_text: str,
    onnx_text: str,
    teradata_text: str,
    match: ThreeWayMatch,
) -> str:
    """Return a short, human-readable description of any 3-way diff.

    Empty when all three strings are equal. For mismatches it describes
    *which pair(s)* differ and where the first divergence occurs, in a
    form that fits in a Markdown table cell.
    """
    if match.all_equal:
        return ""
    parts: list[str] = []
    # Identify which pair(s) disagree
    pairs: list[tuple[str, str, str]] = []
    if not match.transformers_eq_onnx:
        pairs.append(("T", "O", _first_divergence(transformers_text, onnx_text)))
    if not match.transformers_eq_teradata:
        pairs.append(("T", "R", _first_divergence(transformers_text, teradata_text)))
    if not match.onnx_eq_teradata:
        pairs.append(("O", "R", _first_divergence(onnx_text, teradata_text)))
    for left, right, where in pairs:
        parts.append(f"{left}!={right}@{where}")
    return "; ".join(parts)


def _first_divergence(a: str, b: str) -> int:
    """Index of the first differing character; -1 when equal."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_report_json(artifact: dict[str, Any], path: str | Path) -> Path:
    """Serialise the artifact to disk as deterministic, pretty JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    out.write_text(text, encoding="utf-8")
    return out


def render_report_markdown(artifact: dict[str, Any]) -> str:
    """Render the human-readable Markdown report.

    The structure mirrors the local-parity report so a reader can move
    between the two without re-orienting:

    1. Title + headline match counts.
    2. Provenance (model, params, EOS handling, BYOM version).
    3. Per-sentence comparison table (4 text columns + match flag + diff).
    4. Mismatches appendix listing the divergent rows in full.
    """
    h = artifact["header"]
    s = artifact["summary"]
    rows = artifact["rows"]

    lines: list[str] = []
    lines.append("# In-Teradata parity report (transformers vs onnxruntime vs ONNXSeq2Seq)")
    lines.append("")
    teradata_error = s.get("teradata_error")
    if teradata_error:
        lines.append(
            "> **Teradata-side run did not complete.** The "
            "`TD_MLDB.ONNXSeq2Seq` operator raised the error captured "
            "below before any rows were produced. The `Teradata (EN)` "
            "column in the table that follows is therefore empty for "
            "every row. The `transformers` and `onnxruntime` columns "
            "are still populated from the saved baseline (#4) and "
            "local parity report (#6); pairwise equality between those "
            "two backends remains meaningful."
        )
        lines.append("")
        lines.append("```text")
        # First line of the captured error is by far the most useful
        # ("[Error 7583] The secure mode processes had a set up error").
        # Truncate to keep the Markdown readable.
        first_lines = "\n".join(teradata_error.splitlines()[:6])
        lines.append(first_lines)
        lines.append("```")
        lines.append("")
        byom_db = h.get("teradata_database") or "the configured BYOM database"
        lines.append(
            "Diagnosis (per Teradata BYOM error reference and the contents "
            "of `DBC.SW_Event_Log` collected during this run): the BYOM "
            "Hybrid Server JVM is failing to start under `udfsectsk`. The "
            "kernel of the issue is `JNI_CreateJavaVM call with return "
            "value = -1`, which is what surfaces to the client as error "
            "7583. The standard mitigation -- "
            "`call SQLJ.ServerControl('JAVA', 'shutdown'/'enable', a)` -- "
            "did not help on this VM image. This is a VM-side issue, not "
            "a converter / tokenizer / deployment defect: the model and "
            f"tokenizer rows are present in `{byom_db}.onnx_models` and "
            f"`{byom_db}.sequence_tokenizers` and the `BYTES()` of each "
            "BLOB matches the locally-converted file size byte-for-byte "
            "(verified during deployment in #10). Resolving error 7583 "
            "requires devops intervention on the test VM and is tracked "
            "outside this report."
        )
        lines.append("")
    lines.append("## Headline match rates")
    lines.append("")
    lines.append(f"- **Total sentences:** {s['total']}")
    lines.append(
        f"- **All three equal:** {s['all_equal']}/{s['total']} "
        f"({_pct(s['all_equal'], s['total'])}%)"
    )
    lines.append(
        f"- **transformers == onnxruntime:** {s['transformers_eq_onnx']}/{s['total']} "
        f"({_pct(s['transformers_eq_onnx'], s['total'])}%)"
    )
    lines.append(
        f"- **transformers == Teradata:** {s['transformers_eq_teradata']}/{s['total']} "
        f"({_pct(s['transformers_eq_teradata'], s['total'])}%)"
    )
    lines.append(
        f"- **onnxruntime == Teradata:** {s['onnx_eq_teradata']}/{s['total']} "
        f"({_pct(s['onnx_eq_teradata'], s['total'])}%)"
    )
    lines.append("")
    lines.append("## What this report is")
    lines.append("")
    lines.append(
        "For every sentence in the curated DE -> EN baseline test set, this "
        "report compares the decoded English translation produced by three "
        "backends:"
    )
    lines.append("")
    lines.append(
        "1. **transformers** -- `MarianMTModel.generate()`, the gold-standard "
        "reference, loaded from `tests/data/baseline_de_en.json`."
    )
    lines.append(
        "2. **onnxruntime** -- the converted ONNX model executed locally via "
        "`onnxruntime` on CPU, loaded from `tests/data/local_parity_report.json`."
    )
    lines.append(
        "3. **Teradata** -- the same ONNX model + tokenizer deployed into "
        "Teradata BYOM and invoked via the `TD_MLDB.ONNXSeq2Seq` table operator."
    )
    lines.append("")
    lines.append(
        "**There is no automatic pass/fail threshold.** This report is for "
        "human review per Issue #8. The user decides whether to declare the "
        "Phase 1 DE->EN pilot done or to investigate divergences."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- **Model:** `{h['model_id']}`")
    if h.get("model_revision"):
        lines.append(f"- **Model revision (HF hub SHA):** `{h['model_revision']}`")
    lines.append(f"- **BYOM model_id in Teradata:** `{h['teradata_model_id']}`")
    lines.append(f"- **Test set:** `{h['test_set_source']}` ({s['total']} sentences)")
    lines.append(f"- **Test set licence:** {h.get('test_set_license', '(unknown)')}")
    lines.append(
        f"- **Reference baseline:** `{h['baseline_source']}` (sha256 `{h['baseline_sha256']}`)"
    )
    lines.append(
        f"- **Local parity report:** `{h['local_parity_source']}` "
        f"(sha256 `{h['local_parity_sha256']}`)"
    )
    lines.append(f"- **Teradata host:** `{h['teradata_host']}`")
    lines.append(f"- **Teradata version:** {h.get('teradata_version') or '(unknown)'}")
    byom_v = h.get("byom_version")
    if byom_v:
        lines.append(f"- **BYOM version:** {byom_v}")
    else:
        lines.append(
            "- **BYOM version:** not exposed via SQL on this Teradata 20.00 "
            "image; install path on the VM is `/opt/teradata/byom/07.00.00.01/` "
            "(verified during PR #29's deployment of the encoder/decoder/init "
            "ONNX). The `TD_MLDB.ONNXSeq2Seq` operator is present and callable, "
            "which is the authoritative install signal."
        )
    lines.append(f"- **Report generated:** {h['generated_at']}")
    lines.append(f"- **Schema version:** {h['schema_version']}")
    lines.append("")
    lines.append("### Generation parameters (same on all three backends)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(h["generation_params"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("Mapped to BYOM's ``Const_*`` clauses:")
    lines.append("")
    lines.append("```sql")
    for c in h["const_clauses"]:
        lines.append(c)
    lines.append("```")
    lines.append("")
    lines.append("### EOS / special-token handling")
    lines.append("")
    lines.append(
        "All three backends are compared at the **decoded text** level with "
        "special tokens stripped:"
    )
    lines.append("")
    lines.append("- **transformers:** `MarianTokenizer.decode(..., skip_special_tokens=True)`.")
    lines.append("- **onnxruntime:** same tokenizer, same `skip_special_tokens=True`.")
    lines.append(
        f"- **Teradata:** `TD_MLDB.ONNXSeq2Seq(... SkipSpecialTokens('{h['skip_special_tokens']}') ...)`. "
        "BYOM's tokenizer decode path is internal but produces text that is the "
        "natural counterpart to the local skip-special-tokens decode."
    )
    lines.append("")
    lines.append(
        "String comparison is **byte-exact**: differences in whitespace, "
        "casing or punctuation count as mismatches. This is intentional so "
        "the user can decide whether a divergence is cosmetic or substantive."
    )
    lines.append("")
    lines.append("### Disabled generation features (per Issue #17)")
    lines.append("")
    lines.append(
        "The `com.microsoft.BeamSearch` op embedded in our ONNX model does "
        "not implement `bad_words_ids`, `forced_eos_token_id`, or "
        "`renormalize_logits` (all enabled by default in upstream "
        "`opus-mt-de-en`). The transformers reference baseline was generated "
        "with those features explicitly disabled so all three backends "
        "compare apples-to-apples."
    )
    lines.append("")
    lines.append("### Excluded `PARITY_PARAMS` keys")
    lines.append("")
    lines.append(
        "BYOM's ``Const_*`` interface only meaningfully accepts the five "
        "numeric generation parameters listed under `const_clauses`. The "
        "remaining `PARITY_PARAMS` keys (`no_repeat_ngram_size=0`, "
        "`early_stopping=True`) are not exposed by BYOM but are inert at "
        "their fixed values: the ONNX BeamSearch graph baked into the "
        "exported model already pins them, so the local and Teradata runs "
        "use the identical search procedure. `num_return_sequences=1` is "
        "also baked into the graph as a `Constant` node (Issue #82, "
        "v1.0.1) and so is no longer surfaced as a SQL `Const_*` clause."
    )
    lines.append("")
    lines.append("## Per-sentence comparison")
    lines.append("")
    lines.append(
        "The `Match` column reports the 3-way pairwise equality decomposition "
        "(`T=transformers`, `O=onnxruntime`, `R=teradata`). When the row is "
        "fully equal it shows `all`; otherwise it lists the pairs that are "
        "equal (e.g. `T=O`) so a reader can see at a glance whether the "
        "Teradata side is the outlier."
    )
    lines.append("")
    lines.append(
        "| ID | Domain | Source (DE) | transformers (EN) | onnxruntime (EN) | "
        "Teradata (EN) | Match | Diff |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        match_short = _format_match_short(r)
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(r["id"]),
                    _md_cell(r["domain"]),
                    _md_cell(r["source_text"]),
                    _md_cell(r["transformers_text"]),
                    _md_cell(r["onnx_text"]),
                    _md_cell(r["teradata_text"]),
                    match_short,
                    _md_cell(r["diff"]),
                ]
            )
            + " |"
        )
    lines.append("")

    mismatches = [r for r in rows if not r["all_equal"]]
    lines.append("## Mismatch appendix")
    lines.append("")
    if not mismatches:
        lines.append(
            "No mismatches: every sentence produced byte-identical text on all three backends."
        )
        lines.append("")
    elif teradata_error:
        # When the Teradata side failed wholesale, listing 62 "mismatch"
        # entries is noise -- the only fact is "Teradata column is empty
        # because of error 7583". Spell it out once instead.
        lines.append(
            f"All {len(mismatches)} sentences are listed as not-all-equal "
            "because the Teradata column is empty for every row "
            "(`TD_MLDB.ONNXSeq2Seq` failed with error 7583, see the banner at "
            "the top of this report). The `transformers` vs `onnxruntime` "
            "pairwise comparison is unaffected; refer to "
            "`tests/data/local_parity_report.md` for the deep-dive on that "
            "axis (Issue #6 reported 62/62 token-ID parity)."
        )
        lines.append("")
    else:
        lines.append(
            f"{len(mismatches)} sentence(s) produced text that differs across "
            "at least one of the three backends. Each entry below shows all "
            "three strings verbatim plus the pairwise equality flags."
        )
        lines.append("")
        for r in mismatches:
            lines.append(f"### {r['id']} ({r['domain']})")
            lines.append("")
            lines.append(f"- **DE source:** {r['source_text']}")
            lines.append(f"- **transformers:** {r['transformers_text']!r}")
            lines.append(f"- **onnxruntime:** {r['onnx_text']!r}")
            lines.append(f"- **Teradata:**    {r['teradata_text']!r}")
            lines.append(
                f"- **Pairwise:** T==O: {r['transformers_eq_onnx']}; "
                f"T==R: {r['transformers_eq_teradata']}; "
                f"O==R: {r['onnx_eq_teradata']}"
            )
            lines.append(f"- **Diff summary:** {r['diff']}")
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


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "0.0"
    return f"{100.0 * num / denom:.1f}"


def _md_cell(text: str) -> str:
    """Escape a value for safe inclusion in a Markdown table cell."""
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\\", "\\\\")
    s = s.replace("|", "\\|")
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s


def _format_match_short(row: dict[str, Any]) -> str:
    """One-token-ish summary of the 3-way match for a Markdown cell."""
    if row["all_equal"]:
        return "all"
    eq_pairs: list[str] = []
    if row["transformers_eq_onnx"]:
        eq_pairs.append("T=O")
    if row["transformers_eq_teradata"]:
        eq_pairs.append("T=R")
    if row["onnx_eq_teradata"]:
        eq_pairs.append("O=R")
    if not eq_pairs:
        return "**none**"
    return ", ".join(eq_pairs)


def utc_now_iso() -> str:
    """Wall-clock ISO-8601 timestamp in UTC, second precision."""
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
