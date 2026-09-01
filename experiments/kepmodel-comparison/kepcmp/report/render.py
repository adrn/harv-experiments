"""Assemble analysis sections into the bundle.

A section carries its own prose, the tables it cites, its figures, and the
machine-readable findings it establishes. Rendering is deliberately dumb: it writes
what the analyses produced and never computes a number of its own, so nothing in the
report can disagree with the CSV beside it.
"""

from __future__ import annotations

__all__ = ("Section", "write_bundle")

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

MAX_INLINE_ROWS = 40
"""Above this a table is written to CSV and linked rather than inlined."""


@dataclass
class Section:
    """One analysis's contribution to the report."""

    key: str
    title: str
    body: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    findings: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)


def _md_table(frame: pd.DataFrame, float_fmt: str = "{:.4g}") -> str:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "" if pd.isna(v) else float_fmt.format(v)
            )
    return out.to_markdown(index=True)


def write_bundle(
    sections: list[Section],
    out_dir: Path,
    *,
    title: str,
    preamble: str,
    provenance: dict[str, Any],
) -> Path:
    """Write ``REPORT.md``, ``findings.json``, ``tables/`` and ``figures/``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    lines = [f"# {title}", "", preamble.strip(), ""]
    findings: dict[str, Any] = {}
    caveats: list[str] = []

    for sec in sections:
        lines += [f"## {sec.title}", "", sec.body.strip(), ""]

        for name, frame in sec.tables.items():
            stem = f"{sec.key}__{name}"
            frame.to_csv(out_dir / "tables" / f"{stem}.csv")
            if len(frame) <= MAX_INLINE_ROWS:
                lines += [f"**{name}**", "", _md_table(frame), ""]
            else:
                link = f"[`tables/{stem}.csv`](tables/{stem}.csv)"
                lines += [f"**{name}** -- {len(frame)} rows, see {link}", ""]

        for name, fig in sec.figures.items():
            stem = f"{sec.key}__{name}"
            fig.savefig(out_dir / "figures" / f"{stem}.png", dpi=150,
                        bbox_inches="tight")
            lines += [f"![{name}](figures/{stem}.png)", ""]

        if sec.findings:
            findings[sec.key] = sec.findings
        caveats += sec.caveats

    if caveats:
        lines += ["## Limits of this report", ""]
        lines += [f"- {c}" for c in caveats]
        lines.append("")

    lines += [
        "## Provenance",
        "",
        "```json",
        json.dumps(provenance, indent=2, default=str),
        "```",
        "",
    ]

    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines))
    (out_dir / "findings.json").write_text(
        json.dumps(
            {"provenance": provenance, "findings": findings, "caveats": caveats},
            indent=2,
            default=str,
        )
        + "\n"
    )
    return report
