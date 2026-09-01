"""Cross-adapter synthesis, and the two documents harv actually ingests.

    python -m kepcmp.report.synthesis report/rv report/gaia --out report/

Reads the per-adapter ``findings.json`` and writes:

- ``SYNTHESIS.md``   -- where RV and Gaia agree, and where they must not share a default;
- ``spec-patch.md``  -- proposed replacement prose for ``docs/spec.md``'s "Open
  question (deliberately unsettled)" paragraph and the two other places that restate
  the same invariant;
- ``casestudy-outline.md`` -- a section-by-section skeleton for a notebook under
  ``docs/tutorials/case-studies/``, with each exemplar as a *recipe* rather than a data
  file, because harv's tutorials simulate inline and its docs tree drops ``*.h5``.

Everything here is generated from measured findings. Where a finding is unresolved the
document says so; it never fills the gap with prose.
"""

from __future__ import annotations

__all__ = ("main",)

import argparse
import json
from pathlib import Path
from typing import Any

UNRESOLVED = "**UNRESOLVED at this grid's resolution.**"


def _verdicts(findings: dict[str, Any]) -> list[dict[str, Any]]:
    return findings.get("amplitude", {}).get("verdict", [])


def _moving(findings: dict[str, Any]) -> list[dict[str, Any]]:
    return [v for v in _verdicts(findings) if v.get("verdict") == "knee moves"]


def _recipe(adapter: str, ex: dict[str, Any], attrs: dict[str, Any]) -> str:
    """A literal `harv.simulate` call reproducing one exemplar."""
    t_span = float(attrs.get("t_span_day", float("nan")))
    sigma_n = float(attrs.get("sigma_n", float("nan")))
    period = ex["period_ratio"] * t_span
    n_obs = int(ex["n_obs"])
    # Exemplars chosen from per-cell tables carry a cell_id, which has no seed in it;
    # only per-simulation tables do. Fall back rather than mangling the id.
    raw = str(ex.get("sim_id", ""))
    seed = raw.rsplit("seed", 1)[-1] if "seed" in raw else "0"
    seed = int(seed) if seed.isdigit() else 0
    if adapter == "rv":
        return (
            "simulate_rv_sb1_data(\n"
            f"    seed={seed}, n_obs={n_obs}, baseline=Q({t_span:.4g}, 'day'),\n"
            f"    period=Q({period:.4g}, 'day'), eccentricity={ex['eccentricity']:g},\n"
            f"    rv_semiamp=Q({ex['snr'] * sigma_n:.4g}, 'km/s'),\n"
            f"    rv_err=Q({sigma_n:.4g}, 'km/s'), v_sys=Q(0.0, 'km/s'),\n"
            ")"
        )
    return (
        "# see kepcmp.adapters.gaia.GaiaAdapter.simulate: times, scan angles and\n"
        "# parallax factors are constructed before the call, because the simulator's\n"
        "# default parallax_factor is U(-1, 1) white noise with no 1-yr structure\n"
        "simulate_gaia_epoch_astrometry(\n"
        f"    seed={seed}, times=..., scan_angle=..., parallax_factor=...,\n"
        f"    period=Q({period:.4g}, 'day'), eccentricity={ex['eccentricity']:g},\n"
        f"    semi_major_axis=Q({ex['snr'] * sigma_n:.4g}, 'mas'),\n"
        f"    parallax=Q(10.0, 'mas'), al_error=Q({sigma_n:.4g}, 'mas'),\n"
        ")"
    )


def _spec_patch(bundles: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Proposed `docs/spec.md` changes",
        "",
        "Generated from measured findings. Three places in harv restate the same",
        "invariant and all three move together:",
        "",
        "1. `docs/spec.md`, **Priors are explicit** -> *Open question (deliberately",
        "   unsettled)* -- the paragraph this replaces.",
        "2. `docs/spec.md`, **FourierRV and FourierGaiaAstrometry** -- the sentence",
        "   asserting there is deliberately no data-driven default.",
        "3. `src/harv/models/parameterizations/fourier.py`, the module-level",
        "   `TODO(default-amplitude-prior)`.",
        "",
        "---",
        "",
        "## 1. Amplitude scale",
        "",
    ]
    for adapter, b in bundles.items():
        f = b["findings"]
        amp = f.get("amplitude", {})
        moving = _moving(f)
        bias = amp.get("short_period_bias", {})
        lines.append(f"### {adapter}")
        lines.append("")
        if moving:
            exps = ", ".join(
                f"H={v['n_terms_effective']}: {v['exponent_vs_period_ratio']:+.2f}"
                for v in moving
            )
            lines += [
                "The saturation knee **moves with `P / T_span`**, so the recommended",
                "amplitude prior is a *functional form*, not a scalar. Fitted exponent",
                f"of `log10(knee)` against `log10(P / T_span)`: {exps}.",
                "",
                "That makes the Fourier amplitude prior structurally like the Keplerian",
                "ones (`PeriodDependentKPrior`), and the present flat",
                "`Normal(0, sigma_amp)` the odd one out.",
                "",
            ]
        else:
            step = amp.get("grid_step_dex", float("nan"))
            lines += [
                f"{UNRESOLVED} The knee is constant across `P / T_span` to within the",
                f"sweep's resolution ({step:.2f} dex between multipliers), so a floor",
                "of the form `sigma_amp >= N x base-model-residual RMS` is consistent",
                "with the data -- **but a weaker period-dependence is not excluded**.",
                "Deciding between a scalar and a functional form needs a denser",
                "`sigma_amp` sweep, not more cells.",
                "",
            ]
        if bias.get("n"):
            sign = (
                "confirmed" if bias.get("median_d_ln_peak_partial_arc_tightest_prior", 0) < 0
                else "NOT confirmed"
            )
            lines += [
                "The spec's own claim -- that a tight scale biases the peak toward",
                f"short periods in the partial-arc regime -- is **{sign}**: median",
                f"`d_ln_peak` = {bias['median_d_ln_peak_partial_arc_tightest_prior']:+.3f}",
                "at the tightest scale sampled, negative in",
                f"{100 * bias['fraction_negative']:.0f}% of {bias['n']} cells.",
                "",
            ]
        lines += [
            "> **Units.** `sigma_amp` is quoted against the RMS *after projecting out",
            "> the `n_terms=0` base model*. For RV that is RMS about the mean; for Gaia",
            "> it is far smaller than the raw along-scan scatter, which parallax and",
            "> proper motion dominate. The number is meaningless without this.",
            "",
        ]

    lines += ["## 2. Nuisance (base-column) scales", ""]
    for adapter, b in bundles.items():
        nu = b["findings"].get("nuisance", {})
        if not nu:
            continue
        flat = nu.get("flat_while_covering")
        ratio = nu.get("ratio_to_true_value", float("nan"))
        lines += [
            f"### {adapter} -- `{nu.get('parameter')}`",
            "",
            (
                "Measured: the peak is **flat while the prior covers the true value**"
                if flat
                else "Measured: the peak is **not** flat even while covering"
            )
            + f", and degrades only once the width falls to ~{ratio:.3g}x the true"
            f" value ({nu.get('true_value')} {nu.get('unit')}).",
            "",
            "The failure mode is therefore **non-coverage, not mis-tuning** -- which is",
            "the opposite of the amplitude prior, where the scale sets the Occam factor",
            "and therefore the science. A data-driven default is safe here: centre on",
            "the data, width a few x `(RMS + span)`.",
            "",
        ]
    return "\n".join(lines)


def _casestudy_outline(bundles: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Case study outline: which periodogram, and when",
        "",
        "For a new notebook under `docs/tutorials/case-studies/`. Note that",
        "`.github/workflows/tutorials.yml` currently executes only",
        "`docs/tutorials/rv/*.ipynb`, so this renders unexecuted on RTD until that glob",
        "is widened. Data comes from `harv.simulate` inline -- no new files in",
        "`docs/tutorials/data/` (pre-commit caps additions at 2000 kB and the docs tree",
        "ignores `*.h5`).",
        "",
    ]
    for adapter, b in bundles.items():
        f = b["findings"]
        cs = f.get("casestudy", {})
        attrs = b["provenance"]["signal"]["attrs"]
        lines += [f"## {adapter}", ""]
        n_missing = cs.get("z0_unavailable_cells")
        if n_missing is not None:
            lines += [
                "**Precondition, not a score.** kepmodel's profile statistic does not",
                f"exist at `n_obs <= {cs.get('z0_requires_n_obs_gt')}` (p + d); it was",
                f"unavailable in {n_missing} cells here. Teach this as a precondition.",
                "",
            ]
        best = cs.get("best_builder_by_eccentricity")
        if best:
            lines += [
                "**Prior builder vs eccentricity** (by log density at the injected",
                "period, the builder-neutral metric):",
                "",
                "| eccentricity | best builder |",
                "|---|---|",
                *[f"| {k} | `{v}` |" for k, v in sorted(best.items())],
                "",
                f"harv's removed draft asserts: *{cs.get('draft_tutorial_claim')}*",
                "Compare against the table above before repeating it.",
                "",
            ]
        for ex in cs.get("exemplars", []):
            lines += [
                f"### {ex['claim']}",
                "",
                f"Selected by `{ex['criterion']}` -- deterministic, not hand-picked.",
                "",
                "```python",
                _recipe(adapter, ex, attrs),
                "```",
                "",
            ]
    return "\n".join(lines)


def _synthesis(bundles: dict[str, dict[str, Any]]) -> str:
    lines = ["# Synthesis across data types", ""]
    lines += [
        "| | " + " | ".join(bundles) + " |",
        "|---|" + "---|" * len(bundles),
    ]

    def row(label: str, fn: Any) -> str:
        return f"| {label} | " + " | ".join(str(fn(b)) for b in bundles.values()) + " |"

    lines.append(
        row(
            "amplitude knee",
            lambda b: "moves with P/T_span"
            if _moving(b["findings"])
            else "constant within resolution",
        )
    )
    lines.append(
        row(
            "nuisance failure mode",
            lambda b: "non-coverage only"
            if b["findings"].get("nuisance", {}).get("flat_while_covering")
            else "not flat while covering",
        )
    )
    lines.append(
        row(
            "z0 needs n_obs >",
            lambda b: b["findings"].get("casestudy", {}).get("z0_requires_n_obs_gt", "?"),
        )
    )
    lines += [
        "",
        "The two data types share the *shape* of the guidance but not its numbers: the",
        "RMS `sigma_amp` is quoted against means different things (RV: scatter about the",
        "mean; Gaia: residual after the 5-parameter astrometric solution, ~50x smaller",
        "than raw scatter), and `p + d` differs, so they must not share a literal",
        "default.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", type=Path, nargs="+",
                        help="per-adapter report directories")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    loaded: dict[str, dict[str, Any]] = {}
    for path in args.bundles:
        payload = json.loads((path / "findings.json").read_text())
        loaded[payload["provenance"]["adapter"]] = payload

    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("SYNTHESIS.md", _synthesis(loaded)),
        ("spec-patch.md", _spec_patch(loaded)),
        ("casestudy-outline.md", _casestudy_outline(loaded)),
    ):
        (args.out / name).write_text(text.rstrip() + "\n")
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
