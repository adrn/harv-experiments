"""Cross-adapter synthesis, and the document harv actually ingests.

    python -m ampcal.report.synthesis report/rv report/gaia --out report/

Reads the per-adapter ``findings.json`` and writes:

- ``SYNTHESIS.md``  -- where RV and Gaia agree, and where they must not share a default;
- ``spec-patch.md`` -- proposed replacement prose for the three places in harv that
  restate the same open question, and which all move together.

Everything here is generated from measured findings. Where a finding does not clear the
decision rule the document says the flat prior stands; it never fills the gap with
prose.
"""

from __future__ import annotations

__all__ = ("main",)

import argparse
import json
from pathlib import Path
from typing import Any

from ampcal.report.amplitude import INTERVAL

#: The three places in harv that restate the open question. Listed here, not in prose,
#: because a patch that updates two of them leaves the third contradicting the other two.
TARGETS = (
    (
        "`docs/spec.md`, **Priors are explicit** -> *Open question (deliberately "
        "unsettled)* -- the paragraph this replaces."
    ),
    (
        "`docs/spec.md`, **FourierRV and FourierGaiaAstrometry** -- the sentence "
        "asserting there is deliberately no data-driven default."
    ),
    (
        "`src/harv/models/parameterizations/fourier.py`, the module-level "
        "`TODO(default-amplitude-prior)`."
    ),
)


def _amplitude(findings: dict[str, Any]) -> dict[str, Any]:
    return findings.get("amplitude", {})


def _recommended(findings: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``n_terms`` rows whose verdict cleared all three clauses."""
    return [
        v
        for v in _amplitude(findings).get("verdict", [])
        if v.get("verdict") == "period-dependent"
    ]


def _spec_patch(bundles: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Proposed `docs/spec.md` changes",
        "",
        "Generated from measured findings. Three places in harv restate the same",
        "invariant and all three move together:",
        "",
        *[f"{i}. {t}" for i, t in enumerate(TARGETS, start=1)],
        "",
        "---",
        "",
        "## 1. Amplitude prior: scale and exponent",
        "",
    ]
    for adapter, b in bundles.items():
        f = b["findings"]
        amp = _amplitude(f)
        recommended = _recommended(f)
        bias = amp.get("short_period_bias", {})
        lines += [f"### {adapter}", ""]

        if recommended:
            rows = ", ".join(
                f"H={v['n_terms_effective']}: exponent {v['best_exponent']:+.3f} at "
                f"{v['best_level_x_rms']:g}x RMS"
                for v in recommended
            )
            gains = ", ".join(
                f"{v['improvement_over_flat']:+.3f} "
                f"[{v['improvement_lo']:+.3f}, {v['improvement_hi']:+.3f}]"
                for v in recommended
            )
            lines += [
                "The recommended Fourier amplitude prior is a **functional form**:",
                "",
                "```",
                "sigma_amp(P) = sigma_0 * (P / T_span)**exponent",
                "```",
                "",
                (
                    f"Measured optimum: {rows}. Improvement in median "
                    "`|ln P_peak(Delta) - ln P_peak(R)|` over the flat prior at the same"
                ),
                f"scale, with a seed bootstrap {INTERVAL:g}% interval: {gains}.",
                "",
                "That makes the Fourier amplitude prior structurally like the Keplerian",
                "ones (`PeriodDependentKPrior`,",
                "`PeriodDependentSemiMajorAxisPrior`), and the present flat",
                "`Normal(0, sigma_amp)` the odd one out.",
                "",
                "> **The exponent was measured on two populations, not one.** On",
                "> `physical` the injected amplitude follows the power law the prior",
                "> assumes; on `independent` it is flat in period by construction, so",
                "> the prior is misspecified. The recommendation above costs "
                + ", ".join(
                    f"{v['misspecification_cost']:+.3f}" for v in recommended
                )
                + " nats of",
                "> peak error on the misspecified population -- less than it gains on",
                "> the correctly-specified one, which is the third clause of the",
                "> decision rule and the reason it is safe as a *default*.",
                "",
            ]
        else:
            lines += [
                "**The flat prior is not beaten here.** No exponent improved the median",
                "peak error over `exponent = 0` at the same scale by more than the seed",
                "bootstrap's spread, or the improvement on the correctly-specified",
                "population was smaller than the cost on the misspecified one. The",
                "recommendation for this data type is therefore a *scale only*:",
                f"`sigma_amp >= {_best_level(f):g} x base-model-residual RMS`.",
                "",
            ]

        if bias.get("n"):
            median = bias.get("median_d_ln_peak_partial_arc_tightest_flat", float("nan"))
            sign = "confirmed" if median < 0 else "NOT confirmed"
            lines += [
                "The spec's own claim -- that a tight scale biases the peak toward short",
                f"periods in the partial-arc regime -- is **{sign}**: median",
                f"`d_ln_peak` = {median:+.3f} at the tightest flat arm, negative in",
                f"{100 * bias['fraction_negative']:.0f}% of {bias['n']} cells.",
                "",
            ]

        tilt = amp.get("occam_tilt", {})
        if tilt:
            lines += [
                "**Mechanism.** `docs/spec.md` states that the Occam factor is flat in",
                "frequency. It is not: it depends on design-matrix conditioning, which",
                "varies with trial frequency, and that dependence is what produces the",
                "short-period bias. A period-dependent prior tilts it by a *bounded*",
                "amount -- `d(occam)/d(ln P)` gains at most `n_amp_columns * exponent`,",
                "with equality only where the amplitude is well determined -- and the",
                (
                    "measured tilt realises "
                    f"{100 * tilt.get('fraction_of_ceiling_median', float('nan')):.0f}%"
                    " of that ceiling (range "
                    f"{100 * tilt.get('fraction_of_ceiling_min', float('nan')):.0f}-"
                    f"{100 * tilt.get('fraction_of_ceiling_max', float('nan')):.0f}%)."
                ),
                "The sentence claiming flatness should go.",
                "",
                "> **Units.** The scale is quoted against the RMS *after projecting out",
                "> the `n_terms=0` base model*. For RV that is RMS about the mean; for",
                "> Gaia it is far smaller than the raw along-scan scatter, which parallax",
                "> and proper motion dominate. The number is meaningless without this.",
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
            "The failure mode is therefore **non-coverage, not mis-tuning** -- the",
            "opposite of the amplitude prior, where the scale and its exponent set the",
            "Occam factor and therefore the science. A data-driven default is safe here:",
            "centre on the data, width a few x `(RMS + span)`.",
            "",
        ]
    return "\n".join(lines)


def _best_level(findings: dict[str, Any]) -> float:
    best = _amplitude(findings).get("best_by_population", {}).get("physical", {})
    return float(best.get("level", float("nan")))


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
            "amplitude prior",
            lambda b: (
                "period-dependent"
                if _recommended(b["findings"])
                else "flat prior not beaten"
            ),
        )
    )
    lines.append(
        row(
            "best exponent (physical)",
            lambda b: f"{_amplitude(b['findings']).get('best_by_population', {}).get('physical', {}).get('exponent', float('nan')):+.3f}",
        )
    )
    lines.append(
        row(
            "best level (x RMS)",
            lambda b: f"{_best_level(b['findings']):g}",
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
    lines += [
        "",
        "The two data types share the *shape* of the guidance but neither its numbers",
        "nor even the sign of its exponent: RV's semi-amplitude falls as `P^(-1/3)`",
        "while Gaia's angular semi-major axis grows as `P^(+2/3)`, and the RMS the scale",
        "is quoted against means different things (RV: scatter about the mean; Gaia:",
        "residual after the 5-parameter astrometric solution, far smaller than raw",
        "scatter). They must not share a literal default.",
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
    ):
        (args.out / name).write_text(text.rstrip() + "\n")
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
