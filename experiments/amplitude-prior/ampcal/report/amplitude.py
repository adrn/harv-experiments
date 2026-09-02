"""The deliverable: a scale *and* an exponent for the Fourier amplitude prior.

Every *Keplerian* amplitude in harv already carries a parameter-dependent prior
(``PeriodDependentKPrior``, ``PeriodDependentSemiMajorAxisPrior``,
``ParallaxDependentProperMotionPrior``), while the *Fourier* harmonic amplitude gets a
flat ``Normal(0, sigma_amp)`` with no ``P`` dependence --- although at trial period
``P`` it is measuring the same physical thing. This section decides whether that
asymmetry survives, by sweeping the exponent directly rather than inferring it from
where a constant-prior knee lands.

The decision rule is fixed here, before the numbers, and has three clauses. A
period-dependent default is recommended only if:

1. the best exponent on the **physical** population --- where ``K ~ P^(-1/3)`` really
   holds, so the prior *can* be right --- is non-zero;
2. its improvement over the flat arm at the same level survives a seed bootstrap
   (:data:`N_BOOT` resamples, :data:`INTERVAL`% interval excluding zero);
3. it does not lose more on the **independent** population --- where the prior is
   misspecified by construction --- than it gains on the physical one. A default has to
   survive being wrong about the population, because a real catalogue is a mixture.

Clause 3 is the one the old constant-scale study could not ask at all: its grid built
``K`` independently of ``P`` by construction, so it had only the misspecified
population and no way to know it.
"""

from __future__ import annotations

__all__ = ("best_arm", "build_section", "improvement_bootstrap")

from typing import Any

import numpy as np
import pandas as pd

from ampcal.report.render import Section

N_BOOT = 400
"""Seed bootstrap draws. Resampling is over **seeds**, with multiplicity, because that
is the axis the grid replicates; resampling rows would treat the 24 arms of one
simulation as independent when they share a dataset and a reference curve."""

INTERVAL = 95.0
"""Bootstrap interval width, in percent. **Not 68.** A 16-84 interval excluding zero is
a one-sigma claim: for a genuinely null effect the observed difference exceeds its own
bootstrap sd about a third of the time, so that rule would recommend an exponent on a
third of null grids. ``tests/test_report.py`` plants a null and requires this rule to
refuse it."""

MIN_SEEDS = 4
"""Below this the seed bootstrap is refused outright rather than reported with a
zero-width interval, which a smoke run would otherwise present as a verdict."""

PRIMARY = "abs_d_ln_peak"
"""The metric the verdict is decided on. ``tv`` breaks ties --- on well-sampled cells
every arm recovers the reference peak exactly and ``d_ln_peak`` is identically zero, so
without a tiebreak the winner among those would be whichever arm sorted first."""


def _score(frame: pd.DataFrame) -> tuple[float, float]:
    """``(median |d_ln_peak|, median TV)`` --- the ranking key, primary first."""
    return float(frame[PRIMARY].median()), float(frame["tv"].median())


def best_arm(frame: pd.DataFrame) -> dict[str, Any]:
    """The (exponent, level) minimizing :func:`_score` within ``frame``."""
    scored = {
        key: _score(grp) for key, grp in frame.groupby(["exponent", "level"])
    }
    exponent, level = min(scored, key=lambda k: scored[k])
    return {
        "exponent": float(exponent),
        "level": float(level),
        "median_abs_d_ln_peak": scored[(exponent, level)][0],
        "median_tv": scored[(exponent, level)][1],
    }


def improvement_bootstrap(
    frame: pd.DataFrame, *, exponent: float, level: float, seed: int = 0,
    n_boot: int = N_BOOT,
) -> dict[str, float]:
    """Improvement of ``(exponent, level)`` over ``(0, level)``, resampled over seeds.

    Positive means the tilted arm peaks closer to the reference than the flat arm at the
    *same* level, so the comparison isolates the exponent from the scale.
    """
    at_level = frame[frame["level"] == level]
    tilted = at_level[at_level["exponent"] == exponent]
    flat = at_level[at_level["exponent"] == 0.0]
    n_seeds = at_level["seed"].nunique()
    if tilted.empty or flat.empty or n_seeds < MIN_SEEDS:
        # A bootstrap over one or two seeds returns a zero-width interval, which would
        # read as a resolved result. Refusing is the honest answer for a smoke run.
        return {
            "improvement": float("nan"), "lo": float("nan"), "hi": float("nan"),
            "excludes_zero": False, "n_seeds": int(n_seeds),
        }

    def gain(t: pd.DataFrame, f: pd.DataFrame) -> float:
        return float(f[PRIMARY].median() - t[PRIMARY].median())

    seeds = np.sort(at_level["seed"].unique())
    by_seed_t = {s: g for s, g in tilted.groupby("seed")}
    by_seed_f = {s: g for s, g in flat.groupby("seed")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = [s for s in rng.choice(seeds, size=seeds.size, replace=True)]
        # Both arms are resampled on the *same* seed draw: they share the dataset and
        # the reference curve, so an independent resample would inflate the spread with
        # variance that cancels in the difference.
        t = pd.concat([by_seed_t[s] for s in pick if s in by_seed_t])
        f = pd.concat([by_seed_f[s] for s in pick if s in by_seed_f])
        if len(t) and len(f):
            draws.append(gain(t, f))
    arr = np.asarray(draws, dtype=float)
    tail = 0.5 * (100.0 - INTERVAL)
    lo, hi = (
        (float(np.percentile(arr, tail)), float(np.percentile(arr, 100.0 - tail)))
        if arr.size
        else (float("nan"), float("nan"))
    )
    return {
        "improvement": gain(tilted, flat),
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(arr.size and (lo > 0.0 or hi < 0.0)),
        "n_seeds": int(n_seeds),
    }


def _n_amp_columns(adapter: Any, n_terms: int) -> int:
    """Amplitude columns at ``n_terms``: 2 per harmonic for RV, 4 for Gaia."""
    return adapter.d * n_terms


def build_section(data: Any) -> Section:
    """The exponent verdict, the mechanism behind it, and the tables under both."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data.calibrate
    tables: dict[str, pd.DataFrame] = {}
    findings: dict[str, Any] = {
        "exponents": sorted(float(e) for e in rows["exponent"].unique()),
        "levels": sorted(float(m) for m in rows["level"].unique()),
        "primary_metric": PRIMARY,
        "n_boot": N_BOOT,
    }
    caveats: list[str] = [
        (
            "The harmonic index k is not passed to a linear prior in harv, so at "
            "n_terms > 1 every harmonic here is scaled by the fundamental's period "
            "rather than by P/k. The H=1 rows are free of this; treat a disagreement "
            "between the H=1 and H=2 verdicts as evidence about that approximation, "
            "not about the exponent."
        )
    ]

    # --- the decisive table ------------------------------------------------
    tables["abs_d_ln_peak_by_exponent_level_population"] = rows.pivot_table(
        index=["exponent", "level"], columns="population", values=PRIMARY,
        aggfunc="median",
    )
    tables["tv_by_exponent_level_population"] = rows.pivot_table(
        index=["exponent", "level"], columns="population", values="tv",
        aggfunc="median",
    )

    # --- the verdict -------------------------------------------------------
    verdicts: list[dict[str, Any]] = []
    for h in sorted(rows["n_terms_effective"].dropna().unique()):
        at_h = rows[rows["n_terms_effective"] == h]
        physical = at_h[at_h["population"] == "physical"]
        independent = at_h[at_h["population"] == "independent"]
        if physical.empty or independent.empty:
            verdicts.append({"n_terms_effective": int(h), "verdict": "insufficient"})
            continue

        best = best_arm(physical)
        boot = improvement_bootstrap(
            physical, exponent=best["exponent"], level=best["level"]
        )
        # Clause 3: what the physical winner costs where the prior is wrong.
        mis = independent[
            (independent["exponent"] == best["exponent"])
            & (independent["level"] == best["level"])
        ]
        mis_flat = independent[
            (independent["exponent"] == 0.0) & (independent["level"] == best["level"])
        ]
        cost = (
            float(mis[PRIMARY].median() - mis_flat[PRIMARY].median())
            if len(mis) and len(mis_flat)
            else float("nan")
        )

        recommend = bool(
            best["exponent"] != 0.0
            and boot["excludes_zero"]
            and boot["improvement"] > 0.0
            and not (cost > boot["improvement"])
        )
        verdicts.append(
            {
                "n_terms_effective": int(h),
                "best_exponent": best["exponent"],
                "best_level_x_rms": best["level"],
                "median_abs_d_ln_peak": best["median_abs_d_ln_peak"],
                "improvement_over_flat": boot["improvement"],
                "improvement_lo": boot["lo"],
                "improvement_hi": boot["hi"],
                "misspecification_cost": cost,
                "verdict": (
                    "period-dependent" if recommend else "flat prior not beaten"
                ),
            }
        )
    verdict_frame = pd.DataFrame(verdicts)
    tables["verdict"] = verdict_frame.set_index("n_terms_effective")
    findings["verdict"] = verdicts

    # Best arm per population, ignoring n_terms, as the headline number.
    findings["best_by_population"] = {
        pop: best_arm(grp) for pop, grp in rows.groupby("population")
    }

    # --- the spec's own claim, and whether the exponent fixes it ------------
    matched = rows[rows["level"] == _matched_level(findings["levels"])]
    bias = matched.pivot_table(
        index="exponent", columns=["population", "period_ratio"], values="d_ln_peak",
        aggfunc="median",
    )
    tables["d_ln_peak_by_exponent_and_period_ratio"] = bias

    partial = rows[(rows["period_ratio"] >= 1.0) & (rows["level"] == min(findings["levels"]))]
    tight_flat = partial[partial["exponent"] == 0.0]["d_ln_peak"]
    findings["short_period_bias"] = {
        "claim": (
            "docs/spec.md: a scale comparable to the data RMS under-estimates the "
            "amplitude in the partial-arc regime and biases the peak toward short "
            "periods"
        ),
        "median_d_ln_peak_partial_arc_tightest_flat": (
            float(tight_flat.median()) if len(tight_flat) else float("nan")
        ),
        "fraction_negative": (
            float((tight_flat < 0).mean()) if len(tight_flat) else float("nan")
        ),
        "n": len(tight_flat),
    }

    # --- the mechanism, measured against its own ceiling -------------------
    #
    # Writing the arm's amplitude block as Lambda(P) = g(P)^2 Lambda_0 with scalar
    # g = (P/P0)**exponent,
    #     occam = 1/2 ln det(I + Lambda A),   d occam / d ln g  in  [0, k_amp]
    # -- it is a sum of k_amp eigenvalue terms lambda/(1+lambda), each below 1 and
    # approaching 1 only where the amplitude is well determined. So the tilt the
    # exponent adds to d(occam)/d(ln P) is bounded by ``k_amp * exponent`` and reaches
    # it in the well-determined limit. Reporting the *ratio* is therefore the honest
    # statement: it must lie in (0, 1], and a value outside that means the tilt is not
    # coming from the prior at all.
    slopes = rows.pivot_table(
        index="exponent", columns="n_terms_effective", values="occam_slope",
        aggfunc="median",
    )
    predicted = pd.DataFrame(
        {
            h: [float(e) * _n_amp_columns(data.adapter, int(h)) for e in slopes.index]
            for h in slopes.columns
        },
        index=slopes.index,
    )
    flat_slope = slopes.loc[0.0] if 0.0 in slopes.index else slopes.iloc[0] * 0.0
    measured = slopes - flat_slope
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = measured / predicted
    tables["occam_slope_measured_minus_flat"] = measured
    tables["occam_slope_ceiling"] = predicted
    tables["occam_tilt_fraction_of_ceiling"] = fraction

    realised = fraction.to_numpy()[np.isfinite(fraction.to_numpy())]
    findings["occam_tilt"] = {
        "ceiling": "d(occam)/d(ln P) - flat <= n_amp_columns * exponent",
        "fraction_of_ceiling_median": (
            float(np.median(realised)) if realised.size else float("nan")
        ),
        "fraction_of_ceiling_min": (
            float(realised.min()) if realised.size else float("nan")
        ),
        "fraction_of_ceiling_max": (
            float(realised.max()) if realised.size else float("nan")
        ),
        "within_bound": bool(
            realised.size and realised.min() > 0.0 and realised.max() <= 1.0 + 1e-9
        ),
    }

    # --- supporting interactions -------------------------------------------
    tables["abs_d_ln_peak_by_exponent_and_period_ratio"] = rows.pivot_table(
        index="exponent", columns="period_ratio", values=PRIMARY, aggfunc="median"
    )
    tables["abs_d_ln_peak_by_exponent_and_n_obs"] = rows.pivot_table(
        index="exponent", columns="n_obs", values=PRIMARY, aggfunc="median"
    )
    tables["abs_d_ln_peak_by_exponent_and_eccentricity"] = rows.pivot_table(
        index="exponent", columns="eccentricity", values=PRIMARY, aggfunc="median"
    )

    # --- figure -------------------------------------------------------------
    pops = sorted(rows["population"].unique())
    fig, axes = plt.subplots(1, len(pops), figsize=(5.5 * len(pops), 4), squeeze=False,
                             sharey=True)
    for ax, pop in zip(axes[0], pops, strict=False):
        sub = matched[matched["population"] == pop]
        curve = sub.pivot_table(
            index="period_ratio", columns="exponent", values="d_ln_peak",
            aggfunc="median",
        )
        for exponent in curve.columns:
            ax.plot(curve.index, curve[exponent], marker="o",
                    label=f"alpha = {exponent:+.3f}")
        ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"$P / T_{\rm span}$")
        ax.set_title(pop)
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel(r"median $\ln P_{\rm peak}(\Delta) - \ln P_{\rm peak}(R)$")
    fig.suptitle(
        f"{data.adapter_name}: peak bias vs partial arc, by amplitude-prior exponent"
    )

    body = """
Arms are `sigma(P) = level x RMS x (P / T_span)**exponent` applied to every harmonic
amplitude. `RMS` is the **base-model-residual RMS** -- the scatter left after projecting
out the `n_terms=0` model. For RV that is RMS about the mean; for Gaia it is far smaller
than the raw along-scan scatter, which parallax and proper motion dominate. A
recommended number is meaningless without that definition.

`exponent = 0` is today's flat prior. The population axis is the control: on
`physical` the injected amplitude really does follow the power law the prior assumes,
on `independent` it is flat in period by construction, and the recommendation has to
survive both. The verdict's three clauses are stated in this module's docstring and
were fixed before the numbers.
""".strip()

    return Section(
        key="amplitude",
        title="(a) Amplitude prior: scale and exponent",
        body=body,
        tables=tables,
        figures={"peak_bias": fig},
        findings=findings,
        caveats=caveats,
    )


def _matched_level(levels: list[float]) -> float:
    """The level nearest 1x RMS --- the one the exponent is compared at.

    Comparing exponents at the tightest level would confound the two knobs: a tight flat
    prior is already badly biased, so any exponent looks like an improvement.
    """
    return float(min(levels, key=lambda m: abs(np.log10(m))))
