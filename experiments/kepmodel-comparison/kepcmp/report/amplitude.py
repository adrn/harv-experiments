"""Deliverable (a), part 1: the Fourier amplitude prior --- number or functional form?

The question is not "what value for ``sigma_amp``". Every *Keplerian* amplitude in
harv already carries a parameter-dependent prior (``PeriodDependentKPrior``,
``PeriodDependentSemiMajorAxisPrior``, ``ParallaxDependentProperMotionPrior``), while
the *Fourier* harmonic amplitude gets a flat ``Normal(0, sigma_amp)`` with no ``P``
dependence --- although at trial period ``P`` it is measuring the same physical thing
``K`` is. Whether that asymmetry survives is decided by where the saturation knee sits
as ``P / T_span`` grows:

- **knee constant** -> the guidance is a floor, period-independent, and ships as
  documentation;
- **knee moves right** -> the guidance is a functional form, and its exponent is the
  slope of ``log10(knee)`` against ``log10(P / T_span)``.

The estimator, its uncertainty, and the decision threshold are all fixed here rather
than chosen after looking, so the report cannot hedge.
"""

from __future__ import annotations

__all__ = ("build_section", "knee_with_uncertainty", "saturation_knee")

from typing import Any

import numpy as np
import pandas as pd

from kepcmp.report.render import Section

TOL_ABS = 0.01
"""Absolute TV slack counted as "saturated"."""

TOL_FRAC = 0.05
"""...or this fraction of the curve's own range, whichever is larger."""

N_BOOT = 200


def saturation_knee(
    mults: np.ndarray, tv: np.ndarray, *, tol_abs: float = TOL_ABS,
    tol_frac: float = TOL_FRAC,
) -> float:
    """``log10`` of the smallest multiplier at which ``tv`` has saturated.

    Saturation is "within ``tol`` of the wide-prior asymptote", where the asymptote is
    the median of the two widest scales sampled and ``tol`` is the larger of an
    absolute slack and a fraction of the curve's own range.

    The crossing is **interpolated in ``log10(mult)``**, which is the only reason a
    5-point sweep can say anything finer than "somewhere between two grid points".
    Returns ``nan`` when the curve has not flattened by the widest scale sampled,
    which is itself a result: the prior is still binding there and no knee exists
    inside the grid.

    Interpolating a convex curve linearly *compresses* the estimate, so a drift
    between two knees is under-stated by ~10-15% (measured against a synthetic curve
    shifted by a known amount). That biases the verdict toward "constant", never
    toward manufacturing a movement that is not there.
    """
    mults = np.asarray(mults, dtype=float)
    tv = np.asarray(tv, dtype=float)
    order = np.argsort(mults)
    x, y = np.log10(mults[order]), tv[order]
    if x.size < 3 or not np.all(np.isfinite(y)):
        return float("nan")

    tv_inf = float(np.median(y[-2:]))
    tol = max(tol_abs, tol_frac * (float(np.max(y)) - tv_inf))
    target = tv_inf + tol

    # The asymptote is estimated from the two widest scales, so it is only meaningful
    # if the curve has actually flattened there. If it is still falling at the top of
    # the sweep, any "knee" would be an artifact of where the grid happens to stop --
    # the honest answer is that the prior is still binding at the widest scale sampled.
    if abs(y[-2] - y[-1]) > tol:
        return float("nan")

    below = np.flatnonzero(y <= target)
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))


def knee_with_uncertainty(
    frame: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = 0
) -> dict[str, float]:
    """Point estimate plus a seed-bootstrap interval, in dex.

    Resampling is over **seeds**, with multiplicity, because that is the axis the grid
    replicates; resampling rows would treat the five multipliers of one simulation as
    independent when they share a dataset.
    """
    mults = np.sort(frame["sigma_amp_mult"].unique())
    med = frame.groupby("sigma_amp_mult")["tv"].median()
    point = saturation_knee(mults, med.reindex(mults).to_numpy())

    seeds = np.sort(frame["seed"].unique())
    by_seed = {s: g for s, g in frame.groupby("seed")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(seeds, size=seeds.size, replace=True)
        sub = pd.concat([by_seed[s] for s in pick], copy=False)
        m = sub.groupby("sigma_amp_mult")["tv"].median().reindex(mults).to_numpy()
        draws.append(saturation_knee(mults, m))
    arr = np.asarray(draws, dtype=float)
    good = arr[np.isfinite(arr)]
    # A nan point estimate with a finite interval would read as a resolved knee, so
    # the interval is suppressed too and `resolved_fraction` carries what the
    # resamples did manage -- "resolvable in 30% of resamples" is a result.
    resolved = float(good.size) / float(arr.size) if arr.size else float("nan")
    usable = good.size and np.isfinite(point)
    return {
        "knee_dex": point,
        "lo_dex": float(np.percentile(good, 16)) if usable else float("nan"),
        "hi_dex": float(np.percentile(good, 84)) if usable else float("nan"),
        "sd_dex": float(np.std(good)) if usable else float("nan"),
        "resolved_fraction": resolved,
    }


def _grid_step_dex(mults: list[float]) -> float:
    lg = np.log10(np.sort(np.asarray(mults, dtype=float)))
    return float(np.median(np.diff(lg)))


def build_section(data: Any) -> Section:
    """The decisive split, its verdict, and the supporting interactions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = data.delta_rows()
    mults = data.sigma_amp_mults
    step = _grid_step_dex(mults)
    tables: dict[str, pd.DataFrame] = {}
    findings: dict[str, Any] = {"grid_step_dex": step, "sigma_amp_mults": mults}
    caveats: list[str] = []

    # --- the decisive split ------------------------------------------------
    decisive = delta.pivot_table(
        index="sigma_amp_mult", columns="period_ratio", values="tv", aggfunc="median"
    )
    tables["tv_by_sigma_amp_and_period_ratio"] = decisive

    per_h: dict[int, pd.DataFrame] = {}
    knee_rows: list[dict[str, Any]] = []
    for h in sorted(delta["n_terms_effective"].dropna().unique()):
        sub_h = delta[delta["n_terms_effective"] == h]
        per_h[int(h)] = sub_h.pivot_table(
            index="sigma_amp_mult", columns="period_ratio", values="tv",
            aggfunc="median",
        )
        for ratio, grp in sub_h.groupby("period_ratio"):
            k = knee_with_uncertainty(grp)
            knee_rows.append({"n_terms_effective": int(h), "period_ratio": float(ratio), **k})
    knees = pd.DataFrame(knee_rows)
    tables["knee_by_period_ratio"] = knees.set_index(["n_terms_effective", "period_ratio"])
    for h, frame in per_h.items():
        tables[f"tv_by_sigma_amp_and_period_ratio_H{h}"] = frame

    # --- the pre-registered verdict ----------------------------------------
    verdicts = []
    for h, grp in knees.groupby("n_terms_effective"):
        ok = grp[np.isfinite(grp["knee_dex"])].sort_values("period_ratio")
        if len(ok) < 2:
            verdicts.append({"n_terms_effective": int(h), "verdict": "insufficient"})
            continue
        lo_row, hi_row = ok.iloc[0], ok.iloc[-1]
        drift = float(hi_row["knee_dex"] - lo_row["knee_dex"])
        unc = float(np.hypot(lo_row["sd_dex"], hi_row["sd_dex"]))
        threshold = max(2.0 * unc, step / 2.0)
        # |drift|: a knee that moves *left* with period is just as much a functional
        # form as one that moves right -- it only changes the sign of the exponent.
        # Testing the signed drift would call that "constant" and miss it.
        moves = abs(drift) > threshold
        slope = float("nan")
        if moves:
            slope = float(
                np.polyfit(np.log10(ok["period_ratio"]), ok["knee_dex"], 1)[0]
            )
        verdicts.append(
            {
                "n_terms_effective": int(h),
                "drift_dex": drift,
                "bootstrap_unc_dex": unc,
                "threshold_dex": threshold,
                "verdict": "knee moves" if moves else "constant within resolution",
                "exponent_vs_period_ratio": slope,
                "knee_at_min_ratio_x_rms": 10.0 ** float(lo_row["knee_dex"]),
                "knee_at_max_ratio_x_rms": 10.0 ** float(hi_row["knee_dex"]),
            }
        )
    verdict_frame = pd.DataFrame(verdicts)
    tables["verdict"] = verdict_frame.set_index("n_terms_effective")
    findings["verdict"] = verdicts

    if not any(v.get("verdict") == "knee moves" for v in verdicts):
        caveats.append(
            f"The amplitude-prior knee is constant within this grid's resolution. The "
            f"sweep has {len(mults)} multipliers spaced {step:.2f} dex apart, so a "
            f"period-dependence weaker than ~{step / 2:.2f} dex across the sampled "
            "P/T_span range is NOT excluded -- it is unresolved. Resolving it needs a "
            "denser sigma_amp sweep, not more cells."
        )

    # --- the spec's own claim, tested head-on ------------------------------
    bias = delta.pivot_table(
        index="sigma_amp_mult", columns="period_ratio", values="d_ln_peak",
        aggfunc="median",
    )
    tables["d_ln_peak_by_sigma_amp_and_period_ratio"] = bias
    partial = delta[delta["period_ratio"] >= 1.0]
    tight = partial[partial["sigma_amp_mult"] == min(mults)]["d_ln_peak"]
    findings["short_period_bias"] = {
        "claim": (
            "docs/spec.md: a scale comparable to the data RMS under-estimates the "
            "amplitude in the partial-arc regime and biases the peak toward short "
            "periods"
        ),
        "median_d_ln_peak_partial_arc_tightest_prior": float(tight.median())
        if len(tight)
        else float("nan"),
        "fraction_negative": float((tight < 0).mean()) if len(tight) else float("nan"),
        "n": len(tight),
    }

    # --- interactions -------------------------------------------------------
    tables["tv_by_sigma_amp_and_n_terms"] = delta.pivot_table(
        index="sigma_amp_mult", columns="n_terms_effective", values="tv",
        aggfunc="median",
    )
    delta = delta.assign(capped=delta["n_terms_effective"] < delta.groupby("sim_id")[
        "n_terms_effective"
    ].transform("max"))
    tables["tv_by_sigma_amp_and_n_obs"] = delta.pivot_table(
        index="sigma_amp_mult", columns="n_obs", values="tv", aggfunc="median"
    )

    # --- the basis-only floor ----------------------------------------------
    z0_tv = data.z0_rows()["tv"]
    findings["basis_only_floor_tv_z0"] = (
        float(z0_tv.median()) if len(z0_tv) else float("nan")
    )

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(
        1, max(len(per_h), 1), figsize=(5 * max(len(per_h), 1), 4), squeeze=False
    )
    for ax, (h, frame) in zip(axes[0], sorted(per_h.items()), strict=False):
        for ratio in frame.columns:
            ax.plot(frame.index, frame[ratio], marker="o", label=f"P/T={ratio:g}")
        kh = knees[knees["n_terms_effective"] == h]
        for _, row in kh.iterrows():
            if np.isfinite(row["knee_dex"]):
                ax.axvline(10.0 ** row["knee_dex"], color="0.8", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma_{\rm amp}$ / data RMS")
        ax.set_ylabel("TV to reference")
        ax.set_title(f"H = {h}")
        ax.legend(fontsize=7)
    fig.suptitle(
        f"{data.adapter_name}: saturation knee vs partial arc "
        "(vertical lines = fitted knees)"
    )

    body = f"""
`sigma_amp` is quoted as a multiple of the **base-model-residual RMS** -- the scatter
left after projecting out the `n_terms=0` model. For RV that is RMS about the mean;
for Gaia it is ~50x smaller than the raw along-scan scatter, which is dominated by
parallax and proper motion the base columns absorb. A recommended number is
meaningless without that definition.

The knee is the smallest multiplier whose median TV is within
`max({TOL_ABS}, {TOL_FRAC} x range)` of the wide-prior asymptote, interpolated in
`log10`, with a {N_BOOT}-draw bootstrap over seeds. The grid spaces multipliers
{step:.2f} dex apart, and the verdict threshold is `max(2 x bootstrap sd, {step / 2:.2f} dex)`
-- both fixed before looking at the answer.
""".strip()

    return Section(
        key="amplitude",
        title="(a) Amplitude prior: number or functional form",
        body=body,
        tables=tables,
        figures={"knee": fig},
        findings=findings,
        caveats=caveats,
    )
