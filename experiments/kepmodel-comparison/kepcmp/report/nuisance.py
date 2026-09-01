"""Deliverable (a), part 2: the nuisance (base-column) scales.

The grid never swept these -- ``StatRecord`` carries ``sigma_amp`` and nothing for the
offset scales -- so this section measures them directly, and cheaply, by re-running
periodograms rather than re-running the grid.

The prediction under test: because the periodogram does **no centering**
(``docs/spec.md``, "Priors are explicit"), ``Delta`` and its peak are flat in the
nuisance scale until the prior stops *covering* the true value, and then degrade
sharply. If that holds, the two prior groups take opposite defaults --- a data-driven
default is inappropriate for amplitudes, because it sets the Occam factor and
therefore the science, but safe for nuisance offsets, where the only failure mode is
non-coverage.

RV needs fresh simulations because the grid pins ``v_sys = 0``, and non-coverage
cannot be provoked at zero. Gaia does not: parallax is pinned at 10 mas and proper
motions are drawn non-zero, so stored datasets already have something to fail to
cover.
"""

from __future__ import annotations

__all__ = ("build_section",)

import dataclasses
from typing import Any

import harv.periodogram as hp
import numpy as np
import pandas as pd
from unxt import Q, ustrip

from kepcmp.artifact import ArtifactReader
from kepcmp.grid import Cell, shared_frequency_grid
from kepcmp.report.render import Section

#: Systemic velocity for the RV sweep. Large, and deliberately so: it matches the
#: source in harv's tutorial 1, and coverage cannot fail against zero.
RV_V_SYS = Q(-135.0, "km/s")

RV_SIGMA_V0 = np.logspace(-1, 4, 11)      # km/s, ~5 decades bracketing |v_sys|
GAIA_BASE_SCALE = np.logspace(-4, 2, 13)  # x the adapter's default base-prior widths

N_TERMS = 1
STRIDE = 8
"""Frequency-grid stride. This section asks where the peak *moves*, not its shape."""


def _peak_and_height(data: Any, freq: Any, prior: Any) -> tuple[float, float]:
    res = hp.periodogram(data, freq, prior=prior, n_terms=N_TERMS)
    delta = np.asarray(res.delta_ln_likelihood, dtype=float)
    period = np.asarray(ustrip("day", 1.0 / freq), dtype=float)
    return float(period[int(np.nanargmax(delta))]), float(np.nanmax(delta))


def _rv_sweep(data: Any, n_sims: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter = dataclasses.replace(data.adapter, v_sys=RV_V_SYS)
    freq = shared_frequency_grid(adapter)[::STRIDE]
    cell = Cell(period_ratio=0.3, snr=10.0, eccentricity=0.0, n_obs=32)
    rows = []
    for seed in range(n_sims):
        sim, truth = adapter.simulate(cell, seed)
        p_true = float(ustrip("day", truth["period"]))
        for scale in RV_SIGMA_V0:
            a = dataclasses.replace(adapter, sigma_v0=Q(float(scale), "km/s"))
            prior = a.trial_prior(sim, n_terms=N_TERMS, sigma_amp=a.data_rms(sim))
            peak, height = _peak_and_height(sim, freq, prior)
            rows.append(
                {
                    "seed": seed,
                    "sigma_nuisance": float(scale),
                    "covers_truth": float(scale) >= abs(float(ustrip("km/s", RV_V_SYS))),
                    "abs_d_ln_peak": abs(np.log(peak) - np.log(p_true)),
                    "max_delta": height,
                }
            )
    meta = {
        "parameter": "sigma_v0",
        "unit": "km/s",
        # Magnitude: coverage is about |v_sys| against the prior half-width, and a
        # signed value would put the figure's reference line at a negative x on a log
        # axis and make the reported ratio negative.
        "true_value": abs(float(ustrip("km/s", RV_V_SYS))),
        "note": "fresh simulations; the grid pins v_sys = 0 so coverage cannot fail",
    }
    return pd.DataFrame(rows), meta


def _gaia_sweep(data: Any, n_sims: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter = data.adapter
    rows = []
    with ArtifactReader(data.signal_path) as art:
        sim_ids = art.sim_ids()
        # Stratified but deterministic: an evenly spaced walk through the sorted list
        # rather than the first N, which would be one corner of the grid.
        step = max(1, len(sim_ids) // max(n_sims, 1))
        picked = sim_ids[::step][:n_sims]
        freq = Q(art.frequency[::STRIDE], "1/day")
        for sim_id in picked:
            attrs = art.attrs(sim_id)
            sim = adapter.data_from_arrays(art.data(sim_id))
            p_true = float(attrs["truth_period"])
            rms = adapter.data_rms(sim)
            for scale in GAIA_BASE_SCALE:
                a = dataclasses.replace(
                    adapter,
                    sigma_pos=adapter.sigma_pos * float(scale),
                    sigma_pm=adapter.sigma_pm * float(scale),
                    sigma_parallax=adapter.sigma_parallax * float(scale),
                )
                prior = a.trial_prior(sim, n_terms=N_TERMS, sigma_amp=rms)
                peak, height = _peak_and_height(sim, freq, prior)
                width = float(ustrip("mas", a.sigma_parallax))
                rows.append(
                    {
                        "seed": int(attrs["seed"]),
                        "sigma_nuisance": width,
                        "covers_truth": width >= float(attrs["truth_parallax"]),
                        "abs_d_ln_peak": abs(np.log(peak) - np.log(p_true)),
                        "max_delta": height,
                    }
                )
    meta = {
        "parameter": "sigma_pos / sigma_pm / sigma_parallax (scaled together)",
        "unit": "mas (parallax width shown)",
        "true_value": float(ustrip("mas", adapter.parallax)),
        "note": "replayed from stored datasets; no new simulations needed",
    }
    return pd.DataFrame(rows), meta


def build_section(data: Any, *, n_sims: int = 24) -> Section:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep, meta = (
        _rv_sweep(data, n_sims)
        if data.adapter_name == "rv"
        else _gaia_sweep(data, n_sims)
    )
    summary = sweep.pivot_table(
        index="sigma_nuisance",
        values=["abs_d_ln_peak", "max_delta", "covers_truth"],
        aggfunc="median",
    )

    # The threshold: the widest scale at which the peak has already moved. Compared
    # against the true value, this is the whole recommendation.
    wide = sweep[sweep["sigma_nuisance"] >= meta["true_value"]]
    baseline = float(wide["abs_d_ln_peak"].median()) if len(wide) else float("nan")
    tol = max(0.05, 3.0 * baseline)
    broken = summary.index[summary["abs_d_ln_peak"] > tol]
    threshold = float(broken.max()) if len(broken) else float("nan")

    findings = {
        **meta,
        "baseline_abs_d_ln_peak_when_covering": baseline,
        "degradation_tolerance": tol,
        "breaks_at_or_below": threshold,
        "ratio_to_true_value": (
            threshold / meta["true_value"] if np.isfinite(threshold) else float("nan")
        ),
        "flat_while_covering": bool(np.isfinite(baseline) and baseline <= tol),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary.index, summary["abs_d_ln_peak"], marker="o")
    ax.axvline(meta["true_value"], color="crimson", ls="--",
               label=f"true value = {meta['true_value']:g}")
    ax.axhline(tol, color="0.7", ls=":", label="degradation tolerance")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel(f"nuisance prior width [{meta['unit']}]")
    ax.set_ylabel(r"median $|\Delta \ln P_{\rm peak}|$")
    ax.set_title(f"{data.adapter_name}: peak stability vs nuisance prior width")
    ax.legend(fontsize=8)

    body = f"""
Sweeping `{meta["parameter"]}` over ~6 decades while everything else is held fixed.
{meta["note"]}.

The prediction is coverage, not tuning: the peak should be **flat** while the prior
covers the true value ({meta["true_value"]:g} {meta["unit"]}) and degrade sharply
below it. If that is what the table shows, a *data-driven* default is safe here --
centred on the data with a width a few times `(RMS + span)` -- precisely because the
only failure mode is non-coverage, and it is the opposite conclusion from the
amplitude prior, which sets the Occam factor and therefore the science.
""".strip()

    return Section(
        key="nuisance",
        title="(a) Nuisance scales: coverage, not tuning",
        body=body,
        tables={"sweep": summary},
        figures={"coverage": fig},
        findings=findings,
    )
