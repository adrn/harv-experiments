"""Deliverable (b): when to reach for which periodogram.

Two axes, because harv exposes exactly one ``periodogram()`` and the user's real
choices are not all external:

- **external** --- harv's marginal ``Delta`` against kepmodel's profile ``z0``. Some of
  this is a *capability* statement rather than a performance one: ``z0`` does not exist
  at ``n_obs <= p + d``.
- **internal** --- which prior builder the curve is mapped through, and at what
  ``n_terms``. harv's own (removed) draft tutorial asserts that *"peak_period_prior is
  the more robust default across unknown eccentricities; tempered_period_prior with
  beta=1 can concentrate on an alias for eccentric orbits"* with no evidence behind it.
  The grid has an eccentricity axis, so that claim is testable here -- and testing it
  costs nothing, because it maps *stored* curves through a different builder.

The headline metric for the internal axis is the interim prior's **log density at the
injected period**, not a distance to the reference. Density-at-truth is what governs
how often a rejection sampler proposes near the answer, and unlike a distance it does
not quietly favour whichever builder shares a functional form with the reference.
"""

from __future__ import annotations

__all__ = ("build_section",)

from typing import Any

import harv.periodogram as hp
import numpy as np
import pandas as pd

from kepcmp.artifact import ArtifactReader
from kepcmp.reduce.common import (
    as_periodogram_result,
    common_ln_p_grid,
    ln_period_density,
    tv_distance,
)
from kepcmp.report.render import Section

N_GRID = 1024

BUILDERS: dict[str, Any] = {
    "tempered_beta0.5": lambda r: hp.tempered_period_prior(r, beta=0.5, floor=0.1),
    "tempered_beta1": lambda r: hp.tempered_period_prior(r, beta=1.0, floor=0.1),
    "tempered_beta2": lambda r: hp.tempered_period_prior(r, beta=2.0, floor=0.1),
    "peak_height_drop10": lambda r: hp.peak_period_prior(
        r, height_drop=10.0, max_peaks=8, floor=0.1
    ),
}


def _internal_axis(data: Any, config: str = "H1_s1") -> pd.DataFrame:
    """Map every stored curve through each prior builder. No re-simulation."""
    rows: list[dict[str, Any]] = []
    with ArtifactReader(data.signal_path) as art:
        frequency = art.frequency
        period = art.period
        t_span = float(art.meta["t_span_day"])
        ln_p = common_ln_p_grid(period, N_GRID)
        ln_p_uniform = -np.log(ln_p[-1] - ln_p[0])  # the log-uniform control

        for sim_id in art.sim_ids():
            attrs = art.attrs(sim_id)
            if bool(attrs["is_null"]):
                continue
            ref = art.reference(sim_id)
            if ref is None or config not in set(art.stat_keys(sim_id)):
                continue
            values = art.stat(sim_id, config)["delta"]
            ln_p_true = np.log(float(attrs["truth_period"]))
            if not np.isfinite(ln_p_true):
                continue

            rho_ref = ln_period_density(
                hp.tempered_period_prior(
                    as_periodogram_result(frequency, ref, t_span=t_span), floor=0.1
                ),
                ln_p,
            )
            base = {
                "sim_id": sim_id,
                "period_ratio": float(attrs["period_ratio"]),
                "snr": float(attrs["snr"]),
                "eccentricity": float(attrs["eccentricity"]),
                "n_obs": int(attrs["n_obs"]),
            }
            rows.append(
                {**base, "builder": "loguniform",
                 "ln_density_at_truth": float(ln_p_uniform),
                 "tv_to_reference": tv_distance(
                     np.full_like(ln_p, np.exp(ln_p_uniform)), rho_ref, ln_p
                 )}
            )
            result = as_periodogram_result(frequency, values, t_span=t_span)
            for name, builder in BUILDERS.items():
                # A builder can legitimately fail to produce a prior -- peak-finding
                # on a curve with no resolvable peak, for instance. That is a result
                # about the builder, so it is recorded as a failed row rather than
                # dropped, which would silently improve that builder's medians.
                try:
                    rho = ln_period_density(builder(result), ln_p)
                except Exception as exc:  # noqa: BLE001  (harv's builders, any failure is theirs)
                    rows.append(
                        {
                            **base,
                            "builder": name,
                            "ln_density_at_truth": np.nan,
                            "tv_to_reference": np.nan,
                            "failed": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                at_truth = float(np.interp(ln_p_true, ln_p, rho))
                rows.append(
                    {
                        **base,
                        "builder": name,
                        "ln_density_at_truth": float(np.log(max(at_truth, 1e-300))),
                        "tv_to_reference": tv_distance(rho, rho_ref, ln_p),
                        "failed": "",
                    }
                )
    return pd.DataFrame(rows)


def _exemplars(data: Any) -> pd.DataFrame:
    """Deterministic, criterion-based picks -- never hand-chosen.

    Each row is a teaching claim plus the simulation that shows it. Delivered as a
    *recipe* (cell axes + seed), not as data: harv's tutorials simulate inline, its
    `docs/tutorials/.gitignore` drops `*.h5`, and pre-commit caps added files at 2 MB.
    """
    dec = data.decompose
    reg = data.regime[data.regime["available"]]
    picks: list[dict[str, Any]] = []

    def _add(claim: str, frame: pd.DataFrame, col: str, *, largest: bool) -> None:
        f = frame[np.isfinite(frame[col])]
        if f.empty:
            return
        row = f.loc[f[col].idxmax() if largest else f[col].idxmin()]
        picks.append(
            {
                "claim": claim,
                "criterion": f"{'max' if largest else 'min'}({col})",
                "sim_id": row.get("sim_id", row.get("cell_id")),
                "period_ratio": float(row["period_ratio"]),
                "snr": float(row["snr"]),
                "eccentricity": float(row["eccentricity"]),
                "n_obs": int(row["n_obs"]),
                col: float(row[col]),
            }
        )

    comparable = dec[dec.get("comparable_to_z0", pd.Series(dtype=bool)).fillna(False)]
    if not comparable.empty:
        _add("agreement where the amplitude is well constrained",
             comparable, "ln_amp_constraint", largest=True)
        _add("divergence where it is not", comparable,
             "ln_amp_constraint", largest=False)
        _add("Occam explains the disagreement", comparable,
             "occam_range", largest=True)
    if not reg.empty and "tv" in reg:
        _add("best regime for the marginal statistic",
             reg[reg["arm"] == "delta_default"], "tv", largest=False)
    return pd.DataFrame(picks)


def build_section(data: Any) -> Section:
    tables: dict[str, pd.DataFrame] = {}
    findings: dict[str, Any] = {}
    caveats: list[str] = []

    reg = data.regime[data.regime["available"]]

    # --- external: harv vs kepmodel ----------------------------------------
    tables["external_tv_by_period_ratio"] = reg.pivot_table(
        index="period_ratio", columns="arm", values="tv", aggfunc="median"
    )
    tables["external_tv_by_n_obs"] = reg.pivot_table(
        index="n_obs", columns="arm", values="tv", aggfunc="median"
    )
    ident = reg[reg["period_identifiable"]]
    if not ident.empty:
        tables["external_recovered_by_n_obs_identifiable"] = ident.pivot_table(
            index="n_obs", columns="arm", values="recovered", aggfunc="median"
        )
    unavailable = data.regime[~data.regime["available"]]
    z0_missing = unavailable[unavailable["arm"] == "z0"]
    findings["z0_unavailable_cells"] = len(z0_missing)
    findings["z0_requires_n_obs_gt"] = (
        data.adapter.n_base_columns + data.adapter.d
    )
    if data.roc is not None:
        tables["external_tpr_by_snr"] = data.roc.pivot_table(
            index="snr", columns="statistic", values="tpr", aggfunc="median"
        )

    # --- internal: which prior builder -------------------------------------
    internal = _internal_axis(data)
    if not internal.empty:
        tables["internal_density_at_truth_by_eccentricity"] = internal.pivot_table(
            index="builder", columns="eccentricity",
            values="ln_density_at_truth", aggfunc="median",
        )
        tables["internal_density_at_truth_by_period_ratio"] = internal.pivot_table(
            index="builder", columns="period_ratio",
            values="ln_density_at_truth", aggfunc="median",
        )
        tables["internal_tv_to_reference"] = internal.pivot_table(
            index="builder", columns="eccentricity",
            values="tv_to_reference", aggfunc="median",
        )
        by_ecc = internal.pivot_table(
            index="builder", columns="eccentricity",
            values="ln_density_at_truth", aggfunc="median",
        )
        findings["best_builder_by_eccentricity"] = {
            str(col): str(by_ecc[col].idxmax()) for col in by_ecc.columns
        }
        if "failed" in internal:
            fails = internal[internal["failed"].fillna("") != ""]
            findings["builder_failures"] = (
                fails.groupby("builder").size().to_dict() if not fails.empty else {}
            )
        findings["draft_tutorial_claim"] = (
            "peak_period_prior is the more robust default across unknown "
            "eccentricities; tempered beta=1 can concentrate on an alias for "
            "eccentric orbits"
        )
        caveats.append(
            "`tv_to_reference` compares each builder against the reference mapped "
            "through tempered(beta=1), so it structurally favours the tempered "
            "family. `ln_density_at_truth` is the builder-neutral metric and is what "
            "the verdict is based on."
        )

    exemplars = _exemplars(data)
    if not exemplars.empty:
        tables["exemplars"] = exemplars.set_index("claim")
        findings["exemplars"] = exemplars.to_dict("records")

    body = """
The external axis is partly a capability statement: kepmodel's profile statistic does
not exist at `n_obs <= p + d`, where it fits every trial period exactly. That is not
kepmodel losing a race, and it should be taught as a precondition rather than a score.

The internal axis maps the *same stored curves* through different prior builders, so
it costs no simulation and isolates the builder from everything else. It is the axis
harv's users actually control.
""".strip()

    return Section(
        key="casestudy",
        title="(b) Which periodogram, and when",
        body=body,
        tables=tables,
        figures={},
        findings=findings,
        caveats=caveats,
    )
