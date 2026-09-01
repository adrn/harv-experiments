"""Which periodogram is better, per regime, on three metrics side by side.

Aggregates over seeds within each grid cell and reports, for three arms:

- ``delta_default`` --- harv at ``n_terms=2``, ``sigma_amp = 1x`` data RMS. The
  "default settings" arm: neither side tuned per cell.
- ``delta_h1`` --- harv at ``n_terms=1``, same ``sigma_amp``. The controlled contrast:
  its columns match kepmodel's ``d=2`` exactly, so ``delta_h1`` vs ``z0`` isolates the
  *prior* effect and ``delta_default`` vs ``delta_h1`` isolates the *basis* effect.
- ``z0`` --- kepmodel's profile statistic.

Three metrics, reported together rather than collapsed, because they answer different
questions and genuinely disagree in places:

1. ``tv`` --- total-variation distance to the prior built from the Keplerian reference
   ``R``. Distributional fidelity: is the interim prior the right shape? Needs
   ``--reference-n-mc`` to have been set at run time.
2. ``peak_err`` --- median ``|ln P_peak - ln P_true|``. Operational accuracy.
3. ``tpr`` --- true-positive rate at a fixed FPR calibrated on null simulations.
   Detection rather than accuracy; needs ``--null-artifact``.

Two gates keep the map honest, both driven by measured structural facts rather than
taste:

- **``z0`` is dropped where it is not a measurement.** At ``n_obs <= 3`` kepmodel's
  enlarged model has more columns (3) than data, so it fits any trial period exactly:
  ``chi2_K`` varies by ~1e-13 and ``z0`` is flat, while its explicit ``inv(N N^T)`` can
  return non-physical ``chi2``. Those cells report ``z0`` as unavailable instead of
  scoring noise against truth.
- **``peak_err`` is marked unidentifiable where the period cannot be localized.** At
  ``P >= 3 T_span`` the injected frequency approaches one peak width of zero, so no
  method can localize it; at ``P = 10 T_span`` it is ~10x inside one peak width. In
  those columns only ``tv`` is informative.
"""

from __future__ import annotations

__all__ = ("main", "reduce_regime_map")

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import harv.periodogram as hp
import numpy as np

from kepcmp.artifact import ArtifactReader
from kepcmp.reduce.common import (
    as_periodogram_result,
    common_ln_p_grid,
    detection_scalars,
    ln_period_density,
    peak_period,
    tv_distance,
)

#: ``P / T_span`` at or above which the period is not localizable from the data.
IDENTIFIABILITY_LIMIT = 3.0

#: ``|ln P|`` tolerance counted as recovering the injected period.
RECOVERY_LN_TOL = 0.1


def _thresholds(
    null_path: Path, fpr: float
) -> tuple[dict[tuple[int, str], float], dict[int, int]]:
    """Per ``(n_obs, statistic)`` detection threshold from the nulls."""
    scal, attr = detection_scalars(null_path)
    pooled: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sim_id, s in scal.items():
        n_obs = int(attr[sim_id]["n_obs"])
        for stat, v in s.items():
            pooled[n_obs][stat].append(v)
    thr: dict[tuple[int, str], float] = {}
    counts: dict[int, int] = {}
    for n_obs, per_stat in pooled.items():
        for stat, vals in per_stat.items():
            arr = np.asarray(vals, dtype=float)
            thr[(n_obs, stat)] = float(np.quantile(arr, 1.0 - fpr))
            counts[n_obs] = arr.size
    return thr, counts


def reduce_regime_map(
    signal_path: Path,
    null_path: Path | None = None,
    *,
    fpr: float = 0.01,
    beta: float = 1.0,
    floor: float = 0.1,
    default_config: str = "H2_s1",
    control_config: str = "H1_s1",
    n_grid: int = 1024,
) -> list[dict]:
    """One row per (cell, arm), aggregated over seeds."""
    thr: dict[tuple[int, str], float] = {}
    null_counts: dict[int, int] = {}
    if null_path is not None:
        thr, null_counts = _thresholds(null_path, fpr)

    arms = {
        "delta_default": f"delta:{default_config}",
        "delta_h1": f"delta:{control_config}",
        "z0": "z0",
    }
    per_cell: dict[str, dict[str, Any]] = {}

    with ArtifactReader(signal_path) as art:
        frequency = art.frequency
        period = art.period
        t_span = float(art.meta["t_span_day"])
        ln_p = common_ln_p_grid(period, n_grid)

        def density(values: np.ndarray) -> np.ndarray:
            prior = hp.tempered_period_prior(
                as_periodogram_result(frequency, values, t_span=t_span),
                beta=beta,
                floor=floor,
            )
            return ln_period_density(prior, ln_p)

        for sim_id in art.sim_ids():
            a = art.attrs(sim_id)
            if bool(a["is_null"]):
                continue
            cell_id = str(a["cell_id"])
            slot = per_cell.setdefault(
                cell_id,
                {
                    "cell_id": cell_id,
                    "period_ratio": float(a["period_ratio"]),
                    "snr": float(a["snr"]),
                    "eccentricity": float(a["eccentricity"]),
                    "n_obs": int(a["n_obs"]),
                    "integrated_snr": float(a["integrated_snr"]),
                    "n_seeds": 0,
                    "z0_usable_n": 0,
                    "per_arm": defaultdict(
                        lambda: {"tv": [], "peak_err": [], "det": [], "n_eff": []}
                    ),
                },
            )
            slot["n_seeds"] += 1
            z0_usable = art.z0_usable(sim_id)
            slot["z0_usable_n"] += int(z0_usable)

            p_true = float(a.get("truth_period", np.nan))
            ref = art.reference(sim_id)
            rho_ref = density(ref) if ref is not None else None
            keys = set(art.stat_keys(sim_id))

            for arm, source in arms.items():
                if source == "z0":
                    if not z0_usable:
                        continue
                    values = art.z0(sim_id)
                    n_eff = 1
                else:
                    key = source.split(":", 1)[1]
                    if key not in keys:
                        continue
                    st = art.stat(sim_id, key)
                    values = st["delta"]
                    n_eff = int(st["n_terms_effective"])

                rec = slot["per_arm"][arm]
                rec["n_eff"].append(n_eff)
                if rho_ref is not None:
                    rec["tv"].append(tv_distance(density(values), rho_ref, ln_p))
                if np.isfinite(p_true) and p_true > 0:
                    rec["peak_err"].append(
                        abs(np.log(peak_period(period, values)) - np.log(p_true))
                    )
                key_thr = (int(a["n_obs"]), source)
                if key_thr in thr:
                    rec["det"].append(float(np.max(values)) > thr[key_thr])

    rows: list[dict] = []
    for _cell_id, slot in sorted(per_cell.items()):
        identifiable = slot["period_ratio"] < IDENTIFIABILITY_LIMIT
        for arm in arms:
            rec = slot["per_arm"].get(arm)
            base = {
                k: slot[k]
                for k in (
                    "cell_id", "period_ratio", "snr", "eccentricity", "n_obs",
                    "integrated_snr", "n_seeds",
                )
            }
            if rec is None or not rec["n_eff"]:
                rows.append(
                    {
                        **base,
                        "arm": arm,
                        "available": False,
                        "reason": (
                            "z0 informationless (n_obs <= 3)"
                            if arm == "z0"
                            else "config absent (overfitting cap)"
                        ),
                        "n_terms_effective": np.nan,
                        "tv": np.nan,
                        "peak_err": np.nan,
                        "recovered": np.nan,
                        "tpr": np.nan,
                        "period_identifiable": identifiable,
                        "n_null": null_counts.get(slot["n_obs"], 0),
                    }
                )
                continue
            peak_err = np.asarray(rec["peak_err"], dtype=float)
            rows.append(
                {
                    **base,
                    "arm": arm,
                    "available": True,
                    "reason": "",
                    "n_terms_effective": float(np.median(rec["n_eff"])),
                    "tv": float(np.median(rec["tv"])) if rec["tv"] else np.nan,
                    "peak_err": (
                        float(np.median(peak_err)) if peak_err.size else np.nan
                    ),
                    "recovered": (
                        float(np.mean(peak_err <= RECOVERY_LN_TOL))
                        if peak_err.size
                        else np.nan
                    ),
                    "tpr": float(np.mean(rec["det"])) if rec["det"] else np.nan,
                    "period_identifiable": identifiable,
                    "n_null": null_counts.get(slot["n_obs"], 0),
                }
            )
    if not rows:
        raise ValueError("no signal cells found in the artifact")
    return rows


def _verdict(rows: list[dict], metric: str, *, lower_is_better: bool) -> dict:
    """Per-cell winner among available arms on one metric."""
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["available"] and np.isfinite(r[metric]):
            by_cell[r["cell_id"]].append(r)
    tally: dict[str, int] = defaultdict(int)
    for cell_rows in by_cell.values():
        pick = (min if lower_is_better else max)(cell_rows, key=lambda r: r[metric])
        tally[pick["arm"]] += 1
    tally["_cells"] = len(by_cell)
    return dict(tally)


def _summarize(rows: list[dict]) -> str:
    lines = ["regime map: three metrics, aggregated over seeds within each cell", ""]

    n_cells = len({r["cell_id"] for r in rows})
    unavailable = [r for r in rows if not r["available"]]
    z0_out = len([r for r in unavailable if r["arm"] == "z0"])
    cfg_out = len([r for r in unavailable if r["arm"] != "z0"])
    lines.append(f"cells: {n_cells}")
    if z0_out:
        lines.append(
            f"  z0 unavailable in {z0_out} cells (n_obs <= 3: profile statistic is"
            " informationless, not merely noisy)"
        )
    if cfg_out:
        lines.append(
            f"  requested harv config absent in {cfg_out} cells (harv's overfitting"
            " cap forces n_terms=1 at n_obs <= 8)"
        )

    for metric, lower in (("tv", True), ("peak_err", True), ("tpr", False)):
        sub = [r for r in rows if np.isfinite(r[metric])]
        if not sub:
            lines.append(f"\n{metric}: not computed (missing reference or nulls)")
            continue
        lines.append(f"\n{metric} (per-cell winner; {'lower' if lower else 'higher'} better):")
        v = _verdict(rows, metric, lower_is_better=lower)
        cells = v.pop("_cells")
        for arm, n in sorted(v.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {arm:<16} wins {n:>4}/{cells}")
        lines.append("  medians by arm:")
        for arm in ("delta_default", "delta_h1", "z0"):
            vals = [r[metric] for r in sub if r["arm"] == arm]
            if vals:
                lines.append(f"    {arm:<16} {np.median(vals):.4f}  (n={len(vals)})")

    # Sparse-data slice, which is the regime that motivated this axis.
    lines.append("\nsparse-data slice (median tv by n_obs):")
    for n_obs in sorted({r["n_obs"] for r in rows}):
        parts = []
        for arm in ("delta_default", "delta_h1", "z0"):
            vals = [
                r["tv"] for r in rows
                if r["n_obs"] == n_obs and r["arm"] == arm and np.isfinite(r["tv"])
            ]
            parts.append(f"{arm}={np.median(vals):.3f}" if vals else f"{arm}=n/a")
        lines.append(f"  n_obs={n_obs:<3} " + "  ".join(parts))

    non_ident = sorted({r["period_ratio"] for r in rows if not r["period_identifiable"]})
    if non_ident:
        lines.append(
            f"\nNOTE: peak_err is not meaningful at P/T_span in {non_ident}"
            " -- the period is not localizable from the data there."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--null-artifact", type=Path, default=None)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=0.1)
    parser.add_argument("--default-config", type=str, default="H2_s1")
    parser.add_argument("--control-config", type=str, default="H1_s1")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = reduce_regime_map(
        args.artifact,
        args.null_artifact,
        fpr=args.fpr,
        beta=args.beta,
        floor=args.floor,
        default_config=args.default_config,
        control_config=args.control_config,
    )
    print(_summarize(rows))
    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
