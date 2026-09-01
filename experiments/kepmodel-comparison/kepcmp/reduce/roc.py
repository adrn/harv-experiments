"""Reduction 2: detection performance as an ROC.

Thresholds come from the null (``K = 0``) simulations, so the two statistics never
have to be put on a common scale --- which matters, because ``Delta`` is in nats with
a meaningful zero while ``z0`` is a non-negative profile ratio. Each statistic is
reduced to its own detection scalar (the grid maximum), the null distribution of that
scalar sets the threshold at a requested false-positive rate, and the true-positive
rate is measured per cell against that threshold.

Null simulations must be matched in ``n_obs`` (and hence in the number of degrees of
freedom) to the signal simulations they calibrate, so thresholds are computed
per ``n_obs`` rather than pooled.
"""

from __future__ import annotations

__all__ = ("main", "reduce_roc")

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from kepcmp.reduce.common import detection_scalars as _detection_scalars


def reduce_roc(
    signal_path: Path,
    null_path: Path,
    *,
    fpr: float = 0.01,
) -> list[dict]:
    """TPR per (cell, statistic) at a threshold set to ``fpr`` on the nulls."""
    sig_scal, sig_attr = _detection_scalars(signal_path)
    null_scal, null_attr = _detection_scalars(null_path)

    # Thresholds per (n_obs, statistic): the (1 - fpr) quantile of the nulls.
    null_by_nobs: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sim_id, scal in null_scal.items():
        n_obs = int(null_attr[sim_id]["n_obs"])
        for stat, v in scal.items():
            # Drop z0 where it is informationless (n_obs <= 3): its grid maximum is
            # then noise on a flat curve, and calibrating a threshold against that
            # would produce a meaningless TPR rather than an honest absence.
            if stat == "z0" and not null_attr[sim_id].get("z0_usable", True):
                continue
            null_by_nobs[n_obs][stat].append(v)

    thresholds: dict[tuple[int, str], float] = {}
    n_nulls: dict[int, int] = {}
    for n_obs, per_stat in null_by_nobs.items():
        for stat, vals in per_stat.items():
            arr = np.asarray(vals, dtype=float)
            thresholds[(n_obs, stat)] = float(np.quantile(arr, 1.0 - fpr))
            n_nulls[n_obs] = arr.size

    # Group signal sims by cell.
    by_cell: dict[tuple[str, int], list[str]] = defaultdict(list)
    for sim_id, a in sig_attr.items():
        if bool(a["is_null"]):
            continue
        by_cell[(str(a["cell_id"]), int(a["n_obs"]))].append(sim_id)

    rows: list[dict] = []
    for (cell_id, n_obs), sim_ids in sorted(by_cell.items()):
        stats = sorted(sig_scal[sim_ids[0]].keys())
        a0 = sig_attr[sim_ids[0]]
        for stat in stats:
            key = (n_obs, stat)
            if key not in thresholds:
                continue
            usable = [
                s
                for s in sim_ids
                if stat != "z0" or sig_attr[s].get("z0_usable", True)
            ]
            if not usable:
                continue
            thr = thresholds[key]
            vals = np.array([sig_scal[s][stat] for s in usable], dtype=float)
            rows.append(
                {
                    "cell_id": cell_id,
                    "period_ratio": float(a0["period_ratio"]),
                    "snr": float(a0["snr"]),
                    "eccentricity": float(a0["eccentricity"]),
                    "n_obs": n_obs,
                    "integrated_snr": float(a0["integrated_snr"]),
                    "statistic": stat,
                    "fpr": fpr,
                    "threshold": thr,
                    "n_null": n_nulls[n_obs],
                    "n_signal": int(vals.size),
                    "tpr": float(np.mean(vals > thr)),
                }
            )
    if not rows:
        raise ValueError(
            "no signal cells found; check that --artifact holds signal simulations "
            "and --null-artifact holds K=0 simulations"
        )
    return rows


def _summarize(rows: list[dict], fpr: float) -> str:
    stats = sorted({r["statistic"] for r in rows})
    lines = [f"TPR at FPR={fpr:g}, averaged over cells:"]
    for stat in stats:
        vals = [r["tpr"] for r in rows if r["statistic"] == stat]
        n_null = {r["n_null"] for r in rows if r["statistic"] == stat}
        lines.append(f"  {stat:28s} {np.mean(vals):.3f}  (nulls: {sorted(n_null)})")
    smallest = min(r["n_null"] for r in rows)
    if smallest < 1.0 / fpr * 10:
        lines.append(
            f"  NOTE: {smallest} null simulations cannot resolve FPR={fpr:g};"
            f" need >~{int(10 / fpr)} for a stable threshold."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--null-artifact", type=Path, required=True)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = reduce_roc(args.artifact, args.null_artifact, fpr=args.fpr)
    print(_summarize(rows, args.fpr))
    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
