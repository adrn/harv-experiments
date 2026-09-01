"""Reduction 1: where the two statistics disagree, and which term explains it.

Tests the design doc's central prediction: ``Delta`` and ``z0`` agree on peak
location wherever the amplitude is well constrained, and diverge exactly where it is
not. The mechanism is that the Occam factor is frequency-independent to first order
--- it shifts the periodogram bodily --- and its frequency-dependence enters only
through ``A_tilde(nu)``, i.e. the window function.

So the reported quantities are:

- ``peak_agree`` --- do the two peak at the same trial period, within half a
  periodogram peak width (``1 / t_span`` in frequency)?
- ``occam_range`` --- ``max - min`` of the Occam term across the grid. This *is* the
  frequency-dependent part, and the prediction says disagreement should track it.
- ``occam_mean`` --- the bodily shift, which should explain nothing.
- ``shrinkage_range`` --- expected negligible once amplitudes are well measured.
- ``ln_amp_constraint`` --- ``occam_mean / d``, i.e. roughly
  ``ln(sigma_amp / sigma_post)``: how many nats of amplitude constraint the data
  supply relative to the prior. This is the "well constrained" axis the prediction
  is stated against.

Only ``n_terms=1`` configurations are compared against ``z0``, since that is the
only case where the two models have the same columns (harv ``d=2`` = kepmodel
``d=2``). Higher ``n_terms`` rows are reported separately as harv-vs-harv.

**Neither statistic is the yardstick here.** ``d_ln_peak`` is a *symmetric*
disagreement measure --- it says the two differ, not which is right. Both are
additionally scored against the injected period (``d_ln_peak_truth_delta``,
``d_ln_peak_truth_z0``), so nothing in this reduction implicitly privileges
kepmodel's profile statistic. Truth for the calibration question is the Keplerian
reference ``R``, handled in :mod:`kepcmp.reduce.calibrate`, where ``z0`` is scored
as just another statistic.
"""

from __future__ import annotations

__all__ = ("main", "reduce_decompose")

import argparse
from pathlib import Path

import numpy as np

from kepcmp.artifact import ArtifactReader
from kepcmp.reduce.common import peak_period


def reduce_decompose(path: Path) -> list[dict]:
    """One row per (sim, harv config)."""
    rows: list[dict] = []
    with ArtifactReader(path) as art:
        period = art.period
        t_span = float(art.meta["t_span_day"])
        # Half a peak width in frequency, converted to a ln-period tolerance at the
        # peak: |d ln P| = |d f| / f.
        for sim_id in art.sim_ids():
            a = art.attrs(sim_id)
            z0 = art.z0(sim_id)
            p_z0 = peak_period(period, z0)
            for key in art.stat_keys(sim_id):
                st = art.stat(sim_id, key)
                delta = st["delta"]
                occam = st["occam"]
                shrink = st["shrinkage"]
                p_delta = peak_period(period, delta)

                half_width_frac = 0.5 / (t_span / max(p_delta, 1e-30))
                d_ln_p = abs(np.log(p_delta) - np.log(p_z0))
                n_cols_per_harmonic = 2
                d_eff = n_cols_per_harmonic * int(st["n_terms_effective"])

                # Score both against the injected period, so this reduction never
                # implicitly treats either statistic as the reference.
                p_true = float(a.get("truth_period", np.nan))
                if np.isfinite(p_true) and p_true > 0 and not bool(a["is_null"]):
                    err_delta = abs(np.log(p_delta) - np.log(p_true))
                    err_z0 = abs(np.log(p_z0) - np.log(p_true))
                else:
                    err_delta = err_z0 = np.nan

                rows.append(
                    {
                        "sim_id": sim_id,
                        "cell_id": a["cell_id"],
                        "seed": int(a["seed"]),
                        "period_ratio": float(a["period_ratio"]),
                        "snr": float(a["snr"]),
                        "eccentricity": float(a["eccentricity"]),
                        "n_obs": int(a["n_obs"]),
                        "is_null": bool(a["is_null"]),
                        "config": key,
                        "n_terms_effective": int(st["n_terms_effective"]),
                        "sigma_amp_mult": float(st["sigma_amp_mult"]),
                        "truth_period": float(a.get("truth_period", np.nan)),
                        "peak_period_delta": p_delta,
                        "peak_period_z0": p_z0,
                        "d_ln_peak": d_ln_p,
                        "d_ln_peak_truth_delta": err_delta,
                        "d_ln_peak_truth_z0": err_z0,
                        "peak_agree": bool(d_ln_p <= half_width_frac),
                        "comparable_to_z0": int(st["n_terms_effective"]) == 1,
                        "occam_mean": float(np.mean(occam)),
                        "occam_range": float(np.ptp(occam)),
                        "shrinkage_mean": float(np.mean(shrink)),
                        "shrinkage_range": float(np.ptp(shrink)),
                        "ln_amp_constraint": float(np.mean(occam)) / max(d_eff, 1),
                        "max_cond": float(np.nanmax(st["cond"])),
                    }
                )
    return rows


def _summarize(rows: list[dict]) -> str:
    comparable = [r for r in rows if r["comparable_to_z0"] and not r["is_null"]]
    if not comparable:
        return "no n_terms=1 signal rows to compare"
    agree = np.array([r["peak_agree"] for r in comparable])
    constraint = np.array([r["ln_amp_constraint"] for r in comparable])
    occam_range = np.array([r["occam_range"] for r in comparable])
    shrink_range = np.array([np.abs(r["shrinkage_range"]) for r in comparable])

    err_d = np.array([r["d_ln_peak_truth_delta"] for r in comparable], dtype=float)
    err_z = np.array([r["d_ln_peak_truth_z0"] for r in comparable], dtype=float)

    lines = [
        f"n_terms=1 signal rows: {len(comparable)}",
        "  both scored against the INJECTED period (neither is the yardstick):",
        f"    |d ln P| delta vs truth: median {np.nanmedian(err_d):.4f}",
        f"    |d ln P| z0    vs truth: median {np.nanmedian(err_z):.4f}",
        f"  symmetric delta-vs-z0 agreement : {agree.mean():.3f}",
        f"  occam frequency-range   : median {np.median(occam_range):.3f} nats"
        f"  (bodily shift median {np.median([r['occam_mean'] for r in comparable]):.2f})",
        f"  |shrinkage| range       : median {np.median(shrink_range):.2e} nats",
    ]
    # The prediction: agreement should track amplitude constraint.
    if agree.any() and not agree.all():
        lines.append(
            f"  ln_amp_constraint       : agree median"
            f" {np.median(constraint[agree]):.2f} vs disagree median"
            f" {np.median(constraint[~agree]):.2f}"
        )
        lines.append(
            f"  occam_range             : agree median"
            f" {np.median(occam_range[agree]):.3f} vs disagree median"
            f" {np.median(occam_range[~agree]):.3f}"
        )
    else:
        lines.append(
            "  (peak agreement is uniform on this artifact, so the prediction"
            " cannot be tested from it)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = reduce_decompose(args.artifact)
    print(_summarize(rows))
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
