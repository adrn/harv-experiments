"""The reduction: one row per (simulation, arm), scored against the reference ``R``.

Three families of number per row, in the order they decide things:

- ``d_ln_peak`` --- **primary**. Signed peak-location error
  ``ln P_peak(Delta) - ln P_peak(R)``. The *sign* is the diagnostic: negative means the
  arm peaks short of the reference, which is the short-period bias ``docs/spec.md``
  predicts when the amplitude prior is too tight for a partial arc. The previous run
  measured this as exactly 0 for ``P/T <= 0.3`` and -1.75 / -3.37 at ``P/T = 3`` / ``10``
  under a tight flat prior, so it is the metric with the dynamic range to separate
  exponents.
- ``tv`` --- secondary. Total-variation distance between the arm's tempered period
  prior and the reference's, both built by harv's own ``tempered_period_prior``.
  Deliberately *not* a distance between raw nats curves: the frequency-independent part
  of the Occam factor shifts ``Delta`` bodily while changing no inference, and a
  raw-curve metric would score that shift as error.
- ``occam_*`` / ``shrinkage_*`` / ``cond_*`` --- the decomposition. No longer only a
  diagnostic: the Occam tilt *is* the mechanism a period-dependent prior acts through,
  so ``occam_slope`` (the regression of ``occam`` on ``ln P`` across the grid) is the
  direct measurement of that mechanism.
"""

from __future__ import annotations

__all__ = ("main", "reduce_calibrate")

import argparse
from collections import defaultdict
from pathlib import Path

import harv.periodogram as hp
import numpy as np

from ampcal.artifact import ArtifactReader
from ampcal.reduce.common import (
    as_periodogram_result,
    common_ln_p_grid,
    ln_period_density,
    peak_period,
    tv_distance,
)


def _density(
    values: np.ndarray,
    frequency: np.ndarray,
    ln_p: np.ndarray,
    *,
    t_span: float,
    beta: float,
    floor: float,
) -> np.ndarray:
    """Map a nats curve through harv's own prior builder, then to a ln-P density."""
    prior = hp.tempered_period_prior(
        as_periodogram_result(frequency, values, t_span=t_span),
        beta=beta,
        floor=floor,
    )
    return ln_period_density(prior, ln_p)


def _occam_slope(occam: np.ndarray, ln_period: np.ndarray) -> float:
    """``d occam / d ln P``, the tilt a period-dependent amplitude prior imposes.

    A flat arm's Occam factor still varies with frequency --- it depends on the design
    matrix conditioning, which is why the previous run measured
    ``occam_range ~ 0.8 x occam_mean`` and contradicted ``docs/spec.md``'s claim that
    Occam is flat in frequency. The *difference* in this slope between two arms is the
    part the exponent contributes, and that is what the report reads.
    """
    finite = np.isfinite(occam)
    if finite.sum() < 2:
        return float("nan")
    return float(np.polyfit(ln_period[finite], occam[finite], 1)[0])


def reduce_calibrate(
    artifact: Path,
    *,
    beta: float = 1.0,
    floor: float = 0.1,
    n_grid: int = 2048,
) -> list[dict]:
    """One row per (simulation, arm). Simulations without a reference are skipped."""
    rows: list[dict] = []
    with ArtifactReader(artifact) as art:
        frequency = art.frequency
        period = art.period
        ln_period = np.log(period)
        t_span = float(art.meta["t_span_day"])
        ln_p = common_ln_p_grid(period, n_grid)

        def density(values: np.ndarray) -> np.ndarray:
            return _density(
                values, frequency, ln_p, t_span=t_span, beta=beta, floor=floor
            )

        for sim_id in art.sim_ids():
            ref = art.reference(sim_id)
            if ref is None:
                continue
            a = art.attrs(sim_id)
            rho_ref = density(ref)
            p_ref = peak_period(period, ref)

            for name in art.arm_names(sim_id):
                arm = art.arm(sim_id, name)
                delta = arm["delta"]
                p_peak = peak_period(period, delta)
                rows.append(
                    {
                        "sim_id": sim_id,
                        "cell_id": a["cell_id"],
                        "seed": int(a["seed"]),
                        "population": str(a["population"]),
                        "period_ratio": float(a["period_ratio"]),
                        "snr": float(a["snr"]),
                        "eccentricity": float(a["eccentricity"]),
                        "n_obs": int(a["n_obs"]),
                        "truth_period": float(a.get("truth_period", np.nan)),
                        "arm": name,
                        "exponent": float(arm["exponent"]),
                        "level": float(arm["level"]),
                        "n_terms_requested": int(arm["n_terms_requested"]),
                        "n_terms_effective": int(arm["n_terms_effective"]),
                        "sigma_amp": float(arm["sigma_amp"]),
                        "d_ln_peak": float(np.log(p_peak) - np.log(p_ref)),
                        "abs_d_ln_peak": float(
                            abs(np.log(p_peak) - np.log(p_ref))
                        ),
                        "tv": tv_distance(density(delta), rho_ref, ln_p),
                        "peak_period": p_peak,
                        "peak_period_reference": p_ref,
                        "occam_mean": float(np.nanmean(arm["occam"])),
                        "occam_slope": _occam_slope(arm["occam"], ln_period),
                        "shrinkage_mean": float(np.nanmean(arm["shrinkage"])),
                        "cond_max": float(np.nanmax(arm["cond"])),
                    }
                )
    if not rows:
        raise ValueError(
            "no simulations in this artifact carry a reference statistic; re-run "
            "ampcal.run with --reference-n-mc > 0"
        )
    return rows


def _summarize(rows: list[dict]) -> str:
    lines = ["median |d_ln_peak| against the Keplerian reference (lower is better):", ""]
    by: dict[tuple[str, float, float], list[dict]] = defaultdict(list)
    for r in rows:
        by[(r["population"], r["exponent"], r["level"])].append(r)

    lines.append(
        f"{'population':<13}{'exponent':>10}{'level':>10}"
        f"{'|d_ln_peak|':>13}{'d_ln_peak':>12}{'TV':>8}{'n':>6}"
    )
    for key in sorted(by):
        pop, exponent, level = key
        sub = by[key]
        lines.append(
            f"{pop:<13}{exponent:>10.4f}{level:>10.4g}"
            f"{np.median([r['abs_d_ln_peak'] for r in sub]):>13.4f}"
            f"{np.median([r['d_ln_peak'] for r in sub]):>12.4f}"
            f"{np.median([r['tv'] for r in sub]):>8.3f}"
            f"{len(sub):>6d}"
        )

    for pop in sorted({r["population"] for r in rows}):
        sub = {k: v for k, v in by.items() if k[0] == pop}
        best = min(sub, key=lambda k: np.median([r["abs_d_ln_peak"] for r in sub[k]]))
        lines.append(
            f"\nbest on {pop}: exponent {best[1]:+.4f}, level {best[2]:g} x RMS "
            f"(median |d_ln_peak| "
            f"{np.median([r['abs_d_ln_peak'] for r in sub[best]]):.4f})"
        )
    lines.append("  sign of d_ln_peak: negative = Delta peaks short of the reference")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=0.1)
    parser.add_argument("--n-grid", type=int, default=2048)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = reduce_calibrate(
        args.artifact, beta=args.beta, floor=args.floor, n_grid=args.n_grid
    )
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
