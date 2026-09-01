"""Reduction 4: which ``sigma_amp`` makes ``Delta`` reproduce the Keplerian reference.

For every ``sigma_amp`` in the sweep, ``Delta`` and the reference ``R`` are each
mapped through ``tempered_period_prior`` at matched ``beta`` and ``floor``, and the
two resulting densities in ``ln P`` are compared by:

- ``tv`` --- total-variation distance. Deliberately *not* a distance between raw nats
  curves: the frequency-independent part of the Occam factor shifts ``Delta`` bodily
  while changing no inference, and a raw-curve metric would score that shift as
  error.
- ``d_ln_peak`` --- signed peak-location error ``ln P_peak(Delta) - ln P_peak(R)``.
  The *sign* is the diagnostic: negative means ``Delta`` peaks short of the
  reference, which is the short-period bias the design doc predicts when
  ``sigma_amp`` is too tight for a partial arc.

``z0`` is scored the same way as a free reference point: it has the Fourier-basis
approximation but no amplitude prior, so ``tv(z0, R)`` bounds how much of any
``Delta`` discrepancy can be blamed on the basis rather than on ``sigma_amp``.
"""

from __future__ import annotations

__all__ = ("main", "reduce_calibrate")

import argparse
from collections import defaultdict
from pathlib import Path

import harv.periodogram as hp
import numpy as np

from kepcmp.artifact import ArtifactReader
from kepcmp.reduce.common import (
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


def _row(
    label: str,
    values: np.ndarray,
    extra: dict,
    *,
    sim_id: str,
    attrs: dict,
    period: np.ndarray,
    rho: np.ndarray,
    rho_ref: np.ndarray,
    p_ref: float,
    ln_p: np.ndarray,
) -> dict:
    """One output row. Module-level, so nothing closes over loop variables."""
    return {
        "sim_id": sim_id,
        "cell_id": attrs["cell_id"],
        "seed": int(attrs["seed"]),
        "period_ratio": float(attrs["period_ratio"]),
        "snr": float(attrs["snr"]),
        "eccentricity": float(attrs["eccentricity"]),
        "n_obs": int(attrs["n_obs"]),
        "truth_period": float(attrs.get("truth_period", np.nan)),
        "statistic": label,
        "tv": tv_distance(rho, rho_ref, ln_p),
        "d_ln_peak": float(np.log(peak_period(period, values)) - np.log(p_ref)),
        "peak_period": peak_period(period, values),
        "peak_period_reference": p_ref,
        **extra,
    }


def reduce_calibrate(
    artifact: Path,
    *,
    beta: float = 1.0,
    floor: float = 0.1,
    n_grid: int = 2048,
) -> list[dict]:
    """One row per (sim, harv config), plus one ``z0`` row per sim."""
    rows: list[dict] = []
    with ArtifactReader(artifact) as art:
        frequency = art.frequency
        period = art.period
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

            # z0 is only scored where it exists: at n_obs <= p + d it is flat to
            # numerical noise or was uncomputable, and building a prior from an
            # all-nan curve raises inside tempered_period_prior anyway.
            candidates: list[tuple[str, np.ndarray, dict]] = []
            if art.z0_usable(sim_id):
                candidates.append(
                    ("z0", art.z0(sim_id),
                     {"n_terms_effective": 1, "sigma_amp_mult": np.nan})
                )
            for key in art.stat_keys(sim_id):
                st = art.stat(sim_id, key)
                candidates.append(
                    (
                        f"delta:{key}",
                        st["delta"],
                        {
                            "n_terms_effective": int(st["n_terms_effective"]),
                            "sigma_amp_mult": float(st["sigma_amp_mult"]),
                        },
                    )
                )
            for label, values, extra in candidates:
                rows.append(
                    _row(
                        label,
                        values,
                        extra,
                        sim_id=sim_id,
                        attrs=a,
                        period=period,
                        rho=density(values),
                        rho_ref=rho_ref,
                        p_ref=p_ref,
                        ln_p=ln_p,
                    )
                )
    if not rows:
        raise ValueError(
            "no simulations in this artifact carry a reference statistic; re-run "
            "kepcmp.run with --reference-n-mc > 0"
        )
    return rows


def _summarize(rows: list[dict]) -> str:
    lines = ["TV distance to the Keplerian reference (lower is better):"]
    by_stat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stat[r["statistic"]].append(r)

    lines.append(f"{'statistic':<20}{'TV':>8}{'d_ln_peak':>12}{'n':>5}")
    for stat in sorted(by_stat):
        sub = by_stat[stat]
        lines.append(
            f"{stat:<20}"
            f"{np.median([r['tv'] for r in sub]):>8.3f}"
            f"{np.median([r['d_ln_peak'] for r in sub]):>12.3f}"
            f"{len(sub):>5d}"
        )

    deltas = [r for r in rows if r["statistic"].startswith("delta:")]
    if deltas:
        by_mult: dict[float, list[dict]] = defaultdict(list)
        for r in deltas:
            by_mult[r["sigma_amp_mult"]].append(r)
        best = min(by_mult, key=lambda m: np.median([r["tv"] for r in by_mult[m]]))
        lines.append(
            f"\nbest sigma_amp multiplier by median TV: {best:g} x data RMS"
            f"  (TV {np.median([r['tv'] for r in by_mult[best]]):.3f})"
        )
        lines.append("  sign of d_ln_peak: negative = Delta peaks short of reference")
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
