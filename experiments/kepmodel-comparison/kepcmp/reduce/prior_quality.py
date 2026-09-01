"""Reduction 3: which statistic makes the better interim period prior.

This is the reduction that measures what harv actually uses the periodogram for.
Both ``Delta`` and kepmodel's ``z0`` are wrapped in a ``PeriodogramResult`` and fed
through harv's own ``tempered_period_prior``, so the only thing that differs between
the two arms is the statistic --- no duplicated prior code, no scale matching.

**Accept count is deliberately not the headline metric.** Per ``docs/spec.md``
("Interpreting acceptance"), the rejection step accepts with probability
``exp(L - max L)`` where ``max L`` is the maximum over the *drawn* samples, so a
better prior can find the peak, raise ``max L``, and report *fewer* accepted samples
against a correctly higher bar. The reported metrics are therefore:

- ``max_log_likelihood`` --- has the mode actually been found? Higher is strictly
  better and is comparable across arms.
- ``evidence_ess`` --- is the run resolved at all? O(1) means the evidence integral
  is dominated by one lucky draw and nothing else in the row is trustworthy.
- ``recovered`` --- is any accepted sample within ``ln`` tolerance of the true period?
- ``n_accepted`` --- reported, but read only alongside the two above.

A log-uniform arm is included as the control: without it, "the periodogram prior
accepted more" has no baseline.
"""

from __future__ import annotations

__all__ = ("main", "reduce_prior_quality")

import argparse
import warnings
from pathlib import Path

import harv.periodogram as hp
import numpy as np
import numpyro.distributions as dist
from harv.data import AbstractData
from harv.distributions import QuantityDistribution as QD
from harv.samplers import RejectionSampler
from unxt import ustrip

from kepcmp.adapters import get_adapter
from kepcmp.adapters.base import Adapter
from kepcmp.artifact import ArtifactReader
from kepcmp.reduce.common import as_periodogram_result

TIME_UNIT = "day"


def _run_arm(
    adapter: Adapter,
    data: AbstractData,
    period_prior: object,
    *,
    n_prior_samples: int,
    seed: int,
    truth_period: float,
    ln_tol: float,
) -> dict:
    """One rejection run. Warnings are captured, not printed."""
    prior = adapter.science_prior(data, period_prior)
    sampler = RejectionSampler(prior, adapter.science_model())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        samples = sampler.run(
            data,
            n_prior_samples=n_prior_samples,
            seed=seed,
            return_evidence_stats=True,
            return_logprobs=True,
        )
    under_resolved = any(
        "Under-resolved" in str(w.message) for w in caught
    )
    diag = samples.acceptance_diagnostics()
    n_acc = int(samples.n_samples)
    recovered = False
    if n_acc:
        p = np.asarray(ustrip(TIME_UNIT, samples["period"]), dtype=float)
        recovered = bool(
            np.any(np.abs(np.log(p) - np.log(truth_period)) <= ln_tol)
        )
    return {
        "n_accepted": n_acc,
        "max_log_likelihood": float(diag["max_log_likelihood"]),
        "evidence_ess": float(diag["evidence_ess"]),
        "logZ_int": float(diag["logZ_int"]),
        "well_resolved": bool(diag["well_resolved"]),
        "under_resolved_warning": under_resolved,
        "recovered": recovered,
    }


def reduce_prior_quality(
    artifact: Path,
    *,
    config: str | None = None,
    n_prior_samples: int = 100_000,
    beta: float = 1.0,
    floor: float = 0.1,
    ln_tol: float = 0.05,
    seed: int = 0,
    limit: int | None = None,
) -> list[dict]:
    """One row per (sim, arm). Arms: ``delta``, ``z0``, ``loguniform`` (control)."""
    rows: list[dict] = []
    with ArtifactReader(artifact) as art:
        adapter = get_adapter(art.adapter_name)
        frequency = art.frequency
        period = art.period
        t_span = float(art.meta["t_span_day"])
        sim_ids = art.sim_ids()
        if limit:
            sim_ids = sim_ids[:limit]

        for sim_id in sim_ids:
            a = art.attrs(sim_id)
            if bool(a["is_null"]):
                continue
            truth_period = float(a["truth_period"])
            data = adapter.data_from_arrays(art.data(sim_id))

            keys = art.stat_keys(sim_id)
            chosen = config if config in keys else keys[0]
            delta = art.stat(sim_id, chosen)["delta"]

            arms = {
                "delta": hp.tempered_period_prior(
                    as_periodogram_result(frequency, delta, t_span=t_span),
                    beta=beta,
                    floor=floor,
                ),
                "z0": hp.tempered_period_prior(
                    as_periodogram_result(frequency, art.z0(sim_id), t_span=t_span),
                    beta=beta,
                    floor=floor,
                ),
                "loguniform": QD(
                    dist.LogUniform(float(period.min()), float(period.max())),
                    TIME_UNIT,
                ),
            }
            for arm, period_prior in arms.items():
                res = _run_arm(
                    adapter,
                    data,
                    period_prior,
                    n_prior_samples=n_prior_samples,
                    seed=seed,
                    truth_period=truth_period,
                    ln_tol=ln_tol,
                )
                rows.append(
                    {
                        "sim_id": sim_id,
                        "cell_id": a["cell_id"],
                        "period_ratio": float(a["period_ratio"]),
                        "snr": float(a["snr"]),
                        "eccentricity": float(a["eccentricity"]),
                        "n_obs": int(a["n_obs"]),
                        "config": chosen,
                        "arm": arm,
                        "truth_period": truth_period,
                        **res,
                    }
                )
    return rows


def _summarize(rows: list[dict]) -> str:
    arms = sorted({r["arm"] for r in rows})
    lines = [
        f"{len({r['sim_id'] for r in rows})} simulations x {len(arms)} arms",
        f"{'arm':<12}{'recovered':>10}{'max_lnL':>12}{'ess':>9}"
        f"{'resolved':>10}{'n_acc':>9}",
    ]
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        lines.append(
            f"{arm:<12}"
            f"{np.mean([r['recovered'] for r in sub]):>10.3f}"
            f"{np.median([r['max_log_likelihood'] for r in sub]):>12.2f}"
            f"{np.median([r['evidence_ess'] for r in sub]):>9.2f}"
            f"{np.mean([r['well_resolved'] for r in sub]):>10.3f}"
            f"{np.median([r['n_accepted'] for r in sub]):>9.1f}"
        )
    lines.append(
        "  (max_lnL and ess are the comparable metrics; n_acc is not comparable"
        " across arms until max_lnL has converged)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--config", type=str, default=None,
                        help="harv config key, e.g. H2_s1; default: first available")
    parser.add_argument("--n-prior-samples", type=int, default=100_000)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=0.1)
    parser.add_argument("--ln-tol", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = reduce_prior_quality(
        args.artifact,
        config=args.config,
        n_prior_samples=args.n_prior_samples,
        beta=args.beta,
        floor=args.floor,
        ln_tol=args.ln_tol,
        seed=args.seed,
        limit=args.limit,
    )
    if not rows:
        print("no signal simulations in artifact")
        return 1
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
