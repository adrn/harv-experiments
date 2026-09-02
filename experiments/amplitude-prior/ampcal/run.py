"""Execute grid cells and write the artifact.

Per simulation this runs, for each arm, one harv periodogram (harv's real
``hp.periodogram`` code path, so the stored ``delta`` is exactly what a user would get)
plus one batched decomposition for ``occam`` / ``shrinkage`` / ``cond``. The Keplerian
reference ``R`` is computed once per simulation and only when ``--reference-n-mc`` is
given; it is the dominant cost.

Parallelism
-----------
SPMD over MPI. Every rank runs this same code, asks ``COMM_WORLD`` which rank it is,
takes its own share of the simulation list, and writes its own ``<out>.rank<NNN>.h5``.
Ranks never communicate during the work and no two write the same file, so nothing here
needs MPI-IO or parallel HDF5 --- :mod:`ampcal.merge` combines the per-rank artifacts
afterwards. There is one ``gather`` at the end, purely so rank 0 can print a summary.
MPI is a launcher, not a message bus.

    mpirun python -m ampcal.run --out signal.h5 --which signal      # a cluster job
    python -m ampcal.run --out signal.h5 --which signal --max-sims 8  # a laptop run

Without mpi4py this is a single rank and writes one unsuffixed file, which is what
makes it runnable on a laptop and from the test suite.

Work is dealt **longest-processing-time-first** rather than by a contiguous slice or a
stride. Per-simulation cost is close to linear in ``n_obs``, and the grid is enumerated
with ``n_obs`` varying fastest, so a stride whose length shares a factor with the
``n_obs`` ladder would hand one rank every cheap cell and another every expensive one.
See :func:`ampcal.mpi.balance`.
"""

from __future__ import annotations

__all__ = ("main", "run_one")

import argparse
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import harv.periodogram as hp
import jax.numpy as jnp
import numpy as np
from unxt import Q, ustrip

from ampcal import linalg
from ampcal.adapters import get_adapter
from ampcal.artifact import ArmRecord, ArtifactWriter, SimRecord
from ampcal.grid import (
    SAMPLES_PER_PEAK,
    Arm,
    Cell,
    enumerate_arms,
    enumerate_cells,
    seeds,
    shared_frequency_grid,
)
from ampcal.mpi import balance, banner, gather, mpi_context, peak_rss_gb
from ampcal.population import POPULATIONS

if TYPE_CHECKING:
    from ampcal.adapters.base import Adapter


def _estimate_cost(cell: Cell) -> float:
    """Relative cost of one simulation, for the longest-first deal.

    Close to linear in ``n_obs``: the frequency grid is fixed, so the per-frequency
    algebra is ``O(n_obs * k^2)`` and the reference MC is ``O(n_obs)`` too. Only ratios
    matter to :func:`~ampcal.mpi.balance`, so a rough fit is enough --- and being wrong
    here costs a little imbalance, never a wrong answer.
    """
    return 5.3 + 0.107 * cell.n_obs


def _rank_path(out: Path, adapter: str, rank: int, size: int) -> Path:
    """This rank's output path. Unsuffixed when there is only one rank.

    **The adapter is in the filename, not only in the directory.** ``OUT`` defaults to
    ``output/<adapter>/``, but it is an override point, and pointing two adapters at one
    directory made both runs write ``signal.rank000.h5`` -- same rank count, same names.
    The result was not a clean overwrite but shards with one run's link heap and the
    other's object headers: names that list fine and objects that will not open, on every
    shard, with a 60 core-hour grid behind them. Putting the adapter here makes that
    collision impossible rather than merely detectable.
    """
    if size <= 1:
        return out
    return out.with_name(f"{out.stem}.{adapter}.rank{rank:03d}{out.suffix}")


def run_one(
    adapter: Adapter,
    cell: Cell,
    seed: int,
    frequency: Q,
    *,
    arms: list[Arm],
    reference_n_mc: int = 0,
    reference_seed: int = 0,
) -> SimRecord:
    """Simulate one dataset and compute every arm on it."""
    data, truth = adapter.simulate(cell, seed)
    rms = adapter.data_rms(data)
    periods = 1.0 / frequency

    base_prior = adapter.trial_prior(data, n_terms=0)
    ln_z_base = adapter.harv_log_prob(
        data, periods[0], n_terms=0, prior=base_prior
    )
    base_blocks = adapter.marg_blocks(data, periods[0], n_terms=0, prior=base_prior)
    terms_h = linalg.evaluate(*base_blocks[:5])

    records: list[ArmRecord] = []
    for arm in arms:
        prior = adapter.arm_prior(data, arm, rms)
        # harv's real code path -> the delta a user would get, cap warning included
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = hp.periodogram(
                data, frequency, prior=prior, n_terms=arm.n_terms
            )
        del caught

        # Decomposition at the *effective* n_terms, which is what delta used.
        eff = int(result.n_terms)
        eff_prior = (
            prior
            if eff == arm.n_terms
            else adapter.arm_prior(data, Arm(eff, arm.exponent, arm.level), rms)
        )
        x_all, ref_blocks = adapter.marg_blocks_batched(
            data, periods, n_terms=eff, prior=eff_prior
        )
        terms_k = linalg.evaluate_batched(
            x_all,
            jnp.asarray(ref_blocks.y),
            jnp.asarray(ref_blocks.cov),
            jnp.asarray(ref_blocks.prior_mu),
            jnp.asarray(ref_blocks.prior_scale_tril),
        )

        records.append(
            ArmRecord(
                name=arm.name,
                n_terms_requested=arm.n_terms,
                n_terms_effective=eff,
                exponent=float(arm.exponent),
                level=float(arm.level),
                sigma_amp=float(ustrip(adapter.obs_unit, arm.level * rms)),
                ln_z_base=float(ln_z_base),
                delta=np.asarray(result.delta_ln_likelihood, dtype=float),
                occam=np.asarray(terms_k["occam"], dtype=float) - terms_h.occam,
                shrinkage=(
                    np.asarray(terms_k["shrinkage"], dtype=float) - terms_h.shrinkage
                ),
                cond=np.asarray(terms_k["cond"], dtype=float),
            )
        )

    reference = None
    if reference_n_mc > 0:
        reference = adapter.reference_ln_z(
            data, periods, n_mc=reference_n_mc, seed=reference_seed
        )

    return SimRecord(
        sim_id=f"{cell.cell_id}_seed{seed:03d}",
        cell_id=cell.cell_id,
        seed=seed,
        population=cell.population,
        period_ratio=cell.period_ratio,
        snr=cell.snr,
        eccentricity=cell.eccentricity,
        n_obs=cell.n_obs,
        integrated_snr=cell.integrated_snr,
        data_rms=float(ustrip(adapter.obs_unit, rms)),
        truth=adapter.truth_floats(truth),
        data=adapter.data_arrays(data),
        arms=records,
        reference=reference,
        reference_n_mc=reference_n_mc,
        reference_seed=reference_seed,
    )


def _smoke_cells(adapter: Adapter) -> list[Cell]:
    """A handful of cells spanning the adapter's own grid, for plumbing validation.

    Both populations and both ends of ``period_ratio``, because the whole experiment
    is a contrast across exactly those two axes --- a smoke set that covered one
    population would validate half the pipeline.
    """
    pr, n = adapter.period_ratios, adapter.n_obs_values
    return [
        Cell(population=POPULATIONS[0], period_ratio=pr[1], snr=10.0,
             eccentricity=0.0, n_obs=n[-1]),
        Cell(population=POPULATIONS[0], period_ratio=pr[-1], snr=30.0,
             eccentricity=0.6, n_obs=n[-1]),
        Cell(population=POPULATIONS[1], period_ratio=pr[1], snr=10.0,
             eccentricity=0.0, n_obs=n[-1]),
        Cell(population=POPULATIONS[1], period_ratio=pr[-1], snr=3.0,
             eccentricity=0.0, n_obs=n[-2]),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", choices=("rv", "gaia"), default="rv")
    parser.add_argument(
        "--which",
        choices=("signal", "smoke"),
        default="smoke",
        help="which cell set to run; 'smoke' is a few cells for pipeline validation",
    )
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1,
                        help="subsample the shared frequency grid")
    parser.add_argument("--n-terms", type=int, nargs="*", default=None)
    parser.add_argument("--exponents", type=float, nargs="*", default=None,
                        help="override the adapter's amplitude-prior exponents")
    parser.add_argument("--levels", type=float, nargs="*", default=None,
                        help="override the sigma_0 multipliers")
    parser.add_argument("--reference-n-mc", type=int, default=0,
                        help="0 disables the reference statistic")
    parser.add_argument(
        "--max-sims", type=int, default=None,
        help="use only N simulations, spread evenly across the whole grid "
             "(smoke tests). Applied before the deal, so every rank agrees",
    )
    parser.add_argument(
        "--mpi", action="store_true",
        help="force mpi4py even if no launcher variable is recognised; without it "
             "MPI is used when OMPI_COMM_WORLD_SIZE / PMI_SIZE / SLURM_PROCID etc. "
             "are set, and skipped entirely otherwise (importing mpi4py runs "
             "MPI_Init, which hangs on a broken MPI)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=None,
        help="print a line every N simulations on non-zero ranks (default 50); "
             "rank 0 always prints every simulation",
    )
    args = parser.parse_args(argv)

    comm, rank, size = mpi_context(force=args.mpi)
    adapter = get_adapter(args.adapter)
    frequency = shared_frequency_grid(adapter)[:: args.stride]

    if args.which == "signal":
        cells, n_seeds = enumerate_cells(adapter), args.n_seeds or 16
    else:  # smoke
        cells, n_seeds = _smoke_cells(adapter), args.n_seeds or 2

    arm_kwargs = {}
    if args.exponents:
        arm_kwargs["exponents"] = tuple(args.exponents)
    if args.levels:
        arm_kwargs["levels"] = tuple(args.levels)
    if args.n_terms:
        arm_kwargs["n_terms_values"] = tuple(args.n_terms)
    arms = enumerate_arms(adapter, **arm_kwargs)

    all_jobs = [(c, s) for c in cells for s in seeds(n_seeds, offset=args.seed_offset)]
    if args.max_sims and args.max_sims < len(all_jobs):
        # A fixed-seed sample without replacement, not the first N and not a stride.
        #
        # Truncation returns one corner of the grid: cells come from
        # itertools.product, so the first N all share one population and the smallest
        # period_ratio. But a *stride* is no better, because the stride length aliases
        # against the axis lengths -- a step that is a multiple of the n_obs ladder
        # makes a "grid-spanning" sample silently cover one rung. (The same aliasing is
        # why ampcal.mpi.balance deals longest-first rather than by stride.)
        #
        # Sub-sampled before the deal, so every rank agrees on the same short list.
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(all_jobs), size=args.max_sims, replace=False))
        all_jobs = [all_jobs[i] for i in keep]
    jobs = balance(all_jobs, [_estimate_cost(c) for c, _ in all_jobs], rank, size)

    if rank == 0:
        banner(
            comm,
            size,
            len(all_jobs),
            adapter=adapter.name,
            which=args.which,
            grid=f"{frequency.shape[0]} frequencies (stride {args.stride})",
            arms=f"{len(arms)} arms"
                 + (f" + reference n_mc={args.reference_n_mc}"
                    if args.reference_n_mc else ""),
            output=_rank_path(args.out, adapter.name, rank, size).name,
        )

    started = time.perf_counter()
    out_path = _rank_path(args.out, adapter.name, rank, size)
    every = args.progress_every if args.progress_every is not None else 50
    n_failed = 0

    with ArtifactWriter(
        out_path,
        frequency=np.asarray(ustrip(f"1/{adapter.time_unit}", frequency), dtype=float),
        which=args.which,
        samples_per_peak=SAMPLES_PER_PEAK,
        stride=args.stride,
        **adapter.meta(),
    ) as writer:
        for i, (cell, seed) in enumerate(jobs, start=1):
            sim_id = f"{cell.cell_id}_seed{seed:03d}"
            t0 = time.perf_counter()
            try:
                rec = run_one(
                    adapter,
                    cell,
                    seed,
                    frequency,
                    arms=arms,
                    reference_n_mc=args.reference_n_mc,
                    reference_seed=seed,
                )
            except Exception as exc:  # noqa: BLE001
                # One bad cell must not lose the hundreds this rank is holding.
                # Counted, reported, and surfaced in the exit status.
                n_failed += 1
                print(
                    f"[rank {rank:05d}] FAILED {sim_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            writer.write(rec)
            # Rank 0 narrates; the rest report periodically. A unit is seconds here,
            # but 1000 ranks x 450 simulations is half a million lines in one log.
            if rank == 0 or (every and i % every == 0):
                print(
                    f"[rank {rank:05d}] [{i}/{len(jobs)}] {sim_id} "
                    f"{time.perf_counter() - t0:.2f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    mine = {
        "rank": rank,
        "n_sims": len(jobs) - n_failed,
        "n_failed": n_failed,
        "seconds": elapsed,
        "peak_rss_gb": peak_rss_gb(),
    }
    print(
        f"[rank {rank:05d}] {mine['n_sims']} simulations in {elapsed / 60:.1f} min "
        f"-> {out_path.name}",
        flush=True,
    )

    everyone = gather(comm, mine)
    if rank != 0:
        return 0

    total = sum(r["n_sims"] for r in everyone)
    failed = sum(r["n_failed"] for r in everyone)
    slowest = max(r["seconds"] for r in everyone)
    core_seconds = sum(r["seconds"] for r in everyone)
    print(f"\ndone: {total:,} simulations across {size} rank(s)")
    print(f"  slowest rank : {slowest / 3600:.2f} h")
    if total:
        print(
            f"  per sim      : {slowest * size / total:.2f} s/rank-sim "
            f"(core-hours: {core_seconds / 3600:,.1f})"
        )
        # The number that says whether more ranks would have helped. Anything much
        # below ~85% means the deal, not the compute, is the limit.
        print(f"  balance      : {100 * core_seconds / (slowest * size):.0f}% of the "
              f"allocation busy")
    worst = max(r["peak_rss_gb"] for r in everyone)
    print(f"  peak RSS     : {worst:.2f} GB/rank")
    if failed:
        print(f"  FAILED       : {failed:,} simulations -- see the rank logs")
    if size > 1:
        print(f"  next         : python -m ampcal.merge --out {args.out} "
              f"{args.out.with_suffix('')}.{adapter.name}.rank*{args.out.suffix}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
