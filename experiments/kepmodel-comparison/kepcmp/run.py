"""Execute grid cells and write the artifact.

Per simulation this does one kepmodel scan, then for each ``(n_terms, sigma_amp)``
configuration one harv periodogram (harv's real ``hp.periodogram`` code path, so the
stored ``delta`` is exactly what a user would get) plus one batched decomposition
for ``occam`` / ``shrinkage`` / ``cond``.

The reference statistic ``R`` is computed only when ``--reference-n-mc`` is given.

Parallelism
-----------
The grid is embarrassingly parallel, so this is sharded rather than threaded: the
matrices are tiny (``n_obs <= 80``, ``k <= 17``) and do not parallelize within a
process. Two equivalent ways to shard, both producing files that
:mod:`kepcmp.merge` combines:

- **MPI**: ``mpirun -n 32 python -m kepcmp.run --out signal.h5 ...``. Each rank runs
  its own share and writes ``signal.rank<NNN>.h5``; jobs are dealt round-robin, which
  matters because per-simulation cost grows with ``n_obs`` and a contiguous split
  would leave ranks idle at the end.
- **Explicit**: ``--shard i --n-shards N``, or the seed-splitting in
  ``launch_full_grid.sh``, for a machine without MPI.

The ranks never talk to each other --- each writes its own artifact and
:mod:`kepcmp.merge` combines them afterwards --- so the rank is read from the
launcher's environment variables rather than by linking ``libmpi``. That keeps
``mpi4py`` off the dependency list entirely and, more usefully, means a broken or
mismatched MPI runtime cannot take the run down at ``MPI_Init``. The identity found
is printed at startup, so a launcher whose variables are not recognised shows up
immediately as ``1 shard (serial)`` instead of silently having every rank overwrite
one file.
"""

from __future__ import annotations

__all__ = ("main", "run_one", "shard_of")

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import harv.periodogram as hp
import jax.numpy as jnp
import numpy as np
from unxt import Q, ustrip

from kepcmp import linalg
from kepcmp.adapters import get_adapter
from kepcmp.artifact import ArtifactWriter, SimRecord, StatRecord
from kepcmp.grid import (
    N_TERMS_VALUES,
    SAMPLES_PER_PEAK,
    SIGMA_AMP_MULTIPLIERS,
    Cell,
    enumerate_cells,
    enumerate_null_cells,
    seeds,
    shared_frequency_grid,
)

if TYPE_CHECKING:
    from kepcmp.adapters.base import Adapter


#: Rank/size environment variables, in priority order, by launcher.
_RANK_ENV: tuple[tuple[str, str, str], ...] = (
    ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE", "Open MPI"),
    ("PMI_RANK", "PMI_SIZE", "MPICH/Hydra"),
    ("PMIX_RANK", "PMIX_SIZE", "PMIx"),
    ("SLURM_PROCID", "SLURM_NTASKS", "Slurm"),
)


def shard_of(
    shard: int | None, n_shards: int | None
) -> tuple[int, int, str]:
    """``(index, count, source)`` for this process.

    Explicit ``--shard/--n-shards`` wins; otherwise the MPI launcher's own rank
    variables are read. Nothing here links ``libmpi``: the shards are independent
    processes writing independent files, so the rank is all that is needed.
    """
    if shard is not None or n_shards is not None:
        return int(shard or 0), int(n_shards or 1), "--shard"
    for rank_var, size_var, label in _RANK_ENV:
        rank, size = os.environ.get(rank_var), os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size), label
    return 0, 1, "serial"


def _shard_path(out: Path, index: int, count: int) -> Path:
    """Per-shard output path. Unsuffixed when there is only one shard."""
    if count <= 1:
        return out
    return out.with_name(f"{out.stem}.rank{index:03d}{out.suffix}")


def run_one(
    adapter: Adapter,
    cell: Cell,
    seed: int,
    frequency: Q,
    *,
    n_terms_values: tuple[int, ...] = N_TERMS_VALUES,
    sigma_amp_mults: tuple[float, ...] = SIGMA_AMP_MULTIPLIERS,
    reference_n_mc: int = 0,
    reference_seed: int = 0,
) -> SimRecord:
    """Simulate one dataset and compute every statistic on it."""
    data, truth = adapter.simulate(cell, seed)
    rms = adapter.data_rms(data)
    periods = 1.0 / frequency

    # z0 with graceful degradation: on sparse data the profile statistic can be
    # informationless or outright uncomputable, and that is a result to record, not an
    # error to crash on. See HarvKepmodelAdapter.kepmodel_z0.
    chi2_base, z0, z0_degenerate, z0_reason = adapter.kepmodel_z0(data, frequency)

    stats: list[StatRecord] = []
    for n_terms in n_terms_values:
        for mult in sigma_amp_mults:
            sigma_amp = mult * rms
            prior = adapter.trial_prior(data, n_terms=n_terms, sigma_amp=sigma_amp)
            # harv's real code path -> the delta a user would get, cap warning included
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = hp.periodogram(data, frequency, prior=prior, n_terms=n_terms)
            del caught
            base_prior = adapter.trial_prior(data, n_terms=0)
            ln_z_base = adapter.harv_log_prob(
                data, periods[0], n_terms=0, prior=base_prior
            )

            # Decomposition at the *effective* n_terms, which is what delta used.
            eff = int(result.n_terms)
            eff_prior = (
                prior
                if eff == n_terms
                else adapter.trial_prior(data, n_terms=eff, sigma_amp=sigma_amp)
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
            base_blocks = adapter.marg_blocks(
                data, periods[0], n_terms=0, prior=base_prior
            )
            terms_h = linalg.evaluate(*base_blocks[:5])

            stats.append(
                StatRecord(
                    n_terms_requested=n_terms,
                    n_terms_effective=eff,
                    sigma_amp=float(ustrip(adapter.obs_unit, sigma_amp)),
                    sigma_amp_mult=float(mult),
                    ln_z_base=float(ln_z_base),
                    delta=np.asarray(result.delta_ln_likelihood, dtype=float),
                    occam=np.asarray(terms_k["occam"], dtype=float) - terms_h.occam,
                    shrinkage=(
                        np.asarray(terms_k["shrinkage"], dtype=float)
                        - terms_h.shrinkage
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
        period_ratio=cell.period_ratio,
        snr=cell.snr,
        eccentricity=cell.eccentricity,
        n_obs=cell.n_obs,
        integrated_snr=cell.integrated_snr,
        is_null=cell.is_null,
        data_rms=float(ustrip(adapter.obs_unit, rms)),
        truth=adapter.truth_floats(truth),
        data=adapter.data_arrays(data),
        z0=np.asarray(z0, dtype=float),
        chi2_base=float(chi2_base),
        z0_degenerate=z0_degenerate,
        z0_reason=z0_reason,
        stats=stats,
        reference=reference,
        reference_n_mc=reference_n_mc,
        reference_seed=reference_seed,
    )


def _smoke_cells(adapter: Adapter) -> list[Cell]:
    """A handful of cells spanning the adapter's own grid, for plumbing validation."""
    pr, n = adapter.period_ratios, adapter.n_obs_values
    return [
        Cell(period_ratio=pr[1], snr=10.0, eccentricity=0.0, n_obs=n[-1]),
        Cell(period_ratio=pr[2], snr=30.0, eccentricity=0.6, n_obs=n[-1]),
        Cell(period_ratio=pr[-2], snr=3.0, eccentricity=0.0, n_obs=n[-2]),
        Cell(period_ratio=pr[2], snr=0.0, eccentricity=0.0, n_obs=n[-1]),  # null
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", choices=("rv", "gaia"), default="rv")
    parser.add_argument(
        "--which",
        choices=("signal", "null", "smoke"),
        default="smoke",
        help="which cell set to run; 'smoke' is a few cells for pipeline validation",
    )
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1,
                        help="subsample the shared frequency grid")
    parser.add_argument("--n-terms", type=int, nargs="*", default=None)
    parser.add_argument("--sigma-amp-mults", type=float, nargs="*", default=None)
    parser.add_argument("--reference-n-mc", type=int, default=0,
                        help="0 disables the reference statistic")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many simulations *per shard*")
    parser.add_argument("--shard", type=int, default=None,
                        help="shard index; default: this process's MPI rank")
    parser.add_argument("--n-shards", type=int, default=None,
                        help="shard count; default: the MPI world size")
    args = parser.parse_args(argv)

    adapter = get_adapter(args.adapter)
    index, count, source = shard_of(args.shard, args.n_shards)
    tag = f"[rank {index}/{count}] " if count > 1 else ""
    frequency = shared_frequency_grid(adapter)[:: args.stride]

    if args.which == "signal":
        cells, n_seeds = enumerate_cells(adapter), args.n_seeds or 16
    elif args.which == "null":
        cells, n_seeds = enumerate_null_cells(adapter), args.n_seeds or 500
    else:  # smoke
        cells, n_seeds = _smoke_cells(adapter), args.n_seeds or 2

    n_terms_values = tuple(args.n_terms) if args.n_terms else N_TERMS_VALUES
    mults = (
        tuple(args.sigma_amp_mults) if args.sigma_amp_mults else SIGMA_AMP_MULTIPLIERS
    )

    all_jobs = [(c, s) for c in cells for s in seeds(n_seeds, offset=args.seed_offset)]
    # Round-robin, not contiguous: cost grows with n_obs, so a contiguous split would
    # hand every expensive cell to the same rank.
    jobs = all_jobs[index::count]
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print(f"{tag}no jobs for this shard")
        return 0

    out_path = _shard_path(args.out, index, count)
    print(
        f"{tag}{len(jobs)}/{len(all_jobs)} simulations x "
        f"{len(n_terms_values) * len(mults)} harv configs on "
        f"{frequency.shape[0]} frequencies, adapter={adapter.name}, "
        f"{count} shard{'s' if count > 1 else ' (serial)'} via {source}"
        + (f", reference n_mc={args.reference_n_mc}" if args.reference_n_mc else ""),
        flush=True,
    )

    t_start = time.perf_counter()
    with ArtifactWriter(
        out_path,
        frequency=np.asarray(ustrip(f"1/{adapter.time_unit}", frequency), dtype=float),
        which=args.which,
        samples_per_peak=SAMPLES_PER_PEAK,
        stride=args.stride,
        **adapter.meta(),
    ) as writer:
        for i, (cell, seed) in enumerate(jobs, start=1):
            t0 = time.perf_counter()
            rec = run_one(
                adapter,
                cell,
                seed,
                frequency,
                n_terms_values=n_terms_values,
                sigma_amp_mults=mults,
                reference_n_mc=args.reference_n_mc,
                reference_seed=seed,
            )
            writer.write(rec)
            print(
                f"{tag}  [{i}/{len(jobs)}] {rec.sim_id} "
                f"{time.perf_counter() - t0:.2f}s",
                flush=True,
            )

    total = time.perf_counter() - t_start
    print(
        f"{tag}wrote {out_path}  ({total:.1f}s total, {total / len(jobs):.2f}s/sim)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
