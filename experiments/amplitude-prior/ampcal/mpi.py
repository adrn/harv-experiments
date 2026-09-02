"""MPI launcher plumbing.

The runner is SPMD: every rank runs the same code, asks ``COMM_WORLD`` which rank it
is, takes its own share of the simulation list, and writes its own artifact. Ranks
talk exactly twice --- nothing at all during the work, and one ``gather`` at the end
to print a summary --- so nothing here needs MPI-IO or parallel HDF5, and
:mod:`ampcal.merge` combines the per-rank files afterwards. MPI is a launcher, not a
message bus.

Same shape as ``epochalypse``'s ``scripts/*_mpi.py``, deliberately: it is a pattern
that has already survived a 1024-rank production run. Reimplemented here rather than
imported because this directory has to stay location-independent.
"""

from __future__ import annotations

__all__ = (
    "balance",
    "banner",
    "gather",
    "mpi_context",
    "peak_rss_gb",
    "under_mpi_launcher",
)

import os
import resource
import sys
from typing import Any

#: Variables an MPI launcher sets in every rank's environment.
_LAUNCHER_VARS: tuple[str, ...] = (
    "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "PMI_SIZE",  # MPICH / Hydra
    "PMIX_RANK",  # PMIx
    "MV2_COMM_WORLD_SIZE",  # MVAPICH2
    "SLURM_PROCID",  # srun
)


def under_mpi_launcher() -> bool:
    """Whether this process was started by ``mpirun`` / ``srun``."""
    return any(var in os.environ for var in _LAUNCHER_VARS)


def mpi_context(force: bool = False) -> tuple[Any, int, int]:
    """``(comm, rank, size)``, importing mpi4py only when it is actually needed.

    **The import is guarded, and that is not fussiness.** ``import mpi4py.MPI`` runs
    ``MPI_Init`` as a side effect, so on a host whose MPI is broken or ABI-mismatched
    it does not raise --- it hangs, or aborts inside ``PMIx_Finalize``. Importing it
    unconditionally would make a plain ``python -m ampcal.run``, and every test that
    touches the runner, hostage to an MPI installation the serial path has no use for.
    Measured: exactly that, on the machine this was written on.

    So mpi4py is imported when a launcher's variables are present, or when ``force``
    is set (``--mpi``, for a launcher whose variables are not in
    :data:`_LAUNCHER_VARS`). Otherwise this is one rank and no MPI is initialised at
    all. Under ``mpirun`` or ``srun`` you get the real thing.
    """
    if not (force or under_mpi_launcher()):
        return None, 0, 1
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def balance(items: list, costs: list[float], rank: int, size: int) -> list:
    """This rank's share, longest-processing-time-first.

    Sort by cost descending and give each item to the currently least-loaded rank ---
    the classic LPT heuristic, provably within 4/3 of optimal and near-perfect once no
    single item dominates the total. Deterministic given the same ``costs``, so every
    rank computes the same assignment with no communication.

    Why not ``items[rank::size]``: the grid is enumerated with ``n_obs`` varying
    *fastest*, so a stride whose length shares a factor with the ``n_obs`` ladder deals
    every rank simulations of a single ``n_obs``. At ``size = 5`` on the RV grid (5
    ``n_obs`` values) one rank would get every ``n_obs = 32`` cell and another every
    ``n_obs = 2`` cell --- the worst possible split of a cost that is linear in
    ``n_obs``. LPT cannot fall into that hole.
    """
    import heapq

    order = sorted(range(len(items)), key=lambda i: -costs[i])
    heap = [(0.0, r) for r in range(size)]
    heapq.heapify(heap)
    mine = []
    for i in order:
        load, owner = heapq.heappop(heap)
        if owner == rank:
            mine.append(items[i])
        heapq.heappush(heap, (load + float(costs[i]), owner))
    return mine


def gather(comm: Any, summary: dict) -> list[dict]:
    """Every rank's summary on rank 0, or just this one without mpi4py."""
    return comm.gather(summary, root=0) if comm else [summary]


def peak_rss_gb() -> float:
    """Peak resident set size, in GB. Decides ranks-per-node, so measure it."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def banner(comm: Any, size: int, n_items: int, item: str = "simulations", **extra: Any) -> None:
    """Rank 0's header: fleet size, work per rank, and the threading warning.

    The threading line is not decoration. XLA's CPU thread pool ignores
    ``OMP_NUM_THREADS`` and takes ~2.4 cores per rank by default; with tens of ranks
    per node that oversubscribes silently, and the only symptom is that the job is
    slow. ``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`` is what the job script sets.
    """
    print(
        f"ranks       : {size}"
        + ("" if comm else "  (mpi4py not found -- running as a single rank)")
    )
    print(f"{item:<12}: {n_items:,}  ->  ~{n_items // max(size, 1):,} per rank")
    for key, value in extra.items():
        print(f"{key:<12}: {value}")
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    xla = os.environ.get("XLA_FLAGS", "unset")
    print(f"threads/rank: OMP_NUM_THREADS={threads}"
          + ("" if threads == "1" else "   <- set this to 1 to avoid oversubscription"))
    print(
        f"xla_flags   : {xla}"
        + (
            ""
            if "multi_thread_eigen=false" in xla
            else "   <- set --xla_cpu_multi_thread_eigen=false; XLA ignores OMP_NUM_THREADS"
        ),
        flush=True,
    )
