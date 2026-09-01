"""The work deal must be a partition. Nothing else in the harness checks that.

A rank that silently drops or duplicates simulations produces an artifact that looks
fine and is missing cells, or has two copies of one seed --- and `merge.py` only
catches the duplicate, only if it lands in a different rank's file.
"""

from __future__ import annotations

import pytest
from kepcmp.adapters import get_adapter
from kepcmp.grid import enumerate_cells, seeds
from kepcmp.mpi import balance, under_mpi_launcher
from kepcmp.run import _estimate_cost


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 16])
def test_balance_is_a_partition(size: int) -> None:
    adapter = get_adapter("rv")
    jobs = [(c, s) for c in enumerate_cells(adapter) for s in seeds(2)]
    costs = [_estimate_cost(c) for c, _ in jobs]

    dealt = [balance(jobs, costs, rank, size) for rank in range(size)]
    flat = [j for share in dealt for j in share]
    assert len(flat) == len(jobs), "the deal lost or duplicated work"
    assert {(c.cell_id, s) for c, s in flat} == {(c.cell_id, s) for c, s in jobs}


@pytest.mark.parametrize("size", [5, 16])
def test_balance_beats_a_stride_on_this_grid(size: int) -> None:
    """`size = 5` is the trap: the RV grid has 5 ``n_obs`` values varying fastest, so
    ``jobs[rank::5]`` deals every rank a single ``n_obs`` --- one rank gets all the
    cheap cells and another all the expensive ones. LPT cannot do that.
    """
    adapter = get_adapter("rv")
    jobs = [(c, s) for c in enumerate_cells(adapter) for s in seeds(2)]
    costs = [_estimate_cost(c) for c, _ in jobs]

    def spread(shares: list[list]) -> float:
        loads = [sum(_estimate_cost(c) for c, _ in sh) for sh in shares]
        return max(loads) / min(loads)

    lpt = spread([balance(jobs, costs, r, size) for r in range(size)])
    strided = spread([jobs[r::size] for r in range(size)])
    assert lpt <= 1.05, f"LPT imbalance {lpt:.3f}"
    assert lpt < strided, f"LPT {lpt:.3f} vs stride {strided:.3f}"


def test_no_mpi_import_without_a_launcher(monkeypatch) -> None:
    """Importing mpi4py runs MPI_Init, which *hangs* on a broken MPI rather than
    raising. So the serial path must not import it at all --- this is what keeps a
    laptop run and this test suite independent of the host's MPI.
    """
    from kepcmp import mpi as kmpi

    for var in kmpi._LAUNCHER_VARS:
        monkeypatch.delenv(var, raising=False)
    assert not under_mpi_launcher()

    import builtins

    real_import = builtins.__import__

    def guard(name, *a, **kw):
        if name.startswith("mpi4py"):
            raise AssertionError("serial path must not import mpi4py")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guard)
    comm, rank, size = kmpi.mpi_context()
    assert (comm, rank, size) == (None, 0, 1)
