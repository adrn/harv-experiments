"""Merge sharded artifacts into one.

The grid is embarrassingly parallel by seed, so a full run is executed as several
processes each writing its own artifact. The reductions all take a single artifact
path, so shards are merged here rather than teaching four reductions about sharding.

The merge is a straight copy of ``/sims/<sim_id>`` groups. It refuses to proceed if the
shards disagree on the frequency grid, since every reduction assumes a shared grid and
silently mixing two grids would corrupt every comparison.
"""

from __future__ import annotations

__all__ = ("main", "merge")

import argparse
from pathlib import Path

import h5py
import numpy as np


def merge(inputs: list[Path], output: Path) -> tuple[int, int]:
    """Copy every simulation from ``inputs`` into ``output``. Returns (sims, shards)."""
    if not inputs:
        raise ValueError("no input artifacts given")
    output.parent.mkdir(parents=True, exist_ok=True)

    n_sims = 0
    with h5py.File(output, "w") as out:
        out.create_group("sims")
        reference_grid: np.ndarray | None = None
        for i, path in enumerate(inputs):
            with h5py.File(path, "r") as src:
                grid = np.asarray(src["frequency"])
                if reference_grid is None:
                    reference_grid = grid
                    out.create_dataset("frequency", data=grid)
                    for k, v in src.attrs.items():
                        out.attrs[k] = v
                elif grid.shape != reference_grid.shape or not np.allclose(
                    grid, reference_grid, rtol=0, atol=0
                ):
                    raise ValueError(
                        f"{path} has a different frequency grid from {inputs[0]}; "
                        "every reduction assumes one shared grid, so merging these "
                        "would corrupt the comparisons."
                    )
                for sim_id in src["sims"]:
                    if sim_id in out["sims"]:
                        raise ValueError(
                            f"duplicate sim_id {sim_id!r} in {path}; shards must "
                            "cover disjoint seeds"
                        )
                    src.copy(src[f"sims/{sim_id}"], out["sims"], name=sim_id)
                    n_sims += 1
            del i
        out.attrs["merged_from"] = [str(p) for p in inputs]
        out.attrs["n_shards"] = len(inputs)
    return n_sims, len(inputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args(argv)

    n_sims, n_shards = merge(args.inputs, args.out)
    print(f"merged {n_sims} simulations from {n_shards} shards -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
