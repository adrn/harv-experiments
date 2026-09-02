"""Merge sharded artifacts into one.

The grid is embarrassingly parallel by seed, so a full run is executed as several
hundred processes each writing its own artifact. The reductions all take a single
artifact path, so shards are merged here rather than teaching every reduction about
sharding.

The merge is a straight copy of ``/sims/<sim_id>`` groups. It refuses to proceed if the
shards disagree on the frequency grid, since every reduction assumes a shared grid and
silently mixing two grids would corrupt every comparison.

**One unreadable shard must not discard the run.** The merge is the last step of a job
that has already spent tens of core-hours, and it reads several hundred files written
minutes earlier by different nodes. A shard can be short (a filesystem writeback that
failed after ``close()`` succeeded) or stale (left behind by an earlier run with a
different rank count, which the ``signal.rank*.h5`` glob picks up regardless). Either way
an opaque ``h5py`` traceback that names no file is useless at 2am, so every shard is
opened defensively, failures are collected with the file that caused them, and
``--allow-partial`` merges what is good after saying exactly what it dropped.

``--check`` does the same inspection and writes nothing:

    python -m ampcal.merge --check output/rv/signal.rank*.h5
"""

from __future__ import annotations

__all__ = ("ShardReport", "inspect_shards", "main", "merge")

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


def _mtime(stat: object) -> str:
    """Local, timezone-aware modification time, to the second.

    Local rather than UTC on purpose: this is read against a slurm log to decide whether
    a shard belongs to this run or an earlier one, and the log is in local time.
    """
    return (
        dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.UTC)  # type: ignore[attr-defined]
        .astimezone()
        .isoformat(timespec="seconds")
    )


@dataclass(frozen=True)
class ShardReport:
    """What one shard file turned out to be."""

    path: Path
    bytes: int
    mtime: str
    n_sims: int | None
    """``None`` when the shard could not be read."""

    error: str = ""

    @property
    def ok(self) -> bool:
        return self.error == ""

    def line(self) -> str:
        state = f"{self.n_sims:>5} sims" if self.ok else "  UNREADABLE"
        return (
            f"  {self.path.name:<28} {self.bytes:>14,} B  {self.mtime}  {state}"
            + (f"\n      {self.error}" if self.error else "")
        )


def inspect_shards(inputs: list[Path]) -> list[ShardReport]:
    """Open every shard and report what it holds, without writing anything.

    ``mtime`` is reported because it is the discriminator between the two failure
    modes: a shard from *this* run clusters with its siblings, while one left over from
    an earlier run with a different rank count is visibly older -- and an orphan of a
    killed run is exactly the kind of file that is truncated.

    The link names are **iterated**, not counted. ``len(group)`` reads the group's
    stored object count and need not touch the link heap, so a file whose heap block is
    unwritten passes a length check and then fails in the merge's own
    ``for sim_id in src["sims"]`` -- which is precisely the observed failure
    (``unable to offset into local heap data block``). The check has to perform the same
    operation the merge does or it is not a check.
    """
    out: list[ShardReport] = []
    for path in inputs:
        stat = path.stat()
        mtime = _mtime(stat)
        try:
            with h5py.File(path, "r") as src:
                n = len(list(src["sims"]))
        except Exception as exc:  # noqa: BLE001  (their I/O; any failure is the file's)
            out.append(
                ShardReport(
                    path, stat.st_size, mtime, None, f"{type(exc).__name__}: {exc}"
                )
            )
        else:
            out.append(ShardReport(path, stat.st_size, mtime, n))
    return out


def _summarize(reports: list[ShardReport]) -> str:
    bad = [r for r in reports if not r.ok]
    good = [r for r in reports if r.ok]
    lines = [
        (
            f"{len(reports)} shard(s): {len(good)} readable holding "
            f"{sum(r.n_sims or 0 for r in good):,} simulations, {len(bad)} unreadable"
        ),
    ]
    if bad:
        lines.append("\nunreadable:")
        lines += [r.line() for r in bad]
        # Timestamps decide whether these are this run's shards or an earlier run's
        # orphans, and the two need opposite responses: re-run, or delete and re-merge.
        span = sorted(r.mtime for r in good)
        if span:
            lines.append(
                f"\nreadable shards were written {span[0]} .. {span[-1]}; compare the "
                "timestamps above. An unreadable shard from outside that window is an "
                "orphan of an earlier run -- delete it and re-merge. One inside it was "
                "written by this run and its simulations are genuinely lost."
            )
    return "\n".join(lines)


def merge(
    inputs: list[Path], output: Path, *, allow_partial: bool = False
) -> tuple[int, int, list[ShardReport]]:
    """Copy every simulation from ``inputs`` into ``output``.

    Returns ``(n_sims, n_shards_merged, skipped)``. Unreadable shards raise unless
    ``allow_partial``, because silently dropping a shard silently drops grid cells --
    and a reduction cannot tell a missing cell from one that was never run.
    """
    if not inputs:
        raise ValueError("no input artifacts given")

    reports = inspect_shards(inputs)
    bad = [r for r in reports if not r.ok]
    if bad and not allow_partial:
        raise ValueError(
            f"{len(bad)} of {len(inputs)} shards could not be read:\n"
            + "\n".join(r.line() for r in bad)
            + "\n\n"
            + _summarize(reports)
            + "\n\nRe-run those cells, or pass --allow-partial to merge the rest."
        )
    usable = [r.path for r in reports if r.ok]
    if not usable:
        raise ValueError("no readable shards; nothing to merge")

    output.parent.mkdir(parents=True, exist_ok=True)
    n_sims = 0
    merged: list[Path] = []
    failed: list[ShardReport] = list(bad)
    try:
        with h5py.File(output, "w") as out:
            out.create_group("sims")
            reference_grid: np.ndarray | None = None
            for path in usable:
                # Inspection walks the link names; copying additionally reads every
                # dataset block, so a shard with an unwritten *data* block passes the
                # check and fails here. Guarded for the same reason the check exists.
                try:
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
                                f"{path} has a different frequency grid from "
                                f"{usable[0]}; every reduction assumes one shared "
                                "grid, so merging these would corrupt the comparisons."
                            )
                        for sim_id in src["sims"]:
                            if sim_id in out["sims"]:
                                raise ValueError(
                                    f"duplicate sim_id {sim_id!r} in {path}; shards "
                                    "must cover disjoint seeds. If an earlier run left "
                                    "orphaned rank files behind, delete them and "
                                    "re-merge."
                                )
                            src.copy(src[f"sims/{sim_id}"], out["sims"], name=sim_id)
                            n_sims += 1
                except ValueError:
                    # A grid mismatch or a duplicate id is a wiring error, not a broken
                    # file: --allow-partial must not paper over it.
                    raise
                except Exception as exc:  # their I/O; the file is bad
                    stat = path.stat()
                    failed.append(
                        ShardReport(
                            path,
                            stat.st_size,
                            _mtime(stat),
                            None,
                            f"failed while copying: {type(exc).__name__}: {exc}",
                        )
                    )
                    if not allow_partial:
                        raise ValueError(
                            f"{path} passed inspection but failed while copying:\n"
                            f"  {type(exc).__name__}: {exc}\n\n"
                            "Re-run those cells, or pass --allow-partial to merge the "
                            "rest."
                        ) from exc
                else:
                    merged.append(path)
            out.attrs["merged_from"] = [str(p) for p in merged]
            out.attrs["n_shards"] = len(merged)
            # Recorded in the artifact, not just printed: a partial merge that only ever
            # said so on stdout would produce a file no reduction can tell from a full
            # one.
            out.attrs["n_shards_skipped"] = len(failed)
            out.attrs["shards_skipped"] = [str(r.path) for r in failed]
    except BaseException:
        # Never leave a half-written artifact behind. A reduction cannot tell one from a
        # complete grid, and the whole point of failing loudly is not to be believed.
        output.unlink(missing_ok=True)
        raise
    return n_sims, len(merged), failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help="output artifact; required unless --check")
    parser.add_argument("--check", action="store_true",
                        help="inspect the shards and report; write nothing")
    parser.add_argument("--allow-partial", action="store_true",
                        help="merge the readable shards even if some are unreadable; "
                             "the count of dropped shards is recorded in the artifact")
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.check:
        reports = inspect_shards(args.inputs)
        print(_summarize(reports))
        return 1 if any(not r.ok for r in reports) else 0

    if args.out is None:
        parser.error("--out is required unless --check is given")

    n_sims, n_shards, skipped = merge(
        args.inputs, args.out, allow_partial=args.allow_partial
    )
    print(f"merged {n_sims:,} simulations from {n_shards} shards -> {args.out}")
    if skipped:
        print(f"SKIPPED {len(skipped)} unreadable shard(s):")
        for r in skipped:
            print(r.line())
        print("The artifact records this in its `n_shards_skipped` attribute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
