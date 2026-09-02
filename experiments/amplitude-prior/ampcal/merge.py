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

``--check`` inspects and writes nothing. It is **deep** by default -- it opens every
simulation's object header, not just the ``/sims`` link names -- because a shallow pass
once reported "576 shards, 0 bad" on shards that then failed the copy with an
object-header error::

    python -m ampcal.merge --check output/rv/signal.rank*.h5
"""

from __future__ import annotations

__all__ = ("ShardReport", "inspect_shards", "main", "merge")

import argparse
import contextlib
import datetime as dt
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

SPACE_HEADROOM = 1.1
"""Free space required, as a multiple of the shards' total size. The merged file is
about the same size as its inputs; the 10% covers HDF5's own metadata overhead."""

RETRY_ATTEMPTS = 4
RETRY_DELAY_S = 5.0
"""Backoff between attempts, multiplied by the attempt number: 5s, 10s, 15s.

Measured against the failure this exists for: 576 ranks across six nodes closed ~10 GB
seconds before the merge started reading it, one shard came back with an unreadable link
heap, and the same file read perfectly a few minutes later. Thirty seconds of patience
against a 60 core-hour run is not a trade worth thinking about.
"""


def _with_retry[T](fn: Callable[[], T], *, attempts: int, delay: float) -> T:
    """Run ``fn``, retrying I/O failures with linear backoff.

    The merge reads several hundred files that *other nodes* closed seconds earlier. On
    a network filesystem the reading client can still hold a stale, partial view of one
    of them --- close-to-open consistency covers the data a client wrote, not the
    attribute cache of a client that never had the file open. So a shard can be
    genuinely intact on the servers and unreadable here, which is a transient, and a
    transient that discards a completed run is a bug rather than bad luck.

    This deliberately retries *everything*, not a curated list of exceptions: HDF5
    surfaces this class of problem as ``OSError``, ``RuntimeError`` and ``KeyError``
    depending on which block was missed, and guessing the set wrong reintroduces exactly
    the failure being fixed. A genuinely corrupt shard costs the full backoff and is then
    reported normally.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception:  # see docstring: the exception type varies
            if attempt == attempts:
                raise
            time.sleep(delay * attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def _require_space(inputs: list[Path], output: Path) -> None:
    """Fail in one second if the destination cannot hold a second copy of the grid.

    The merge writes a full copy of every shard into one file, so it needs as much free
    space again as the shards occupy -- and the shards are normally sitting on the same
    filesystem, so the run has already spent half the budget. Running out partway does
    not fail cleanly: HDF5 write failures corrupt the output's object headers, every
    subsequent lookup in it fails with ``message not aligned``, and the merge reports
    that as *every shard* being unreadable. That is a spectacularly misleading symptom
    for a full disk, and it costs a completed multi-core-hour run.

    ``shutil.disk_usage`` reports the filesystem, not the caller's quota, so this catches
    a genuinely full device and not a quota that is exhausted while the device has room.
    A quota failure lands in the copy loop's guard instead, which now says to check space
    first.
    """
    need = sum(p.stat().st_size for p in inputs)
    free = shutil.disk_usage(output.parent if output.parent.exists() else Path.cwd()).free
    if free < need * SPACE_HEADROOM:
        raise ValueError(
            f"not enough room to merge: {len(inputs)} shards total "
            f"{need / 1e9:.1f} GB and {output.parent} has {free / 1e9:.1f} GB free "
            f"(want {need * SPACE_HEADROOM / 1e9:.1f} GB, including headroom).\n"
            "Point --out at a filesystem with room -- the reductions take a path, so "
            "the merged artifact does not have to live beside the shards. Pass "
            "--no-space-check to override (a quota, not the device, is the usual reason "
            "this check is wrong in either direction)."
        )


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


def _read_shard(path: Path, *, deep: bool) -> int:
    """Number of simulations in ``path``, exercising what the merge will need.

    Shallow walks the ``/sims`` link names. That is enough to catch an unreadable link
    heap, and it is what the merge's own ``for sim_id in src["sims"]`` does first.

    Deep additionally *opens* every simulation's object header, its attributes and its
    ``arms`` subgroup. This distinction is not academic: a shallow pass reported "576
    shards, 0 bad" on a set of shards that then failed during the copy with
    ``Unable to open object (message not aligned)`` -- an object-header error, on
    headers a shallow pass never touches. Listing names proves the names are readable
    and nothing else.

    Deep still does not read dataset *contents*; only ``merge`` does that, which is why
    the copy loop keeps its own guard.
    """
    with h5py.File(path, "r") as src:
        sim_ids = list(src["sims"])
        if deep:
            for sim_id in sim_ids:
                group = src[f"sims/{sim_id}"]
                dict(group.attrs)
                list(group["arms"])
        return len(sim_ids)


def inspect_shards(
    inputs: list[Path],
    *,
    deep: bool = False,
    attempts: int | None = None,
    delay: float | None = None,
) -> list[ShardReport]:
    """Open every shard and report what it holds, without writing anything.

    ``mtime`` is reported because it is the discriminator between two failure modes: a
    shard from *this* run clusters with its siblings, while one left over from an
    earlier run with a different rank count is visibly older -- and an orphan of a
    killed run is exactly the kind of file that is truncated.

    ``deep`` decides how much of each shard is proved readable; see :func:`_read_shard`.
    It defaults to ``False`` for :func:`merge`'s preflight, which only needs to reject
    obviously-broken files cheaply before writing anything -- the copy loop reports and
    rolls back anything that fails later. ``--check`` defaults it to ``True``, because a
    post-mortem exists to find the damage rather than to be quick.

    Each shard is retried (see :func:`_with_retry`), since one observed failure turned
    out to be transient. Pass ``attempts=1`` to inspect without waiting.
    """
    # Resolved here rather than bound as defaults: a default argument is evaluated once
    # at import, so `RETRY_ATTEMPTS` would document a policy this function had already
    # frozen and could no longer be changed at runtime.
    attempts = RETRY_ATTEMPTS if attempts is None else attempts
    delay = RETRY_DELAY_S if delay is None else delay

    out: list[ShardReport] = []
    for path in inputs:
        stat = path.stat()
        mtime = _mtime(stat)

        def read(p: Path = path) -> int:
            return _read_shard(p, deep=deep)

        try:
            n = _with_retry(read, attempts=attempts, delay=delay)
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
    inputs: list[Path],
    output: Path,
    *,
    allow_partial: bool = False,
    check_space: bool = True,
    attempts: int | None = None,
    delay: float | None = None,
) -> tuple[int, int, list[ShardReport]]:
    """Copy every simulation from ``inputs`` into ``output``.

    Returns ``(n_sims, n_shards_merged, skipped)``. Unreadable shards raise unless
    ``allow_partial``, because silently dropping a shard silently drops grid cells --
    and a reduction cannot tell a missing cell from one that was never run.
    """
    if not inputs:
        raise ValueError("no input artifacts given")

    reports = inspect_shards(inputs, attempts=attempts, delay=delay)
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
    if check_space:
        _require_space(usable, output)

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
                from_this_shard: list[str] = []
                try:
                    with h5py.File(path, "r") as src:
                        grid = np.asarray(src["frequency"])
                        if reference_grid is None:
                            reference_grid = grid
                            out.create_dataset("frequency", data=grid)
                            for k, v in src.attrs.items():
                                out.attrs[k] = v
                        elif src.attrs.get("adapter") != out.attrs.get("adapter"):
                            # Checked before the grid, because it is the same mistake
                            # with a far better message. Two adapters were pointed at
                            # one OUT and wrote each other's shards; the grids differ
                            # too (RV 1828 frequencies, Gaia 914), so the grid check
                            # would catch it, but only as an unexplained mismatch.
                            raise ValueError(
                                f"{path} was written by adapter "
                                f"{src.attrs.get('adapter')!r} but {usable[0]} by "
                                f"{out.attrs.get('adapter')!r}. Two runs shared an "
                                "output directory. Their grids and cells are not "
                                "comparable; re-run with a separate OUT per adapter."
                            )
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
                            from_this_shard.append(sim_id)
                except ValueError:
                    # A grid mismatch or a duplicate id is a wiring error, not a broken
                    # file: --allow-partial must not paper over it.
                    raise
                except Exception as exc:  # their I/O; the file is bad
                    # A shard reported as skipped must contribute *nothing*. Leaving the
                    # sims it copied before failing produces an artifact holding half a
                    # shard while claiming the shard was dropped -- which is precisely
                    # the silently-missing-cells failure --allow-partial exists to avoid.
                    for sim_id in from_this_shard:
                        with contextlib.suppress(Exception):
                            del out["sims"][sim_id]
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
                    n_sims += len(from_this_shard)
            if not merged:
                # Every shard failed during the copy. That is not a partial merge, it is
                # a failed one, and --allow-partial must not turn it into a zero-content
                # artifact and an exit status of 0.
                raise ValueError(
                    f"all {len(usable)} shards failed while copying; nothing merged.\n"
                    + "\n".join(r.line() for r in failed[:5])
                    + (f"\n  ... and {len(failed) - 5} more" if len(failed) > 5 else "")
                    + "\n\nAn identical error on every shard is a problem with the "
                    "destination, not the shards -- check free space on "
                    f"{output.parent} first."
                )
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
    parser.add_argument("--shallow", action="store_true",
                        help="with --check, only list link names instead of opening "
                             "every simulation's object header. Faster, and strictly "
                             "weaker: a shallow pass passes shards whose headers the "
                             "merge then cannot open")
    parser.add_argument("--no-space-check", action="store_true",
                        help="skip the free-space preflight")
    parser.add_argument("--allow-partial", action="store_true",
                        help="merge the readable shards even if some are unreadable; "
                             "the count of dropped shards is recorded in the artifact")
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args(argv)

    if args.check:
        # One attempt: --check is a post-mortem, and its job is to report what is true
        # right now rather than to wait out a transient the way the merge should. Deep
        # by default: a check that skips what the merge does is not a check.
        reports = inspect_shards(args.inputs, deep=not args.shallow, attempts=1)
        print(_summarize(reports))
        return 1 if any(not r.ok for r in reports) else 0

    if args.out is None:
        parser.error("--out is required unless --check is given")

    n_sims, n_shards, skipped = merge(
        args.inputs,
        args.out,
        allow_partial=args.allow_partial,
        check_space=not args.no_space_check,
    )
    print(f"merged {n_sims:,} simulations from {n_shards} shards -> {args.out}")
    if skipped:
        print(f"SKIPPED {len(skipped)} unreadable shard(s):")
        for r in skipped[:20]:
            print(r.line())
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
        print("The artifact records this in its `n_shards_skipped` attribute.")
    # Non-zero even under --allow-partial: the caller asked to continue past bad shards,
    # not to be told the run was complete. run_grid.sh is `set -e`, and a merge that
    # dropped shards must not silently flow into the reductions.
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
