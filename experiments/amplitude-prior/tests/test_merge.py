"""Merging must survive one bad shard out of hundreds, and say which one.

The merge is the last step of a job that has already spent tens of core-hours across
several hundred ranks. A shard can come back short (a filesystem writeback that failed
after ``close()`` succeeded) or stale (an orphan of an earlier run at a different rank
count, which the ``signal.rank*.h5`` glob picks up regardless). Losing the whole run to
an ``h5py`` traceback that names no file is the failure this module exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from ampcal.merge import inspect_shards, merge

# The suite must never sit through the production backoff.
NO_WAIT = {"attempts": 1}

NF = 32


def _shard(path: Path, sim_ids: list[str]) -> Path:
    with h5py.File(path, "w") as h:
        h.create_dataset("frequency", data=np.linspace(0.001, 0.1, NF))
        h.attrs["adapter"] = "rv"
        sims = h.create_group("sims")
        for sid in sim_ids:
            g = sims.create_group(sid)
            g.attrs["seed"] = 0
            g.create_dataset("x", data=np.zeros(NF))
    return path


def _truncate(path: Path, keep: float = 0.4) -> None:
    """Chop the tail off a real HDF5 file.

    This is the actual observed failure -- the superblock and the early datasets survive,
    so the file opens and ``/frequency`` reads fine, and only iterating ``/sims`` hits a
    heap offset past the end. A zero-byte or garbage file would not exercise that path.
    """
    size = path.stat().st_size
    with path.open("r+b") as fh:
        fh.truncate(int(size * keep))


@pytest.fixture
def shards(tmp_path: Path) -> list[Path]:
    return [
        _shard(tmp_path / f"signal.rank{r:03d}.h5", [f"cell_r{r}_seed{i:03d}" for i in range(3)])
        for r in range(4)
    ]


def test_merges_clean_shards(shards: list[Path], tmp_path: Path) -> None:
    n_sims, n_shards, skipped = merge(shards, tmp_path / "signal.h5", **NO_WAIT)
    assert (n_sims, n_shards, skipped) == (12, 4, [])
    with h5py.File(tmp_path / "signal.h5", "r") as h:
        assert len(h["sims"]) == 12
        assert h.attrs["n_shards_skipped"] == 0


def test_a_truncated_shard_is_named_not_a_traceback(
    shards: list[Path], tmp_path: Path
) -> None:
    """The whole point: the error has to say *which* file, out of 576."""
    _truncate(shards[2])
    with pytest.raises(ValueError, match=r"1 of 4 shards could not be read") as exc:
        merge(shards, tmp_path / "signal.h5", **NO_WAIT)
    assert shards[2].name in str(exc.value)
    # And it must not have left a half-written artifact behind that a reduction would
    # happily read as a complete grid.
    assert not (tmp_path / "signal.h5").exists()


def test_allow_partial_merges_the_rest_and_records_it(
    shards: list[Path], tmp_path: Path
) -> None:
    _truncate(shards[2])
    n_sims, n_shards, skipped = merge(
        shards, tmp_path / "signal.h5", allow_partial=True, **NO_WAIT
    )
    assert (n_sims, n_shards) == (9, 3)
    assert [r.path for r in skipped] == [shards[2]]
    with h5py.File(tmp_path / "signal.h5", "r") as h:
        # Recorded in the file, not only on stdout: a partial merge that only said so in
        # a log produces an artifact no reduction can tell from a complete one.
        assert h.attrs["n_shards_skipped"] == 1
        assert shards[2].name in h.attrs["shards_skipped"][0]


def test_inspect_reports_size_and_mtime_for_every_shard(shards: list[Path]) -> None:
    """``mtime`` is the discriminator between this run's shards and an earlier run's
    orphans, and the two need opposite responses (re-run vs. delete and re-merge)."""
    _truncate(shards[1])
    reports = inspect_shards(shards, **NO_WAIT)
    assert len(reports) == 4
    assert [r.ok for r in reports] == [True, False, True, True]
    assert all(r.bytes > 0 and r.mtime for r in reports)
    assert reports[1].n_sims is None
    assert reports[1].error
    assert reports[0].n_sims == 3


def test_all_shards_unreadable_is_not_an_empty_success(
    shards: list[Path], tmp_path: Path
) -> None:
    for p in shards:
        _truncate(p)
    with pytest.raises(ValueError, match="no readable shards"):
        merge(shards, tmp_path / "signal.h5", allow_partial=True, **NO_WAIT)


def test_duplicate_sim_id_points_at_stale_shards(tmp_path: Path) -> None:
    """Two shards holding the same sim_id means an orphan got into the glob, which is
    the other half of the same hazard -- so the message has to say so."""
    a = _shard(tmp_path / "signal.rank000.h5", ["cell_seed000"])
    b = _shard(tmp_path / "signal.rank001.h5", ["cell_seed000"])
    with pytest.raises(ValueError, match="orphaned rank files"):
        merge([a, b], tmp_path / "signal.h5", **NO_WAIT)


def test_a_shard_that_fails_mid_copy_is_caught(shards: list[Path], tmp_path: Path) -> None:
    """Inspection walks link names; copying also reads dataset blocks. A shard whose
    data blocks are unwritten passes the check and dies in the copy loop, so that loop
    needs the same guard -- and must still not leave a half-written artifact.
    """
    import ampcal.merge as m

    real_copy = h5py.File.copy
    calls = {"n": 0}

    def flaky(self, source, dest, name=None, **kw):
        calls["n"] += 1
        if calls["n"] == 5:
            raise OSError("unable to read data block")
        return real_copy(self, source, dest, name=name, **kw)

    h5py.File.copy = flaky
    try:
        with pytest.raises(ValueError, match="failed while copying"):
            m.merge(shards, tmp_path / "signal.h5", **NO_WAIT)
        assert not (tmp_path / "signal.h5").exists()

        calls["n"] = 0
        n_sims, n_shards, skipped = m.merge(
            shards, tmp_path / "signal.h5", allow_partial=True, **NO_WAIT
        )
        assert len(skipped) == 1 and "failed while copying" in skipped[0].error
        assert n_shards == 3 and n_sims < 12
    finally:
        h5py.File.copy = real_copy


def test_check_iterates_links_rather_than_counting(tmp_path: Path) -> None:
    """`len(group)` can read a stored count without touching the link heap, which is
    exactly the block that was unreadable. The check must do what the merge does."""
    import ampcal.merge as m

    p = _shard(tmp_path / "signal.rank000.h5", ["a_seed000", "b_seed001"])
    iterated: list[str] = []
    real_iter = h5py.Group.__iter__

    def watching_iter(self):
        iterated.append(self.name)
        return real_iter(self)

    h5py.Group.__iter__ = watching_iter
    try:
        report = m.inspect_shards([p], **NO_WAIT)[0]
    finally:
        h5py.Group.__iter__ = real_iter
    assert report.ok and report.n_sims == 2
    # `list()` also calls __len__ as a size hint, which is harmless. What matters is
    # that the links were actually walked -- that is the operation that failed on the
    # cluster, and a check that skips it is not a check.
    assert "/sims" in iterated, "inspect_shards never iterated the links"


def test_a_transient_read_is_retried(tmp_path: Path) -> None:
    """The failure this was built for: intact on the servers, unreadable right now.

    576 ranks closed ~10 GB seconds before the merge read it, one shard came back with an
    unreadable link heap, and the same file read perfectly minutes later. Giving up on
    the first attempt discards a completed 60 core-hour run.
    """
    import ampcal.merge as m

    p = _shard(tmp_path / "signal.rank000.h5", ["a_seed000", "b_seed001"])
    calls = {"n": 0}
    real_open = h5py.File.__init__

    def flaky(self, name, mode="r", **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("unable to offset into local heap data block")
        return real_open(self, name, mode, **kw)

    h5py.File.__init__ = flaky
    try:
        report = m.inspect_shards([p], attempts=4, delay=0.0)[0]
    finally:
        h5py.File.__init__ = real_open
    assert report.ok and report.n_sims == 2
    assert calls["n"] == 3, "retried the wrong number of times"


def test_a_permanently_bad_shard_still_gives_up(tmp_path: Path) -> None:
    """Retry must not turn a genuinely corrupt shard into a hang or a false pass."""
    import ampcal.merge as m

    p = _shard(tmp_path / "signal.rank000.h5", ["a_seed000"])
    _truncate(p)
    report = m.inspect_shards([p], attempts=3, delay=0.0)[0]
    assert not report.ok and report.error


def test_a_failed_shard_leaves_nothing_behind(shards: list[Path], tmp_path: Path) -> None:
    """A shard reported as skipped must contribute *zero* sims, not the ones it managed.

    Observed: a full disk made every shard fail partway, and the merge reported
    "merged 4,032 simulations from 0 shards" -- an artifact holding half of every shard
    while recording that no shard was merged. A reduction cannot tell that from a
    complete grid.
    """
    import ampcal.merge as m

    real_copy = h5py.File.copy
    calls = {"n": 0}

    def flaky(self, source, dest, name=None, **kw):
        calls["n"] += 1
        # Fail on the 2nd sim of the 1st shard: one sim is already copied by then.
        if calls["n"] == 2:
            raise OSError("message not aligned")
        return real_copy(self, source, dest, name=name, **kw)

    h5py.File.copy = flaky
    try:
        n_sims, n_shards, skipped = m.merge(
            shards, tmp_path / "signal.h5", allow_partial=True, **NO_WAIT
        )
    finally:
        h5py.File.copy = real_copy

    assert len(skipped) == 1 and n_shards == 3
    # The headline invariant: the count must equal what the merged shards actually hold.
    assert n_sims == 9
    with h5py.File(tmp_path / "signal.h5", "r") as h:
        assert len(h["sims"]) == n_sims
        assert h.attrs["n_shards"] == n_shards
        # Nothing from the failed shard survived.
        assert not any(sid.startswith("cell_r0_") for sid in h["sims"])


def test_every_shard_failing_is_a_failure_not_a_partial_merge(
    shards: list[Path], tmp_path: Path
) -> None:
    """An identical error on all N shards is a problem with the destination, and
    --allow-partial must not turn it into a zero-content artifact and exit 0."""
    import ampcal.merge as m

    real_copy = h5py.File.copy

    def always_fails(self, source, dest, name=None, **kw):
        raise OSError("message not aligned")

    h5py.File.copy = always_fails
    try:
        with pytest.raises(ValueError, match="all 4 shards failed while copying"):
            m.merge(shards, tmp_path / "signal.h5", allow_partial=True, **NO_WAIT)
    finally:
        h5py.File.copy = real_copy
    assert not (tmp_path / "signal.h5").exists()


def test_skipping_shards_exits_non_zero(
    shards: list[Path], tmp_path: Path, monkeypatch
) -> None:
    """`run_grid.sh` is `set -e`. A merge that dropped shards must not flow silently
    into the reductions just because --allow-partial was passed."""
    import ampcal.merge as m

    # main() owns the retry policy, so the constant is what a test can reach.
    monkeypatch.setattr(m, "RETRY_DELAY_S", 0.0)
    _truncate(shards[1])
    rc = m.main([
        "--out", str(tmp_path / "signal.h5"), "--allow-partial", "--no-space-check",
        *[str(p) for p in shards],
    ])
    assert rc == 1
    rc_clean = m.main([
        "--out", str(tmp_path / "clean.h5"), "--no-space-check",
        *[str(p) for p in shards if p != shards[1]],
    ])
    assert rc_clean == 0


def test_space_preflight_refuses_before_writing(
    shards: list[Path], tmp_path: Path, monkeypatch
) -> None:
    """Fail in one second, not after 6 GB. Running out mid-merge corrupts the output's
    object headers and reports every shard as unreadable -- a spectacularly misleading
    symptom for a full disk."""
    import types

    import ampcal.merge as m

    # `_require_space` reads only `.free`; a stand-in avoids depending on shutil private
    # API for the shape of its result.
    monkeypatch.setattr(
        m.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=1024)
    )
    with pytest.raises(ValueError, match="not enough room to merge"):
        m.merge(shards, tmp_path / "signal.h5", **NO_WAIT)
    assert not (tmp_path / "signal.h5").exists()

    # And the override exists, because the check reads the device and not a quota.
    n_sims, _, _ = m.merge(
        shards, tmp_path / "signal.h5", check_space=False, **NO_WAIT
    )
    assert n_sims == 12


def test_shallow_check_misses_what_deep_catches(tmp_path: Path) -> None:
    """The bug this cost us: listing link names proves the names are readable and
    nothing else, so a shard whose object headers are unopenable passes a shallow check
    and then fails the merge with `Unable to open object (message not aligned)`.
    """
    import ampcal.merge as m

    p = _shard(tmp_path / "signal.rank000.h5", ["a_seed000", "b_seed001"])
    # Give the shard the structure the deep pass walks.
    with h5py.File(p, "a") as h:
        for sid in h["sims"]:
            h["sims"][sid].create_group("arms").create_group("H1_e+0.0000_s1")

    real_getitem = h5py.Group.__getitem__

    def broken_headers(self, key):
        # The merge opens sim groups by path off the File ("sims/<id>"), so matching on
        # the key is what catches it; matching on self.name would not.
        if isinstance(key, str) and key.startswith("sims/"):
            raise KeyError("Unable to open object (message not aligned)")
        return real_getitem(self, key)

    h5py.Group.__getitem__ = broken_headers
    try:
        shallow = m.inspect_shards([p], deep=False, attempts=1)[0]
        deep = m.inspect_shards([p], deep=True, attempts=1)[0]
    finally:
        h5py.Group.__getitem__ = real_getitem

    assert shallow.ok, "shallow pass should still see the link names"
    assert not deep.ok, "deep pass must open the headers and fail"
    assert "message not aligned" in deep.error


def test_mixing_adapters_is_refused(tmp_path: Path) -> None:
    """The failure that cost the RV grid: OUT was exported, so a Gaia run wrote its
    shards into the RV directory under the same rank filenames. Names alone gave it
    away only because n_obs=64 does not exist in the RV grid -- the tool should say so
    directly."""
    import ampcal.merge as m

    a = _shard(tmp_path / "signal.rank000.h5", ["physical_pr0.1_snr3_e0_n32_seed000"])
    b = _shard(tmp_path / "signal.rank001.h5", ["physical_pr0.1_snr3_e0_n64_seed000"])
    with h5py.File(b, "a") as h:
        h.attrs["adapter"] = "gaia"
    with pytest.raises(ValueError, match="Two runs shared an output directory"):
        m.merge([a, b], tmp_path / "signal.h5", **NO_WAIT)
    assert not (tmp_path / "signal.h5").exists()
