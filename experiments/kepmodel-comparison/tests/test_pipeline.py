"""End-to-end: run a tiny grid, write an artifact, and run all four reductions.

Also re-checks the identity at the *artifact* level, which is what validates the
batched production path rather than the per-frequency gate path. Only ``n_terms=1``
rows are comparable to kepmodel's ``d=2``, so higher-order rows are excluded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from kepcmp.adapters import RVAdapter
from kepcmp.adapters.rv import TIME_UNIT
from kepcmp.artifact import ArtifactReader, ArtifactWriter
from kepcmp.grid import Cell, shared_frequency_grid
from kepcmp.reduce.calibrate import reduce_calibrate
from kepcmp.reduce.decompose import reduce_decompose
from kepcmp.reduce.prior_quality import reduce_prior_quality
from kepcmp.reduce.regime_map import reduce_regime_map
from kepcmp.reduce.roc import reduce_roc
from kepcmp.run import run_one
from unxt import ustrip

SIGNAL_CELLS = [
    Cell(period_ratio=0.1, snr=30.0, eccentricity=0.0, n_obs=32),
    Cell(period_ratio=0.3, snr=10.0, eccentricity=0.6, n_obs=32),
    # Sparse rung where the profile statistic does not exist, so the pipeline and
    # every reduction must handle a missing z0 arm rather than crash.
    Cell(period_ratio=0.3, snr=10.0, eccentricity=0.0, n_obs=2),
]
NULL_CELL = Cell(period_ratio=1.0, snr=0.0, eccentricity=0.0, n_obs=32)


def _write(path: Path, cells, seeds, *, reference_n_mc: int = 0) -> Path:
    adapter = RVAdapter()
    frequency = shared_frequency_grid(adapter)[::16]
    with ArtifactWriter(
        path,
        frequency=np.asarray(ustrip(f"1/{TIME_UNIT}", frequency), dtype=float),
        **adapter.meta(),
    ) as w:
        for cell in cells:
            for seed in seeds:
                w.write(
                    run_one(
                        adapter,
                        cell,
                        seed,
                        frequency,
                        n_terms_values=(1, 2),
                        sigma_amp_mults=(0.3, 1.0, 3.0),
                        reference_n_mc=reference_n_mc,
                        reference_seed=seed,
                    )
                )
    return path


@pytest.fixture(scope="module")
def signal_artifact(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("kepcmp") / "signal.h5"
    return _write(path, SIGNAL_CELLS, (0, 1), reference_n_mc=256)


@pytest.fixture(scope="module")
def null_artifact(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("kepcmp") / "null.h5"
    return _write(path, [NULL_CELL], tuple(range(6)))


def test_artifact_roundtrip(signal_artifact: Path) -> None:
    with ArtifactReader(signal_artifact) as art:
        assert art.sim_ids()
        assert art.frequency.ndim == 1
        assert art.meta["kepmodel_version"] != "absent"
        sim = art.sim_ids()[0]
        assert art.z0(sim).shape == art.frequency.shape
        assert set(art.data(sim)) == {"time", "rv", "rv_err"}
        st = art.stat(sim, art.stat_keys(sim)[0])
        for field in ("delta", "occam", "shrinkage", "cond"):
            assert st[field].shape == art.frequency.shape


def test_identity_holds_in_the_artifact(signal_artifact: Path) -> None:
    """The batched production path must satisfy the same identity as the gate."""
    with ArtifactReader(signal_artifact) as art:
        checked = 0
        for sim_id in art.sim_ids():
            if not art.z0_usable(sim_id):
                continue  # no profile statistic here; the identity is undefined
            z0 = art.z0(sim_id)
            chi2_base = float(
                art._h5[f"sims/{sim_id}/kepmodel"].attrs["chi2_base"]
            )
            scale = max(chi2_base, 1.0)
            for key in art.stat_keys(sim_id):
                st = art.stat(sim_id, key)
                if int(st["n_terms_effective"]) != 1:
                    continue
                resid = st["delta"] + st["occam"] + st["shrinkage"] - z0
                assert float(np.abs(resid).max()) < 1e-9 * scale, (
                    f"{sim_id}/{key}: identity broken in the batched path "
                    f"(max residual {np.abs(resid).max():.3e}, chi2 scale {scale:.3e})"
                )
                checked += 1
        assert checked > 0, "no n_terms=1 rows were available to check"


def test_decompose_reduction(signal_artifact: Path) -> None:
    rows = reduce_decompose(signal_artifact)
    assert rows
    assert {"peak_agree", "occam_range", "ln_amp_constraint"} <= set(rows[0])
    # Occam must be positive on average: it is a penalty, not a bonus.
    assert np.median([r["occam_mean"] for r in rows]) > 0.0


def test_roc_reduction(signal_artifact: Path, null_artifact: Path) -> None:
    rows = reduce_roc(signal_artifact, null_artifact, fpr=0.2)
    assert rows
    stats = {r["statistic"] for r in rows}
    assert "z0" in stats
    assert any(s.startswith("delta:") for s in stats)
    assert all(0.0 <= r["tpr"] <= 1.0 for r in rows)


def test_calibrate_reduction(signal_artifact: Path) -> None:
    rows = reduce_calibrate(signal_artifact)
    assert rows
    assert all(0.0 <= r["tv"] <= 1.0 + 1e-9 for r in rows)
    assert any(r["statistic"] == "z0" for r in rows)


def test_regime_map_reduction(signal_artifact: Path, null_artifact: Path) -> None:
    rows = reduce_regime_map(signal_artifact, null_artifact, fpr=0.2)
    assert rows
    assert {r["arm"] for r in rows} == {"delta_default", "delta_h1", "z0"}
    # The n_obs=2 cell must appear with z0 marked unavailable and a stated reason,
    # rather than being silently dropped or scored on garbage.
    sparse = [r for r in rows if r["n_obs"] == 2 and r["arm"] == "z0"]
    assert sparse, "the sparse cell is missing from the map entirely"
    assert all(not r["available"] for r in sparse)
    assert all(r["reason"] for r in sparse)
    # harv's arms must still be available and scored there.
    sparse_delta = [
        r for r in rows if r["n_obs"] == 2 and r["arm"] == "delta_h1"
    ]
    assert sparse_delta and all(r["available"] for r in sparse_delta)
    # P/T_span < 3 for all test cells, so identifiability should hold.
    assert all(r["period_identifiable"] for r in rows)


def test_prior_quality_reduction(signal_artifact: Path) -> None:
    rows = reduce_prior_quality(
        signal_artifact, n_prior_samples=4_000, limit=1, seed=0
    )
    assert rows
    assert {r["arm"] for r in rows} == {"delta", "z0", "loguniform"}
    for r in rows:
        assert np.isfinite(r["max_log_likelihood"])
        assert r["evidence_ess"] >= 0.0
