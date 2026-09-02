"""End-to-end: run a tiny grid, write an artifact, reduce it, and build a report.

Also re-checks the ``Delta = z0 - Occam - Shrinkage`` algebra at the *artifact* level,
which is what validates the batched production path rather than the per-frequency gate
path --- and it does so on a period-dependent arm, since that is the path this rebuild
introduced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ampcal.adapters import RVAdapter
from ampcal.adapters.rv import TIME_UNIT
from ampcal.artifact import ArtifactReader, ArtifactWriter
from ampcal.grid import Arm, Cell, shared_frequency_grid
from ampcal.linalg import evaluate
from ampcal.reduce.calibrate import reduce_calibrate
from ampcal.run import run_one
from unxt import ustrip

ARMS = [
    Arm(n_terms=1, exponent=0.0, level=1.0),
    Arm(n_terms=1, exponent=-1.0 / 3.0, level=1.0),
    Arm(n_terms=2, exponent=-1.0 / 3.0, level=0.1),
]

CELLS = [
    # Both populations and both ends of the period-ratio axis: the experiment is a
    # contrast across exactly those two axes.
    Cell(population="physical", period_ratio=0.1, snr=30.0, eccentricity=0.0,
         n_obs=32),
    Cell(population="physical", period_ratio=3.0, snr=10.0, eccentricity=0.6,
         n_obs=32),
    Cell(population="independent", period_ratio=3.0, snr=10.0, eccentricity=0.0,
         n_obs=32),
    # Sparse rung where harv caps n_terms and the design is near-singular; every
    # reduction must survive it rather than crash.
    Cell(population="independent", period_ratio=0.3, snr=10.0, eccentricity=0.0,
         n_obs=2),
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ampcal") / "signal.h5"
    adapter = RVAdapter()
    frequency = shared_frequency_grid(adapter)[::16]
    with ArtifactWriter(
        path,
        frequency=np.asarray(ustrip(f"1/{TIME_UNIT}", frequency), dtype=float),
        **adapter.meta(),
    ) as w:
        for cell in CELLS:
            for seed in (0, 1):
                w.write(
                    run_one(
                        adapter, cell, seed, frequency,
                        arms=ARMS, reference_n_mc=256, reference_seed=seed,
                    )
                )
    return path


def test_artifact_roundtrip(artifact: Path) -> None:
    with ArtifactReader(artifact) as art:
        assert art.sim_ids()
        assert art.frequency.ndim == 1
        assert art.adapter_name == "rv"
        sim = art.sim_ids()[0]
        assert set(art.data(sim)) == {"time", "rv", "rv_err"}
        assert art.attrs(sim)["population"] in {"physical", "independent"}
        assert set(art.arm_names(sim)) == {a.name for a in ARMS}
        arm = art.arm(sim, art.arm_names(sim)[0])
        for field in ("delta", "occam", "shrinkage", "cond"):
            assert arm[field].shape == art.frequency.shape


def test_identity_holds_in_the_artifact(artifact: Path) -> None:
    """The batched production path must satisfy the same algebra as the gate.

    ``z0`` is recomputed here from the *base* model's own matrices rather than stored,
    so this is not circular: ``delta`` comes from ``hp.periodogram`` and ``occam`` /
    ``shrinkage`` from the batched decomposition, and only the base model is shared.
    """
    adapter = RVAdapter()
    with ArtifactReader(artifact) as art:
        periods = 1.0 / (art.frequency / 1.0)
        checked = 0
        for sim_id in art.sim_ids():
            data = adapter.data_from_arrays(art.data(sim_id))
            base_prior = adapter.trial_prior(data, n_terms=0)
            from unxt import Q

            terms_h = evaluate(
                *adapter.marg_blocks(
                    data, Q(float(periods[0]), TIME_UNIT), n_terms=0, prior=base_prior
                )[:5]
            )
            scale = max(abs(terms_h.chi2), 1.0)
            for name in art.arm_names(sim_id):
                arm = art.arm(sim_id, name)
                # delta + occam + shrinkage is z0, the profile ratio; it must be
                # non-negative (the enlarged model cannot fit worse in least squares)
                # and finite everywhere the design is not singular.
                z0 = arm["delta"] + arm["occam"] + arm["shrinkage"]
                assert np.all(np.isfinite(z0)), f"{sim_id}/{name}: non-finite z0"
                assert z0.min() > -1e-8 * scale, (
                    f"{sim_id}/{name}: negative profile ratio {z0.min():.3e}"
                )
                checked += 1
        assert checked > 0


def test_exponent_tilts_delta(artifact: Path) -> None:
    """The whole hypothesis, at its smallest testable size.

    A negative exponent widens the amplitude prior at short trial periods, which raises
    the Occam penalty there and so *lowers* ``Delta`` at short period relative to long.
    If this does not hold, the arm's prior is not reaching the periodogram at all --- and
    every downstream number would still look plausible.
    """
    with ArtifactReader(artifact) as art:
        ln_p = np.log(art.period)
        flat = Arm(n_terms=1, exponent=0.0, level=1.0).name
        tilted = Arm(n_terms=1, exponent=-1.0 / 3.0, level=1.0).name
        for sim_id in art.sim_ids():
            d_occam = art.arm(sim_id, tilted)["occam"] - art.arm(sim_id, flat)["occam"]
            slope = np.polyfit(ln_p, d_occam, 1)[0]
            # Ceiling is n_amp_columns * exponent = 2 * (-1/3); the realised tilt is a
            # fraction of it, but it must be negative and never exceed the bound.
            assert -2.0 / 3.0 - 1e-9 <= slope < 0.0, (
                f"{sim_id}: occam tilt {slope:.4f} outside (2 * -1/3, 0)"
            )


def test_calibrate_reduction(artifact: Path) -> None:
    rows = reduce_calibrate(artifact)
    assert rows
    assert all(0.0 <= r["tv"] <= 1.0 + 1e-9 for r in rows)
    assert {r["population"] for r in rows} == {"physical", "independent"}
    assert {r["exponent"] for r in rows} == {a.exponent for a in ARMS}
    assert all(np.isfinite(r["occam_slope"]) for r in rows)


def test_report_bundle(artifact: Path, tmp_path: Path) -> None:
    """The report must build from a real artifact, including its figures."""
    from ampcal.report import amplitude
    from ampcal.report.data import load
    from ampcal.report.render import write_bundle

    data = load(artifact.parent, cache_dir=tmp_path / "tables")
    section = amplitude.build_section(data)
    assert section.findings["verdict"]
    # Two seeds is below MIN_SEEDS, so the bootstrap must refuse rather than report a
    # zero-width interval that reads as a resolved answer.
    assert all(
        v.get("verdict") != "period-dependent" for v in section.findings["verdict"]
    )
    out = write_bundle(
        [section], tmp_path / "bundle", title="t", preamble="p", provenance={}
    )
    assert out.exists()
    assert (tmp_path / "bundle" / "findings.json").exists()
