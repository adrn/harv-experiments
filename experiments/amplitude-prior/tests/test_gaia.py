"""The Gaia astrometry adapter: correctness gate, known answer, roundtrip.

The gate matters more here than for RV. The base model has 5 columns and the trial
model contributes ``d = 4``, so the algebra is cancelling a much larger block
factorisation --- and the arm's period-dependent prior is applied to four amplitude
columns per harmonic rather than two.
"""

from __future__ import annotations

import harv.periodogram as hp
import numpy as np
import pytest
from ampcal.adapters import GaiaAdapter
from ampcal.grid import Arm, Cell, shared_frequency_grid
from ampcal.identity import check_identity
from unxt import ustrip

CORNERS = [
    Cell(population="physical", period_ratio=0.1, snr=3.0, eccentricity=0.0,
         n_obs=20),
    # P = 1 yr exactly: the frequency columns sit on top of the parallax column, the
    # worst-conditioned cell in the grid and where a period-dependent Occam tilt has
    # the most room to do damage.
    Cell(population="physical", period_ratio=0.2, snr=30.0, eccentricity=0.0,
         n_obs=20),
    Cell(population="independent", period_ratio=1.0, snr=10.0, eccentricity=0.6,
         n_obs=40),
    Cell(population="independent", period_ratio=5.0, snr=30.0, eccentricity=0.3,
         n_obs=40),
]


@pytest.fixture(scope="module")
def adapter() -> GaiaAdapter:
    return GaiaAdapter()


@pytest.fixture(scope="module")
def frequency(adapter: GaiaAdapter):
    return shared_frequency_grid(adapter)[::32]


@pytest.mark.parametrize("cell", CORNERS, ids=lambda c: c.cell_id)
def test_correctness_gate(adapter: GaiaAdapter, frequency, cell: Cell) -> None:
    """Run on the *physical* exponent, not the flat control: the flat arm exercises
    none of the per-frequency prior machinery this rebuild added."""
    data, _ = adapter.simulate(cell, 0)
    arm = Arm(n_terms=1, exponent=2.0 / 3.0, level=1.0)
    rep = check_identity(
        adapter, data, frequency, arm=arm, rms=adapter.data_rms(data)
    )
    assert rep.passed, rep.report()

    # The marginal path routes through M = I + B^T B >= I and stays accurate whatever
    # the conditioning, so it gets the tight bar even with 9 columns.
    assert rep.ln_z_residual / rep.chi2_scale < 1e-8, rep.report()


def test_amplitude_columns_are_the_ti_block(adapter: GaiaAdapter) -> None:
    """The arm's prior must land on the four Thiele-Innes columns and nothing else.

    Silent failure mode: naming the amplitude columns by *position* rather than by name
    works for RV (where ``v_sys`` is last) and puts the tilt on the astrometric solution
    for Gaia (where the base columns are first).
    """
    assert adapter.amp_names(1) == ("ti_A_1", "ti_B_1", "ti_F_1", "ti_G_1")
    assert len(adapter.amp_names(2)) == 8
    assert not set(adapter.amp_names(2)) & set(adapter.base_linear_names)


def test_all_statistics_peak_at_the_injected_period(adapter: GaiaAdapter) -> None:
    """Known answer for a period-dependent ``Delta`` and the Thiele-Innes reference."""
    frequency = shared_frequency_grid(adapter)[::4]
    cell = Cell(population="independent", period_ratio=0.1, snr=100.0,
                eccentricity=0.0, n_obs=40)
    data, truth = adapter.simulate(cell, 0)
    p_true = float(ustrip("day", truth["period"]))
    tol = p_true / float(ustrip("day", adapter.t_span))

    period = np.asarray(ustrip("day", 1.0 / frequency), dtype=float)

    def peak(values) -> float:
        return float(period[int(np.argmax(np.asarray(values)))])

    arm = Arm(n_terms=1, exponent=2.0 / 3.0, level=1.0)
    prior = adapter.arm_prior(data, arm, adapter.data_rms(data))
    p_delta = peak(
        hp.periodogram(data, frequency, prior=prior, n_terms=1).delta_ln_likelihood
    )
    p_ref = peak(adapter.reference_ln_z(data, 1.0 / frequency, n_mc=512, seed=0))

    for label, got in (("delta", p_delta), ("reference", p_ref)):
        assert abs(np.log(got) - np.log(p_true)) < tol, (
            f"{label} peaked at {got:.4f} d, injected {p_true:.4f} d"
        )


def test_physical_population_scales_as_expected(adapter: GaiaAdapter) -> None:
    """``a_0 ~ P^(2/3)`` on the physical population and flat on the independent one.

    Without this the population factor is unfalsifiable: a bug in the mass draw would
    leave both populations identical and every "misspecification cost" would read as
    zero for the wrong reason.
    """
    for pop, want in (("physical", 2.0 / 3.0), ("independent", 0.0)):
        ln_p, ln_a = [], []
        for pr in adapter.period_ratios:
            cell = Cell(population=pop, period_ratio=pr, snr=10.0, eccentricity=0.0,
                        n_obs=8)
            for seed in range(24):
                ln_p.append(np.log(float(ustrip("day", adapter.period(cell)))))
                ln_a.append(
                    np.log(float(ustrip("mas", adapter.amplitude(cell, seed))))
                )
        slope = float(np.polyfit(ln_p, ln_a, 1)[0])
        assert abs(slope - want) < 0.05, f"{pop}: slope {slope:.4f}, want {want:.4f}"


def test_artifact_data_roundtrip(adapter: GaiaAdapter, frequency, tmp_path) -> None:
    """``data_arrays`` -> HDF5 -> ``data_from_arrays`` must be lossless.

    The report's nuisance sweep replays stored datasets rather than re-simulating, so a
    dropped column here would silently change what it measures. Gaia stores five arrays
    where RV stores three, which is why the artifact schema does not name them.
    """
    from ampcal.artifact import ArtifactReader, ArtifactWriter
    from ampcal.run import run_one

    cell = Cell(population="physical", period_ratio=0.2, snr=30.0, eccentricity=0.0,
                n_obs=20)
    rec = run_one(
        adapter, cell, 0, frequency,
        arms=[Arm(n_terms=1, exponent=2.0 / 3.0, level=1.0)],
    )
    assert set(rec.data) == {
        "time", "al_position", "al_position_err", "scan_angle", "parallax_factor",
    }

    path = tmp_path / "gaia.h5"
    with ArtifactWriter(
        path,
        frequency=np.asarray(ustrip("1/day", frequency), dtype=float),
        **adapter.meta(),
    ) as w:
        w.write(rec)
    with ArtifactReader(path) as art:
        assert art.adapter_name == "gaia"
        assert art.attrs(rec.sim_id)["population"] == "physical"
        back = adapter.data_from_arrays(art.data(rec.sim_id))

    original = adapter.simulate(cell, 0)[0]
    for name in ("al_position", "al_position_err", "scan_angle"):
        np.testing.assert_allclose(
            np.asarray(getattr(back, name).value),
            np.asarray(getattr(original, name).value),
        )
    np.testing.assert_allclose(
        np.asarray(back.parallax_factor), np.asarray(original.parallax_factor)
    )
