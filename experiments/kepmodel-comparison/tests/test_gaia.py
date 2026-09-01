"""The Gaia astrometry adapter: identity gate, degeneracy boundary, known answer.

The identity gate matters more here than for RV. The base model has 5 columns and
the trial model contributes ``d = 4``, so ``Delta + Occam + Shrinkage == z0`` is
cancelling a much larger block factorisation --- and the five base columns are handed
to kepmodel out of harv's own design matrix, which this is what verifies.
"""

from __future__ import annotations

import harv.periodogram as hp
import numpy as np
import pytest
from kepcmp.adapters import GaiaAdapter
from kepcmp.grid import Cell, shared_frequency_grid
from kepcmp.identity import check_identity
from unxt import ustrip

CORNERS = [
    Cell(period_ratio=0.05, snr=3.0, eccentricity=0.0, n_obs=20),
    # P = 1 yr exactly: the frequency columns sit on top of the parallax column, which
    # is the degeneracy the whole Gaia grid is built to probe.
    Cell(period_ratio=0.2, snr=30.0, eccentricity=0.0, n_obs=20),
    Cell(period_ratio=1.0, snr=10.0, eccentricity=0.6, n_obs=40),
    Cell(period_ratio=2.0, snr=100.0, eccentricity=0.3, n_obs=40),
]


@pytest.fixture(scope="module")
def adapter() -> GaiaAdapter:
    return GaiaAdapter()


@pytest.fixture(scope="module")
def frequency(adapter: GaiaAdapter):
    return shared_frequency_grid(adapter)[::32]


@pytest.mark.parametrize("cell", CORNERS, ids=lambda c: c.cell_id)
def test_identity_gate(adapter: GaiaAdapter, frequency, cell: Cell) -> None:
    data, _ = adapter.simulate(cell, 0)
    rep = check_identity(adapter, data, frequency, sigma_amp=adapter.data_rms(data))
    assert rep.passed, rep.report()

    # The marginal path routes through M = I + B^T B >= I and stays accurate whatever
    # the conditioning, so it gets the tight bar even with 9 columns.
    assert rep.ln_z_residual / rep.chi2_scale < 1e-11, rep.report()
    # The cross-code residual carries kepmodel's explicit inverse and goes as
    # eps * cond**2; cond is genuinely large here because the frequency columns really
    # do go degenerate with parallax near 1 yr. A wiring bug is O(1).
    assert rep.identity_residual_relative < 1e-5, rep.report()


def test_bases_are_column_identical_to_kepmodel(adapter: GaiaAdapter) -> None:
    """harv's ``d=4`` frequency block spans exactly kepmodel's ``_perio_phi``.

    The identity gate would catch a mismatch, but only as a number; this says *why*
    it passes, and fails loudly if either package reorders or redefines its columns.
    """
    cell = Cell(period_ratio=0.2, snr=30.0, eccentricity=0.0, n_obs=20)
    data, _ = adapter.simulate(cell, 0)
    period = adapter.period(cell)

    blocks = adapter.marg_blocks(
        data, period, n_terms=1, prior=adapter.trial_prior(data, n_terms=1,
                                                           sigma_amp=adapter.data_rms(data))
    )
    harv_freq_cols = blocks.X[:, adapter.n_base_columns :]
    assert harv_freq_cols.shape[1] == adapter.d == 4

    km = adapter.kepmodel_model(data)
    t = np.asarray(ustrip("day", data.time - data.t_ref), dtype=float)
    nu = 2.0 * np.pi / float(ustrip("day", period))
    kep_freq_cols = np.stack(km._perio_phi(np.cos(nu * t), np.sin(nu * t)), axis=-1)

    # Same span, checked without assuming an ordering: each set must be fully
    # explained by the other in least squares.
    for a, b in ((harv_freq_cols, kep_freq_cols), (kep_freq_cols, harv_freq_cols)):
        coef, *_ = np.linalg.lstsq(b, a, rcond=None)
        resid = np.abs(a - b @ coef).max()
        assert resid < 1e-10 * max(np.abs(a).max(), 1.0), f"residual {resid:.3e}"


@pytest.mark.parametrize("n_obs", [4, 9])
def test_profile_statistic_absent_below_p_plus_d(
    adapter: GaiaAdapter, frequency, n_obs: int
) -> None:
    """``z0`` needs ``n_obs > p + d = 9``, four times the RV threshold.

    The runner must record that as a first-class "no answer", not a number.
    """
    cell = Cell(period_ratio=0.2, snr=30.0, eccentricity=0.0, n_obs=n_obs)
    data, _ = adapter.simulate(cell, 0)
    _, z0, degenerate, reason = adapter.kepmodel_z0(data, frequency)
    assert degenerate
    assert "p+d=9" in reason
    assert np.all(np.isnan(z0))


def test_all_three_statistics_peak_at_the_injected_period(
    adapter: GaiaAdapter,
) -> None:
    """Known answer for ``Delta``, ``z0`` and the Thiele-Innes reference ``R``."""
    frequency = shared_frequency_grid(adapter)[::4]
    cell = Cell(period_ratio=0.1, snr=100.0, eccentricity=0.0, n_obs=40)
    data, truth = adapter.simulate(cell, 0)
    p_true = float(ustrip("day", truth["period"]))
    tol = p_true / float(ustrip("day", adapter.t_span))

    period = np.asarray(ustrip("day", 1.0 / frequency), dtype=float)

    def peak(values) -> float:
        return float(period[int(np.argmax(np.asarray(values)))])

    prior = adapter.trial_prior(data, n_terms=1, sigma_amp=adapter.data_rms(data))
    p_delta = peak(hp.periodogram(data, frequency, prior=prior,
                                  n_terms=1).delta_ln_likelihood)
    _, z0, degenerate, _ = adapter.kepmodel_z0(data, frequency)
    assert not degenerate
    p_z0 = peak(z0)
    p_ref = peak(adapter.reference_ln_z(data, 1.0 / frequency, n_mc=512, seed=0))

    for label, got in (("delta", p_delta), ("z0", p_z0), ("reference", p_ref)):
        assert abs(np.log(got) - np.log(p_true)) < tol, (
            f"{label} peaked at {got:.4f} d, injected {p_true:.4f} d"
        )


def test_artifact_data_roundtrip(adapter: GaiaAdapter, frequency) -> None:
    """``data_arrays`` -> HDF5 -> ``data_from_arrays`` must be lossless.

    ``prior_quality`` re-runs the sampler on the stored dataset rather than
    re-simulating, so a dropped column here would silently change what it samples.
    Gaia stores five arrays where RV stores three, which is why the artifact schema
    had to stop naming them.
    """
    from kepcmp.artifact import ArtifactReader, ArtifactWriter
    from kepcmp.run import run_one

    cell = Cell(period_ratio=0.2, snr=30.0, eccentricity=0.0, n_obs=20)
    rec = run_one(
        adapter, cell, 0, frequency, n_terms_values=(1,), sigma_amp_mults=(1.0,)
    )
    assert set(rec.data) == {
        "time",
        "al_position",
        "al_position_err",
        "scan_angle",
        "parallax_factor",
    }

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gaia.h5"
        with ArtifactWriter(
            path, frequency=np.asarray(1.0 / ustrip("day", 1.0 / frequency)),
            **adapter.meta(),
        ) as w:
            w.write(rec)
        with ArtifactReader(path) as art:
            assert art.adapter_name == "gaia"
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


def test_science_prior_builds(adapter: GaiaAdapter) -> None:
    """``prior_quality`` is the only caller of ``science_prior`` and it is not in the
    launcher, so nothing else would notice this being wrong until a long run ended.

    harv's Thiele-Innes proper-motion prior is parameterized in *velocity*
    (``sigma_vtan``), not in ``mas/yr``, and omitting it raises rather than defaulting.
    """
    import numpyro.distributions as dist
    from harv.distributions import QuantityDistribution as QD
    from unxt import Q

    period_prior = QD(dist.LogUniform(20.0, 4000.0), "day")
    prior = adapter.science_prior(None, period_prior)
    assert set(prior.linear_priors) == {
        "ra0", "dec0", "pmra", "pmdec", "parallax", "ti_A", "ti_B", "ti_F", "ti_G",
    }
    assert set(prior.nonlinear_priors) == {"period", "eccentricity", "phase_peri"}

    # Two of these are LinearPriorCallables that only resolve at sample time, so
    # checking the keys alone is not enough: a sigma_a0 in mas instead of AU builds a
    # perfectly well-formed prior and then raises UnitConversionError inside the
    # sampler, several minutes into a run.
    params = {
        "period": Q(500.0, "day"),
        "eccentricity": 0.3,
        "parallax": Q(10.0, "mas"),
    }
    for name in ("ti_A", "pmra"):
        entry = prior.linear_priors[name]
        resolved = entry(params) if callable(entry) else entry
        assert resolved.unit is not None
