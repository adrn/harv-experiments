"""The design doc's remaining tests: known answer, FAP calibration, MC convergence."""

from __future__ import annotations

import harv.periodogram as hp
import numpy as np
import pytest
from kepcmp.adapters import RVAdapter
from kepcmp.adapters.common import angular_frequency_args
from kepcmp.grid import Cell, shared_frequency_grid
from unxt import ustrip


@pytest.fixture(scope="module")
def adapter() -> RVAdapter:
    return RVAdapter()


@pytest.fixture(scope="module")
def frequency(adapter: RVAdapter):
    return shared_frequency_grid(adapter)


def _peak_period(freq, values) -> float:
    period = np.asarray(ustrip("day", 1.0 / freq), dtype=float)
    return float(period[int(np.argmax(np.asarray(values)))])


def test_all_three_statistics_peak_at_the_injected_period(
    adapter: RVAdapter, frequency
) -> None:
    """Noiseless-limit known answer, per the design doc's "Testing" section.

    A genuinely noiseless dataset is degenerate (``chi2_H`` finite but the enlarged
    model fits exactly), so this uses a very high SNR instead, which is the same test
    with well-defined arithmetic.
    """
    cell = Cell(period_ratio=0.1, snr=300.0, eccentricity=0.0, n_obs=40)
    data, truth = adapter.simulate(cell, 0)
    p_true = float(ustrip("day", truth["period"]))
    # Within one periodogram peak width in ln P.
    tol = p_true / float(ustrip("day", adapter.t_span))

    prior = adapter.trial_prior(data, n_terms=1, sigma_amp=adapter.data_rms(data))
    delta = hp.periodogram(data, frequency, prior=prior, n_terms=1)
    p_delta = _peak_period(frequency, delta.delta_ln_likelihood)

    chi2_base, chi2_k = adapter.kepmodel_chi2ogram(data, frequency)
    p_z0 = _peak_period(frequency, 0.5 * (chi2_base - chi2_k))

    ref = adapter.reference_ln_z(data, 1.0 / frequency, n_mc=512, seed=0)
    p_ref = _peak_period(frequency, ref)

    for label, got in (("delta", p_delta), ("z0", p_z0), ("reference", p_ref)):
        assert abs(np.log(got) - np.log(p_true)) < tol, (
            f"{label} peaked at {got:.4f} d, injected {p_true:.4f} d"
        )


def test_reference_mc_converges(adapter: RVAdapter) -> None:
    """``R`` at two ``n_mc`` must agree in peak location and in shape.

    Common random numbers make the *shape* converge much faster than the level, and
    shape is all the calibration reduction reads --- so the shape tolerance is the
    meaningful one.
    """
    cell = Cell(period_ratio=0.3, snr=10.0, eccentricity=0.0, n_obs=40)
    data, _ = adapter.simulate(cell, 0)
    freq = shared_frequency_grid(adapter)[::8]
    periods = 1.0 / freq

    r_lo = adapter.reference_ln_z(data, periods, n_mc=256, seed=0)
    r_hi = adapter.reference_ln_z(data, periods, n_mc=2048, seed=0)

    assert abs(np.log(_peak_period(freq, r_lo)) - np.log(_peak_period(freq, r_hi))) < 0.05

    # Shape: compare after removing the level, relative to the curve's own range.
    span = float(np.ptp(r_hi))
    assert span > 1.0, "reference curve is flat; the test would be vacuous"
    resid = (r_lo - r_lo.mean()) - (r_hi - r_hi.mean())
    assert float(np.abs(resid).max()) / span < 0.25


def test_kepmodel_analytic_fap_matches_empirical_null_rate(adapter: RVAdapter) -> None:
    """Validate our wiring of kepmodel against its own published calibration.

    On ``K = 0`` data, the fraction of realizations whose peak exceeds a threshold
    must match kepmodel's analytic ``fap()`` for that threshold. This is the cheapest
    available proof that we are driving their code correctly --- it tests our setup,
    not their theory.
    """
    freq = shared_frequency_grid(adapter)[::4]
    nu0, dnu, nfreq = angular_frequency_args(freq)
    numax = nu0 + (nfreq - 1) * dnu

    n_trials = 300
    cell = Cell(period_ratio=1.0, snr=0.0, eccentricity=0.0, n_obs=40)
    peaks = np.empty(n_trials)
    fap_model = None
    for i in range(n_trials):
        data, _ = adapter.simulate(cell, 1000 + i)
        model = adapter.kepmodel_model(data)
        _, power = model.periodogram(nu0, dnu, nfreq)
        peaks[i] = float(np.max(power))
        if fap_model is None:
            fap_model = model

    # Compare at the empirical median: analytic FAP there should be ~0.5.
    zmed = float(np.median(peaks))
    fap_analytic = float(fap_model.fap(zmed, numax))
    empirical = float(np.mean(peaks > zmed))

    assert abs(fap_analytic - empirical) < 0.15, (
        f"analytic FAP {fap_analytic:.3f} vs empirical {empirical:.3f} at z={zmed:.4f}"
        " -- suggests the kepmodel model is wired up wrong (noise, columns, or grid)"
    )


def test_null_statistics_are_small(adapter: RVAdapter, frequency) -> None:
    """On K=0 data, ``Delta`` should not show a confident detection.

    ``z0 >= 0`` always, but ``Delta`` has a meaningful zero, so a well-behaved
    marginal statistic should stay near or below it on pure noise.
    """
    cell = Cell(period_ratio=1.0, snr=0.0, eccentricity=0.0, n_obs=40)
    worst = -np.inf
    for seed in range(8):
        data, _ = adapter.simulate(cell, 500 + seed)
        prior = adapter.trial_prior(data, n_terms=1, sigma_amp=adapter.data_rms(data))
        delta = hp.periodogram(data, frequency, prior=prior, n_terms=1)
        worst = max(worst, float(np.max(delta.delta_ln_likelihood)))
    assert worst < 10.0, f"max Delta on null data reached {worst:.2f} nats"
