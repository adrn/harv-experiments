"""The identity gate as a test, across the grid corners.

``Delta + Occam + Shrinkage == z0`` with ``z0`` from kepmodel is the one check that
validates the cross-code wiring: frequency convention, time reference, noise model,
base columns. If this fails, nothing downstream means anything.
"""

from __future__ import annotations

import pytest
from kepcmp.adapters import RVAdapter
from kepcmp.grid import Cell, shared_frequency_grid
from kepcmp.identity import check_identity

CORNERS = [
    Cell(period_ratio=0.02, snr=1.0, eccentricity=0.0, n_obs=4),
    Cell(period_ratio=0.1, snr=100.0, eccentricity=0.0, n_obs=32),
    Cell(period_ratio=1.0, snr=3.0, eccentricity=0.6, n_obs=8),
    Cell(period_ratio=3.0, snr=100.0, eccentricity=0.6, n_obs=8),
    Cell(period_ratio=10.0, snr=30.0, eccentricity=0.0, n_obs=16),
    Cell(period_ratio=3.0, snr=30.0, eccentricity=0.0, n_obs=32),
]

#: Where the profile statistic does not exist, so the gate is undefined.
DEGENERATE = [
    Cell(period_ratio=0.3, snr=10.0, eccentricity=0.0, n_obs=2),
    Cell(period_ratio=0.3, snr=10.0, eccentricity=0.0, n_obs=3),
]


@pytest.fixture(scope="module")
def adapter() -> RVAdapter:
    return RVAdapter()


@pytest.fixture(scope="module")
def frequency(adapter: RVAdapter):
    # Strided: the gate is a python loop per frequency and does not need full density
    # to establish that the conventions line up.
    return shared_frequency_grid(adapter)[::24]


@pytest.mark.parametrize("cell", CORNERS, ids=lambda c: c.cell_id)
def test_identity_gate(adapter: RVAdapter, frequency, cell: Cell) -> None:
    data, _ = adapter.simulate(cell, 0)
    rep = check_identity(adapter, data, frequency, sigma_amp=adapter.data_rms(data))
    assert rep.passed, rep.report()

    # Independent teeth, so the test is not just re-running the gate's own arithmetic.
    #
    # The marginal path is conditioning-independent (measured ~3e-16 relative from
    # cond 2e3 to 2e10), so a fixed tight bar is legitimate there.
    assert rep.ln_z_residual / rep.chi2_scale < 1e-11, rep.report()
    # The cross-code residual is not: it scales as eps * cond**2 because kepmodel forms
    # the normal matrix explicitly. So the only fixed claim worth making is that it sits
    # far below wiring-bug scale, which is O(1) --- see the 2 pi test below.
    assert rep.identity_residual_relative < 1e-6, rep.report()
    assert rep.n_asserted == rep.n_frequencies, "unexpected ill-conditioned frequencies"


@pytest.mark.parametrize("cell", DEGENERATE, ids=lambda c: c.cell_id)
def test_sparse_data_reports_no_profile_statistic(
    adapter: RVAdapter, frequency, cell: Cell
) -> None:
    """At ``n_obs <= p + d`` the profile statistic must be reported absent, not faked.

    kepmodel either returns non-physical ``chi2`` or raises ``LinAlgError`` here. Small
    ``n_obs`` is a primary use case, so "this method has no answer" has to be a
    first-class result rather than a crash or a silently stored garbage number.
    """
    import numpy as np

    data, _ = adapter.simulate(cell, 0)
    chi2_base, z0, degenerate, reason = adapter.kepmodel_z0(data, frequency)
    assert degenerate, "expected the profile statistic to be flagged degenerate"
    assert reason
    assert np.all(np.isnan(z0)), "degenerate z0 must be nan, not a plausible number"
    del chi2_base


@pytest.mark.parametrize("cell", DEGENERATE, ids=lambda c: c.cell_id)
def test_marginal_statistic_survives_sparse_data(
    adapter: RVAdapter, frequency, cell: Cell
) -> None:
    """The marginal statistic stays finite where the profile one cannot be computed.

    ``S = A + Lambda^-1`` is positive-definite even when ``A`` is singular, so the
    prior supplies the missing rank. Being finite is not the same as being *right* ---
    at these ``n_obs`` the peak is driven by prior and window structure rather than
    signal, which is why the mixture ``floor`` matters.
    """
    import harv.periodogram as hp
    import numpy as np

    data, _ = adapter.simulate(cell, 0)
    prior = adapter.trial_prior(data, n_terms=1, sigma_amp=adapter.data_rms(data))
    delta = np.asarray(
        hp.periodogram(data, frequency, prior=prior, n_terms=1).delta_ln_likelihood
    )
    assert np.all(np.isfinite(delta))
    assert np.ptp(delta) > 0.0


def test_gate_detects_a_broken_frequency_convention(
    adapter: RVAdapter, frequency
) -> None:
    """Sanity: the gate must actually fail when the conventions are wrong.

    A gate that cannot fail is not a gate. Feeding kepmodel a grid off by ``2 pi``
    --- the exact mistake the design doc calls out as a silent failure mode --- has
    to blow the identity residual up.
    """
    import numpy as np

    cell = Cell(period_ratio=0.1, snr=10.0, eccentricity=0.0, n_obs=40)
    data, _ = adapter.simulate(cell, 0)

    original = adapter.kepmodel_chi2ogram

    def wrong(d, freq):
        # Pretend harv's cycles-per-day grid is already angular.
        return original(d, freq / (2.0 * np.pi))

    object.__setattr__(adapter, "kepmodel_chi2ogram", wrong)
    try:
        rep = check_identity(adapter, data, frequency, sigma_amp=adapter.data_rms(data))
    finally:
        object.__setattr__(adapter, "kepmodel_chi2ogram", original)

    assert not rep.passed
    assert rep.identity_residual_relative > 1e-6
