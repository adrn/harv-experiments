"""Pin the fast batched path to the validated scalar one.

:func:`kepcmp.linalg.evaluate` is what the identity gate validates against harv and
kepmodel. :func:`~kepcmp.linalg.evaluate_batched` is the vectorized equivalent used
for the production grid. If they drift, the gate would still pass while every stored
``occam`` / ``shrinkage`` was wrong --- so they are pinned here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from kepcmp import linalg
from kepcmp.adapters import RVAdapter
from kepcmp.grid import Cell, shared_frequency_grid
from unxt import Q


@pytest.fixture(scope="module")
def adapter() -> RVAdapter:
    return RVAdapter()


@pytest.mark.parametrize("n_terms", [1, 2, 3])
@pytest.mark.parametrize(
    "cell",
    [
        Cell(period_ratio=0.1, snr=10.0, eccentricity=0.0, n_obs=40),
        Cell(period_ratio=3.0, snr=100.0, eccentricity=0.6, n_obs=40),
    ],
    ids=["well-sampled", "partial-arc-high-snr"],
)
def test_batched_matches_scalar(adapter: RVAdapter, cell: Cell, n_terms: int) -> None:
    data, _ = adapter.simulate(cell, 0)
    freq = shared_frequency_grid(adapter)[::64]
    periods = 1.0 / freq
    prior = adapter.trial_prior(data, n_terms=n_terms, sigma_amp=Q(3.0, "km/s"))

    x_all, ref = adapter.marg_blocks_batched(
        data, periods, n_terms=n_terms, prior=prior
    )
    got = linalg.evaluate_batched(
        x_all,
        jnp.asarray(ref.y),
        jnp.asarray(ref.cov),
        jnp.asarray(ref.prior_mu),
        jnp.asarray(ref.prior_scale_tril),
    )

    scalar = [
        linalg.evaluate(
            *adapter.marg_blocks(data, p, n_terms=n_terms, prior=prior)[:5]
        )
        for p in periods
    ]
    want = linalg.to_arrays(scalar)

    scale = max(float(np.abs(want["chi2"]).max()), 1.0)
    cond = max(float(np.nanmax(want["cond"])), 1.0)

    # The marginal quantities route through M = I + B^T B >= I and are accurate
    # regardless of conditioning: measured ~3e-16 relative at cond from 2e3 to 2e10.
    for field in ("ln_z", "occam"):
        assert np.allclose(
            np.asarray(got[field]), want[field], rtol=0, atol=1e-12 * scale
        ), f"{field} drifted between batched and scalar paths"

    # The profile quantities need a pseudo-inverse of a near-singular matrix, and the
    # batched SVD and scalar lstsq pick their rank cutoffs independently. Measured
    # relative divergence grows linearly in cond (4e-15 at 2.4e3 -> 4.2e-8 at 2.4e10),
    # so the bar scales with it; 10x eps*cond leaves ~100x margin over the trend.
    profile_atol = max(1e-12, 10.0 * float(np.finfo(float).eps) * cond) * scale
    for field in ("chi2", "shrinkage"):
        assert np.allclose(
            np.asarray(got[field]), want[field], rtol=0, atol=profile_atol
        ), f"{field} drifted beyond the conditioning-scaled bar (cond={cond:.2e})"

    # cond is a diagnostic, and at cond ~ 1e10 the smallest singular value is itself
    # near the precision limit, so the two SVDs report it to ~1e-9 relative. Its exact
    # value never enters a result.
    assert np.allclose(np.asarray(got["cond"]), want["cond"], rtol=1e-6)


def test_decomposition_is_exact(adapter: RVAdapter) -> None:
    """``ln Z = ln L_hat - shrinkage - occam`` per model, to machine precision."""
    cell = Cell(period_ratio=0.3, snr=30.0, eccentricity=0.3, n_obs=40)
    data, _ = adapter.simulate(cell, 1)
    prior = adapter.trial_prior(data, n_terms=2, sigma_amp=Q(3.0, "km/s"))
    terms = linalg.evaluate(
        *adapter.marg_blocks(data, Q(200.0, "day"), n_terms=2, prior=prior)[:5]
    )
    rebuilt = terms.ln_l_hat - terms.shrinkage - terms.occam
    assert abs(rebuilt - terms.ln_z) < 1e-9 * max(abs(terms.chi2), 1.0)


def test_occam_and_shrinkage_are_non_negative(adapter: RVAdapter) -> None:
    """Both terms are non-negative per model (the *differences* may have any sign)."""
    cell = Cell(period_ratio=1.0, snr=10.0, eccentricity=0.0, n_obs=40)
    data, _ = adapter.simulate(cell, 2)
    prior = adapter.trial_prior(data, n_terms=2, sigma_amp=Q(3.0, "km/s"))
    for p in (50.0, 500.0, 5000.0):
        terms = linalg.evaluate(
            *adapter.marg_blocks(data, Q(p, "day"), n_terms=2, prior=prior)[:5]
        )
        assert terms.occam >= 0.0
        assert terms.shrinkage >= -1e-9 * max(abs(terms.chi2), 1.0)


def test_rejects_full_covariance() -> None:
    """A 2-d covariance is a correlated-noise run, which is out of scope."""
    with pytest.raises(ValueError, match="white-noise only"):
        linalg.evaluate(
            np.ones((4, 2)), np.zeros(4), np.eye(4), np.zeros(2), np.eye(2)
        )
