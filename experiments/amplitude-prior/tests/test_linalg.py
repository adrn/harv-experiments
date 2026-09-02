"""Pin the fast batched path to the validated scalar one.

:func:`ampcal.linalg.evaluate` is what the correctness gate validates against harv.
:func:`~ampcal.linalg.evaluate_batched` is the vectorized equivalent used for the
production grid. If they drift, the gate would still pass while every stored ``occam`` /
``shrinkage`` was wrong --- so they are pinned here, **for period-dependent arms as well
as flat ones**. The per-frequency prior is the new machinery in this rebuild, and it is
the only thing standing between "the exponent tilts Occam" and "the exponent silently
did nothing to the recorded decomposition".
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from ampcal import linalg
from ampcal.adapters import RVAdapter
from ampcal.grid import Arm, Cell, shared_frequency_grid
from unxt import Q


@pytest.fixture(scope="module")
def adapter() -> RVAdapter:
    return RVAdapter()


CELLS = [
    Cell(population="independent", period_ratio=0.1, snr=10.0, eccentricity=0.0,
         n_obs=40),
    Cell(population="physical", period_ratio=3.0, snr=30.0, eccentricity=0.6,
         n_obs=40),
]


@pytest.mark.parametrize("exponent", [0.0, -1.0 / 3.0, -0.5])
@pytest.mark.parametrize("n_terms", [1, 2])
@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.cell_id)
def test_batched_matches_scalar(
    adapter: RVAdapter, cell: Cell, n_terms: int, exponent: float
) -> None:
    data, _ = adapter.simulate(cell, 0)
    freq = shared_frequency_grid(adapter)[::64]
    periods = 1.0 / freq
    arm = Arm(n_terms=n_terms, exponent=exponent, level=1.0)
    prior = adapter.arm_prior(data, arm, Q(3.0, "km/s"))

    x_all, ref = adapter.marg_blocks_batched(
        data, periods, n_terms=n_terms, prior=prior
    )
    # The shape itself is a result: a period-dependent arm must produce a per-frequency
    # prior, and a flat one must not pay for one.
    expected_ndim = 3 if exponent != 0.0 else 2
    assert ref.prior_scale_tril.ndim == expected_ndim

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


def test_batched_prior_carries_the_arms_power_law(adapter: RVAdapter) -> None:
    """The inferred per-frequency scales must be the arm's own, not a re-fit.

    ``marg_blocks_batched`` extrapolates the prior scales from two evaluations. If it
    inferred the exponent from the wrong pair of periods, or applied it to the wrong
    columns, everything downstream would still run --- with the exponent quietly
    rescaled. This checks the extrapolation against the arm's definition directly.
    """
    cell = Cell(population="physical", period_ratio=0.3, snr=10.0, eccentricity=0.0,
                n_obs=32)
    data, _ = adapter.simulate(cell, 0)
    freq = shared_frequency_grid(adapter)[::64]
    periods = 1.0 / freq
    arm = Arm(n_terms=2, exponent=-1.0 / 3.0, level=1.0)
    rms = adapter.data_rms(data)

    _, ref = adapter.marg_blocks_batched(
        data, periods, n_terms=arm.n_terms, prior=adapter.arm_prior(data, arm, rms)
    )
    p_vals = np.asarray((periods).value, dtype=float)
    diag = np.diagonal(ref.prior_scale_tril, axis1=-2, axis2=-1)

    names = ref.names
    amp = [i for i, n in enumerate(names) if n not in adapter.base_linear_names]
    base = [i for i, n in enumerate(names) if n in adapter.base_linear_names]
    assert amp and base, f"expected both column kinds, got {names}"

    sigma_0 = float((arm.level * rms).value)
    p0 = float(adapter.t_span.to("day").value)
    want = sigma_0 * (p_vals / p0) ** arm.exponent
    np.testing.assert_allclose(
        diag[:, amp], np.repeat(want[:, None], len(amp), axis=1), rtol=1e-12
    )

    # The base columns must not move: their prior is not the arm's.
    np.testing.assert_array_equal(
        diag[:, base], np.repeat(diag[:1, base], diag.shape[0], axis=0)
    )


def test_batched_rejects_a_non_power_law_prior(adapter: RVAdapter) -> None:
    """The extrapolation is exact for power laws and must refuse anything else."""
    import equinox as eqx
    import numpyro.distributions as dist
    from harv.distributions import QuantityDistribution
    from harv.models.parameterizations import FourierRV

    class Wiggly(eqx.Module):
        def __call__(self, params):
            p = float(params["period"].to("day").value)
            return QuantityDistribution(
                dist.Normal(0.0, 1.0 + 0.5 * np.sin(np.log(p))), "km/s"
            )

    cell = Cell(population="independent", period_ratio=0.3, snr=10.0,
                eccentricity=0.0, n_obs=32)
    data, _ = adapter.simulate(cell, 0)
    freq = shared_frequency_grid(adapter)[::64]
    prior = FourierRV(n_terms=1).default_prior(
        period_min=adapter.period_min,
        period_max=adapter.period_max,
        sigma_v0=adapter.sigma_v0,
        cos_amp_1=Wiggly(),
        sin_amp_1=Wiggly(),
    )
    with pytest.raises(ValueError, match="power law"):
        adapter.marg_blocks_batched(data, 1.0 / freq, n_terms=1, prior=prior)


def test_decomposition_is_exact(adapter: RVAdapter) -> None:
    """``ln Z = ln L_hat - shrinkage - occam`` per model, to machine precision."""
    cell = Cell(population="physical", period_ratio=0.3, snr=30.0, eccentricity=0.3,
                n_obs=40)
    data, _ = adapter.simulate(cell, 1)
    arm = Arm(n_terms=2, exponent=-1.0 / 3.0, level=1.0)
    prior = adapter.arm_prior(data, arm, Q(3.0, "km/s"))
    terms = linalg.evaluate(
        *adapter.marg_blocks(data, Q(200.0, "day"), n_terms=2, prior=prior)[:5]
    )
    rebuilt = terms.ln_l_hat - terms.shrinkage - terms.occam
    assert abs(rebuilt - terms.ln_z) < 1e-9 * max(abs(terms.chi2), 1.0)


def test_occam_and_shrinkage_are_non_negative(adapter: RVAdapter) -> None:
    """Both terms are non-negative per model (the *differences* may have any sign)."""
    cell = Cell(population="independent", period_ratio=1.0, snr=10.0,
                eccentricity=0.0, n_obs=40)
    data, _ = adapter.simulate(cell, 2)
    arm = Arm(n_terms=2, exponent=-0.5, level=1.0)
    prior = adapter.arm_prior(data, arm, Q(3.0, "km/s"))
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
