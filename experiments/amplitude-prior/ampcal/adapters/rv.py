"""The RV adapter.

Everything data-type specific about the RV arm of this experiment lives here: the
simulator call, the Fourier trial prior, and the Keplerian reference statistic ``R``.
The parts that are *not* RV-specific --- pulling harv's marginalization blocks and
building the arm's period-dependent prior --- live in :mod:`ampcal.adapters.common`.

One ``sigma_v0`` is shared by the trial prior and the reference, so both statistics
have a genuinely common null model and their zero points coincide.
"""

from __future__ import annotations

__all__ = ("RVAdapter",)

from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from harv.custom_types import ScalarQSpeed, ScalarQTime
from harv.data import RVData
from harv.models import RVModel
from harv.models.parameterizations import FourierRV, StandardRV
from harv.models.priors import HarvPrior
from harv.samplers._prior_resolution import effective_linear_prior_from_prior
from harv.simulate import simulate_rv_sb1_data
from numpy.typing import NDArray
from unxt import Q, ustrip

from ampcal import population
from ampcal.adapters.common import HarvAdapter
from ampcal.grid import Cell

TIME_UNIT = "day"
OBS_UNIT = "km/s"


def _sample_prior_1d(dist_like: Any, key: jax.Array, n: int, unit: str | None) -> Any:
    """Draw ``n`` samples from a prior entry, returned unit-stripped in ``unit``."""
    inner = getattr(dist_like, "distribution", dist_like)
    own_unit = getattr(dist_like, "unit", None)
    x = inner.sample(key, (n,))
    if own_unit is not None and unit is not None and own_unit != unit:
        x = ustrip(unit, Q(x, own_unit))
    return np.asarray(x, dtype=float)


@dataclass(frozen=True)
class RVAdapter(HarvAdapter):
    """RV implementation of :class:`~ampcal.adapters.base.Adapter`."""

    name: ClassVar[str] = "rv"
    d: ClassVar[int] = 2
    n_base_columns: ClassVar[int] = 1
    obs_unit: ClassVar[str] = OBS_UNIT
    time_unit: ClassVar[str] = TIME_UNIT
    base_linear_names: ClassVar[tuple[str, ...]] = ("v_sys",)
    truth_units: ClassVar[dict[str, str]] = {
        "period": TIME_UNIT,
        "rv_semiamp": OBS_UNIT,
        "eccentricity": "",
        "arg_peri": "rad",
        "t_peri": TIME_UNIT,
        "v_sys": OBS_UNIT,
    }

    # --- grid geometry -----------------------------------------------------

    t_span: ScalarQTime = field(default_factory=lambda: Q(5.0, "yr"))
    """Observing baseline, fixed across the grid; ``period = period_ratio * t_span``.

    Also the arms' reference period ``P0``, so a ``level`` is the prior width at
    ``P = T_span`` and the exponent tilts it away from there.
    """

    sigma_n: ScalarQSpeed = field(default_factory=lambda: Q(0.5, OBS_UNIT))
    """Per-observation RV uncertainty, fixed. See :meth:`amplitude` for what the
    ``snr`` axis means against it on each population."""

    v_sys: ScalarQSpeed = field(default_factory=lambda: Q(0.0, OBS_UNIT))
    """Systemic velocity, pinned to zero.

    The simulator would otherwise draw it at random. Offset absorption is not the
    question here, and harv's Fourier priors do no centering, so pinning it keeps the
    ``v_sys`` prior scale from becoming a hidden confound. The nuisance section of the
    report re-simulates with a large ``v_sys`` precisely because coverage cannot fail
    against zero.
    """

    period_min: ScalarQTime = field(default_factory=lambda: Q(10.0, TIME_UNIT))
    period_max: ScalarQTime = field(default_factory=lambda: Q(73050.0, TIME_UNIT))
    """``40 * t_span``, so the ``P / t_span = 10`` cell sits well inside the grid.

    Extending this is essentially free: the grid size is set by
    ``f_max = 1 / period_min``, so lowering ``f_min`` leaves the point count at ~1826
    either way. It buys headroom so long-period peaks are not pinned to the boundary
    --- which would be an artifact on top of the genuine non-identifiability there.
    """

    period_ratios: tuple[float, ...] = (0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0)
    """``P / t_span``, deliberately dense above 1.

    The previous run measured ``d_ln_peak`` identically zero for ``P/T <= 0.3`` at
    every prior scale, and -1.75 / -3.37 at ``P/T = 3`` / ``10`` --- the whole effect
    lives in the partial-arc regime, which the old grid sampled at only three rungs
    (1, 3, 10). ``0.1`` and ``0.3`` are kept as the null controls; everything else is
    spent above 1. At ``P = 10 t_span`` the injected frequency is ~10x *smaller* than
    one periodogram peak width, so the period is not localizable by any method --- that
    is information content, not a grid artifact, and the reference ``R`` is subject to
    it too, which is why ``d_ln_peak`` remains meaningful there.
    """

    snrs: tuple[float, ...] = (3.0, 10.0, 30.0)
    eccentricities: tuple[float, ...] = (0.0, 0.6)
    """Two rungs, not three. ``population`` doubles the grid and the middle
    eccentricity never separated from its neighbours in the previous run."""

    n_obs_values: tuple[int, ...] = (2, 4, 8, 16, 32)
    """Sparse-data ladder. Small ``n_obs`` is a primary use case for harv, not an edge.

    harv's overfitting cap forces ``n_terms = 1`` for ``n_obs <= 8``, so the ``H = 2``
    arms exist only from ``n_obs = 12`` up; below that the ``H = 1`` arms carry the
    comparison. The marginal statistic itself stays well defined at every rung, because
    ``S = A + Lambda^-1`` is positive-definite even where ``A`` is singular --- and at
    ``n_obs <= 3`` it peaks on prior-and-window structure rather than signal, which is
    exactly the regime where the amplitude prior's shape should matter most.
    """

    exponents: tuple[float, ...] = (0.0, -1.0 / 6.0, -1.0 / 3.0, -1.0 / 2.0)
    """``sigma_K(P) = sigma_0 (P/P0)^exponent``.

    ``0`` is today's flat prior. ``-1/3`` is
    :class:`~harv.models.priors.PeriodDependentKPrior`'s exponent, i.e. constant in
    companion mass. ``-1/6`` brackets it from below and ``-1/2`` from above, so the
    optimum can be located rather than merely confirmed or rejected.
    """

    # --- prior scales ------------------------------------------------------

    sigma_v0: Q = field(default_factory=lambda: Q(50.0, OBS_UNIT))
    """Systemic-velocity prior scale, shared by both statistics.

    Wide relative to any injected signal so the offset column is effectively
    unconstrained by the prior: harv does no centering, and ``Delta`` is only
    invariant to a constant offset in the limit that this prior absorbs it.
    """

    sigma_k_reference: Q = field(default_factory=lambda: Q(500.0, OBS_UNIT))
    """Semi-amplitude prior scale for the reference statistic ``R``.

    Deliberately very wide so ``R`` is effectively amplitude-prior-free. ``R`` is the
    fixed yardstick every arm is scored against, so it must not itself carry an opinion
    about amplitudes --- least of all a period-dependent one, which is the very thing
    under test.
    """

    # --- simulation --------------------------------------------------------

    def amplitude(self, cell: Cell, seed: int) -> Q:
        """The injected ``K``.

        ``independent``: ``snr * sigma_n``, flat in period by construction.
        ``physical``: the same, scaled by ``K(P, e, m2, i) / K(P0, 0, reference)`` from
        :mod:`ampcal.population`, so ``K ~ P^(-1/3) (1 - e^2)^(-1/2)`` on average and
        the ``snr`` axis still reads as "per-point SNR for a reference companion at
        ``P = T_span``".
        """
        base = cell.snr * self.sigma_n
        if cell.population == "independent":
            return base
        return base * population.rv_amplitude_ratio(
            self.period(cell), cell.eccentricity, seed, p0=self.t_span
        )

    def simulate(self, cell: Cell, seed: int) -> tuple[RVData, dict[str, Any]]:
        """Simulate one dataset. ``seed`` drives times, ``arg_peri``, ``t_peri`` and
        (on the ``physical`` population) the companion-mass and inclination draw."""
        return simulate_rv_sb1_data(
            seed=seed,
            n_obs=cell.n_obs,
            baseline=self.t_span,
            period=self.period(cell),
            eccentricity=cell.eccentricity,
            rv_semiamp=self.amplitude(cell, seed),
            rv_err=self.sigma_n,
            v_sys=self.v_sys,
        )

    def _obs(self, data: RVData) -> NDArray[np.float64]:
        return np.asarray(ustrip(OBS_UNIT, data.rv), dtype=float)

    def data_arrays(self, data: RVData) -> dict[str, NDArray[np.float64]]:
        return {
            "time": np.asarray(ustrip(TIME_UNIT, data.time - data.t_ref), dtype=float),
            "rv": np.asarray(ustrip(OBS_UNIT, data.rv), dtype=float),
            "rv_err": np.asarray(ustrip(OBS_UNIT, data.rv_err), dtype=float),
        }

    def data_from_arrays(self, arrays: dict[str, NDArray[np.float64]]) -> RVData:
        return RVData(
            time=Q(arrays["time"], TIME_UNIT),
            rv=Q(arrays["rv"], OBS_UNIT),
            rv_err=Q(arrays["rv_err"], OBS_UNIT),
            t_ref=Q(0.0, TIME_UNIT),
        )

    # --- harv trial model --------------------------------------------------

    def harv_model(self, n_terms: int) -> RVModel:
        return RVModel(parameterization=FourierRV(n_terms=n_terms))

    def trial_prior(
        self,
        data: RVData,
        *,
        n_terms: int,
        sigma_amp: Q | None = None,
        exponent: float = 0.0,
    ) -> HarvPrior:
        """Fourier trial prior. ``data`` is unused --- no data-driven scales."""
        del data
        base = {
            "period_min": self.period_min,
            "period_max": self.period_max,
            "sigma_v0": self.sigma_v0,
        }
        if n_terms == 0:
            return FourierRV(n_terms=0).default_prior(**base)
        if sigma_amp is None:
            raise TypeError("sigma_amp is required for n_terms > 0")
        return FourierRV(n_terms=n_terms).default_prior(
            **base, **self.amp_overrides(n_terms, sigma_amp, exponent)
        )

    # --- reference statistic R ---------------------------------------------

    def reference_prior(self) -> HarvPrior:
        """Keplerian prior for ``R``: wide amplitude, shared ``sigma_v0``."""
        import numpyro.distributions as dist
        from harv.distributions import QuantityDistribution

        return StandardRV().default_prior(
            period_min=self.period_min,
            period_max=self.period_max,
            sigma_v0=self.sigma_v0,
            rv_semiamp=QuantityDistribution(
                dist.Normal(0.0, ustrip(OBS_UNIT, self.sigma_k_reference)), OBS_UNIT
            ),
        )

    def reference_ln_z(
        self,
        data: RVData,
        periods: Q,
        *,
        n_mc: int,
        seed: int = 0,
        chunk: int = 64,
    ) -> NDArray[np.float64]:
        """The Keplerian marginal-likelihood reference ``R``, in nats.

        ``Z(P) = E_{e, omega, M0}[Z_lin(P, e, omega, M0)]`` with ``rv_semiamp`` and
        ``v_sys`` marginalized analytically by harv's own machinery, minus the base
        model's ``ln Z``. Uses **common random numbers**: one set of
        ``(e, phase_peri, arg_peri)`` draws is reused at every trial period, so the MC
        noise is shared across the curve and its *shape* --- the only thing the
        reductions read --- converges far faster than the level.
        """
        model = RVModel(parameterization=StandardRV())
        prior = self.reference_prior()
        lp = effective_linear_prior_from_prior(prior, model) or {}
        lp = {n: lp[n] for n in model._all_linear_names()}

        keys = jax.random.split(jax.random.key(seed), 3)
        ecc = _sample_prior_1d(prior.nonlinear_priors["eccentricity"], keys[0], n_mc, "")
        phase = _sample_prior_1d(
            prior.nonlinear_priors["phase_peri"], keys[1], n_mc, ""
        )
        argp = _sample_prior_1d(
            prior.nonlinear_priors["arg_peri"], keys[2], n_mc, "rad"
        )
        ecc_j, phase_j, argp_j = (jnp.asarray(a) for a in (ecc, phase, argp))

        def one(p: jax.Array, e: jax.Array, ph: jax.Array, w: jax.Array) -> jax.Array:
            nl = {
                "period": Q(p, TIME_UNIT),
                "eccentricity": e,
                "phase_peri": ph,
                "arg_peri": Q(w, "rad"),
            }
            return model.log_prob(nl, data, linear_priors=lp)

        over_draws = jax.vmap(one, in_axes=(None, 0, 0, 0))

        @jax.jit
        def ln_z_at(p: jax.Array) -> jax.Array:
            lnl = over_draws(p, ecc_j, phase_j, argp_j)
            return jax.scipy.special.logsumexp(lnl) - jnp.log(float(n_mc))

        p_vals = np.asarray(ustrip(TIME_UNIT, periods), dtype=float)
        out = np.empty(p_vals.size, dtype=float)
        for start in range(0, p_vals.size, chunk):
            stop = min(start + chunk, p_vals.size)
            out[start:stop] = np.asarray(
                jax.lax.map(ln_z_at, jnp.asarray(p_vals[start:stop]))
            )

        base = self.harv_log_prob(
            data,
            Q(p_vals[0], TIME_UNIT),
            n_terms=0,
            prior=self.trial_prior(data, n_terms=0),
        )
        return out - base
