"""The RV adapter.

Everything data-type specific about the RV comparison lives here: the simulator
call, the Fourier trial prior, the kepmodel model construction, and the Keplerian
reference statistic ``R``. The parts that are *not* RV-specific --- pulling harv's
marginalization blocks, driving kepmodel's chi2ogram, the ``p + d`` degeneracy gate
--- live in :mod:`kepcmp.adapters.common`.

Convention notes that the identity gate depends on (see ``../README.md``,
"Component 1"):

- kepmodel is handed ``t - t_ref``, not raw ``t``, because harv's mean longitude is
  ``2 pi (t - t_ref) / P``. Passing raw ``t`` still produces a plausible-looking
  periodogram while silently breaking the column match.
- kepmodel takes *angular* frequency; harv's grid is cycles per unit time.
  :func:`~kepcmp.adapters.common.angular_frequency_args` does that conversion in one
  place.
- The noise covariance is the diagonal reported errors on both sides. No spleaf GP
  term is ever added, because harv's periodogram cannot marginalize one.
- One ``sigma_v0`` is shared by the trial prior, the reference, and the science
  prior, so all three statistics have a genuinely common null model and their zero
  points coincide.
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

from kepcmp.adapters.common import HarvKepmodelAdapter
from kepcmp.grid import Cell

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
class RVAdapter(HarvKepmodelAdapter):
    """RV implementation of :class:`~kepcmp.adapters.base.Adapter`."""

    name: ClassVar[str] = "rv"
    d: ClassVar[int] = 2
    n_base_columns: ClassVar[int] = 1
    obs_unit: ClassVar[str] = OBS_UNIT
    time_unit: ClassVar[str] = TIME_UNIT
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
    """Observing baseline, fixed across the grid; ``period = period_ratio * t_span``."""

    sigma_n: ScalarQSpeed = field(default_factory=lambda: Q(0.5, OBS_UNIT))
    """Per-observation RV uncertainty, fixed; ``rv_semiamp = snr * sigma_n``."""

    v_sys: ScalarQSpeed = field(default_factory=lambda: Q(0.0, OBS_UNIT))
    """Systemic velocity, pinned to zero.

    The simulator would otherwise draw it at random. Offset absorption is not one of
    the four questions, and harv's Fourier priors do no centering, so pinning it keeps
    the ``v_sys`` prior scale from becoming a hidden confound.
    """

    period_min: ScalarQTime = field(default_factory=lambda: Q(10.0, TIME_UNIT))
    period_max: ScalarQTime = field(default_factory=lambda: Q(73050.0, TIME_UNIT))
    """``40 * t_span``, so the ``P / t_span = 10`` cell sits well inside the grid.

    Extending this is essentially free: the grid size is set by
    ``f_max = 1 / period_min``, so lowering ``f_min`` leaves the point count at ~1826
    either way. It buys headroom so long-period peaks are not pinned to the boundary
    --- which would be an artifact on top of the genuine non-identifiability there.
    """

    period_ratios: tuple[float, ...] = (0.02, 0.1, 0.3, 1.0, 3.0, 10.0)
    """``P / t_span``. ``10.0`` is deliberately beyond identifiability.

    At ``P = 10 t_span`` the injected frequency is ~10x *smaller* than one periodogram
    peak width (``1 / t_span``), so the period is not localizable from the data by any
    method --- that is information content, not a grid artifact. Peak-location metrics
    are meaningless in that column and only the distributional (TV) metric is
    informative; :mod:`kepcmp.reduce.regime_map` marks it accordingly.
    """

    snrs: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0)
    eccentricities: tuple[float, ...] = (0.0, 0.3, 0.6)

    n_obs_values: tuple[int, ...] = (2, 4, 8, 16, 32)
    """Sparse-data ladder. Small ``n_obs`` is a primary use case for harv, not an edge.

    Two structural facts drive how this axis must be read, both measured:

    - **The profile statistic is informationless at ``n_obs <= 3``.** kepmodel's
      enlarged model has ``p + d = 1 + 2 = 3`` columns, so with <= 3 observations it
      fits any trial period exactly: ``chi2_K`` varies by ~1e-13 across the grid and
      ``z0`` is flat. See
      :meth:`~kepcmp.adapters.common.HarvKepmodelAdapter.kepmodel_z0`.
    - **harv's overfitting cap forces ``n_terms = 1`` for ``n_obs <= 8``.** So the
      ``H=2`` head-to-head arm does not exist below ``n_obs = 12``; there the
      comparison is the ``H=1`` control (matched columns with kepmodel's ``d=2``),
      which is still the cleaner contrast anyway since it isolates the prior from the
      basis.

    The marginal statistic stays well defined at every rung, because
    ``S = A + Lambda^-1`` is positive-definite even where ``A`` is singular. At
    ``n_obs = 4`` it recovers the injected period where ``z0`` does not; at
    ``n_obs <= 3`` it is non-flat but peaks on prior-and-window structure rather than
    signal, which is a *silent* failure and the one the ``floor`` mixture exists to
    contain.
    """

    # --- prior scales ------------------------------------------------------

    sigma_v0: Q = field(default_factory=lambda: Q(50.0, OBS_UNIT))
    """Systemic-velocity prior scale, shared by all three statistics.

    Wide relative to any injected signal so the offset column is effectively
    unconstrained by the prior: harv does no centering, and ``Delta`` is only
    invariant to a constant offset in the limit that this prior absorbs it.
    """

    sigma_k_reference: Q = field(default_factory=lambda: Q(500.0, OBS_UNIT))
    """Semi-amplitude prior scale for the reference statistic ``R``.

    Deliberately very wide so ``R`` is effectively amplitude-prior-free, matching
    its row in the design doc's statistics table ("amplitude prior: no"). ``R`` is
    the fixed yardstick the ``sigma_amp`` sweep is calibrated against, so it must
    not itself carry an opinion about amplitudes.
    """

    # --- simulation --------------------------------------------------------

    def simulate(self, cell: Cell, seed: int) -> tuple[RVData, dict[str, Any]]:
        """Simulate one dataset. ``seed`` drives times, ``arg_peri`` and ``t_peri``."""
        return simulate_rv_sb1_data(
            seed=seed,
            n_obs=cell.n_obs,
            baseline=self.t_span,
            period=self.period(cell),
            eccentricity=cell.eccentricity,
            rv_semiamp=self.amplitude(cell),
            rv_err=self.sigma_n,
            v_sys=self.v_sys,
        )

    def _obs(self, data: RVData) -> NDArray[np.float64]:
        return np.asarray(ustrip(OBS_UNIT, data.rv), dtype=float)

    def _nl(self, period: Q) -> dict[str, Any]:
        # ``eccentricity`` is ignored by FourierRV but was present in the validated
        # harness; kept so the marginalization path is byte-identical to it.
        return {"period": period, "eccentricity": 0.0}

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
        self, data: RVData, *, n_terms: int, sigma_amp: Q | None = None
    ) -> HarvPrior:
        """Fourier trial prior. ``data`` is unused --- no data-driven scales."""
        del data
        if n_terms == 0:
            return FourierRV(n_terms=0).default_prior(
                period_min=self.period_min,
                period_max=self.period_max,
                sigma_v0=self.sigma_v0,
            )
        return FourierRV(n_terms=n_terms).default_prior(
            period_min=self.period_min,
            period_max=self.period_max,
            sigma_amp=sigma_amp,
            sigma_v0=self.sigma_v0,
        )

    # --- kepmodel ----------------------------------------------------------

    def kepmodel_model(self, data: RVData) -> Any:
        """An ``RvModel`` with a single constant column and diagonal errors.

        The base model matches harv's ``FourierRV(n_terms=0)``: one offset column.
        """
        from kepmodel.rv import RvModel
        from spleaf.term import Error

        t = np.asarray(ustrip(TIME_UNIT, data.time - data.t_ref), dtype=float)
        y = np.asarray(ustrip(OBS_UNIT, data.rv), dtype=float)
        err = np.asarray(ustrip(OBS_UNIT, data.rv_err), dtype=float)
        model = RvModel(t, y, err=Error(err))
        model.add_lin(np.ones(t.size), name="v_sys")
        return model

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
        ``(e, phase_peri, arg_peri)`` draws is reused at every trial period, so the
        MC noise is shared across the curve and its *shape* --- the only thing the
        calibration reduction reads --- converges far faster than the level.
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

    # --- downstream science model -----------------------------------------

    def science_model(self) -> RVModel:
        return RVModel(parameterization=StandardRV())

    def science_prior(self, data: RVData, period_prior: Any) -> HarvPrior:
        """Keplerian prior for the rejection run, with the interim period prior."""
        del data
        return StandardRV().default_prior(
            period=period_prior,
            sigma_K0=self.sigma_k_reference,
            sigma_v0=self.sigma_v0,
        )
