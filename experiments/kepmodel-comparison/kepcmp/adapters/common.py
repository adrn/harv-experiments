"""Plumbing shared by every adapter.

Pulling harv's marginalization blocks, driving kepmodel's chi2ogram, and deciding
when the profile statistic has stopped existing are *identical* operations for RV and
for Gaia astrometry --- only the model class, the observable array and the column
counts differ. They live here so there is one implementation to get right, and so the
RV test suite doubles as a regression test for the code the Gaia adapter runs on.

What a subclass supplies: the class-level constants, :meth:`harv_model`,
:meth:`kepmodel_model`, :meth:`_obs`, :meth:`_nl`, and the data-type-specific science
(simulation, trial prior, reference statistic).
"""

from __future__ import annotations

__all__ = ("HarvKepmodelAdapter", "angular_frequency_args")

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from harv.samplers._prior_resolution import effective_linear_prior_from_prior
from numpy.typing import NDArray
from unxt import Q, ustrip

from kepcmp.adapters.base import MargBlocks

if TYPE_CHECKING:
    from harv.custom_types import ScalarQTime
    from harv.data import AbstractData
    from harv.models.priors import HarvPrior

    from kepcmp.grid import Cell


def angular_frequency_args(
    frequency: Q, time_unit: str = "day"
) -> tuple[float, float, int]:
    """Convert a harv cycles-per-time frequency grid to kepmodel ``(nu0, dnu, n)``.

    harv's grid is uniform in frequency and ascending, so the angular grid is
    uniform too. This is the single place the ``2 pi`` lives --- a factor of ``2 pi``
    here still yields a plausible-looking periodogram, so it must not be duplicated.
    """
    f = np.asarray(ustrip(f"1/{time_unit}", frequency), dtype=float)
    if f.size < 2:
        raise ValueError("need at least two grid points")
    df = np.diff(f)
    if not np.allclose(df, df[0], rtol=1e-10, atol=0.0):
        raise ValueError("frequency grid is not uniform; kepmodel requires uniform")
    return float(2.0 * np.pi * f[0]), float(2.0 * np.pi * df[0]), int(f.size)


@dataclass(frozen=True)
class HarvKepmodelAdapter:
    """Shared implementation of the :class:`~kepcmp.adapters.base.Adapter` protocol."""

    name: ClassVar[str] = "?"
    d: ClassVar[int] = 0
    """Frequency-dependent columns per harmonic (kepmodel's ``d`` at one harmonic)."""

    n_base_columns: ClassVar[int] = 0
    """Columns in the base model (kepmodel's ``p``)."""

    obs_unit: ClassVar[str] = ""
    time_unit: ClassVar[str] = "day"

    truth_units: ClassVar[dict[str, str]] = {}
    """Which truth-dict entries to store, and the unit to store each one in.

    An empty string means the value is already dimensionless.
    """

    # --- grid geometry -----------------------------------------------------

    def period(self, cell: Cell) -> ScalarQTime:
        return cell.period_ratio * self.t_span  # type: ignore[attr-defined]

    def amplitude(self, cell: Cell) -> Q:
        return cell.snr * self.sigma_n  # type: ignore[attr-defined]

    def meta(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "time_unit": self.time_unit,
            "obs_unit": self.obs_unit,
            "d": self.d,
            "n_base_columns": self.n_base_columns,
            # Kept as ``*_day`` because every adapter works in days and four
            # reductions read these names; ``time_unit`` records the unit explicitly.
            "t_span_day": float(ustrip(self.time_unit, self.t_span)),  # type: ignore[attr-defined]
            "period_min_day": float(ustrip(self.time_unit, self.period_min)),  # type: ignore[attr-defined]
            "period_max_day": float(ustrip(self.time_unit, self.period_max)),  # type: ignore[attr-defined]
            "sigma_n": float(ustrip(self.obs_unit, self.sigma_n)),  # type: ignore[attr-defined]
        }

    # --- hooks a subclass must implement -----------------------------------

    def harv_model(self, n_terms: int) -> Any:
        raise NotImplementedError

    def kepmodel_model(self, data: AbstractData) -> Any:
        raise NotImplementedError

    def _obs(self, data: AbstractData) -> NDArray[np.float64]:
        """The observable array, unit-stripped to ``obs_unit``."""
        raise NotImplementedError

    def _nl(self, period: Q) -> dict[str, Any]:
        """Nonlinear values passed to ``log_prob`` for the Fourier trial model."""
        return {"period": period}

    # --- generic --------------------------------------------------------

    def data_rms(self, data: AbstractData) -> Q:
        """RMS of the data *after* projecting out the base model's columns.

        This is the scale ``sigma_amp`` is quoted relative to, so it has to measure
        the signal the trial columns could plausibly explain --- not structure the
        base model already absorbs.

        For RV the base is a single constant column, so this is exactly the RMS about
        the mean the validated harness used. For Gaia it is emphatically not: the raw
        along-scan scatter is dominated by parallax and proper motion (~30 mas for a
        1 mas orbit at the default pinning), so quoting ``sigma_amp`` against it would
        put every multiplier ~30x above the amplitude it is supposed to bracket, by a
        factor that moves with the astrometric solution from seed to seed.
        """
        blocks = self.marg_blocks(
            data,
            self.period_min,  # type: ignore[attr-defined]
            n_terms=0,
            prior=self.trial_prior(data, n_terms=0),  # type: ignore[attr-defined]
        )
        coef, *_ = np.linalg.lstsq(blocks.X, blocks.y, rcond=None)
        resid = blocks.y - blocks.X @ coef
        return Q(float(np.sqrt(np.mean(resid**2))), self.obs_unit)

    def truth_floats(self, truth: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, unit in self.truth_units.items():
            if k not in truth:
                continue
            v = truth[k]
            out[k] = float(ustrip(unit, v)) if unit else float(v)
        return out

    # --- harv marginalization blocks ---------------------------------------

    def _linear_priors(self, model: Any, prior: HarvPrior) -> dict[str, Any]:
        lp = effective_linear_prior_from_prior(prior, model) or {}
        return {n: lp[n] for n in model._all_linear_names()}

    def marg_blocks(
        self, data: AbstractData, period: Q, *, n_terms: int, prior: HarvPrior
    ) -> MargBlocks:
        """Pull harv's own ``(X, y, cov, mu, L)`` at one trial period.

        Mirrors the call path ``log_prob`` takes in auto mode, so the matrices are
        byte-for-byte the ones harv marginalizes over.
        """
        model = self.harv_model(n_terms)
        lp = self._linear_priors(model, prior)
        marg_names = model._auto_marginalized_names(lp)
        blocks = model._build_marg_blocks(self._nl(period), marg_names, {}, data, lp)
        return MargBlocks(
            X=np.asarray(blocks.X, dtype=float),
            y=np.asarray(blocks.y, dtype=float),
            cov=np.asarray(blocks.cov, dtype=float),
            prior_mu=np.asarray(blocks.prior_mu, dtype=float),
            prior_scale_tril=np.asarray(blocks.prior_scale_tril, dtype=float),
            names=tuple(blocks.marg_names),
        )

    def marg_blocks_batched(
        self, data: AbstractData, periods: Q, *, n_terms: int, prior: HarvPrior
    ) -> tuple[jax.Array, MargBlocks]:
        """Design matrices at every period, plus the shared pieces.

        Returns ``(X_all, reference_blocks)`` with ``X_all`` of shape
        ``(n_freq, n_obs, k)``. Two things are *checked* rather than assumed, since
        both would silently corrupt the batched path:

        - the linear prior is period-independent (it is for plain Normal amplitude
          priors, but a ``LinearPriorCallable`` such as ``PeriodDependentKPrior``
          would break the assumption);
        - ``_full_design_matrix`` column order matches the marginalized column
          order harv slices to, i.e. the permutation is the identity.
        """
        model = self.harv_model(n_terms)
        lp = self._linear_priors(model, prior)
        marg_names = model._auto_marginalized_names(lp)

        p_vals = np.asarray(ustrip(self.time_unit, periods), dtype=float)

        def blocks_at(p: float) -> Any:
            return model._build_marg_blocks(
                self._nl(Q(p, self.time_unit)), marg_names, {}, data, lp
            )

        ref = blocks_at(float(p_vals[0]))
        mid = blocks_at(float(p_vals[p_vals.size // 2]))
        if not (
            np.allclose(ref.prior_mu, mid.prior_mu, rtol=0, atol=0)
            and np.allclose(ref.prior_scale_tril, mid.prior_scale_tril, rtol=0, atol=0)
        ):
            raise ValueError(
                "the linear prior varies with period, so it cannot be hoisted out of "
                "the batched path. Use the per-frequency path instead."
            )

        def full_design(p: jax.Array) -> jax.Array:
            return model._full_design_matrix(self._nl(Q(p, self.time_unit)), data)

        x_all = jax.jit(jax.vmap(full_design))(jnp.asarray(p_vals))
        # Tolerance, not exact equality: the k >= 2 harmonic columns are built by an
        # angle-addition recurrence, and the jitted/vmapped fusion rounds it
        # differently from the direct call --- measured at 1.1e-16 (half an ulp) in
        # exactly those columns, and 0 for k = 1. A genuine column permutation would
        # show O(1) differences, so this stays 12 orders of magnitude away from a
        # false pass.
        col_diff = np.abs(np.asarray(x_all[0]) - np.asarray(ref.X)).max()
        if not col_diff < 1e-12:
            raise ValueError(
                "_full_design_matrix column order does not match the marginalized "
                "column order (max column difference "
                f"{col_diff:.3e}); the batched path would pair columns with the "
                "wrong priors."
            )

        return x_all, MargBlocks(
            X=np.asarray(ref.X, dtype=float),
            y=np.asarray(ref.y, dtype=float),
            cov=np.asarray(ref.cov, dtype=float),
            prior_mu=np.asarray(ref.prior_mu, dtype=float),
            prior_scale_tril=np.asarray(ref.prior_scale_tril, dtype=float),
            names=tuple(ref.marg_names),
        )

    def harv_log_prob(
        self, data: AbstractData, period: Q, *, n_terms: int, prior: HarvPrior
    ) -> float:
        """``model.log_prob`` at one trial period --- the number harv itself reports."""
        model = self.harv_model(n_terms)
        lp = self._linear_priors(model, prior)
        return float(model.log_prob(self._nl(period), data, linear_priors=lp))

    # --- kepmodel ----------------------------------------------------------

    def kepmodel_chi2ogram(
        self, data: AbstractData, frequency: Q
    ) -> tuple[float, NDArray[np.float64]]:
        """``(chi2_H, chi2_K(nu))`` from kepmodel on harv's grid.

        Also asserts kepmodel's public ``periodogram()`` power equals
        ``1 - chi2_K/chi2_H``, i.e. that ``z0 = 0.5 * chi2_H * power`` is the right
        conversion for this version. Verified to hold for ``RvModel`` and
        ``AstroModel`` alike.
        """
        model = self.kepmodel_model(data)
        nu0, dnu, nfreq = angular_frequency_args(frequency, self.time_unit)
        chi2_h, chi2_k = model._chi2ogram(nu0, dnu, nfreq)
        _, power = model.periodogram(nu0, dnu, nfreq)
        if not np.allclose(power, 1.0 - chi2_k / chi2_h, rtol=1e-10, atol=1e-12):
            raise AssertionError(
                "kepmodel power != 1 - chi2_K/chi2_H; the z0 conversion assumed by "
                "this harness does not hold for the installed kepmodel version."
            )
        return float(chi2_h), np.asarray(chi2_k, dtype=float)

    def kepmodel_z0(
        self, data: AbstractData, frequency: Q
    ) -> tuple[float, NDArray[np.float64], bool, str]:
        """``(chi2_base, z0, degenerate, reason)`` with graceful degradation.

        The profile statistic is not merely noisy on sparse data --- it stops
        existing. kepmodel's enlarged model has ``p + d`` columns, so with no more
        observations than that it fits any trial period exactly and, because
        ``_chi2ogram`` forms ``inv(N N^T)`` explicitly, it either returns non-physical
        ``chi2`` (measured ``-1.5e14`` for RV at ``n_obs = 2``) or raises
        ``LinAlgError`` on an exactly singular matrix.

        Rather than let either outcome enter the artifact as if it were a measurement,
        this returns ``z0 = nan`` with ``degenerate=True`` and a reason string. Sparse
        data is a primary use case, so the harness has to represent "this method has
        no answer here" as a first-class result. The threshold moves with the data
        type: ``p + d`` is 3 for RV but 9 for Gaia astrometry.
        """
        n_obs = int(data.time.shape[0])
        n_cols = self.n_base_columns + self.d
        nan_curve = np.full(int(frequency.shape[0]), np.nan)

        if n_obs <= n_cols:
            return (
                float("nan"),
                nan_curve,
                True,
                f"n_obs={n_obs} <= p+d={n_cols}: profile statistic informationless",
            )
        try:
            chi2_h, chi2_k = self.kepmodel_chi2ogram(data, frequency)
        except Exception as exc:  # noqa: BLE001  (their linalg, any failure is theirs)
            return (
                float("nan"),
                nan_curve,
                True,
                f"kepmodel failed: {type(exc).__name__}: {exc}",
            )

        scale = max(abs(chi2_h), 1.0)
        if (
            not np.all(np.isfinite(chi2_k))
            or np.any(chi2_k < -1e-8 * scale)
            or np.any(chi2_k > chi2_h * (1.0 + 1e-8) + 1e-8)
        ):
            return (
                chi2_h,
                nan_curve,
                True,
                "kepmodel chi2 outside [0, chi2_base]: singular normal matrix",
            )
        return chi2_h, 0.5 * (chi2_h - chi2_k), False, ""
