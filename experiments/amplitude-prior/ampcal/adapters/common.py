"""Plumbing shared by every adapter.

Pulling harv's marginalization blocks and building the arm's prior are *identical*
operations for RV and for Gaia astrometry --- only the model class, the observable
array and the column names differ. They live here so there is one implementation to get
right, and so the RV test suite doubles as a regression test for the code the Gaia
adapter runs on.

What a subclass supplies: the class-level constants, :meth:`harv_model`, :meth:`_obs`,
:meth:`_nl`, and the data-type-specific science (simulation, trial prior, reference
statistic).
"""

from __future__ import annotations

__all__ = ("HarvAdapter",)

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from harv.samplers._prior_resolution import effective_linear_prior_from_prior
from numpy.typing import NDArray
from unxt import Q, ustrip

from ampcal.adapters.base import MargBlocks

if TYPE_CHECKING:
    from harv.custom_types import ScalarQTime
    from harv.data import AbstractData
    from harv.models.priors import HarvPrior

    from ampcal.grid import Arm, Cell


@dataclass(frozen=True)
class HarvAdapter:
    """Shared implementation of the :class:`~ampcal.adapters.base.Adapter` protocol."""

    name: ClassVar[str] = "?"
    d: ClassVar[int] = 0
    """Frequency-dependent columns per harmonic."""

    n_base_columns: ClassVar[int] = 0
    """Columns in the base (``n_terms = 0``) model."""

    obs_unit: ClassVar[str] = ""
    time_unit: ClassVar[str] = "day"

    base_linear_names: ClassVar[tuple[str, ...]] = ()
    """Linear parameters that are not harmonic amplitudes."""

    truth_units: ClassVar[dict[str, str]] = {}
    """Which truth-dict entries to store, and the unit to store each one in.

    An empty string means the value is already dimensionless.
    """

    # --- grid geometry -----------------------------------------------------

    def period(self, cell: Cell) -> ScalarQTime:
        return cell.period_ratio * self.t_span  # type: ignore[attr-defined]

    def meta(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "time_unit": self.time_unit,
            "obs_unit": self.obs_unit,
            "d": self.d,
            "n_base_columns": self.n_base_columns,
            # Kept as ``*_day`` because every adapter works in days and the reductions
            # read these names; ``time_unit`` records the unit explicitly.
            "t_span_day": float(ustrip(self.time_unit, self.t_span)),  # type: ignore[attr-defined]
            "period_min_day": float(ustrip(self.time_unit, self.period_min)),  # type: ignore[attr-defined]
            "period_max_day": float(ustrip(self.time_unit, self.period_max)),  # type: ignore[attr-defined]
            "sigma_n": float(ustrip(self.obs_unit, self.sigma_n)),  # type: ignore[attr-defined]
            "exponents": list(self.exponents),  # type: ignore[attr-defined]
        }

    # --- hooks a subclass must implement -----------------------------------

    def harv_model(self, n_terms: int) -> Any:
        raise NotImplementedError

    def _obs(self, data: AbstractData) -> NDArray[np.float64]:
        """The observable array, unit-stripped to ``obs_unit``."""
        raise NotImplementedError

    def _nl(self, period: Q) -> dict[str, Any]:
        """Nonlinear values passed to ``log_prob`` for the Fourier trial model.

        Matches :func:`harv.periodogram.core._nl` exactly --- ``eccentricity = 0`` is
        carried so eccentricity-dependent amplitude priors resolve --- so the
        marginalization path is the one ``hp.periodogram`` itself takes.
        """
        return {"period": period, "eccentricity": 0.0}

    # --- generic --------------------------------------------------------

    def data_rms(self, data: AbstractData) -> Q:
        """RMS of the data *after* projecting out the base model's columns.

        This is the scale the arm's ``level`` is quoted relative to, so it has to
        measure the signal the trial columns could plausibly explain --- not structure
        the base model already absorbs.

        For RV the base is a single constant column, so this is exactly the RMS about
        the mean. For Gaia it is emphatically not: the raw along-scan scatter is
        dominated by parallax and proper motion (~30 mas for a 1 mas orbit at the
        default pinning), so quoting the level against it would put every multiplier
        ~30x above the amplitude it is supposed to bracket, by a factor that moves with
        the astrometric solution from seed to seed.
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

    # --- the arm's prior ---------------------------------------------------

    def amp_names(self, n_terms: int) -> tuple[str, ...]:
        """Harmonic-amplitude parameter names at ``n_terms``, in harv's own order."""
        return tuple(
            p.name
            for p in self.harv_model(n_terms).parameterization.linear_params()
            if p.name not in self.base_linear_names
        )

    def amp_overrides(
        self, n_terms: int, sigma_amp: Q, exponent: float
    ) -> dict[str, Any]:
        """``{amplitude name: PowerLawAmpPrior}`` for ``default_prior``'s kwargs.

        The callable is used even at ``exponent = 0``, where it returns a constant
        ``Normal``. One code path through harv for every arm means the flat arm is a
        *measurement* of the same machinery rather than a separately-wired control.
        """
        from ampcal.grid import PowerLawAmpPrior

        prior = PowerLawAmpPrior(
            sigma_0=sigma_amp,
            p0=self.t_span,  # type: ignore[attr-defined]
            exponent=float(exponent),
        )
        return dict.fromkeys(self.amp_names(n_terms), prior)

    def arm_prior(self, data: AbstractData, arm: Arm, rms: Q) -> HarvPrior:
        """:meth:`trial_prior` for one arm: level x RMS at ``P0``, tilted by exponent."""
        return self.trial_prior(  # type: ignore[attr-defined]
            data,
            n_terms=arm.n_terms,
            sigma_amp=arm.level * rms,
            exponent=arm.exponent,
        )

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
        """Design matrices at every period, plus the prior at every period.

        Returns ``(X_all, blocks)`` with ``X_all`` of shape ``(n_freq, n_obs, k)``. The
        returned ``prior_scale_tril`` is ``(k, k)`` when the prior does not move with
        period, and ``(n_freq, k, k)`` when it does.

        **How the batched prior is built, and why it is not a python loop.** The
        decomposition needs ``Lambda(P)`` at ~1800 frequencies for each of ~24 arms.
        Calling ``_build_marg_blocks`` at each one would be ~40k python-level model
        builds per simulation. Instead the scales are read at three periods and the
        per-column exponent is *inferred*:

            s_j(P) = s_j(P0) * (P / P0)**e_j,   e_j = ln(s_j(P1)/s_j(P0)) / ln(P1/P0)

        with the third period a verification, not a fit. That is exact for every arm
        this experiment defines (:class:`~ampcal.grid.PowerLawAmpPrior` and plain
        Normals, which come out at ``e_j = 0``), and it *raises* rather than
        approximating for anything else. Two structural facts are checked here too,
        since either would silently corrupt the batched path: the prior mean must not
        move with period, and ``_full_design_matrix``'s column order must match the
        marginalized order harv slices to.

        ponytail: power-law prior scales only, verified at a third period. If a
        non-power-law arm is ever wanted, evaluate ``_build_marg_blocks`` per frequency
        and pass the stack straight through --- ``evaluate_batched`` already takes it.
        """
        model = self.harv_model(n_terms)
        lp = self._linear_priors(model, prior)
        marg_names = model._auto_marginalized_names(lp)

        p_vals = np.asarray(ustrip(self.time_unit, periods), dtype=float)

        def blocks_at(p: float) -> Any:
            return model._build_marg_blocks(
                self._nl(Q(p, self.time_unit)), marg_names, {}, data, lp
            )

        i_mid, i_end = p_vals.size // 2, p_vals.size - 1
        ref = blocks_at(float(p_vals[0]))
        mid = blocks_at(float(p_vals[i_mid]))
        chk = blocks_at(float(p_vals[i_end]))

        if not np.allclose(ref.prior_mu, mid.prior_mu, rtol=0, atol=0):
            raise ValueError(
                "the prior mean varies with period; the batched path assumes it does "
                "not. Every arm here is zero-mean, so this is a wiring bug."
            )

        s0 = np.asarray(np.diag(np.asarray(ref.prior_scale_tril, dtype=float)))
        s1 = np.asarray(np.diag(np.asarray(mid.prior_scale_tril, dtype=float)))
        s2 = np.asarray(np.diag(np.asarray(chk.prior_scale_tril, dtype=float)))
        for label, blocks in (("first", ref), ("middle", mid)):
            off = np.asarray(blocks.prior_scale_tril, dtype=float).copy()
            np.fill_diagonal(off, 0.0)
            if np.abs(off).max() > 0.0:
                raise ValueError(
                    f"the {label} prior covariance is not diagonal; harv builds "
                    "diagonal linear priors, so the per-column scaling used here "
                    "would be wrong."
                )

        varies = not np.allclose(s0, s1, rtol=0, atol=0)
        if varies:
            exps = np.log(s1 / s0) / np.log(p_vals[i_mid] / p_vals[0])
            predicted = s0 * (p_vals[i_end] / p_vals[0]) ** exps
            if not np.allclose(predicted, s2, rtol=1e-12, atol=0.0):
                raise ValueError(
                    "the amplitude prior does not vary as a power law in period, so "
                    "its scales cannot be extrapolated from two evaluations. Build "
                    "prior_scale_tril per frequency and pass the (n_freq, k, k) stack."
                )
            scale = s0[None, :] * (p_vals[:, None] / p_vals[0]) ** exps[None, :]
            tril = scale[:, :, None] * np.eye(s0.size)[None, :, :]
        else:
            tril = np.asarray(ref.prior_scale_tril, dtype=float)

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
            prior_scale_tril=tril,
            names=tuple(ref.marg_names),
        )

    def harv_log_prob(
        self, data: AbstractData, period: Q, *, n_terms: int, prior: HarvPrior
    ) -> float:
        """``model.log_prob`` at one trial period --- the number harv itself reports."""
        model = self.harv_model(n_terms)
        lp = self._linear_priors(model, prior)
        return float(model.log_prob(self._nl(period), data, linear_priors=lp))
