"""The Gaia epoch-astrometry adapter.

The along-scan case is the *stronger* test of the same algebra: the base model has
5 columns instead of 1 and the trial model contributes ``d = 4`` per harmonic instead
of 2, so the identity gate exercises far more of the block factorisation. The bases
line up column for column --- harv's ``FourierGaiaAstrometry(n_terms=1)`` frequency
block is ``[cos*cos_psi, cos*sin_psi, sin*cos_psi, sin*sin_psi]`` and kepmodel's
``astro.AstroModel._perio_phi`` is ``[cth*cos, sth*cos, cth*sin, sth*sin]``, the same
four columns in the same order.

Three things differ from the RV case in ways that matter scientifically:

- **The base columns are handed to kepmodel from harv's own design matrix** rather
  than rebuilt. There are five of them with a proper-motion time origin and a
  parallax factor, and rebuilding them by hand is exactly the kind of silent
  convention mismatch the identity gate exists to catch. Taking them from
  ``_full_design_matrix`` makes the match structural instead of hopeful.
- **The reference ``R`` is cheaper here, not more expensive.** The design document
  assumed a 5-dimensional MC over ``(e, omega, Omega, cos i, M0)``. With the
  Thiele-Innes parameterization ``A, B, F, G`` are *linear*, so only
  ``(eccentricity, phase_peri)`` need integrating --- a 2-dimensional MC, one
  dimension cheaper than RV's.
- **``P ~ 1 yr`` is a headline axis, not a filler value.** ``docs/spec.md`` claims
  ``Delta`` suppresses parallax and proper-motion power because those columns are in
  both models and cancel. :attr:`GaiaAdapter.period_ratios` puts a cell exactly at
  1 yr to test it. kepmodel's ``z0`` shares the cancellation, so this is a test of
  the claim rather than a comparison between the two codes.
"""

from __future__ import annotations

__all__ = ("GaiaAdapter",)

from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from harv.custom_types import ScalarQAngle, ScalarQTime
from harv.data import GaiaAstrometryData
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.parameterizations import (
    FourierGaiaAstrometry,
    ThieleInnesGaiaAstrometry,
)
from harv.models.priors import HarvPrior
from harv.samplers._prior_resolution import effective_linear_prior_from_prior
from harv.simulate import fake_parallax_factor, simulate_gaia_epoch_astrometry
from numpy.typing import NDArray
from unxt import Q, ustrip

from kepcmp.adapters.common import HarvKepmodelAdapter
from kepcmp.grid import Cell

TIME_UNIT = "day"
OBS_UNIT = "mas"

_LINEAR_UNITS = {
    "ra0": "mas",
    "dec0": "mas",
    "pmra": "mas/yr",
    "pmdec": "mas/yr",
    "parallax": "mas",
    "ti_A": "mas",
    "ti_B": "mas",
    "ti_F": "mas",
    "ti_G": "mas",
}


def _wide_normal(scale: Q) -> Any:
    """A zero-mean Normal wide enough to carry no opinion, in ``scale``'s unit."""
    import numpyro.distributions as dist
    from harv.distributions import QuantityDistribution

    unit = str(scale.unit)
    return QuantityDistribution(dist.Normal(0.0, float(ustrip(unit, scale))), unit)


def _sample_prior_1d(dist_like: Any, key: jax.Array, n: int, unit: str | None) -> Any:
    """Draw ``n`` samples from a prior entry, returned unit-stripped in ``unit``."""
    inner = getattr(dist_like, "distribution", dist_like)
    own_unit = getattr(dist_like, "unit", None)
    x = inner.sample(key, (n,))
    if own_unit is not None and unit is not None and own_unit != unit:
        x = ustrip(unit, Q(x, own_unit))
    return np.asarray(x, dtype=float)


@dataclass(frozen=True)
class GaiaAdapter(HarvKepmodelAdapter):
    """Gaia along-scan implementation of :class:`~kepcmp.adapters.base.Adapter`."""

    name: ClassVar[str] = "gaia"
    d: ClassVar[int] = 4
    n_base_columns: ClassVar[int] = 5
    obs_unit: ClassVar[str] = OBS_UNIT
    time_unit: ClassVar[str] = TIME_UNIT
    truth_units: ClassVar[dict[str, str]] = {
        "period": TIME_UNIT,
        "semi_major_axis": OBS_UNIT,
        "eccentricity": "",
        "t_peri": TIME_UNIT,
        "arg_peri": "rad",
        "lon_asc_node": "rad",
        "inclination": "rad",
        "parallax": OBS_UNIT,
        "mu_alpha": "mas/yr",
        "mu_delta": "mas/yr",
        "A": OBS_UNIT,
        "B": OBS_UNIT,
        "F": OBS_UNIT,
        "G": OBS_UNIT,
    }

    # --- grid geometry -----------------------------------------------------

    t_span: ScalarQTime = field(default_factory=lambda: Q(5.0, "yr"))
    """Mission baseline. Unlike RV this is *not* a free knob --- it is set by the data
    release, so ``period_ratio`` is a reparameterization of period rather than an
    independent axis."""

    sigma_n: Q = field(default_factory=lambda: Q(0.1, OBS_UNIT))
    """Per-epoch along-scan uncertainty; ``semi_major_axis = snr * sigma_n``."""

    parallax: Q = field(default_factory=lambda: Q(10.0, OBS_UNIT))
    """Pinned, not drawn.

    The simulator's default is ``Exp(10 mas)``, which would vary the strength of the
    parallax column by orders of magnitude between seeds --- and the parallax/1-yr
    degeneracy is a *headline axis* here, so its strength must be controlled rather
    than randomised. Same reasoning as pinning ``v_sys`` in the RV adapter.
    """

    ra: ScalarQAngle = field(default_factory=lambda: Q(180.0, "deg"))
    dec: ScalarQAngle = field(default_factory=lambda: Q(45.0, "deg"))
    """Sky position, fixed. Only enters through the parallax factor's phase."""

    period_min: ScalarQTime = field(default_factory=lambda: Q(20.0, TIME_UNIT))
    period_max: ScalarQTime = field(default_factory=lambda: Q(29220.0, TIME_UNIT))
    """``16 * t_span``. Long-period peaks must not pin to the grid edge."""

    period_ratios: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 1.0, 2.0)
    """``P / t_span`` at ``t_span = 5 yr``, i.e. ``P`` = 0.25, 0.5, **1**, 2, 5, 10 yr.

    ``0.2`` is exactly one year: the parallax column is a 1-yr sinusoid in the base
    model of *both* codes, so this cell tests the ``docs/spec.md`` cancellation claim
    directly, with ``0.1`` and ``0.4`` flanking it to show how wide the suppression
    is. ``2.0`` is past identifiability, the analogue of the RV grid's ``10.0``.
    """

    snrs: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0)
    eccentricities: tuple[float, ...] = (0.0, 0.3, 0.6)

    n_obs_values: tuple[int, ...] = (10, 20, 40, 80)
    """Field-of-view transit counts, not the RV ladder.

    Two structural boundaries, both moved by the wider model:

    - ``z0`` needs ``n_obs > p + d = 9``, so ``10`` is the first rung where the
      profile statistic exists at all --- with one residual degree of freedom. It is
      the direct analogue of ``n_obs = 4`` in the RV grid, which is where the marginal
      statistic was measured to recover the period and the profile statistic was not.
    - harv's overfitting cap is ``H_max = (n_obs/2 - 5) // 4``, so ``H=2`` needs
      ``n_obs >= 26`` and ``H=3`` needs ``n_obs >= 34``. Only the ``40`` and ``80``
      rungs carry the multi-harmonic arms.

    Going below 10 would be measuring nothing: both statistics are undefined and the
    comparison has no content, unlike RV where the marginal statistic still worked.
    """

    # --- prior scales ------------------------------------------------------

    sigma_pos: Q = field(default_factory=lambda: Q(1000.0, OBS_UNIT))
    sigma_pm: Q = field(default_factory=lambda: Q(1000.0, "mas/yr"))
    sigma_parallax: Q = field(default_factory=lambda: Q(1000.0, OBS_UNIT))
    """Base-column prior scales, wide enough to be effectively unconstraining.

    They are identical in the base and enlarged models, so their Occam contribution
    cancels exactly in ``Delta``; what matters is only that they do not shrink the
    astrometric solution, which would leak signal into the frequency columns.
    """

    sigma_a0_science: Q = field(default_factory=lambda: Q(5.0, "AU"))
    """Semi-major-axis scale for the *science* prior. **Physical length, not angular.**

    ``PeriodDependentSemiMajorAxisPrior`` is
    ``sigma_a0 * (P/P0)**(2/3) * parallax``, so ``sigma_a0`` carries length units and
    the parallax does the conversion to mas. Passing an angle here constructs without
    complaint and then raises ``UnitConversionError`` deep inside the sampler, which is
    why this is a separate field from :attr:`sigma_ti_reference` rather than the same
    number reused --- the two are not interchangeable and the names now say so.
    At 10 mas parallax this is ~50 mas at ``P0 = 1 yr``, far above anything injected.
    """

    sigma_vtan: Q = field(default_factory=lambda: Q(100.0, "km/s"))
    """Tangential-velocity scale for the *science* proper-motion prior.

    harv's ``ParallaxDependentProperMotionPrior`` is parameterized in velocity, not in
    ``mas/yr``, and ``ThieleInnesGaiaAstrometry.default_prior`` raises without it. Wide
    enough not to constrain: 100 km/s is well above the disk velocity dispersion, so
    the base columns stay free. :meth:`reference_prior` never needs this --- it
    overrides ``pmra``/``pmdec`` with plain Normals outright.
    """

    sigma_ti_reference: Q = field(default_factory=lambda: Q(500.0, OBS_UNIT))
    """Thiele-Innes prior scale for ``R``, deliberately very wide. **Angular** (mas).

    harv's *default* Thiele-Innes prior is period-dependent
    (``PeriodDependentSemiMajorAxisPrior``), which is the right physical choice for
    an analysis but wrong for the yardstick: ``R`` must carry no opinion about
    amplitudes, or the ``sigma_amp`` calibration measures the reference's prior
    instead of harv's. :meth:`reference_prior` therefore overrides every linear prior
    with a wide plain Normal.
    """

    # --- simulation --------------------------------------------------------

    def simulate(
        self, cell: Cell, seed: int
    ) -> tuple[GaiaAstrometryData, dict[str, Any]]:
        """Simulate one epoch-astrometry dataset.

        Times, scan angles and parallax factors are built here rather than left to the
        simulator's defaults, because the simulator's default ``parallax_factor`` is
        ``U(-1, 1)`` *white noise* --- uncorrelated with time, so the parallax column
        would carry no 1-yr structure and the degeneracy axis would test nothing.
        """
        # A separate stream from the simulator's own SeedSequence(seed), which drives
        # the orbital angles and proper motions.
        rng = np.random.default_rng(np.random.SeedSequence((seed, 1)))
        t_ref = Q(0.0, "yr")
        span = float(ustrip("yr", self.t_span))

        dt = np.sort(rng.uniform(0.0, span, cell.n_obs))
        times = Q(dt, "yr") + t_ref
        # ponytail: uniform scan angles, not the real scan law. harv.simulate.scanlaw
        # has the DR3/4/5 tables (they download on first use); swap it in here when
        # alias structure from the real scanning pattern becomes the question.
        scan_angle = Q(rng.uniform(0.0, 2.0 * np.pi, cell.n_obs), "rad")
        pf = fake_parallax_factor(times, self.ra, self.dec, scan_angle)

        return simulate_gaia_epoch_astrometry(
            seed=seed,
            times=times,
            scan_angle=scan_angle,
            parallax_factor=jnp.asarray(ustrip("", pf)),
            period=self.period(cell),
            eccentricity=cell.eccentricity,
            semi_major_axis=self.amplitude(cell),
            parallax=self.parallax,
            alpha0=Q(0.0, OBS_UNIT),
            delta0=Q(0.0, OBS_UNIT),
            al_error=self.sigma_n,
            t_ref=t_ref,
        )

    def _obs(self, data: GaiaAstrometryData) -> NDArray[np.float64]:
        return np.asarray(ustrip(OBS_UNIT, data.al_position), dtype=float)

    def data_arrays(
        self, data: GaiaAstrometryData
    ) -> dict[str, NDArray[np.float64]]:
        return {
            "time": np.asarray(ustrip(TIME_UNIT, data.time - data.t_ref), dtype=float),
            "al_position": np.asarray(
                ustrip(OBS_UNIT, data.al_position), dtype=float
            ),
            "al_position_err": np.asarray(
                ustrip(OBS_UNIT, data.al_position_err), dtype=float
            ),
            "scan_angle": np.asarray(ustrip("rad", data.scan_angle), dtype=float),
            "parallax_factor": np.asarray(data.parallax_factor, dtype=float),
        }

    def data_from_arrays(
        self, arrays: dict[str, NDArray[np.float64]]
    ) -> GaiaAstrometryData:
        return GaiaAstrometryData(
            time=Q(arrays["time"], TIME_UNIT),
            al_position=Q(arrays["al_position"], OBS_UNIT),
            al_position_err=Q(arrays["al_position_err"], OBS_UNIT),
            scan_angle=Q(arrays["scan_angle"], "rad"),
            parallax_factor=jnp.asarray(arrays["parallax_factor"]),
            t_ref=Q(0.0, TIME_UNIT),
        )

    # --- harv trial model --------------------------------------------------

    def harv_model(self, n_terms: int) -> GaiaAstrometryModel:
        return GaiaAstrometryModel(
            parameterization=FourierGaiaAstrometry(n_terms=n_terms)
        )

    def trial_prior(
        self, data: GaiaAstrometryData, *, n_terms: int, sigma_amp: Q | None = None
    ) -> HarvPrior:
        """Fourier trial prior. ``data`` is unused --- no data-driven scales."""
        del data
        base = {
            "period_min": self.period_min,
            "period_max": self.period_max,
            "sigma_pos": self.sigma_pos,
            "sigma_pm": self.sigma_pm,
            "sigma_parallax": self.sigma_parallax,
        }
        if n_terms == 0:
            return FourierGaiaAstrometry(n_terms=0).default_prior(**base)
        return FourierGaiaAstrometry(n_terms=n_terms).default_prior(
            sigma_amp=sigma_amp, **base
        )

    # --- kepmodel ----------------------------------------------------------

    def kepmodel_model(self, data: GaiaAstrometryData) -> Any:
        """An ``AstroModel`` carrying **harv's own** five base columns.

        ``AstroModel`` installs no linear columns of its own, so the 5-parameter
        astrometric solution has to be added explicitly. Taking those columns from
        ``_full_design_matrix`` rather than rebuilding them means the proper-motion
        time origin, the along-scan sign convention and the parallax factor cannot
        drift between the two codes --- a mismatch there would look like a scientific
        result, which is precisely what the identity gate is defended against.
        """
        from kepmodel.astro import AstroModel
        from spleaf.term import Error

        t = np.asarray(ustrip(TIME_UNIT, data.time - data.t_ref), dtype=float)
        s = np.asarray(ustrip(OBS_UNIT, data.al_position), dtype=float)
        err = np.asarray(ustrip(OBS_UNIT, data.al_position_err), dtype=float)
        psi = np.asarray(ustrip("rad", data.scan_angle), dtype=float)

        base_model = self.harv_model(0)
        x_base = np.asarray(
            base_model._full_design_matrix(self._nl(self.period_min), data),
            dtype=float,
        )

        model = AstroModel(t, s, np.cos(psi), np.sin(psi), err=Error(err))
        for i, cname in enumerate(base_model._all_linear_names()):
            model.add_lin(x_base[:, i], name=cname)
        return model

    # --- reference statistic R ---------------------------------------------

    def reference_prior(self) -> HarvPrior:
        """Keplerian prior for ``R``, with every linear prior forced wide and plain.

        harv's Thiele-Innes and proper-motion defaults are period- and
        parallax-dependent. Those are good analysis priors and bad yardstick priors,
        so all nine linear columns are overridden with zero-mean Normals whose scales
        are far above anything injected. The parallax prior becomes a Normal rather
        than a HalfNormal for the same reason --- a truncation is an opinion, and it
        would also break the analytic linear marginalization.
        """
        overrides: dict[str, Any] = {
            "ra0": _wide_normal(self.sigma_pos),
            "dec0": _wide_normal(self.sigma_pos),
            "pmra": _wide_normal(self.sigma_pm),
            "pmdec": _wide_normal(self.sigma_pm),
            "parallax": _wide_normal(self.sigma_parallax),
        }
        for name in ("ti_A", "ti_B", "ti_F", "ti_G"):
            overrides[name] = _wide_normal(self.sigma_ti_reference)
        return ThieleInnesGaiaAstrometry().default_prior(
            period_min=self.period_min, period_max=self.period_max, **overrides
        )

    def science_model(self) -> GaiaAstrometryModel:
        return GaiaAstrometryModel(parameterization=ThieleInnesGaiaAstrometry())

    def reference_ln_z(
        self,
        data: GaiaAstrometryData,
        periods: Q,
        *,
        n_mc: int,
        seed: int = 0,
        chunk: int = 64,
    ) -> NDArray[np.float64]:
        """The Keplerian marginal-likelihood reference ``R``, in nats.

        ``Z(P) = E_{e, M0}[Z_lin(P, e, M0)]``: the five astrometric parameters and the
        four Thiele-Innes constants are marginalized analytically by harv's own
        machinery, leaving a **2-dimensional** MC. Common random numbers again --- one
        set of ``(e, phase_peri)`` draws reused at every trial period --- so the curve
        *shape*, which is all the calibration reduction reads, converges quickly.
        """
        model = self.science_model()
        prior = self.reference_prior()
        lp = effective_linear_prior_from_prior(prior, model) or {}
        lp = {n: lp[n] for n in model._all_linear_names()}

        keys = jax.random.split(jax.random.key(seed), 2)
        ecc = _sample_prior_1d(prior.nonlinear_priors["eccentricity"], keys[0], n_mc, "")
        phase = _sample_prior_1d(
            prior.nonlinear_priors["phase_peri"], keys[1], n_mc, ""
        )
        ecc_j, phase_j = jnp.asarray(ecc), jnp.asarray(phase)

        def one(p: jax.Array, e: jax.Array, ph: jax.Array) -> jax.Array:
            nl = {
                "period": Q(p, TIME_UNIT),
                "eccentricity": e,
                "phase_peri": ph,
            }
            return model.log_prob(nl, data, linear_priors=lp)

        over_draws = jax.vmap(one, in_axes=(None, 0, 0))

        @jax.jit
        def ln_z_at(p: jax.Array) -> jax.Array:
            lnl = over_draws(p, ecc_j, phase_j)
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

    def science_prior(
        self, data: GaiaAstrometryData, period_prior: Any
    ) -> HarvPrior:
        """Thiele-Innes prior for the rejection run, with the interim period prior.

        Unlike :meth:`reference_prior` this keeps harv's *physical* defaults --- the
        period-dependent semi-major-axis prior and the parallax-dependent proper
        motion prior --- because ``prior_quality`` is asking what a real analysis
        would do with each periodogram as its interim period prior.
        """
        del data
        return ThieleInnesGaiaAstrometry().default_prior(
            period=period_prior,
            sigma_a0=self.sigma_a0_science,
            sigma_pos=self.sigma_pos,
            sigma_parallax=self.sigma_parallax,
            sigma_vtan=self.sigma_vtan,
        )
