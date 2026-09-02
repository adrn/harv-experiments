"""The two injected populations: ``physical`` and ``independent``.

The whole point of the ``population`` grid axis is to measure a period-dependent
amplitude prior on a population where it *can* be right, and on one where it cannot:

- **``physical``** --- draw a companion mass and an inclination, then compute the
  amplitude from ``(m1, m2, P, e, i)``. Then ``K ~ P^(-1/3)`` (RV) and
  ``a_0 ~ P^(+2/3)`` (Gaia astrometry) hold on average, which is exactly the form
  harv's ``PeriodDependentKPrior`` / ``PeriodDependentSemiMajorAxisPrior`` assume, so
  the physical prior is correctly specified.
- **``independent``** --- ``amplitude = snr * sigma_n``, a product grid in which the
  amplitude is independent of period by construction, so any period-dependent prior
  is misspecified. This measures the cost of that misspecification.

Everything here is the *forward* direction (masses -> elements). ``harv.kepler.masses``
is entirely the inverse (``binary_mass_function``, ``companion_mass_from_mass_function``,
``semi_major_axis_physical``), so only ``KeplerianBody.from_masses`` is reused --- it
owns Kepler's third law and the barycentric conversion, which is the part worth not
re-deriving here.

Normalization
-------------
The ``snr`` grid axis has to keep meaning something on both populations, so the drawn
amplitude is expressed as a *ratio* to a fixed reference system:

    amplitude = snr * sigma_n * f(P, e, m2, cos i) / f(P0, 0, M2_REF, COS_I_REF)

with ``P0 = t_span``. So ``snr`` is "the per-observation SNR this cell would have for
a :data:`M2_REF` companion at :data:`COS_I_REF` and ``P = t_span``", and the ratio
carries the whole ``P^(-1/3)`` (or ``P^(+2/3)``) trend plus the scatter the mass
function and the inclination draw put around it. ``sigma_n`` never moves, so the two
populations are directly comparable at the same ``snr`` rung and only differ in whether
the amplitude tracks the period.
"""

from __future__ import annotations

__all__ = (
    "COS_I_REF",
    "M1",
    "M2_LOG_RANGE",
    "M2_REF",
    "POPULATIONS",
    "angular_semi_major_axis",
    "draw_companion",
    "gaia_amplitude_ratio",
    "rv_amplitude_ratio",
    "rv_semiamp",
)

import numpy as np
from harv.kepler.body import KeplerianBody
from unxt import Q, ustrip

POPULATIONS: tuple[str, ...] = ("physical", "independent")

M1: Q = Q(1.0, "Msun")
"""Primary mass, fixed. Only ratios matter, so one value is enough."""

M2_LOG_RANGE: tuple[float, float] = (np.log(0.05), np.log(1.0))
"""Companion mass drawn **log-uniform** in ``[0.05, 1.0] Msun``.

A log-flat companion-mass function is the conventional uninformative choice in
binary-star period searches (it is what The Joker's defaults are calibrated against),
and it is the distribution ``PeriodDependentKPrior``'s "approximately constant in
companion mass" argument implicitly assumes. The range is a factor of 20, which puts
about +-0.65 dex of scatter around the ``P^(-1/3)`` trend --- enough that the physical
prior is right *on average* and wrong on any individual system, which is the honest
version of "correctly specified".
"""

M2_REF: Q = Q(0.5, "Msun")
COS_I_REF: float = 0.5
"""The reference system the ``snr`` axis is defined against. See the module docstring."""


def draw_companion(seed: int) -> tuple[Q, float]:
    """One ``(m2, cos i)`` draw.

    Its own stream, separate from the simulators' ``SeedSequence(seed)``, so adding
    this axis does not shift the times, angles or noise of any existing cell.
    """
    rng = np.random.default_rng(np.random.SeedSequence((seed, 2)))
    m2 = float(np.exp(rng.uniform(*M2_LOG_RANGE)))
    return Q(m2, "Msun"), float(rng.uniform(0.0, 1.0))


def _a1(period: Q, m2: Q, m1: Q = M1) -> Q:
    """Barycentric semi-major axis **of the primary**, from Kepler's third law.

    ``KeplerianBody.from_masses`` returns ``a_body = a_rel * (1 - m_body / m_total)``,
    so passing ``m_body = m1`` gives ``a_1 = a_rel * m2 / m_total`` --- the primary's
    orbit about the barycenter, which is what both observables measure. Eccentricity
    does not enter ``a_1``; it enters ``K`` through the ``(1 - e^2)^(-1/2)`` factor
    below, and the astrometric ``a_0`` not at all.
    """
    return KeplerianBody.from_masses(
        period=period,
        eccentricity=0.0,
        m_total=m1 + m2,
        m_body=m1,
        t_peri=Q(0.0, "day"),
    ).semi_major_axis


def rv_semiamp(period: Q, eccentricity: float, m2: Q, cos_i: float, m1: Q = M1) -> Q:
    r"""``K = 2 pi a_1 sin i / (P sqrt(1 - e^2))``.

    Combined with ``a_1 ~ P^(2/3)`` this gives ``K ~ P^(-1/3) (1 - e^2)^(-1/2)``, which
    is exactly :class:`~harv.models.priors.PeriodDependentKPrior`'s functional form.
    """
    sin_i = float(np.sqrt(1.0 - cos_i**2))
    return (
        2.0 * np.pi * _a1(period, m2, m1) * sin_i
        / (period * float(np.sqrt(1.0 - eccentricity**2)))
    )


def angular_semi_major_axis(period: Q, m2: Q, parallax: Q, m1: Q = M1) -> Q:
    """Angular ``a_0`` of the primary's orbit, in the parallax's own angular unit.

    The inverse of :func:`harv.kepler.masses.semi_major_axis_physical`: by the
    definition of parallax, ``a_angular = a_physical [AU] * parallax``. Combined with
    ``a_1 ~ P^(2/3)`` this is
    :class:`~harv.models.priors.PeriodDependentSemiMajorAxisPrior`'s form.

    Inclination is deliberately absent: for astrometry it enters through the
    Thiele-Innes projection, which the simulator applies itself, not through ``a_0``.
    """
    return float(ustrip("AU", _a1(period, m2, m1))) * parallax


def rv_amplitude_ratio(
    period: Q, eccentricity: float, seed: int, *, p0: Q, m1: Q = M1
) -> float:
    """``K(P, e, draw) / K(P0, 0, reference)`` --- the factor ``snr * sigma_n`` scales by."""
    m2, cos_i = draw_companion(seed)
    got = rv_semiamp(period, eccentricity, m2, cos_i, m1)
    ref = rv_semiamp(p0, 0.0, M2_REF, COS_I_REF, m1)
    return float(ustrip("", got / ref))


def gaia_amplitude_ratio(period: Q, seed: int, *, p0: Q, m1: Q = M1) -> float:
    """``a_0(P, draw) / a_0(P0, reference)``. Parallax cancels, so it is not an argument."""
    m2, _ = draw_companion(seed)
    plx = Q(1.0, "mas")  # cancels; any value gives the same ratio
    got = angular_semi_major_axis(period, m2, plx, m1)
    ref = angular_semi_major_axis(p0, M2_REF, plx, m1)
    return float(ustrip("", got / ref))


def demo() -> None:
    """Self-check: the injected amplitudes carry the power laws the priors assume."""
    p0 = Q(5.0, "yr")
    periods = Q(np.geomspace(0.05, 10.0, 24), "yr")

    for label, fn, want in (
        ("K", lambda p, s: rv_amplitude_ratio(p, 0.0, s, p0=p0), -1.0 / 3.0),
        ("a_0", lambda p, s: gaia_amplitude_ratio(p, s, p0=p0), 2.0 / 3.0),
    ):
        # Averaged over 64 draws, the mass-function scatter integrates out and the
        # remaining trend must be the pure power law.
        ln_a = np.array(
            [
                np.mean([np.log(fn(Q(float(p), "yr"), s)) for s in range(64)])
                for p in periods.value
            ]
        )
        slope = np.polyfit(np.log(np.asarray(periods.value)), ln_a, 1)[0]
        assert abs(slope - want) < 1e-6, f"{label} slope {slope:.6f}, want {want:.6f}"

    # Eccentricity enters K only, and only as (1 - e^2)^(-1/2).
    ratio = rv_amplitude_ratio(p0, 0.6, 3, p0=p0) / rv_amplitude_ratio(p0, 0.0, 3, p0=p0)
    assert abs(ratio - (1 - 0.6**2) ** -0.5) < 1e-12
    print("population: OK")


if __name__ == "__main__":
    demo()
