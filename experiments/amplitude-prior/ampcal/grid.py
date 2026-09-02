"""Grid cells, arms, and the period-dependent amplitude prior itself.

A :class:`Cell` is *only* a point in the abstract axis space --- population, period
ratio, SNR, eccentricity, ``n_obs``. Everything needed to turn that into a dataset
(what ``T_span`` is, what ``SNR = 1`` means in physical units, which axis values are
worth running) belongs to the adapter, because it differs between RV and Gaia.

An :class:`Arm` is one amplitude-prior configuration:

    sigma(P) = level * data_rms * (P / P0)**exponent          P0 = adapter.t_span

``exponent = 0`` is today's flat ``Normal(0, sigma_amp)``; the adapter's own list of
exponents brackets the physical value (``-1/3`` for RV's ``K``, ``+2/3`` for Gaia's
angular ``a_0``). The *level* is quoted against each simulation's base-model-residual
RMS, exactly as the old constant sweep was, so the two studies' numbers are comparable.

The frequency grid is deliberately *shared* by every cell of one adapter: identical
shapes mean harv JIT-compiles the scan once for the whole run.
"""

from __future__ import annotations

__all__ = (
    "N_SEEDS",
    "N_TERMS_VALUES",
    "SAMPLES_PER_PEAK",
    "SIGMA_AMP_MULTIPLIERS",
    "Arm",
    "Cell",
    "PowerLawAmpPrior",
    "enumerate_arms",
    "enumerate_cells",
    "seeds",
    "shared_frequency_grid",
    "with_n_obs",
)

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import equinox as eqx
import harv.periodogram as hp
import numpy as np
import numpyro.distributions as dist
from harv.custom_types import NFrequency, ScalarQTime
from harv.distributions import QuantityDistribution
from unxt import Q, ustrip

from ampcal.population import POPULATIONS

if TYPE_CHECKING:
    from ampcal.adapters.base import Adapter

# --- knobs shared by every adapter -----------------------------------------

N_SEEDS: int = 16

SIGMA_AMP_MULTIPLIERS: tuple[float, ...] = (10.0**-1.5, 1.0, 10.0**0.75)
"""Amplitude-prior *levels*, as multiples of the per-simulation base-residual RMS.

Three, not the old five: the exponent axis now carries the shape, so the level only has
to say tight / matched / wide. The two outer values are kept at their old grid points
(``10**-1.5`` and ``10**0.75``) precisely so the ``exponent = 0`` arm reproduces the
previous run's ``d_ln_peak`` table --- that is the "reproduce a known result" gate.
"""

N_TERMS_VALUES: tuple[int, ...] = (1, 2)
"""``H = 3`` is dropped: it never separated from ``H = 2`` in the previous run and it
costs a third of the arm budget the exponent axis now needs."""

SAMPLES_PER_PEAK: int = 10
"""Above harv's default of 5.

``docs/spec.md`` warns that on high-SNR data ``exp(Delta)`` can be narrower than the
grid spacing, which smears the peak to about one knot width. The grid must resolve
peaks better than the quantity being measured, and peak *location* is the headline
metric here.
"""


# --- the arm's prior -------------------------------------------------------


class PowerLawAmpPrior(eqx.Module):
    r"""``sigma(P) = sigma_0 (P / P0)^exponent`` as a ``LinearPriorCallable``.

    Reads only ``params["period"]``, which is what
    :func:`harv.periodogram.core._nl` supplies, so it can be attached to every
    amplitude column of ``FourierRV`` *and* ``FourierGaiaAstrometry``.

    That last part is the reason this exists rather than reusing harv's
    :class:`~harv.models.priors.PeriodDependentSemiMajorAxisPrior`, which cannot be
    attached to ``FourierGaiaAstrometry``'s ``ti_*_k`` columns at all: it requires
    ``params["parallax"]``, and in the periodogram ``parallax`` carries a plain Normal,
    so it is analytically marginalized and never reaches the prior --- a ``KeyError``
    inside a jit trace. Defining the arm in **angular** units instead is also the right
    modelling choice: the parallax factor in that prior only converts AU to angle, and
    a scale in a periodogram where parallax is marginalized cannot legitimately depend
    on it. A nominal parallax is folded into ``sigma_0``.

    Known limitation, deliberate and documented: the harmonic index ``k`` is not passed
    to a linear prior, so at ``n_terms > 1`` every harmonic is scaled by the
    *fundamental's* period rather than by ``P / k``.
    """

    sigma_0: Q
    p0: ScalarQTime
    exponent: float = eqx.field(static=True)

    def __call__(self, params: dict[str, Any]) -> QuantityDistribution:
        unit = str(self.sigma_0.unit)
        ratio = ustrip("", params["period"] / self.p0)
        scale = ustrip(unit, self.sigma_0) * ratio**self.exponent
        return QuantityDistribution(dist.Normal(loc=0.0, scale=scale), unit)


@dataclass(frozen=True, slots=True)
class Arm:
    """One amplitude-prior configuration: harmonics, exponent, level."""

    n_terms: int
    exponent: float
    level: float
    """Multiple of the base-model-residual RMS, at ``P = P0``."""

    @property
    def name(self) -> str:
        """Artifact group name. Stable, sortable, and parseable back to the axes."""
        return f"H{self.n_terms}_e{self.exponent:+.4f}_s{self.level:.4g}"

    @property
    def is_flat(self) -> bool:
        return self.exponent == 0.0


def enumerate_arms(
    adapter: Adapter,
    *,
    exponents: tuple[float, ...] | None = None,
    levels: tuple[float, ...] = SIGMA_AMP_MULTIPLIERS,
    n_terms_values: tuple[int, ...] = N_TERMS_VALUES,
) -> list[Arm]:
    """Every arm run per simulation."""
    exps = adapter.exponents if exponents is None else exponents
    return [
        Arm(n_terms=h, exponent=e, level=s)
        for h, e, s in itertools.product(n_terms_values, exps, levels)
    ]


# --- frequency grid --------------------------------------------------------


def shared_frequency_grid(adapter: Adapter) -> NFrequency:
    """The single frequency grid used by every cell and every statistic.

    Bounds and baseline come from the adapter, so RV and Gaia get grids matched to
    their own mission geometry while every cell within one run shares a grid.
    """
    return hp.frequency_grid(
        period_min=adapter.period_min,
        period_max=adapter.period_max,
        t_span=adapter.t_span,
        samples_per_peak=SAMPLES_PER_PEAK,
    )


# --- cells -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cell:
    """One point in the simulation grid.

    Deliberately unitless apart from ``population``. ``period_ratio`` becomes a period
    and ``snr`` becomes an amplitude only once an adapter interprets them
    (``adapter.period(cell)``, ``adapter.amplitude(cell, seed)``), which is what lets
    the RV and Gaia grids share every reduction downstream.
    """

    population: str
    period_ratio: float
    snr: float
    eccentricity: float
    n_obs: int

    def __post_init__(self) -> None:
        if self.population not in POPULATIONS:
            raise ValueError(
                f"unknown population {self.population!r}; expected one of {POPULATIONS}"
            )

    @property
    def cell_id(self) -> str:
        return (
            f"{self.population}_pr{self.period_ratio:g}_snr{self.snr:g}"
            f"_e{self.eccentricity:g}_n{self.n_obs}"
        )

    @property
    def integrated_snr(self) -> float:
        """``amplitude sqrt(n) / sigma_n`` --- what detectability actually scales with.

        Per-point ``snr`` is the grid axis, but comparing across ``n_obs`` without this
        is misleading. On the ``physical`` population this is the *nominal* value at the
        reference system; the realised one moves with the mass draw (see
        :mod:`ampcal.population`), and the artifact records the injected truth.
        """
        return float(self.snr * np.sqrt(self.n_obs))


def enumerate_cells(adapter: Adapter) -> list[Cell]:
    """The adapter's full grid, both populations."""
    return [
        Cell(population=pop, period_ratio=pr, snr=snr, eccentricity=e, n_obs=n)
        for pop, pr, snr, e, n in itertools.product(
            POPULATIONS,
            adapter.period_ratios,
            adapter.snrs,
            adapter.eccentricities,
            adapter.n_obs_values,
        )
    ]


def seeds(n: int = N_SEEDS, *, offset: int = 0) -> list[int]:
    """Deterministic seed list."""
    return [offset + i for i in range(n)]


def with_n_obs(cell: Cell, n_obs: int) -> Cell:
    """A copy of ``cell`` at a different ``n_obs``."""
    return replace(cell, n_obs=n_obs)
