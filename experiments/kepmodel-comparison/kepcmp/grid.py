"""Grid cell definition and enumeration.

A :class:`Cell` is *only* a point in the abstract axis space --- period ratio, SNR,
eccentricity, ``n_obs``. Everything needed to turn that into a dataset (what
``T_span`` is, what ``SNR = 1`` means in physical units, which axis values are worth
running) belongs to the adapter, because it differs between RV and Gaia. See
``../README.md``, "Component 2: simulation grid".

The frequency grid is deliberately *shared* by every cell of one adapter: identical
shapes mean harv JIT-compiles the scan once for the whole run.
"""

from __future__ import annotations

__all__ = (
    "N_SEEDS",
    "N_TERMS_VALUES",
    "SAMPLES_PER_PEAK",
    "SIGMA_AMP_MULTIPLIERS",
    "Cell",
    "enumerate_cells",
    "enumerate_null_cells",
    "seeds",
    "shared_frequency_grid",
    "with_n_obs",
)

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import harv.periodogram as hp
import numpy as np
from harv.custom_types import NFrequency

if TYPE_CHECKING:
    from kepcmp.adapters.base import Adapter

# --- knobs shared by every adapter -----------------------------------------

N_SEEDS: int = 16

SIGMA_AMP_MULTIPLIERS: tuple[float, ...] = tuple(
    float(x) for x in np.logspace(-1.5, 1.5, 5)
)
"""Amplitude-prior scales as multiples of the per-simulation data RMS.

Five values spanning three decades, per the design doc. The absolute value used is
recorded in the artifact, since the RMS differs per simulation.
"""

N_TERMS_VALUES: tuple[int, ...] = (1, 2, 3)

SAMPLES_PER_PEAK: int = 10
"""Above harv's default of 5.

``docs/spec.md`` warns that on high-SNR data ``exp(Delta)`` can be narrower than the
grid spacing, which smears the peak to about one knot width. The grid must resolve
peaks better than the quantity being measured, and peak *location* is a headline
metric here.
"""


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

    Deliberately unitless. ``period_ratio`` becomes a period and ``snr`` becomes an
    amplitude only once an adapter interprets them (``adapter.period(cell)``,
    ``adapter.amplitude(cell)``), which is what lets the RV and Gaia grids share every
    reduction downstream.
    """

    period_ratio: float
    snr: float
    eccentricity: float
    n_obs: int

    @property
    def cell_id(self) -> str:
        return (
            f"pr{self.period_ratio:g}_snr{self.snr:g}"
            f"_e{self.eccentricity:g}_n{self.n_obs}"
        )

    @property
    def integrated_snr(self) -> float:
        """``amplitude sqrt(n) / sigma_n`` --- what detectability actually scales with.

        Per-point ``snr`` is the grid axis, but comparing across ``n_obs`` without
        this is misleading.
        """
        return float(self.snr * np.sqrt(self.n_obs))

    @property
    def is_null(self) -> bool:
        return self.snr == 0.0


def enumerate_cells(adapter: Adapter) -> list[Cell]:
    """The adapter's full signal grid."""
    return [
        Cell(period_ratio=pr, snr=snr, eccentricity=e, n_obs=n)
        for pr, snr, e, n in itertools.product(
            adapter.period_ratios,
            adapter.snrs,
            adapter.eccentricities,
            adapter.n_obs_values,
        )
    ]


def enumerate_null_cells(adapter: Adapter) -> list[Cell]:
    """No-signal cells for the ROC: zero amplitude, one per ``n_obs``.

    ``period_ratio`` and ``eccentricity`` are irrelevant at zero amplitude but must be
    valid simulator inputs, so they are pinned rather than swept.
    """
    return [
        Cell(period_ratio=1.0, snr=0.0, eccentricity=0.0, n_obs=n)
        for n in adapter.n_obs_values
    ]


def seeds(n: int = N_SEEDS, *, offset: int = 0) -> list[int]:
    """Deterministic seed list."""
    return [offset + i for i in range(n)]


def with_n_obs(cell: Cell, n_obs: int) -> Cell:
    """A copy of ``cell`` at a different ``n_obs``."""
    return replace(cell, n_obs=n_obs)
