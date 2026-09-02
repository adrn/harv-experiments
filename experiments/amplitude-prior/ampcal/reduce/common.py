"""Shared helpers for the reduction.

The important one is :func:`as_periodogram_result`: ``PeriodogramResult`` is a plain
``eqx.Module``, so the Keplerian reference ``R`` can be wrapped in one and fed through
harv's own ``tempered_period_prior``. That is what makes the arm-vs-reference
comparison apples-to-apples with no duplicated prior code.
"""

from __future__ import annotations

__all__ = (
    "as_periodogram_result",
    "common_ln_p_grid",
    "ln_period_density",
    "peak_period",
    "tv_distance",
)

import jax.numpy as jnp
import numpy as np
from harv.periodogram import PeriodogramResult
from numpy.typing import NDArray
from unxt import Q

TIME_UNIT = "day"


def as_periodogram_result(
    frequency: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    t_span: float,
    n_terms: int = 1,
    ln_likelihood_base: float = 0.0,
) -> PeriodogramResult:
    """Wrap an arbitrary nats-valued curve as a ``PeriodogramResult``.

    ``frequency`` is in ``1 / day`` and ``t_span`` in days. Used for the reference ``R``
    so it can drive harv's prior builders unchanged.
    """
    return PeriodogramResult(
        frequency=Q(jnp.asarray(np.asarray(frequency, dtype=float)), f"1/{TIME_UNIT}"),
        delta_ln_likelihood=jnp.asarray(np.asarray(values, dtype=float)),
        ln_likelihood_base=jnp.asarray(float(ln_likelihood_base)),
        t_span=Q(float(t_span), TIME_UNIT),
        t_ref=Q(0.0, TIME_UNIT),
        n_terms=n_terms,
    )


def peak_period(period: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    """Trial period at the maximum of ``values``."""
    return float(period[int(np.argmax(values))])


def ln_period_density(
    prior: object, ln_p_grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Normalized density per unit ``ln P`` on ``ln_p_grid``.

    ``prior`` is a ``QuantityDistribution`` wrapping a ``LogGridDensity`` (whatever
    ``tempered_period_prior`` returned). ``log_prob_ln`` is already the density per unit
    ln-period and is unit-invariant, but it is only normalized over the prior's own
    support, so renormalize on the evaluation grid to make the two curves comparable.
    """
    inner = prior.distribution  # type: ignore[attr-defined]
    p = np.exp(ln_p_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_rho = np.asarray(inner.log_prob_ln(jnp.asarray(p)), dtype=float)
    rho = np.where(np.isfinite(ln_rho), np.exp(ln_rho), 0.0)
    norm = np.trapezoid(rho, ln_p_grid)
    return rho / norm if norm > 0 else rho


def tv_distance(
    rho_a: NDArray[np.float64],
    rho_b: NDArray[np.float64],
    ln_p_grid: NDArray[np.float64],
) -> float:
    """Total-variation distance between two densities in ``ln P``.

    ``0`` means identical, ``1`` means disjoint support. Chosen over a distance between
    raw nats curves because the frequency-independent part of the Occam factor shifts
    ``Delta`` bodily without changing any inference, and a raw-curve metric would score
    that shift as error.
    """
    return 0.5 * float(np.trapezoid(np.abs(rho_a - rho_b), ln_p_grid))


def common_ln_p_grid(
    period: NDArray[np.float64], n: int = 2048
) -> NDArray[np.float64]:
    """A uniform ln-period grid spanning the periodogram's own period range."""
    lo, hi = np.log(period.min()), np.log(period.max())
    return np.linspace(lo, hi, n)
