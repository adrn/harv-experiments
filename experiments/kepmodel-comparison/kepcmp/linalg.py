"""Marginal and profile likelihoods of a Gaussian linear model, and the exact
Occam / shrinkage decomposition that relates them.

Every quantity here is computed from ``(X, y, cov, prior_mu, prior_scale_tril)``
--- exactly the tuple harv's own ``AbstractComponentModel._build_marg_blocks``
returns. Nothing re-derives a design matrix, so the decomposition is computed from
harv's matrices rather than from a parallel implementation that could drift.

Notation follows ``../README.md``: with ``C`` the noise covariance, ``Lambda`` the
Gaussian prior covariance on the linear parameters, and ``y_tilde = y - X mu``,

    A = X^T C^-1 X          b = X^T C^-1 y_tilde        S = A + Lambda^-1
    chi2      = y_tilde^T C^-1 y_tilde - b^T A^+ b      (profile / least squares)
    ln Z      = -1/2 y_tilde^T C^-1 y_tilde + 1/2 b^T S^-1 b
                - 1/2 ln det(I + Lambda A) - 1/2 ln det(2 pi C)
    occam     = 1/2 ln det(I + Lambda A)
    shrinkage = 1/2 (b^T A^+ b - b^T S^-1 b)   >= 0

and per model ``ln Z = ln L_hat - shrinkage - occam``, so differencing two models
gives ``Delta = z0 - Occam - Shrinkage``.

Everything is evaluated in the whitened, prior-scaled basis

    B = C^-1/2 X L        (L L^T = Lambda)      y_w = C^-1/2 y_tilde
    M = I + B^T B         c = B^T y_w

which gives ``b^T S^-1 b = c^T M^-1 c`` and ``ln det(I + Lambda A) = ln det M``.
``Lambda^-1`` and ``A^-1`` are therefore never formed. That matters: at long trial
periods the cos/sin columns become degenerate with the constant column, ``A`` goes
singular, and anything built on ``A^-1`` blows up. ``M`` stays >= I, and the profile
term is obtained from a least-squares projection (a pseudo-inverse, which is the
mathematically correct limit here since ``b`` is orthogonal to ``null(A)``).
"""

from __future__ import annotations

__all__ = (
    "DeltaDecomposition",
    "LinearModelTerms",
    "delta_decomposition",
    "evaluate",
    "evaluate_batched",
)

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import jax


class LinearModelTerms(NamedTuple):
    """Marginal / profile quantities for one Gaussian linear model."""

    ln_z: float
    """Marginal log-likelihood, ``ln N(y; X mu, C + X Lambda X^T)``."""

    ln_l_hat: float
    """Profile log-likelihood, ``ln L`` at the least-squares optimum."""

    chi2: float
    """``chi2`` of the residuals after the least-squares fit."""

    occam: float
    """``1/2 ln det(I + Lambda A)`` >= 0."""

    shrinkage: float
    """``1/2 (b^T A^+ b - b^T S^-1 b)`` >= 0."""

    ln_det_2pi_c: float
    """``ln det(2 pi C)``; identical across models, cancels in every difference."""

    cond: float
    """Condition number of the whitened design matrix."""

    rank: int
    """Numerical rank of the whitened design matrix."""

    n_columns: int
    """Number of linear columns."""


def evaluate(
    X: ArrayLike,
    y: ArrayLike,
    cov: ArrayLike,
    prior_mu: ArrayLike,
    prior_scale_tril: ArrayLike,
) -> LinearModelTerms:
    """Evaluate marginal and profile quantities for one Gaussian linear model.

    Parameters
    ----------
    X
        Design matrix, shape ``(n, k)``.
    y
        Observations, shape ``(n,)``.
    cov
        Noise *variances*, shape ``(n,)``. This matches what harv's
        ``_build_marg_blocks`` returns (it starts from ``obs_err ** 2``). A 2-d
        covariance is rejected: this experiment is white-noise only by design,
        since harv's periodogram cannot marginalize a GP.
    prior_mu
        Gaussian prior mean on the linear parameters, shape ``(k,)``.
    prior_scale_tril
        Cholesky factor ``L`` of the prior covariance, shape ``(k, k)``.

    Returns
    -------
        The decomposed quantities. ``ln_z == ln_l_hat - shrinkage - occam`` holds
        identically.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    cov = np.asarray(cov, dtype=float)
    mu = np.asarray(prior_mu, dtype=float)
    chol = np.asarray(prior_scale_tril, dtype=float)

    if cov.ndim != 1:
        raise ValueError(
            "expected diagonal noise variances of shape (n,), got array of shape "
            f"{cov.shape}. This experiment is white-noise only -- see the design "
            "doc, 'Out of scope'."
        )
    _n, k = X.shape
    if mu.shape != (k,) or chol.shape != (k, k):
        raise ValueError(
            f"prior shapes inconsistent with X: X is {X.shape}, prior_mu is "
            f"{mu.shape}, prior_scale_tril is {chol.shape}"
        )

    inv_sigma = 1.0 / np.sqrt(cov)
    xw = X * inv_sigma[:, None]
    yw = (y - X @ mu) * inv_sigma

    yy = float(yw @ yw)
    ln_det_2pi_c = float(np.sum(np.log(2.0 * math.pi * cov)))

    # --- marginal: everything through M = I + B^T B, which is always >= I ---
    b_mat = xw @ chol
    m_mat = np.eye(k) + b_mat.T @ b_mat
    c_vec = b_mat.T @ yw
    _, ln_det_m = np.linalg.slogdet(m_mat)
    quad_marginal = float(c_vec @ np.linalg.solve(m_mat, c_vec))

    # --- profile: least-squares projection, stable when X is rank-deficient ---
    coef, _, rank, svals = np.linalg.lstsq(xw, yw, rcond=None)
    resid = yw - xw @ coef
    chi2 = float(resid @ resid)
    quad_profile = yy - chi2

    occam = 0.5 * float(ln_det_m)
    shrinkage = 0.5 * (quad_profile - quad_marginal)
    ln_l_hat = -0.5 * chi2 - 0.5 * ln_det_2pi_c
    ln_z = -0.5 * yy + 0.5 * quad_marginal - occam - 0.5 * ln_det_2pi_c

    smax = float(svals[0]) if svals.size else 0.0
    smin = float(svals[-1]) if svals.size else 0.0
    cond = smax / smin if smin > 0.0 else math.inf

    return LinearModelTerms(
        ln_z=ln_z,
        ln_l_hat=ln_l_hat,
        chi2=chi2,
        occam=occam,
        shrinkage=shrinkage,
        ln_det_2pi_c=ln_det_2pi_c,
        cond=cond,
        rank=int(rank),
        n_columns=int(k),
    )


class DeltaDecomposition(NamedTuple):
    """``Delta = z0 - occam - shrinkage`` for an enlarged vs base model pair."""

    delta: float
    """``ln Z_K - ln Z_H``, the marginal log-likelihood ratio (harv's statistic)."""

    z0: float
    """``1/2 (chi2_H - chi2_K)``, the profile log-likelihood ratio (kepmodel's)."""

    occam: float
    """``occam_K - occam_H``."""

    shrinkage: float
    """``shrinkage_K - shrinkage_H``. Sign is not fixed; magnitude is small when
    both models' amplitudes are well measured."""

    residual: float
    """``delta + occam + shrinkage - z0``. Zero to machine precision by algebra;
    a non-zero value means the two models were built inconsistently."""


def delta_decomposition(
    enlarged: LinearModelTerms, base: LinearModelTerms
) -> DeltaDecomposition:
    """Difference two models into the ``Delta = z0 - Occam - Shrinkage`` terms."""
    delta = enlarged.ln_z - base.ln_z
    z0 = 0.5 * (base.chi2 - enlarged.chi2)
    occam = enlarged.occam - base.occam
    shrinkage = enlarged.shrinkage - base.shrinkage
    return DeltaDecomposition(
        delta=delta,
        z0=z0,
        occam=occam,
        shrinkage=shrinkage,
        residual=delta + occam + shrinkage - z0,
    )


def to_arrays(terms: list[LinearModelTerms]) -> dict[str, NDArray[np.float64]]:
    """Stack a per-frequency list of terms into named arrays."""
    fields = LinearModelTerms._fields
    return {f: np.array([getattr(t, f) for t in terms], dtype=float) for f in fields}


# --- batched (all frequencies at once) -------------------------------------
#
# :func:`evaluate` is the reference implementation and is what the identity gate
# validates against harv and kepmodel. It is also a python-level call per
# frequency, which is far too slow for the production grid (2400 sims x 15 configs
# x ~1800 frequencies). :func:`evaluate_batched` is the vectorized equivalent;
# ``tests/test_linalg.py`` pins it to :func:`evaluate` so the fast path can never
# silently drift from the validated one.


def _lstsq_rcond(n: int, k: int) -> float:
    """Numpy's ``lstsq(rcond=None)`` singular-value cutoff, reproduced exactly."""
    return float(np.finfo(float).eps * max(n, k))


def evaluate_batched(
    X: jax.Array,
    y: jax.Array,
    cov: jax.Array,
    prior_mu: jax.Array,
    prior_scale_tril: jax.Array,
) -> dict[str, jax.Array]:
    """Vectorized :func:`evaluate` over a leading frequency axis.

    Parameters
    ----------
    X
        Design matrices, shape ``(n_freq, n, k)``.
    y, cov
        Observations and noise variances, shape ``(n,)`` --- shared across
        frequencies.
    prior_mu, prior_scale_tril
        Prior mean ``(k,)`` and Cholesky factor ``(k, k)``, shared.

    Returns
    -------
        Dict of ``(n_freq,)`` arrays: ``ln_z``, ``ln_l_hat``, ``chi2``, ``occam``,
        ``shrinkage``, ``cond``.

    The profile term uses an SVD with numpy's ``lstsq`` cutoff rather than solving
    normal equations, so it stays correct where the design goes rank-deficient (long
    trial periods, where the cos/sin columns collapse onto the constant column).
    """
    import jax.numpy as jnp

    _n_freq, n, k = X.shape
    inv_sigma = 1.0 / jnp.sqrt(cov)
    xw = X * inv_sigma[None, :, None]
    yw = (y[None, :] - jnp.einsum("fnk,k->fn", X, prior_mu)) * inv_sigma[None, :]

    yy = jnp.sum(yw * yw, axis=-1)
    ln_det_2pi_c = jnp.sum(jnp.log(2.0 * jnp.pi * cov))

    b_mat = xw @ prior_scale_tril
    m_mat = jnp.eye(k) + jnp.swapaxes(b_mat, -1, -2) @ b_mat
    c_vec = jnp.einsum("fnk,fn->fk", b_mat, yw)
    ln_det_m = jnp.linalg.slogdet(m_mat)[1]
    quad_marginal = jnp.einsum(
        "fk,fk->f", c_vec, jnp.linalg.solve(m_mat, c_vec[..., None])[..., 0]
    )

    u_mat, svals, _ = jnp.linalg.svd(xw, full_matrices=False)
    cutoff = _lstsq_rcond(n, k) * svals[:, :1]
    keep = svals > cutoff
    ut_y = jnp.einsum("fnk,fn->fk", u_mat, yw)
    quad_profile = jnp.sum(jnp.where(keep, ut_y**2, 0.0), axis=-1)
    chi2 = yy - quad_profile

    occam = 0.5 * ln_det_m
    shrinkage = 0.5 * (quad_profile - quad_marginal)
    return {
        "ln_z": -0.5 * yy + 0.5 * quad_marginal - occam - 0.5 * ln_det_2pi_c,
        "ln_l_hat": -0.5 * chi2 - 0.5 * ln_det_2pi_c,
        "chi2": chi2,
        "occam": occam,
        "shrinkage": shrinkage,
        "cond": svals[:, 0] / jnp.where(svals[:, -1] > 0, svals[:, -1], jnp.nan),
    }
