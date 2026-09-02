"""The correctness gate. Run it before any grid.

Three residuals, none of which can be satisfied by construction:

1. ``ln_z_residual`` --- :func:`ampcal.linalg.evaluate` against harv's own
   ``model.log_prob``, from the matrices harv handed us. Validates that we are reading
   harv's design matrix, noise and *resolved prior* correctly. With a period-dependent
   arm this is the check that ``PowerLawAmpPrior`` reaches the marginalization the same
   way through both paths.
2. ``delta_residual`` --- our ``ln Z_K - ln Z_H`` against ``hp.periodogram``'s stored
   ``delta_ln_likelihood``. Validates the whole public code path, including harv's
   per-trial-period re-resolution of a callable prior.
3. ``algebra_residual`` --- ``Delta + Occam + Shrinkage - z0``, all from our own
   matrices. Pure algebra, so a non-zero value means the two models were built
   inconsistently.

and one that pins the fast path:

4. ``batched_residual`` --- :func:`ampcal.linalg.evaluate_batched` against
   :func:`~ampcal.linalg.evaluate`, on the same arm. The production grid never calls
   the scalar path, so without this the gate would validate code the run does not
   execute. This is the residual that would catch the batched per-frequency prior
   being built wrong, which is the one genuinely new piece of machinery here.

Tolerance is **relative**, not a fixed number of nats. The identity is a cancellation
between quantities of magnitude ``chi2 ~ SNR^2 * n_obs``, so the achievable absolute
residual grows with ``chi2``: measured across four decades of SNR, ``residual / chi2``
is flat at ``4-6e-13``, about 2000x machine epsilon, which is what an ``lstsq`` +
``solve`` + ``slogdet`` chain costs. A fixed ``1e-8`` would pass at SNR 10 and fail at
SNR 100 for purely arithmetic reasons.

Conditioning. The marginal quantities route through ``M = I + B^T B >= I`` and are
accurate regardless of conditioning (measured ~3e-16 relative from ``cond = 2e3`` to
``2e10``), so they get a flat bar. The profile quantities need a pseudo-inverse of a
near-singular matrix and degrade linearly in ``cond``, so residual (3) and the profile
half of (4) get a ``cond``-scaled bar. A genuine wiring bug is O(1) and lands far above
either.
"""

from __future__ import annotations

__all__ = ("IdentityReport", "check_identity", "main")

import argparse
from dataclasses import dataclass

import harv.periodogram as hp
import jax.numpy as jnp
import numpy as np
from harv.data import AbstractData
from unxt import Q, ustrip

from ampcal import linalg
from ampcal.adapters import get_adapter
from ampcal.adapters.base import Adapter
from ampcal.grid import Arm, Cell, shared_frequency_grid


@dataclass(frozen=True)
class IdentityReport:
    """Result of the gate. ``passed`` requires every residual within tolerance."""

    arm: str
    n_frequencies: int
    n_asserted: int
    n_excluded_ill_conditioned: int
    cond_max_allowed: float

    ln_z_residual: float
    """max |our ln Z - harv log_prob| over the trial and base models."""

    delta_residual: float
    """max |our Delta - hp.periodogram's delta_ln_likelihood|."""

    algebra_residual: float
    """max |Delta + Occam + Shrinkage - z0|, all from our own matrices."""

    batched_marginal_residual: float
    """max |batched - scalar| over ``ln_z`` and ``occam``."""

    batched_profile_residual: float
    """max |batched - scalar| over ``chi2`` and ``shrinkage``."""

    max_cond: float
    chi2_scale: float
    """``max(chi2_H, max chi2_K)`` --- the magnitude the identity cancels against."""

    tolerance: float
    """``atol + rtol * chi2_scale``: the bar for conditioning-independent residuals."""

    tolerance_cond: float
    """The bar for the profile-path residuals, which degrade linearly in ``cond``."""

    atol: float
    rtol: float
    passed: bool

    @property
    def relative(self) -> float:
        """Scale-free algebra residual. ~5e-13 is the arithmetic floor; >>1e-10 is a bug."""
        return self.algebra_residual / self.chi2_scale if self.chi2_scale else 0.0

    def report(self) -> str:
        lines = [
            f"correctness gate: {'PASS' if self.passed else 'FAIL'}   arm {self.arm}",
            f"  frequencies            : {self.n_frequencies}",
            (
                f"  asserted on            : {self.n_asserted}"
                f"  (excluded {self.n_excluded_ill_conditioned} with"
                f" cond > {self.cond_max_allowed:g})"
            ),
            f"  max cond(whitened X)   : {self.max_cond:.3e}",
            f"  chi2 scale             : {self.chi2_scale:.3e}",
            (
                f"  tolerance              : {self.tolerance:.3e}"
                f"   (cond-scaled) {self.tolerance_cond:.3e}"
            ),
            f"  (1) ln Z vs harv       : {self.ln_z_residual:.3e}",
            f"  (2) Delta vs periodogram: {self.delta_residual:.3e}",
            (
                f"  (3) internal algebra   : {self.algebra_residual:.3e}"
                f"  ({self.relative:.2e} relative)"
            ),
            (
                f"  (4) batched vs scalar  : {self.batched_marginal_residual:.3e}"
                f" marginal, {self.batched_profile_residual:.3e} profile"
            ),
        ]
        return "\n".join(lines)


def check_identity(
    adapter: Adapter,
    data: AbstractData,
    frequency: Q,
    *,
    arm: Arm,
    rms: Q,
    atol: float = 1e-8,
    rtol: float = 1e-11,
    cond_max: float = 1e8,
) -> IdentityReport:
    """Run the gate for one arm. Pass a period-dependent arm --- that is the point."""
    prior_k = adapter.arm_prior(data, arm, rms)
    prior_h = adapter.trial_prior(data, n_terms=0)

    periods = 1.0 / frequency
    p_vals = np.asarray(ustrip(adapter.time_unit, periods), dtype=float)
    unit = adapter.time_unit

    # The base model carries no Fourier columns and no callable prior, so it is
    # period-independent and one evaluation suffices --- exactly as hp.periodogram
    # decides for itself.
    blocks_h = adapter.marg_blocks(data, Q(p_vals[0], unit), n_terms=0, prior=prior_h)
    terms_h = linalg.evaluate(*blocks_h[:5])
    harv_h = adapter.harv_log_prob(data, Q(p_vals[0], unit), n_terms=0, prior=prior_h)

    n = p_vals.size
    delta = np.empty(n)
    algebra = np.empty(n)
    cond = np.empty(n)
    chi2 = np.empty(n)
    ln_z_err = np.empty(n)
    scalar = []

    for i, p in enumerate(p_vals):
        period = Q(p, unit)
        blocks_k = adapter.marg_blocks(
            data, period, n_terms=arm.n_terms, prior=prior_k
        )
        terms_k = linalg.evaluate(*blocks_k[:5])
        dec = linalg.delta_decomposition(terms_k, terms_h)
        harv_k = adapter.harv_log_prob(
            data, period, n_terms=arm.n_terms, prior=prior_k
        )

        scalar.append(terms_k)
        delta[i] = dec.delta
        algebra[i] = dec.residual
        chi2[i] = terms_k.chi2
        cond[i] = terms_k.cond
        ln_z_err[i] = abs(terms_k.ln_z - harv_k)

    # (2) the public code path, with harv re-resolving the callable prior per period.
    result = hp.periodogram(data, frequency, prior=prior_k, n_terms=arm.n_terms)
    if int(result.n_terms) != arm.n_terms:
        raise ValueError(
            f"harv capped n_terms to {int(result.n_terms)} for this dataset, so the "
            f"gate's own {arm.n_terms}-harmonic model is not what the periodogram ran. "
            "Use more observations or a smaller n_terms."
        )
    delta_err = np.abs(delta - np.asarray(result.delta_ln_likelihood, dtype=float))

    # (4) the path the production grid actually executes.
    x_all, ref_blocks = adapter.marg_blocks_batched(
        data, periods, n_terms=arm.n_terms, prior=prior_k
    )
    batched = linalg.evaluate_batched(
        x_all,
        jnp.asarray(ref_blocks.y),
        jnp.asarray(ref_blocks.cov),
        jnp.asarray(ref_blocks.prior_mu),
        jnp.asarray(ref_blocks.prior_scale_tril),
    )
    want = linalg.to_arrays(scalar)

    ok = np.isfinite(cond) & (cond <= cond_max)
    n_excluded = int((~ok).sum())
    if not ok.any():
        raise ValueError(
            f"every frequency is worse-conditioned than cond_max={cond_max:g}; there "
            "is nothing to assert on. Raise --cond-max or use a denser dataset."
        )

    def worst(arr: np.ndarray) -> float:
        return float(np.abs(arr)[ok].max())

    ln_z_residual = float(max(worst(ln_z_err), abs(terms_h.ln_z - harv_h)))
    batched_marginal = max(
        worst(np.asarray(batched[f]) - want[f]) for f in ("ln_z", "occam")
    )
    batched_profile = max(
        worst(np.asarray(batched[f]) - want[f]) for f in ("chi2", "shrinkage")
    )

    chi2_scale = float(max(terms_h.chi2, np.abs(chi2).max(), 1.0))
    tolerance = atol + rtol * chi2_scale
    cond_for_tol = max(float(np.nanmax(cond[np.isfinite(cond)])), 1.0)
    # 10x eps*cond leaves ~100x margin over the measured trend (4e-15 relative at
    # cond = 2.4e3 rising to 4.2e-8 at 2.4e10).
    cond_rtol = max(rtol, 10.0 * float(np.finfo(float).eps) * cond_for_tol)
    tolerance_cond = atol + cond_rtol * chi2_scale

    passed = (
        ln_z_residual < tolerance
        and worst(delta_err) < tolerance
        and batched_marginal < tolerance
        and worst(algebra) < tolerance_cond
        and batched_profile < tolerance_cond
    )

    return IdentityReport(
        arm=arm.name,
        n_frequencies=n,
        n_asserted=int(ok.sum()),
        n_excluded_ill_conditioned=n_excluded,
        cond_max_allowed=cond_max,
        ln_z_residual=ln_z_residual,
        delta_residual=worst(delta_err),
        algebra_residual=worst(algebra),
        batched_marginal_residual=batched_marginal,
        batched_profile_residual=batched_profile,
        max_cond=cond_for_tol,
        chi2_scale=chi2_scale,
        tolerance=tolerance,
        tolerance_cond=tolerance_cond,
        atol=atol,
        rtol=rtol,
        passed=passed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("rv", "gaia"), default="rv")
    parser.add_argument("--population", default="physical",
                        choices=("physical", "independent"))
    parser.add_argument("--period-ratio", type=float, default=None,
                        help="default: the adapter's second period-ratio rung")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--eccentricity", type=float, default=0.0)
    parser.add_argument("--n-obs", type=int, default=40)

    parser.add_argument("--n-terms", type=int, default=1)
    parser.add_argument("--exponent", type=float, default=None,
                        help="default: the adapter's physical exponent (its last "
                             "rung before the widest); pass 0 for the flat control")
    parser.add_argument("--level", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8,
                        help="subsample the shared grid; the gate is O(n_freq) "
                             "python-loop iterations and does not need full density")
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--cond-max", type=float, default=1e8)
    args = parser.parse_args(argv)

    adapter = get_adapter(args.adapter)
    cell = Cell(
        population=args.population,
        period_ratio=(
            args.period_ratio
            if args.period_ratio is not None
            else adapter.period_ratios[1]
        ),
        snr=args.snr,
        eccentricity=args.eccentricity,
        n_obs=args.n_obs,
    )
    exponent = (
        args.exponent if args.exponent is not None else adapter.exponents[2]
    )
    arm = Arm(n_terms=args.n_terms, exponent=exponent, level=args.level)

    data, _ = adapter.simulate(cell, args.seed)
    freq = shared_frequency_grid(adapter)[:: args.stride]
    rms = adapter.data_rms(data)

    print(f"{adapter.name}: cell {cell.cell_id} seed {args.seed}")
    print(
        f"  injected period {adapter.period(cell)}, "
        f"amplitude {adapter.amplitude(cell, args.seed)}"
    )
    print(f"  base-residual RMS = {rms}, arm sigma_0 = {arm.level} x RMS")

    rep = check_identity(
        adapter, data, freq, arm=arm, rms=rms,
        atol=args.atol, rtol=args.rtol, cond_max=args.cond_max,
    )
    print(rep.report())
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
