"""The exact-identity gate.

``Delta = z0 - Occam - Shrinkage`` is algebra, not an approximation, so with
``n_terms=1`` (harv ``d=2`` = kepmodel ``d=2``), matched noise, matched base column
and matched time reference, the two codes must agree to machine precision once the
two correction terms are added back.

The gate asserts **three separate residuals**, because the headline identity alone
would be circular. If ``Occam`` and ``Shrinkage`` were obtained by subtracting the
two periodograms, the identity would hold by construction and prove nothing:

1. ``ln_z_residual`` --- :func:`kepcmp.linalg.evaluate` vs harv's own
   ``model.log_prob``, from the matrices harv handed us. Validates that we are
   reading harv's design matrix, noise and prior correctly.
2. ``chi2_residual`` --- our least-squares ``chi2`` vs kepmodel's ``_chi2ogram``.
   Validates the cross-code conventions: frequency units, time reference, noise,
   base columns.
3. ``identity_residual`` --- ``Delta + Occam + Shrinkage - z0`` with ``z0`` taken
   from **kepmodel**, not from our own matrices. Only meaningful because (1) and
   (2) are independently checked.

Two tolerances, because two different things are being compared. Residual (1) is our
code against harv's, both of which route the marginal quantities through
``M = I + B^T B >= I``; measured, those are accurate to ~3e-16 *relative, regardless of
conditioning*, so they are held to a tight bar. Residuals (2) and (3) involve
kepmodel's explicit ``inv(N N^T)`` and our pseudo-inverse, and the profile quantities
degrade linearly in ``cond``: measured relative divergence 4e-15 at ``cond = 2.4e3``,
4.7e-12 at ``7.8e6``, 4.2e-8 at ``2.4e10``. Holding those to the marginal bar would
fail on well-wired code in near-degenerate designs, so their tolerance scales as
``eps * cond``. A genuine wiring bug is O(1) --- the ``2 pi`` negative test lands far
above either bar --- so nothing is being hidden.

Tolerance is **relative**, not the fixed ``1e-8`` nats the design doc originally
specified. The identity is a cancellation between quantities of magnitude
``chi2 ~ SNR^2 * n_obs``, so the achievable absolute residual grows with ``chi2``.
Measured across four decades of SNR at fixed everything else, the residual tracks
``chi2`` with ``residual / chi2`` flat at ``4-6e-13`` --- about 2000x machine
epsilon, which is what an ``lstsq`` + ``solve`` + ``slogdet`` chain costs. A fixed
``1e-8`` therefore passes at SNR 10 and fails at SNR 100 for purely arithmetic
reasons. The gate uses ``atol + rtol * chi2_scale`` and reports the scale-free
relative residual, which is the number that actually diagnoses a wiring bug.

Conditioning. kepmodel inverts ``N N^T`` explicitly, so at trial periods where the
cos/sin columns go degenerate with the constant column its ``chi2`` loses accuracy
while ours (a least-squares pseudo-inverse) stays well defined. Those frequencies
are reported but excluded from the assertion, gated on the condition number of the
whitened design: failing the gate on *their* numerics would be a false alarm, and
silently including them would understate the agreement.
"""

from __future__ import annotations

__all__ = ("IdentityReport", "check_identity", "main")

import argparse
from dataclasses import dataclass

import numpy as np
from harv.data import AbstractData
from unxt import Q, ustrip

from kepcmp import linalg
from kepcmp.adapters import get_adapter
from kepcmp.adapters.base import Adapter
from kepcmp.grid import Cell, shared_frequency_grid


@dataclass(frozen=True)
class IdentityReport:
    """Result of the gate. ``passed`` requires all three residuals within tolerance."""

    n_frequencies: int
    n_asserted: int
    n_excluded_ill_conditioned: int
    cond_max_allowed: float

    ln_z_residual: float
    """max |our ln Z - harv log_prob| over enlarged and base models."""

    chi2_residual: float
    """max |our chi2 - kepmodel chi2| over enlarged and base models."""

    identity_residual: float
    """max |Delta + Occam + Shrinkage - z0_kepmodel|."""

    algebra_residual: float
    """max |Delta + Occam + Shrinkage - z0_ours|; pure algebra, ~1e-12."""

    worst_excluded_identity_residual: float
    max_cond: float
    chi2_scale: float
    """``max(chi2_H, max chi2_K)`` --- the magnitude the identity cancels against."""

    tolerance: float
    """``atol + rtol * chi2_scale``: the bar for the marginal-path residuals (1) and
    the internal algebra, which are conditioning-independent."""

    tolerance_cross: float
    """The bar for the cross-code profile residuals (2) and (3), which involve
    kepmodel's explicit inverse and degrade linearly in ``cond``."""

    atol: float
    rtol: float
    passed: bool

    @property
    def identity_residual_relative(self) -> float:
        """Scale-free residual. ~5e-13 is the arithmetic floor; >>1e-10 is a bug."""
        return self.identity_residual / self.chi2_scale if self.chi2_scale else 0.0

    def report(self) -> str:
        lines = [
            f"identity gate: {'PASS' if self.passed else 'FAIL'}",
            f"  frequencies            : {self.n_frequencies}",
            f"  asserted on            : {self.n_asserted}"
            f"  (excluded {self.n_excluded_ill_conditioned} with"
            f" cond > {self.cond_max_allowed:g})",
            f"  max cond(whitened X)   : {self.max_cond:.3e}",
            f"  chi2 scale             : {self.chi2_scale:.3e}",
            f"  tolerance (marginal)   : {self.tolerance:.3e}"
            f"   (cross-code, cond-scaled) {self.tolerance_cross:.3e}",
            f"  (1) ln Z vs harv       : {self.ln_z_residual:.3e}",
            f"  (2) chi2 vs kepmodel   : {self.chi2_residual:.3e}",
            f"  (3) identity residual  : {self.identity_residual:.3e}"
            f"  ({self.identity_residual_relative:.2e} relative)",
            f"      internal algebra   : {self.algebra_residual:.3e}",
        ]
        if self.n_excluded_ill_conditioned:
            lines.append(
                "      worst excluded     : "
                f"{self.worst_excluded_identity_residual:.3e}"
                "  (kepmodel's explicit inverse, not a wiring bug)"
            )
        return "\n".join(lines)


def check_identity(
    adapter: Adapter,
    data: AbstractData,
    frequency: Q,
    *,
    sigma_amp: Q,
    atol: float = 1e-8,
    rtol: float = 1e-11,
    cond_max: float = 1e8,
) -> IdentityReport:
    """Run the gate at ``n_terms=1`` against kepmodel's ``d=2`` periodogram."""
    prior_k = adapter.trial_prior(data, n_terms=1, sigma_amp=sigma_amp)
    prior_h = adapter.trial_prior(data, n_terms=0)

    periods = 1.0 / frequency
    p_vals = np.asarray(ustrip(adapter.time_unit, periods), dtype=float)

    # Base model is period-independent: evaluate once.
    blocks_h = adapter.marg_blocks(
        data, Q(p_vals[0], adapter.time_unit), n_terms=0, prior=prior_h
    )
    terms_h = linalg.evaluate(*blocks_h[:5])
    harv_h = adapter.harv_log_prob(
        data, Q(p_vals[0], adapter.time_unit), n_terms=0, prior=prior_h
    )

    chi2_h_kep, z0_kep, degenerate, reason = adapter.kepmodel_z0(data, frequency)
    if degenerate:
        raise ValueError(
            f"the identity gate needs a computable profile statistic, but {reason}. "
            "The gate is undefined where z0 does not exist; use n_obs > p + d."
        )
    chi2_k_kep = chi2_h_kep - 2.0 * z0_kep

    n = p_vals.size
    delta = np.empty(n)
    occam = np.empty(n)
    shrink = np.empty(n)
    z0_ours = np.empty(n)
    chi2_ours = np.empty(n)
    cond = np.empty(n)
    ln_z_err = np.empty(n)

    for i, p in enumerate(p_vals):
        period = Q(p, adapter.time_unit)
        blocks_k = adapter.marg_blocks(data, period, n_terms=1, prior=prior_k)
        terms_k = linalg.evaluate(*blocks_k[:5])
        dec = linalg.delta_decomposition(terms_k, terms_h)

        harv_k = adapter.harv_log_prob(data, period, n_terms=1, prior=prior_k)

        delta[i] = dec.delta
        occam[i] = dec.occam
        shrink[i] = dec.shrinkage
        z0_ours[i] = dec.z0
        chi2_ours[i] = terms_k.chi2
        cond[i] = terms_k.cond
        ln_z_err[i] = abs(terms_k.ln_z - harv_k)

    identity = delta + occam + shrink - z0_kep
    algebra = delta + occam + shrink - z0_ours
    chi2_err = np.abs(chi2_ours - chi2_k_kep)

    ok = np.isfinite(cond) & (cond <= cond_max)
    n_excluded = int((~ok).sum())

    ln_z_residual = float(max(ln_z_err[ok].max() if ok.any() else 0.0,
                              abs(terms_h.ln_z - harv_h)))
    chi2_residual = float(max(chi2_err[ok].max() if ok.any() else np.inf,
                              abs(terms_h.chi2 - chi2_h_kep)))
    identity_residual = float(np.abs(identity[ok]).max() if ok.any() else np.inf)
    algebra_residual = float(np.abs(algebra).max())
    worst_excluded = float(np.abs(identity[~ok]).max()) if n_excluded else 0.0

    # The identity cancels quantities of magnitude chi2, so that sets the bar.
    chi2_scale = float(max(chi2_h_kep, np.abs(chi2_k_kep).max(), 1.0))
    tolerance = atol + rtol * chi2_scale

    # Cross-code profile residuals carry kepmodel's error too, and kepmodel forms the
    # normal matrix explicitly (`inv(N N^T)`), which *squares* the conditioning. So the
    # bar goes as eps * cond**2, not eps * cond: at cond = 3.1e3 that predicts 2.1e-9
    # against a measured 1.1e-11, and at cond = 2.4e3 it predicts 1.3e-9 against a
    # measured 1.1e-8 -- hence the 100x margin. Our own SVD path needs only eps * cond
    # (see tests/test_linalg.py), which is why the two bars differ.
    cond_for_tol = max(
        float(np.nanmax(cond[np.isfinite(cond)])) if np.isfinite(cond).any() else 1.0,
        1.0,
    )
    cross_rtol = max(rtol, 100.0 * float(np.finfo(float).eps) * cond_for_tol**2)
    tolerance_cross = atol + cross_rtol * chi2_scale

    passed = (
        # our code vs harv, and our own algebra: conditioning-independent
        ln_z_residual < tolerance
        and algebra_residual < tolerance
        # cross-code, via kepmodel's explicit inverse: conditioning-scaled
        and identity_residual < tolerance_cross
        and chi2_residual < tolerance_cross
    )

    return IdentityReport(
        n_frequencies=n,
        n_asserted=int(ok.sum()),
        n_excluded_ill_conditioned=n_excluded,
        cond_max_allowed=cond_max,
        ln_z_residual=ln_z_residual,
        chi2_residual=chi2_residual,
        identity_residual=identity_residual,
        algebra_residual=algebra_residual,
        worst_excluded_identity_residual=worst_excluded,
        max_cond=float(np.nanmax(cond[np.isfinite(cond)])) if np.isfinite(cond).any() else np.inf,
        chi2_scale=chi2_scale,
        tolerance=tolerance,
        tolerance_cross=tolerance_cross,
        atol=atol,
        rtol=rtol,
        passed=passed,
    )


def _thin(frequency: Q, stride: int) -> Q:
    """Subsample the grid, preserving uniform spacing (kepmodel requires it)."""
    return frequency[::stride]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("rv", "gaia"), default="rv")
    parser.add_argument("--period-ratio", type=float, default=None,
                        help="default: the adapter's second period-ratio rung")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--eccentricity", type=float, default=0.0)
    parser.add_argument("--n-obs", type=int, default=40)

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
        period_ratio=(
            args.period_ratio
            if args.period_ratio is not None
            else adapter.period_ratios[1]
        ),
        snr=args.snr,
        eccentricity=args.eccentricity,
        n_obs=args.n_obs,
    )
    data, _ = adapter.simulate(cell, args.seed)
    freq = _thin(shared_frequency_grid(adapter), args.stride)

    sigma_amp = adapter.data_rms(data)
    print(f"{adapter.name}: cell {cell.cell_id} seed {args.seed}")
    print(
        f"  injected period {adapter.period(cell)}, "
        f"amplitude {adapter.amplitude(cell)}"
    )
    print(f"  sigma_amp = data RMS = {sigma_amp}")

    rep = check_identity(
        adapter, data, freq, sigma_amp=sigma_amp,
        atol=args.atol, rtol=args.rtol, cond_max=args.cond_max,
    )
    print(rep.report())
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
