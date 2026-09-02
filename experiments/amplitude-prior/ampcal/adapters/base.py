"""The adapter protocol: the one data-type-specific seam in the harness.

Everything above this layer --- ``identity``, ``run``, ``artifact``, ``reduce/*`` ---
talks only to what is declared here. If a module above needs to know whether it is
looking at radial velocities or along-scan abscissae, that knowledge belongs in an
adapter method instead.
"""

from __future__ import annotations

__all__ = ("Adapter", "MargBlocks")

from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from harv.custom_types import ScalarQTime
    from harv.data import AbstractData
    from harv.models.priors import HarvPrior
    from unxt import Q

    from ampcal.grid import Arm, Cell


class MargBlocks(NamedTuple):
    """The Gaussian-linear-model pieces harv actually uses, as plain arrays.

    Pulled straight out of ``AbstractComponentModel._build_marg_blocks`` so that
    :mod:`ampcal.linalg` decomposes harv's own matrices rather than a
    reimplementation. ``cov`` holds noise *variances*.

    ``prior_scale_tril`` is ``(k, k)`` for a period-independent prior and
    ``(n_freq, k, k)`` when the arm's amplitude prior varies with the trial period;
    :func:`ampcal.linalg.evaluate_batched` accepts either.
    """

    X: NDArray[np.float64]
    y: NDArray[np.float64]
    cov: NDArray[np.float64]
    prior_mu: NDArray[np.float64]
    prior_scale_tril: NDArray[np.float64]
    names: tuple[str, ...]


@runtime_checkable
class Adapter(Protocol):
    """What each data type must provide.

    ``d`` is the number of frequency-dependent columns contributed per harmonic:
    2 for RV (``cos``, ``sin``), 4 for Gaia astrometry (the scan-angle-projected pair).
    ``n_base_columns`` is the base model's column count: 1 for RV (the offset), 5 for
    Gaia (the astrometric solution).
    """

    name: str
    d: int
    n_base_columns: int
    obs_unit: str
    time_unit: str
    base_linear_names: tuple[str, ...]
    """Linear parameters that are *not* harmonic amplitudes, so do not carry the arm's
    period-dependent prior."""

    # --- grid geometry -----------------------------------------------------

    t_span: ScalarQTime
    period_min: ScalarQTime
    period_max: ScalarQTime

    period_ratios: tuple[float, ...]
    snrs: tuple[float, ...]
    eccentricities: tuple[float, ...]
    n_obs_values: tuple[int, ...]
    exponents: tuple[float, ...]
    """The arm's amplitude-prior exponents, signed. ``0.0`` is today's flat prior and
    one rung is the data type's physical value."""

    def period(self, cell: Cell) -> ScalarQTime:
        """The injected period for ``cell``: ``period_ratio * t_span``."""
        ...

    def amplitude(self, cell: Cell, seed: int) -> Q:
        """The injected signal amplitude. ``seed`` drives the ``physical`` mass draw."""
        ...

    def meta(self) -> dict[str, Any]:
        """Adapter-specific scalars to record in the artifact's root attributes."""
        ...

    # --- simulation --------------------------------------------------------

    def simulate(self, cell: Cell, seed: int) -> tuple[AbstractData, dict[str, Any]]:
        """Simulate one dataset; returns the data and the injected truth."""
        ...

    def data_rms(self, data: AbstractData) -> Q:
        """RMS of the observations after projecting out the base model's columns."""
        ...

    def truth_floats(self, truth: dict[str, Any]) -> dict[str, float]:
        """Flatten the simulator's truth dict to plain floats in harness units."""
        ...

    def data_arrays(self, data: AbstractData) -> dict[str, NDArray[np.float64]]:
        """Per-epoch arrays to store, so the report need not re-simulate."""
        ...

    def data_from_arrays(self, arrays: dict[str, NDArray[np.float64]]) -> AbstractData:
        """Rebuild the dataset from :meth:`data_arrays`. Inverse of the above."""
        ...

    # --- harv trial model --------------------------------------------------

    def trial_prior(
        self,
        data: AbstractData,
        *,
        n_terms: int,
        sigma_amp: Any = None,
        exponent: float = 0.0,
    ) -> HarvPrior:
        """Prior for the Kepler-free Fourier trial model at ``n_terms``.

        ``sigma_amp`` is the level at ``P = t_span``; ``exponent`` tilts it as
        ``(P / t_span)**exponent`` via :class:`~ampcal.grid.PowerLawAmpPrior`.
        """
        ...

    def arm_prior(self, data: AbstractData, arm: Arm, rms: Q) -> HarvPrior:
        """:meth:`trial_prior` for one :class:`~ampcal.grid.Arm`."""
        ...

    def harv_model(self, n_terms: int) -> Any:
        """The harv component model wrapping the Fourier parameterization."""
        ...

    def marg_blocks(
        self,
        data: AbstractData,
        period: Any,
        *,
        n_terms: int,
        prior: HarvPrior,
    ) -> MargBlocks:
        """Extract harv's design matrix, noise, and linear prior at one period."""
        ...

    def marg_blocks_batched(
        self,
        data: AbstractData,
        periods: Any,
        *,
        n_terms: int,
        prior: HarvPrior,
    ) -> tuple[Any, MargBlocks]:
        """Design matrices at every period, plus the (possibly batched) prior."""
        ...

    def harv_log_prob(
        self, data: AbstractData, period: Any, *, n_terms: int, prior: HarvPrior
    ) -> float:
        """``model.log_prob`` at one trial period --- the number harv itself reports."""
        ...

    # --- reference ---------------------------------------------------------

    def reference_ln_z(
        self, data: AbstractData, periods: Any, *, n_mc: int, seed: int
    ) -> NDArray[np.float64]:
        """The Keplerian marginal-likelihood reference, ``R``, on ``periods``."""
        ...
