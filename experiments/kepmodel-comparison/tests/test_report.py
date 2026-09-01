"""The knee estimator is the one place the report makes a judgement call.

Everything else in `kepcmp.report` aggregates numbers the reductions already produced,
but the knee decides deliverable (a) -- whether the Fourier amplitude prior wants a
scalar or a functional form -- so it needs to recover a known answer and, just as
importantly, to refuse when the grid cannot resolve one.
"""

from __future__ import annotations

import numpy as np
import pytest
from kepcmp.report.amplitude import saturation_knee

#: The production sweep: five multipliers over three decades.
MULTS = np.logspace(-1.5, 1.5, 5)
GRID_STEP_DEX = 0.75


def _curve(mid_dex: float, floor: float = 0.078, top: float = 0.46) -> np.ndarray:
    """A saturating TV curve whose transition is centred on ``mid_dex``."""
    x = np.log10(MULTS)
    return floor + (top - floor) / (1.0 + np.exp(10.0 * (x - mid_dex)))


def test_returns_a_finite_knee_for_a_saturating_curve() -> None:
    """`knee` is the onset of *saturation*, which necessarily sits above the
    transition midpoint -- the estimator asks where the curve reaches its asymptote,
    not where it starts falling. So this pins finiteness and ordering, not equality.
    """
    early = saturation_knee(MULTS, _curve(-1.5))
    late = saturation_knee(MULTS, _curve(-0.75))
    assert np.isfinite(early) and np.isfinite(late)
    assert early < late, "a curve that saturates later must give a larger knee"


def test_a_one_step_shift_is_detected() -> None:
    """The decisive case: does a knee moving by a full grid step read as moving?"""
    drift = saturation_knee(MULTS, _curve(-0.75)) - saturation_knee(MULTS, _curve(-1.5))
    assert drift > GRID_STEP_DEX / 2, f"drift {drift:.3f} dex was not detected"
    # Linear interpolation of a convex curve compresses the estimate, so the drift is
    # under-stated. That biases toward "constant" -- it never invents a movement.
    assert drift <= GRID_STEP_DEX * 1.05


def test_a_sub_grid_shift_is_not_over_called() -> None:
    """Half the guidance is knowing when the answer is not resolvable."""
    drift = saturation_knee(MULTS, _curve(-1.3)) - saturation_knee(MULTS, _curve(-1.5))
    assert abs(drift) < GRID_STEP_DEX / 2


def test_unflattened_curve_returns_nan() -> None:
    """A curve still falling at the widest scale has no knee inside the grid; the
    asymptote would be an artifact of where the sweep happens to stop.
    """
    assert np.isnan(saturation_knee(MULTS, np.linspace(0.9, 0.5, 5)))


@pytest.mark.parametrize("bad", [np.array([0.4, np.nan, 0.1, 0.1, 0.1]), None])
def test_degenerate_input_returns_nan(bad: np.ndarray | None) -> None:
    values = np.array([0.4, 0.2]) if bad is None else bad
    mults = MULTS[:2] if bad is None else MULTS
    assert np.isnan(saturation_knee(mults, values))
