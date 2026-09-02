"""The verdict is the one place the report makes a judgement call.

Everything else in :mod:`ampcal.report` aggregates numbers the reduction already
produced, but the verdict decides the deliverable --- whether the Fourier amplitude
prior wants an exponent --- so it has to recover a known answer, and just as
importantly, refuse when the evidence does not clear its own three clauses.

The frames here are synthetic, not simulated: the point is to test the *rule*, and a
rule that only ever sees real data cannot be shown to refuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ampcal.report.amplitude import MIN_SEEDS, best_arm, improvement_bootstrap

EXPONENTS = (0.0, -1.0 / 6.0, -1.0 / 3.0, -1.0 / 2.0)
LEVELS = (0.0316, 1.0, 5.62)


def _frame(
    *, best_exponent: float, effect: float, noise: float, n_seeds: int = 16,
    seed: int = 0,
) -> pd.DataFrame:
    """Rows whose ``abs_d_ln_peak`` is minimized at ``best_exponent``.

    ``effect`` is the depth of the minimum in nats and ``noise`` its per-seed scatter,
    so the two together set whether the bootstrap should resolve it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for exponent in EXPONENTS:
        for level in LEVELS:
            penalty = effect * abs(exponent - best_exponent) / 0.5
            for s in range(n_seeds):
                rows.append(
                    {
                        "exponent": exponent,
                        "level": level,
                        "seed": s,
                        "abs_d_ln_peak": max(
                            0.0, penalty + rng.normal(0.0, noise)
                        ),
                        "tv": 0.1 + 0.01 * penalty,
                    }
                )
    return pd.DataFrame(rows)


def test_recovers_a_planted_optimum() -> None:
    frame = _frame(best_exponent=-1.0 / 3.0, effect=1.0, noise=0.05)
    assert best_arm(frame)["exponent"] == pytest.approx(-1.0 / 3.0)


def test_tv_breaks_ties_when_every_arm_is_exact() -> None:
    """On well-sampled cells every arm recovers the reference peak and the primary
    metric is identically zero. Without the tiebreak the winner would be whichever arm
    happened to sort first, which is not a measurement."""
    frame = _frame(best_exponent=0.0, effect=0.0, noise=0.0)
    frame["tv"] = np.where(frame["exponent"] == -0.5, 0.01, 0.5)
    assert best_arm(frame)["exponent"] == pytest.approx(-0.5)


def test_a_real_effect_is_resolved() -> None:
    frame = _frame(best_exponent=-1.0 / 3.0, effect=1.0, noise=0.05)
    boot = improvement_bootstrap(frame, exponent=-1.0 / 3.0, level=1.0)
    assert boot["improvement"] > 0.0
    assert boot["excludes_zero"]
    assert boot["lo"] > 0.0


def test_noise_is_not_over_called() -> None:
    """Half the guidance is knowing when the answer is not resolvable.

    One null frame proves nothing --- it could be the lucky one --- so this measures the
    rule's false-positive rate over a dozen independent nulls. The bar is loose (at most
    one in twelve) because 12 draws cannot resolve 5% precisely; the point is to catch a
    rule that fires half the time, which a 16-84 interval demonstrably does.
    """
    fired = 0
    for seed in range(12):
        frame = _frame(best_exponent=-1.0 / 3.0, effect=0.0, noise=0.5, seed=seed)
        boot = improvement_bootstrap(
            frame, exponent=-1.0 / 3.0, level=1.0, n_boot=200, seed=seed
        )
        fired += bool(boot["excludes_zero"])
    assert fired <= 1, f"the rule fired on {fired}/12 null grids"


def test_too_few_seeds_is_refused() -> None:
    """A bootstrap over one seed has a zero-width interval, which would otherwise read
    as a resolved result rather than as no result at all."""
    frame = _frame(
        best_exponent=-1.0 / 3.0, effect=1.0, noise=0.05, n_seeds=MIN_SEEDS - 1
    )
    boot = improvement_bootstrap(frame, exponent=-1.0 / 3.0, level=1.0)
    assert not boot["excludes_zero"]
    assert np.isnan(boot["improvement"])


def test_missing_flat_arm_is_refused() -> None:
    """The improvement is defined against the flat arm at the same level; without one
    there is nothing to improve on, and returning 0 would read as a null result."""
    frame = _frame(best_exponent=-1.0 / 3.0, effect=1.0, noise=0.05)
    frame = frame[frame["exponent"] != 0.0]
    boot = improvement_bootstrap(frame, exponent=-1.0 / 3.0, level=1.0)
    assert np.isnan(boot["improvement"])
