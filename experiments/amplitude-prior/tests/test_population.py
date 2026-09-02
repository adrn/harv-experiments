"""The population factor has to be falsifiable, or the whole experiment is decorative.

If the ``physical`` and ``independent`` draws were accidentally identical --- a swapped
branch, a ratio that came out constant --- every "misspecification cost" in the report
would read as zero, and it would read as a *result*: "the exponent costs nothing when
the prior is wrong". So the injected amplitudes are checked against the power laws the
priors assume, before anything downstream is trusted.
"""

from __future__ import annotations

import numpy as np
import pytest
from ampcal import population
from ampcal.adapters import get_adapter
from ampcal.grid import Cell
from unxt import Q, ustrip


@pytest.mark.parametrize(
    ("adapter_name", "unit", "want"),
    [("rv", "km/s", -1.0 / 3.0), ("gaia", "mas", 2.0 / 3.0)],
)
def test_physical_population_follows_the_priors_power_law(
    adapter_name: str, unit: str, want: float
) -> None:
    adapter = get_adapter(adapter_name)
    ln_p, ln_a = [], []
    for pr in adapter.period_ratios:
        cell = Cell(population="physical", period_ratio=pr, snr=10.0,
                    eccentricity=0.0, n_obs=8)
        for seed in range(32):
            ln_p.append(np.log(float(ustrip("day", adapter.period(cell)))))
            ln_a.append(np.log(float(ustrip(unit, adapter.amplitude(cell, seed)))))
    slope = float(np.polyfit(ln_p, ln_a, 1)[0])
    assert abs(slope - want) < 0.03, f"slope {slope:.4f}, want {want:.4f}"


@pytest.mark.parametrize("adapter_name", ["rv", "gaia"])
def test_independent_population_is_flat_in_period(adapter_name: str) -> None:
    """Exactly flat, not approximately: ``amplitude = snr * sigma_n`` and nothing else."""
    adapter = get_adapter(adapter_name)
    amps = {
        float(adapter.amplitude(
            Cell(population="independent", period_ratio=pr, snr=10.0,
                 eccentricity=0.0, n_obs=8),
            seed,
        ).value)
        for pr in adapter.period_ratios
        for seed in range(4)
    }
    assert len(amps) == 1, f"the independent population varies with period: {amps}"


def test_the_two_populations_differ() -> None:
    """The contrast the grid is built on must actually exist away from ``P = P0``."""
    adapter = get_adapter("rv")
    far = Cell(population="physical", period_ratio=10.0, snr=10.0, eccentricity=0.0,
               n_obs=8)
    flat = Cell(population="independent", period_ratio=10.0, snr=10.0,
                eccentricity=0.0, n_obs=8)
    ratios = [
        float(adapter.amplitude(far, s).value) / float(adapter.amplitude(flat, s).value)
        for s in range(16)
    ]
    # P/P0 = 10 and alpha = -1/3 predicts a median near 10**(-1/3) = 0.46, blurred by
    # the mass draw. Anything near 1 means the physical branch did nothing.
    assert 0.1 < float(np.median(ratios)) < 0.9


def test_eccentricity_enters_k_and_not_a0() -> None:
    """``K`` carries ``(1 - e^2)^(-1/2)``; the astrometric ``a_0`` does not."""
    p0 = Q(5.0, "yr")
    period = Q(2.0, "yr")
    rv_ratio = population.rv_amplitude_ratio(
        period, 0.6, 3, p0=p0
    ) / population.rv_amplitude_ratio(period, 0.0, 3, p0=p0)
    assert rv_ratio == pytest.approx((1 - 0.6**2) ** -0.5)

    adapter = get_adapter("gaia")
    kwargs = {"population": "physical", "period_ratio": 0.4, "snr": 10.0, "n_obs": 8}
    a_ecc = adapter.amplitude(Cell(eccentricity=0.6, **kwargs), 3)
    a_circ = adapter.amplitude(Cell(eccentricity=0.0, **kwargs), 3)
    assert float(a_ecc.value) == pytest.approx(float(a_circ.value))


def test_draw_is_deterministic_and_seed_dependent() -> None:
    """Reproducibility, and that the seed reaches the draw at all."""
    assert population.draw_companion(7) == population.draw_companion(7)
    assert population.draw_companion(7) != population.draw_companion(8)


def test_the_mass_draw_uses_its_own_stream() -> None:
    """The companion draw must not perturb the times, angles or noise of a dataset.

    It shares a seed with the simulator, so if it drew from the same stream, adding the
    population axis would have silently re-rolled every ``independent`` dataset too ---
    and any comparison with the previous run's numbers would be meaningless.
    """
    adapter = get_adapter("rv")
    cell = Cell(population="independent", period_ratio=0.3, snr=10.0,
                eccentricity=0.0, n_obs=16)
    before, _ = adapter.simulate(cell, 0)
    population.draw_companion(0)
    after, _ = adapter.simulate(cell, 0)
    np.testing.assert_array_equal(
        np.asarray(before.time.value), np.asarray(after.time.value)
    )
    np.testing.assert_array_equal(
        np.asarray(before.rv.value), np.asarray(after.rv.value)
    )
