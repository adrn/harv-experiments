"""Guards on the two things that made an earlier regime map misreport itself.

Neither failure was visible in the output: the tie bug inflated one arm's win count
by more than half the grid, and the identifiability constant silently flagged nothing
on the Gaia grid. Both produced numbers that looked entirely reasonable.
"""

from __future__ import annotations

import numpy as np
import pytest
from kepcmp.reduce.regime_map import (
    ARMS,
    MIN_PEAK_WIDTHS_FROM_ZERO,
    _verdict,
)


def _rows(values: dict[str, float], cell_id: str = "c0") -> list[dict]:
    return [
        {"cell_id": cell_id, "arm": arm, "available": True, "tv": v}
        for arm, v in values.items()
    ]


def test_exact_tie_is_reported_as_a_tie_not_a_win() -> None:
    """The failure mode: harv's overfitting cap makes delta_default bit-identical to
    delta_h1, and a `min()` over an ordered list awards every one of those to
    whichever arm it visits first.
    """
    rows = _rows({"delta_default": 0.5, "delta_h1": 0.5, "z0": 0.9})
    v = _verdict(rows, "tv", lower_is_better=True)

    assert v["n_decisive"] == 0, "a three-way scored cell with a tie is not decisive"
    assert v["tally"] == {"tie(delta_default+delta_h1)": 1}
    assert "delta_default" not in v["tally"]


def test_decisive_winner_still_wins() -> None:
    rows = _rows({"delta_default": 0.5, "delta_h1": 0.6, "z0": 0.9})
    v = _verdict(rows, "tv", lower_is_better=True)
    assert v["tally"] == {"delta_default": 1}
    assert v["n_decisive"] == 1


def test_higher_is_better_flips_the_comparison() -> None:
    rows = _rows({"delta_default": 0.5, "delta_h1": 0.6, "z0": 0.9})
    assert _verdict(rows, "tv", lower_is_better=False)["tally"] == {"z0": 1}


def test_unavailable_and_nonfinite_arms_are_not_scored() -> None:
    """`z0` is absent by construction at ``n_obs <= p + d``; scoring a nan there
    would silently hand it a win or a loss it never earned.
    """
    rows = _rows({"delta_default": 0.5, "delta_h1": 0.6})
    rows.append({"cell_id": "c0", "arm": "z0", "available": False, "tv": np.nan})
    v = _verdict(rows, "tv", lower_is_better=True)
    assert v["tally"] == {"delta_default": 1}
    assert v["n_cells"] == 1


@pytest.mark.parametrize(
    ("period_ratio", "expected"),
    [
        (0.02, True),
        (1.0, True),
        (2.0, True),   # the Gaia grid's widest cell: 0.5 peak widths out
        (3.0, True),   # exactly at the criterion
        (10.0, False), # the RV grid's non-identifiable column
    ],
)
def test_identifiability_is_a_peak_width_criterion(
    period_ratio: float, expected: bool
) -> None:
    """Adapter-independent, and it has to be: the previous hardcoded
    ``period_ratio >= 3`` flagged nothing at all on the Gaia grid, whose widest cell
    sits *further* from zero than RV cells the constant did flag.
    """
    assert (1.0 / period_ratio >= MIN_PEAK_WIDTHS_FROM_ZERO) is expected


def test_arms_are_declared_once() -> None:
    assert ARMS == ("delta_default", "delta_h1", "z0")
