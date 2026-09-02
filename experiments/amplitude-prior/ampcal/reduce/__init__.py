"""The reduction.

:mod:`ampcal.reduce.calibrate` turns the artifact into one row per (simulation, arm),
carrying ``d_ln_peak``, ``tv`` and the Occam/shrinkage decomposition. Everything the
report says is an aggregation of that table, so the report and the CSV beside it cannot
disagree. Run it as ``python -m ampcal.reduce.calibrate``.
"""

__all__ = ()
