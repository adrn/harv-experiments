"""Turn the grid artifact into a report harv can act on.

One deliverable: what to make the linear-parameter prior defaults in
``harv.periodogram``. It has two halves that the evidence says want *opposite*
treatment --- the amplitude prior (:mod:`ampcal.report.amplitude`), whose scale and
exponent set the Occam factor and therefore the science, and the nuisance scales
(:mod:`ampcal.report.nuisance`), whose only failure mode is non-coverage.

Nothing here recomputes a metric. Every number comes from
:mod:`ampcal.reduce.calibrate`, which owns the definitions, so the report and the CSV
beside it cannot disagree. The one exception is the nuisance sweep, which the grid
never ran and which re-runs periodograms of its own; it says so.
"""

__all__ = ()
