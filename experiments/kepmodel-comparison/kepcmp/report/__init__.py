"""Turn the grid artifacts into a report harv can act on.

Two deliverables, one bundle per adapter:

- **(a)** what to make the linear-parameter prior defaults in ``harv.periodogram`` ---
  the amplitude scale (:mod:`kepcmp.report.amplitude`) and the nuisance scales
  (:mod:`kepcmp.report.nuisance`), which the evidence says want *opposite* treatment;
- **(b)** when to reach for which periodogram (:mod:`kepcmp.report.casestudy`).

Nothing here recomputes a metric. Every number comes from the five reductions in
:mod:`kepcmp.reduce`, which own the definitions, so the report and the CSV beside it
cannot disagree.
"""

__all__ = ()
