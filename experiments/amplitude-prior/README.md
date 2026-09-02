# Does a period-dependent amplitude prior belong in harv's periodogram?

Design document, 2026-09-01. This is a research experiment, not harv public API —
`docs/spec.md` remains authoritative for the package itself. Nothing here changes
`harv.periodogram`'s public semantics; if a result argues for a change, that change goes
through `docs/spec.md` first.

## The question

`docs/spec.md` §"Priors are explicit" carries an open question — what amplitude scale to
recommend for the Fourier periodogram — gated by `TODO(default-amplitude-prior)` in
`harv/models/parameterizations/fourier.py`. Every *Keplerian* amplitude in harv already
carries a parameter-dependent prior (`PeriodDependentKPrior`,
`PeriodDependentSemiMajorAxisPrior`, `ParallaxDependentProperMotionPrior`), while the
*Fourier* harmonic amplitude gets a flat `Normal(0, sigma_amp)` with no `P` dependence —
although at trial period `P` it is measuring the same physical thing.

The intended outcome is **a scale and an exponent** for `FourierRV` and
`FourierGaiaAstrometry`, measured on a population where the physical prior can be right
and one where it cannot, so the default is judged on misspecification too.

### Why the predecessor could not answer it

`experiments/kepmodel-comparison` (deleted in the same commit that added this; see git
history) ran and did not settle the question, for structural rather than statistical
reasons:

1. **It only ever swept a constant scale.** The functional-form question was to be
   *inferred* from whether the optimal constant drifts with `P/T_span`.
2. **That inference did not resolve where it mattered.** Its `saturation_knee` returns
   NaN when the TV curve is still falling at the widest scale sampled, which is exactly
   what happens for `period_ratio >= 1` (RV) and `>= 0.4` (Gaia). The verdict was
   computed from the two well-sampled bins; the partial-arc regime — the whole point —
   contributed nothing.
3. **Its grid could not test the physical prior at all.** `period = period_ratio *
   t_span` and `amplitude = snr * sigma_n` were independent `itertools.product` axes, so
   **K was independent of P by construction** and a `K ∝ P^(-1/3)` prior was misspecified
   there by design.

What it *did* establish, and what this rebuild keeps:

- **The peak metric, not TV, is decisive.** Median `d_ln_peak` is exactly 0 for
  `P/T_span <= 0.3` at every scale, and −1.75 / −3.37 (RV) at `P/T = 3` / `10` for a
  tight prior, vanishing at `>= 5.62x` RMS. The spec's predicted short-period bias is
  real, large, and confined to partial arcs.
- **Occam is not flat in frequency**, contradicting `docs/spec.md`: measured
  `occam_range ≈ 0.8 x occam_mean`. It depends on design-matrix conditioning, which
  varies with trial frequency — and that frequency dependence *is* the mechanism behind
  the short-period bias.
- **A period-dependent prior tilts Δ in the direction that fixes it.**
- **Nuisance scales are settled**: flat while the prior covers the truth, degrading only
  at non-coverage. Failure is non-coverage, not mis-tuning, so a data-driven default is
  safe there. The report re-measures this rather than inheriting the claim.

## Design

### Arms

`sigma(P) = level * data_rms * (P / P0)**exponent`, with `P0 = t_span`, applied to every
harmonic amplitude column. `exponent = 0` is today's flat prior.

| | exponents | physical value |
|---|---|---|
| RV | `0, -1/6, -1/3, -1/2` | `-1/3` (`PeriodDependentKPrior`) |
| Gaia | `0, +1/3, +2/3, +1` | `+2/3` (`PeriodDependentSemiMajorAxisPrior`) |

Crossed with three levels — `10**-1.5`, `1`, `10**0.75` × the base-model-residual RMS —
and `n_terms ∈ {1, 2}`. 24 arms per simulation. The two outer levels sit on the
predecessor's grid points on purpose: the `exponent = 0` arm there must reproduce its
`d_ln_peak` table, which is the "reproduce a known result" gate.

**The Gaia blocker, and why the exponent sweep sidesteps it.**
`PeriodDependentSemiMajorAxisPrior` cannot be attached to `FourierGaiaAstrometry`'s
`ti_*_k` columns as it stands: it requires `params["parallax"]`, but the periodogram's
`_nl` supplies only `{"period", "eccentricity"}` and `FourierGaiaAstrometry` gives
`parallax` a plain Normal, so it is marginalized and never reaches the prior — a
`KeyError` inside a jit trace. The arm is therefore defined directly in **angular**
units, folding a nominal parallax into `sigma_0`. That is the right modelling choice
anyway: the prior's parallax factor only converts AU to angle, and in a periodogram
where parallax is marginalized the scale cannot legitimately depend on it. It is a small
`LinearPriorCallable` in this experiment (`ampcal.grid.PowerLawAmpPrior`), not a harv
change.

*Known limitation, documented in the report's caveats:* the harmonic index `k` is not
passed to a linear prior, so for `n_terms > 1` every harmonic is scaled by the
fundamental's period rather than `P/k`.

### Populations

The `population` axis is the control, and it is what the predecessor lacked:

- **`physical`** — draw `m2` from a log-flat companion-mass function at fixed `m1`, plus
  `cos i`, then compute `K` (RV) and angular `a_0` (Gaia) through
  `KeplerianBody.from_masses`. `K ∝ P^(-1/3) (1-e²)^(-1/2)` and `a_0 ∝ P^(+2/3)` then
  hold on average, so the physical prior is correctly specified.
- **`independent`** — `amplitude = snr * sigma_n`, flat in period, where the prior is
  wrong by construction. Measures the misspecification cost.

The `snr` axis keeps one meaning across both: the injected amplitude is normalized to a
reference system (`M2_REF` at `COS_I_REF`, `P = P0`, `e = 0`), so `snr` is "the
per-observation SNR this cell would have for that companion at `P = T_span`". See
`ampcal/population.py`; `tests/test_population.py` asserts both power laws, because a
population factor that cannot be falsified would make every misspecification cost read
as zero for the wrong reason.

### Metrics

`d_ln_peak` (primary), TV to the reference `R` (secondary), and
`occam` / `shrinkage` / `cond`. The decomposition is no longer only diagnostic: the
Occam tilt is the mechanism under test, and its ceiling —
`d(occam)/d(ln P)` gains at most `n_amp_columns * exponent` — is a prediction the report
checks rather than a fit.

`R` is the Keplerian marginal likelihood with a deliberately wide amplitude prior, MC
over `(e, omega, M0)` for RV and `(e, phase)` for Gaia with common random numbers,
`rv_semiamp`/`v_sys` (or the Thiele-Innes constants) marginalized analytically. It is
most of the per-simulation cost, and `--reference-n-mc` is the cost dial.

### Grid

`population` (2) × `period_ratio` (7) × `snr` (3) × `eccentricity` (2) × `n_obs` (5 RV /
4 Gaia) — 420 cells (RV) or 336 (Gaia), × 16 seeds. Doubling for `population` is paid
for by dropping `snr` and `eccentricity` rungs; the decisive axis is `period_ratio` into
the partial-arc regime, which the old grid under-sampled at the top end.

Gaia's `n_obs` floor of 10 is gone. It existed because kepmodel's profile statistic needs
`n_obs > p + d = 9`; with kepmodel gone, the sparse regime — where the marginal statistic
is the only thing that works at all, and where the prior's shape carries the most weight
— is open.

## Layout

```
ampcal/
  population.py   the two injected populations; masses -> amplitudes
  grid.py         Cell, Arm, PowerLawAmpPrior, the shared frequency grid
  adapters/       the one data-type seam (rv, gaia; common plumbing in common.py)
  linalg.py       marginal/profile likelihoods and the Occam/shrinkage decomposition
  identity.py     the correctness gate -- run it before any grid
  run.py          execute cells, write the artifact (SPMD over MPI)
  merge.py        combine per-rank shards
  artifact.py     the HDF5 schema every reduction reads
  reduce/         calibrate.py: one row per (simulation, arm)
  report/         amplitude.py (the verdict), nuisance.py, synthesis.py
```

## Running it

Everything runs with this directory on `PYTHONPATH`:

```sh
EXP=experiments/amplitude-prior

# 1. the correctness gate, on a period-dependent arm. Non-negotiable and cheap.
uv run python -m ampcal.identity --adapter rv --n-obs 40
uv run python -m ampcal.identity --adapter gaia --n-obs 40

# 2. a laptop-sized end-to-end pass
uv run python -m ampcal.run --adapter rv --which smoke \
  --out /tmp/rv/signal.h5 --stride 16 --n-seeds 2 --reference-n-mc 256
uv run python -m ampcal.reduce.calibrate --artifact /tmp/rv/signal.h5

# 3. the real thing
sbatch slurm/run_grid.sh
ADAPTER=gaia sbatch slurm/run_grid.sh
python -m ampcal.report.synthesis report/rv report/gaia --out report/
```

`SMOKE=1 sbatch …` runs the whole chain on a handful of cells. Do that before the real
launch.

## Verification

Four checks, in the order they have to hold. The first three are automated; the fourth
is a comparison against the predecessor's published numbers and has to be read.

1. **Correctness gate**, on a *period-dependent* arm, before any grid. `ampcal.identity`
   asserts our `ln Z` against harv's `log_prob`, our `Delta` against `hp.periodogram`,
   the `Delta = z0 - Occam - Shrinkage` algebra, and the batched path against the scalar
   one. That last residual is the one that would catch the per-frequency prior being
   built wrong — the only genuinely new machinery here.
2. **Population sanity** (`tests/test_population.py`): the injected `K` really does scale
   as `P^(-1/3)` on the physical population, `a_0` as `P^(2/3)`, and neither does on
   `independent`.
3. **The mechanism** (`tests/test_pipeline.py`, and the report's
   `occam_tilt_fraction_of_ceiling` table): the exponent tilts `occam` in the predicted
   direction and never beyond `n_amp_columns * exponent`.
4. **Reproduce a known result**: the `exponent = 0` arm at the outer levels must
   reproduce the predecessor's `d_ln_peak` table — 0 for `P/T <= 0.3`, −1.75 / −3.37 at
   `P/T = 3` / `10` at the tightest scale, vanishing at `>= 5.62x` RMS.

## Out of scope

- **Downstream rejection sampling.** Periodogram only: `d_ln_peak`, TV to `R`, and the
  decomposition. What the resulting interim period prior does to a sampler is a separate
  question and a separate cost.
- **Detection / ROC.** No null grid. Detection was the predecessor's question and it is
  not this one.
- **Correlated noise.** `linalg.evaluate` rejects a 2-d covariance outright, because
  harv's periodogram cannot marginalize a GP.
- **The real Gaia scan law.** Uniform scan angles; see the `ponytail:` comment in
  `adapters/gaia.py` for when to swap it in.
