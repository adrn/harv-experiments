# harv vs kepmodel: marginal versus profile periodograms

Design document, 2026-08-12, revised 2026-09-01 (Gaia adapter, MPI sharding). This is
a research experiment, not harv public API —
`docs/spec.md` remains authoritative for the package itself. Nothing here changes
`harv.periodogram`'s public semantics; if a result argues for a change, that change
goes through `docs/spec.md` first.

> **Picking this up cold? Read [`HANDOFF.md`](HANDOFF.md) first.** It has the state of
> play, the next actions in order, the decisions already made, and the specific traps
> (kepmodel is not the reference; float64 is mandatory; `z0` does not exist at
> `n_obs <= p + d`, which is 3 for RV and 9 for Gaia). This document is the design and
> the detailed measured status; **"Running the experiment"** below is the run guide.

## Purpose

`harv.periodogram` reports a **marginal** log-likelihood ratio. kepmodel reports a
**profile** (maximum) log-likelihood ratio. The two differ by terms we can write
down exactly. This experiment measures what that difference does in practice over
a grid of simulated systems, answering four questions:

1. **Mechanism** — where the two statistics disagree, and which term explains it.
2. **Detection** — which detects better, as an ROC against null simulations.
3. **Prior quality** — which makes a better interim period prior for the rejection
   sampler, i.e. the thing harv actually uses the periodogram for.
4. **Calibration** — what amplitude prior scale `sigma_amp` to recommend. This is
   the open question flagged in `docs/spec.md` ("Priors are explicit") and in the
   `TODO` in `harv.models.parameterizations.fourier`.

All four are reductions over one stored table of periodogram evaluations. The
expensive work — simulating, scanning, and computing the reference statistic — is
done once.

## Background: the exact relation between the two statistics

Both codes are linear-model periodograms with the same skeleton: a base model `H`
with `p` linear columns, an enlarged model `K(nu)` appending `d` frequency-dependent
columns, a fixed Gaussian noise covariance `C`, and a scan over frequency only.
Write

```
A_m = Phi_m^T C^-1 Phi_m      b_m = Phi_m^T C^-1 y      theta_hat_m = A_m^-1 b_m
chi2_m = y^T C^-1 y - b_m^T A_m^-1 b_m
```

**kepmodel** minimises over all linear parameters at every frequency
(`timeseries.py::_chi2ogram`, via a Cholesky-whitened design from spleaf) and
reports

```
z0(nu) = 1/2 (chi2_H - chi2_K(nu)) = ln L(theta_hat_K) - ln L(theta_hat_H)
```

a generalised likelihood ratio at the MLE. `periodogram()` returns
`1 - chi2_K/chi2_H`, Baluev's `z1` normalisation, which is invariant to an unknown
global noise scale. Significance is supplied externally by `fap()`: Baluev (2008)
extreme-value theory generalised to correlated noise by Delisle, Hara & Segransan
(2020), with bandwidth `W = f_max * T_eff` and `T_eff` computed from `C`.

**harv** places `theta ~ N(0, Lambda)` and integrates,
`Z_m = N(y; 0, C + Phi_m Lambda Phi_m^T)`, giving with `S_m = A_m + Lambda_m^-1`

```
ln Z_m = -1/2 y^T C^-1 y + 1/2 b_m^T S_m^-1 b_m - 1/2 ln det(I + Lambda_m A_m) + const(C)
```

Subtracting the two models and using the block factorisation
`det(I + Lambda_K A_K) = det(I + Lambda_H A_H) * det(I + Lambda_phi A_tilde(nu))`:

```
Delta(nu) = z0(nu) - O(nu) - G(nu)

O(nu) = 1/2 ln det(I + Lambda_phi A_tilde(nu))          [Occam]
A_tilde(nu) = A_phi_phi - A_phi_H (Lambda_H^-1 + A_HH)^-1 A_H_phi

G(nu) = 1/2 [theta_hat^T A S^-1 Lambda^-1 theta_hat]_K
      - 1/2 [theta_hat^T A S^-1 Lambda^-1 theta_hat]_H  [shrinkage]
```

**harv's statistic is kepmodel's, minus an Occam factor, minus a shrinkage term.**
This is an exact algebraic identity, not an approximation, and it is the basis of
the validation gate below.

Two properties drive the experiment:

- `O(nu)` depends only on how well the *new* columns are constrained once the base
  is projected out. In the wide-prior limit its eigenvalues are
  `(sigma_amp / sigma_post,j)^2`, so `O ≈ d * ln(sigma_amp sqrt(n) / (sigma_n sqrt(2)))`
  for a well-sampled sinusoid. To first order this is frequency-independent: it
  shifts the whole periodogram and leaves peak *locations* alone. The
  frequency-dependence enters only through `A_tilde(nu)`, i.e. through the window
  function. **Prediction: the two statistics agree on peak location wherever the
  amplitude is well constrained, and diverge exactly where it is not** — partial
  arcs, aliases, and columns near-degenerate with the base.
- `G(nu)` is ridge shrinkage of the amplitudes toward the prior mean. Each of its
  two terms is non-negative and vanishes as `A >> Lambda^-1`, so its sign is not
  fixed but its magnitude is negligible for well-measured amplitudes.

Both expressions assume zero prior means. Non-zero means (a `v_sys` prior centred
away from zero) add a known quadratic term; the identity gate uses zero-mean priors
to keep the check clean.

## Scope

**Two data types, one harness: RV and Gaia epoch astrometry.** They share every
statistic, every reduction and the identity gate; only the adapter differs. Each has
its own grid axes, its own units and its own artifact file — the two are never
merged, because the axes mean different things. See "Running the experiment" next,
and "The Gaia adapter" for what the astrometry case changed.

## Running the experiment

Every entry point takes `--adapter rv` (the default) or `--adapter gaia`. The two
data types have different axes, units and column counts, so **they are separate runs
into separate directories, and their artifacts must never be merged** — `merge.py`
checks the frequency grid, not the adapter, so it will not stop you.

### Prerequisites

`harv` and `kepmodel` are git dependencies of the repository root, so a plain
`uv run` is enough. The only thing to set is `PYTHONPATH`, because `kepcmp` is a
directory on the path rather than an installed package:

```bash
export PYTHONPATH=experiments/kepmodel-comparison   # or wherever this directory lives
uv run python -m pytest $PYTHONPATH/tests -q        # 41 passed
```

`PY_CMD="python -m"` replaces the `uv run` invocation in the launcher if you are
already inside a suitable environment.

### Step 0 — check the gate before spending any compute

The identity gate validates the whole cross-code wiring in one shot. Run it for the
adapter you are about to use; if it fails, nothing downstream means anything.

```bash
uv run python -m kepcmp.identity --adapter rv
uv run python -m kepcmp.identity --adapter gaia --n-obs 20
```

Read the report, not just `PASS`. The number that diagnoses a wiring bug is the
scale-free `identity residual ... (X relative)`: `~5e-13` is the arithmetic floor for
RV, `~1e-7` for Gaia (higher because `cond` is genuinely larger there — see
"The Gaia adapter"), and anything approaching O(1) is a real bug. `--period-ratio`,
`--snr`, `--eccentricity`, `--n-obs`, `--seed` pick the cell; `--stride` thins the
grid, since the gate is a Python loop per frequency and does not need full density.

### Step 1 — validate the plumbing

`SMOKE=1` runs 2 simulations per shard through launch → shards → merge, about a
minute. Do this after any change to the runner, the artifact schema or an adapter.

```bash
EXP=experiments/kepmodel-comparison
SMOKE=1 bash $EXP/launch_full_grid.sh scratch/kepcmp/smoketest
SMOKE=1 ADAPTER=gaia bash $EXP/launch_full_grid.sh scratch/kepcmp/smoketest_gaia
```

### Step 2 — launch the grid

```bash
NPROC=32 bash $EXP/launch_full_grid.sh scratch/kepcmp/rv_full
ADAPTER=gaia NPROC=32 bash $EXP/launch_full_grid.sh scratch/kepcmp/gaia_full
```

Each invocation runs two phases in sequence — `signal` (the full grid, with the
reference statistic) and `null` (`K = 0`, two configs only, for the ROC thresholds)
— and writes per-shard artifacts plus a log per shard under `logs/`. Watch it with
`tail -f scratch/kepcmp/rv_full/logs/*.log`.

Launcher environment:

| variable | default | meaning |
|---|---|---|
| `ADAPTER` | `rv` | `rv` or `gaia` |
| `NPROC` | cores, capped at 32 | processes per phase |
| `MPI` | auto (`1` if `mpirun` is on `PATH`) | `0` forces plain background processes |
| `N_SEEDS` | 16 | signal seeds |
| `NULL_SEEDS` | 1000 | null seeds per `n_obs`; FPR=0.01 needs >~1000 |
| `REFERENCE_N_MC` | 2048 | MC draws for `R`; shape converges by ~256 |
| `SMOKE` | unset | 2 sims per shard, plumbing only |
| `PY_CMD` | `uv run python -m` | replace the invocation |

### Step 3 — merge the shards

```bash
OUT=scratch/kepcmp/rv_full
uv run python -m kepcmp.merge --out $OUT/signal.h5 $OUT/signal.rank*.h5
uv run python -m kepcmp.merge --out $OUT/null.h5   $OUT/null.rank*.h5
```

`merge` refuses on a frequency-grid mismatch or a duplicate `sim_id`, both of which
mean the shards were not from one run. At `NPROC=1` there are no `.rank` files — the
single shard writes `signal.h5` directly and this step is unnecessary.

### Step 4 — reduce

`regime_map` is the deliverable; the other four answer the design document's four
questions individually. All take `--csv` to write a table alongside the printed
summary.

```bash
uv run python -m kepcmp.reduce.regime_map --artifact $OUT/signal.h5 \
     --null-artifact $OUT/null.h5 --csv $OUT/regime_map.csv
```

| reduction | question | required inputs |
|---|---|---|
| `reduce.regime_map` | which periodogram wins where, on `tv` / `peak_err` / `tpr` side by side | `--artifact`, `--null-artifact` |
| `reduce.decompose` | where the two disagree, and whether `O` or `G` explains it | `--artifact` |
| `reduce.roc` | detection: TPR at fixed FPR per cell | `--artifact`, `--null-artifact` |
| `reduce.calibrate` | which `sigma_amp` makes `Delta` reproduce `R` | `--artifact` (needs `R`) |
| `reduce.prior_quality` | which makes a better interim period prior | `--artifact` |

`prior_quality` is the expensive one and is deliberately **not** in the launcher: it
runs a rejection sampler per simulation per arm. Start with `--limit` and raise
`--n-prior-samples` well above the default before believing the result — at 50k
every arm was under-resolved. `regime_map` and `roc` take `--fpr`; everything that
maps a curve through `tempered_period_prior` takes `--beta` and `--floor`, which must
match across arms for the comparison to mean anything.

### What lands on disk

```
scratch/kepcmp/rv_full/
  logs/signal.log, logs/null.log          # or logs/<phase>.rank<i>.log without MPI
  signal.rank000.h5 ... signal.rankNNN.h5 # one per shard, written incrementally
  null.rank000.h5   ... null.rankNNN.h5
  signal.h5, null.h5                      # after merge
  regime_map.csv                          # after reduce
```

Artifacts are written **incrementally**, one simulation at a time, so a run that dies
partway through still leaves a readable file with everything finished so far. There is
no resume: re-running a shard rewrites its file from scratch.

### Parallelism

The grid is embarrassingly parallel and the shards never talk to each other: each
writes its own `<name>.rank<NNN>.h5` and `kepcmp.merge` combines them afterwards. So
`kepcmp.run` reads its rank from the *launcher's environment*
(`OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `PMIX_RANK`, `SLURM_PROCID`) rather than linking
`libmpi`. `mpirun -n 32 python -m kepcmp.run ...` therefore works with no `mpi4py`
dependency at all, and a broken or ABI-mismatched MPI runtime cannot take a run down
at `MPI_Init`. `--shard i --n-shards N` is the explicit equivalent, and
`launch_full_grid.sh` uses `mpirun` when it is on `PATH` and plain background
processes otherwise (`MPI=0` forces the latter).

The shard identity is printed at startup, so an unrecognised launcher shows up
immediately as `1 shard (serial)` instead of every rank silently overwriting one file.
**Check that line in the log before walking away from a long run.**

Jobs are dealt **round-robin** across shards, not in contiguous blocks: per-simulation
cost grows with `n_obs`, so a contiguous split would hand every expensive cell to the
same rank and leave the others idle at the end.

Each process is pinned to one thread. The matrices are tiny (`n_obs <= 80`,
`k <= 17`), so they do not parallelize within a process and N single-threaded shards
beat N shards fighting over BLAS threads.

### Timing

Measured on a 16-core laptop over the full 1828-point grid, 15 harv configs plus
kepmodel plus the reference at `n_mc = 2048`: RV runs 5.5 s/sim at `n_obs = 2` rising
to 8.7 s/sim at `n_obs = 32`, mean 7.0. So 7200 signal sims is ~14 h serial and
~26 min at 32 ranks; the nulls (two configs only) are ~50 min serial. Gaia has not
been timed at full density; its 5760 sims are in the same class, with the reference
*cheaper* (2-D MC) and the per-frequency algebra dearer (9 columns instead of 3).

### Running one cell by hand

`kepcmp.run` takes the same flags the launcher passes, so a single cell or a fast
partial grid is one command. `--stride` thins the frequency grid and `--limit` caps
the simulations per shard — together they turn a 14-hour grid into a 30-second check.

```bash
uv run python -m kepcmp.run --adapter gaia --which smoke --out /tmp/probe.h5 \
    --stride 16 --n-terms 1 --sigma-amp-mults 1.0 --reference-n-mc 256 --limit 3
```

| flag | meaning |
|---|---|
| `--which` | `signal`, `null`, or `smoke` (a few cells spanning the adapter's grid) |
| `--stride` | subsample the shared frequency grid |
| `--limit` | stop after N simulations **per shard** |
| `--n-terms`, `--sigma-amp-mults` | cut the config sweep down from 15 |
| `--reference-n-mc` | `0` disables `R`, which is most of the per-sim cost |
| `--n-seeds`, `--seed-offset` | which seeds to run |
| `--shard`, `--n-shards` | explicit sharding, overriding rank detection |

`gaia_probe.py` next to this file is a standalone 60-line version of the same path
(simulate → harv periodogram → kepmodel `AstroModel`) with no `kepcmp` machinery,
useful when the question is about harv or kepmodel rather than about the harness.

## Architecture

One package, `experiments/kepmodel-comparison/`, with the data-type-specific parts
behind a single seam:

```python
class Adapter(Protocol):
    d: int                      # frequency columns per harmonic: 2 (RV), 4 (Gaia)
    n_base_columns: int         # kepmodel's p: 1 (RV), 5 (Gaia)
    t_span, period_min, period_max, period_ratios, snrs, eccentricities, n_obs_values

    def period(self, cell) -> ScalarQTime: ...          # cell axes -> physical units
    def amplitude(self, cell) -> Q: ...
    def simulate(self, cell: Cell, seed: int) -> tuple[AbstractData, dict]: ...
    def data_arrays(self, data) -> dict[str, np.ndarray]: ...   # what the artifact stores
    def data_from_arrays(self, arrays) -> AbstractData: ...     # and how to read it back
    def trial_prior(self, data, *, n_terms: int, sigma_amp) -> HarvPrior: ...
    def kepmodel_model(self, data): ...        # base linear columns installed, cov set
    def reference_ln_z(self, data, periods, *, n_mc: int) -> np.ndarray: ...
    def science_prior(self, data, period_prior) -> HarvPrior: ...   # for the rejection run
```

`Cell` is deliberately unitless --- `period_ratio`, `snr`, `eccentricity`, `n_obs`
and nothing else. It becomes a dataset only once an adapter interprets it, which is
what lets both grids share every reduction downstream.

Modules:

| module | responsibility |
|---|---|
| `adapters/common.py` | plumbing both adapters share (harv blocks, kepmodel driving, the `p+d` gate) |
| `adapters/rv.py` | the RV adapter |
| `adapters/gaia.py` | the Gaia epoch-astrometry adapter |
| `identity.py` | the exact-identity gate, including `O` and `G` |
| `grid.py` | grid cell definition and enumeration |
| `run.py` | executes cells, writes the artifact |
| `artifact.py` | HDF5 schema, read/write |
| `reduce/{decompose,roc,prior_quality,calibrate}.py` | the four reductions |
| `reduce/regime_map.py` | per-regime verdict on all three metrics side by side |
| `reduce/common.py` | the only thing reductions share |
| `merge.py` | combine seed-sharded artifacts into one |
| `launch_full_grid.sh` | sharded launcher (MPI or background processes); `SMOKE=1` validates plumbing first |

Each reduction reads the artifact and shares nothing else with the others.

## Component 1: the identity gate (build this first)

With `n_terms=1` (harv `d=2` = kepmodel `d=2`), the same `C`, the same base column,
and the same `y`, the two codes must agree to machine precision once `O` and `G` are
added back:

```
| Delta(nu) + O(nu) + G(nu) - z0(nu) | < atol + rtol * chi2_scale     for all nu
    atol = 1e-8, rtol = 1e-11, chi2_scale = max(chi2_H, max_nu chi2_K)
```

This validates the entire harness in one shot — matched noise, matched base model,
matched time reference, matched frequency convention. It is not optional: a wiring
bug looks exactly like a scientific result.

**The tolerance is relative, and an earlier draft of this document was wrong to
state a fixed `1e-8`.** The identity is a cancellation between quantities of
magnitude `chi2 ~ SNR^2 * n_obs`, so the achievable absolute residual grows with
`chi2`. Measured across four decades of SNR at fixed everything else, the residual
tracks `chi2` with `residual / chi2` flat at `4-6e-13` — about 2000x machine
epsilon, which is what an `lstsq` + `solve` + `slogdet` chain costs. A fixed `1e-8`
passes at SNR 10 and fails at SNR 100 for purely arithmetic reasons. The scale-free
`residual / chi2_scale` is the number that diagnoses a wiring bug: `~5e-13` is the
floor, `>> 1e-10` means something is actually wrong.

**float64 must be enabled** (`kepcmp/__init__.py` does this at import). harv does
not enable it, so JAX defaults to float32 with `eps ~ 1e-7` and the gate is
unreachable by three orders of magnitude — while kepmodel/spleaf are numpy and
already double precision, so the two sides would not even be computing at the same
precision.

Measured on eight cells spanning the grid corners (`P/T_span` 0.02–3, SNR 1–1000,
`e` 0–0.6, `n_obs` 12–40): all pass, relative residuals `1e-14` to `4e-13`.

Specific conventions that must line up, each a silent failure mode:

- **Frequency units.** kepmodel takes *angular* frequency (`nu0`, `dnu`); harv's
  grid is in cycles per unit time. A factor of `2*pi` here still yields a
  plausible-looking periodogram.
- **Time reference.** kepmodel uses raw `t`; harv uses `data.time - data.t_ref`. The
  base columns must span the same space, so pass kepmodel `t - t_ref`.
- **Noise.** harv's periodogram rejects nonlinear extensions (`Jitter`, `GP` raise
  `TypeError`), so `C` must be the diagonal reported errors on both sides. Do not
  fit an spleaf GP for the comparison.
- **Statistic conversion.** `z0 = 0.5 * chi20 * power`, since
  `power = 1 - chi2_K/chi2_H`.
- **Base zero point.** kepmodel profiles its offset; harv marginalises `v_sys` under
  a Gaussian. The difference is frequency-independent, so shapes compare directly,
  but store `chi20` and `ln_likelihood_base` so curves can be shifted onto a common
  zero.

`O` and `G` are computed directly from the design matrices in `identity.py`, not
inferred from the difference of the two periodograms — otherwise the gate is
circular.

## Component 2: simulation grid

`harv.simulate.simulate_rv_sb1_data` already exposes every axis needed except the
observing window.

| axis | values | why |
|---|---|---|
| `P / T_span` | 0.02, 0.1, 0.3, 1.0, 3.0, 10.0 | well-sampled through partial arc and past identifiability |
| `K / sigma_n` | 1, 3, 10, 30, 100 | per-point SNR |
| `eccentricity` | 0, 0.3, 0.6 | the harmonics axis |
| `n_obs` | 2, 4, 8, 16, 32 | sparse data is a primary harv use case, not an edge |
| seeds | 16 | |

450 cells x 16 seeds = 7200 simulations.

**`P / T_span = 10` is deliberately past identifiability.** There the injected frequency
is ~10x *smaller* than one periodogram peak width (`1 / t_span`), so no method can
localize the period — that is information content, not a grid artifact. `peak_err` is
meaningless in that column and only the distributional (TV) metric is informative;
`regime_map` marks it. `PERIOD_MAX` was extended to `40 T_span` so those peaks are not
pinned to the grid boundary, which costs nothing: the point count is set by
`f_max = 1 / PERIOD_MIN` and stays at ~1828.

**The `n_obs` ladder has two hard structural consequences, both measured.**

The Fourier trial model has `2H + 1` linear columns and kepmodel's enlarged model has
`p + d = 3`, so on the sparse rungs there are more parameters than data:

- **The profile statistic ceases to exist at `n_obs <= 3`.** It fits any trial period
  exactly: `chi2_K` varies by ~1e-13 across the whole grid, so `z0` is flat and carries
  zero period information. kepmodel's explicit `inv(N N^T)` additionally returns
  non-physical `chi2` (measured `-1.5e14` at `n_obs = 2` on a strided grid) or raises
  `LinAlgError` outright (on the full grid). `HarvKepmodelAdapter.kepmodel_z0` returns
  `nan` with a reason string rather than letting either outcome enter the artifact as a
  measurement, and every reduction gates on `ArtifactReader.z0_usable`. The threshold is
  `p + d`, so it moves with the data type — 9 for Gaia.
- **harv's own overfitting cap forces `n_terms = 1` for `n_obs <= 8`.** So the `H=2`
  head-to-head arm does not exist below `n_obs = 12`; there the comparison is the `H=1`
  control, which is the cleaner contrast anyway since its columns match kepmodel's
  `d = 2` exactly.

The marginal statistic stays well defined at every rung, because `S = A + Lambda^-1` is
positive-definite even where `A` is singular — the prior supplies the missing rank.
Measured at `P/T_span = 0.3`, `SNR = 10`, `e = 0`:

| `n_obs` | `z0` range | `Delta` range | `P_true` | `Delta` peak | `z0` peak |
|---|---|---|---|---|---|
| 2 | 1.1e-13 (flat) | 62.2 | 548 d | 16 d | 11 d |
| 3 | 1.7e-13 (flat) | 83.1 | 548 d | 17 d | 142 d |
| 4 | 71.3 | 80.3 | 548 d | **555 d** | 27 d |
| 8 | 158.8 | 157.7 | 548 d | **555 d** | **555 d** |

Three regimes, and the middle one is the finding: at `n_obs = 4` the marginal statistic
recovers the period and the profile statistic does not. At `n_obs <= 3` neither works,
but they fail differently — `z0` fails visibly flat, `Delta` stays non-flat and peaks
~30x short on prior-and-window structure rather than signal. For an interim prior that
silent, confident failure is the worse mode, and the `floor` mixture is what keeps it
from excluding the truth. `sigma_n` is fixed at 0.5 km/s and
`K = SNR * sigma_n`; `T_span` is fixed and `period = ratio * T_span`. Per-point SNR
is the parameterisation, but detectability scales as `K sqrt(n) / sigma_n`, so the
integrated SNR is recorded per cell to keep results interpretable across `n_obs`.

Each simulation is scanned at `n_terms` in {1, 2, 3} x `sigma_amp` over five values
spanning three decades around the data RMS, plus one kepmodel run. Grid shapes are
constant across cells, so harv JIT-compiles once.

**Known limitation:** sampling is uniform-random over the baseline. The built-in
simulator has no seasonal-gap option, so alias structure is under-represented
relative to real RV campaigns. Adding a window means constructing `RVData` from
custom time arrays. Deferred, and results about aliases are reported with this
caveat attached rather than generalised.

**The Gaia grid is a different grid, not the same one re-run.** `T_span` is fixed by
the data release, so `P / T_span` is a reparameterization of period rather than a free
axis, and the interesting periods are the ones near a year:

| axis | values | why |
|---|---|---|
| `P / T_span` (`T_span = 5 yr`) | 0.05, 0.1, **0.2**, 0.4, 1.0, 2.0 | `P` = 0.25, 0.5, **1**, 2, 5, 10 yr |
| `a_0 / sigma_al` | 1, 3, 10, 30, 100 | per-epoch SNR; `sigma_al = 0.1 mas` |
| `eccentricity` | 0, 0.3, 0.6 | the harmonics axis, unchanged |
| `n_obs` | 10, 20, 40, 80 | field-of-view transits |
| seeds | 16 | |

360 cells x 16 seeds = 5760 simulations.

`P / T_span = 0.2` is **exactly one year**, and it is the point of the Gaia grid. The
parallax column is a 1-yr sinusoid in the base model of *both* codes, so `docs/spec.md`'s
claim that `Delta` suppresses parallax and proper-motion power by cancellation is
directly testable there, with 0.1 and 0.4 flanking it to show how wide the suppression
is. `z0` shares the cancellation, so this is a test of the claim rather than a
comparison between the two codes --- the one place in this experiment where the two
statistics are expected to agree *and that agreement is the result*.

Two structural boundaries move with the wider model, both from `p + d = 5 + 4 = 9`:

- **`z0` needs `n_obs > 9`**, four times the RV threshold. `n_obs = 10` is the first
  rung where the profile statistic exists at all, with one residual degree of freedom
  --- the direct analogue of `n_obs = 4` in the RV grid, which is where the marginal
  statistic was measured to recover the period and the profile statistic was not.
- **harv's cap is `H_max = (n_obs/2 - 5) // 4`**, so `H=2` needs `n_obs >= 26` and
  `H=3` needs `n_obs >= 34`. Only the 40 and 80 rungs carry the multi-harmonic arms.

Going below `n_obs = 10` would measure nothing: both statistics are undefined and the
comparison has no content, unlike RV where the marginal statistic still worked.

**`sigma_amp` is quoted against the base-model residual RMS, not the raw data RMS.**
For RV those are the same thing --- the base is one constant column, so removing it is
subtracting the mean, which is what the validated harness always did (verified
bit-for-bit). For Gaia they differ by a factor of ~50: the raw along-scan scatter is
~30 mas, dominated by parallax and proper motion that the base columns absorb, while a
1 mas orbit contributes ~0.56 mas. Quoting `sigma_amp` against the raw RMS would put
every multiplier ~30x above the amplitude it is meant to bracket, by a factor that
moves with the astrometric solution from seed to seed, and the calibration sweep would
measure nothing.

**Null simulations** for the ROC: zero amplitude, matched `n_obs` / baseline /
`sigma_n`, ~500 per configuration.

**There is no reference slice any more.** The design tiered compute because `R` looked
expensive; measured at ~5 s/sim (and cheaper still for Gaia) it is affordable on the
whole grid, so `--reference-n-mc N` is passed for every cell and every cell gets a truth
curve. `enumerate_reference_cells` and `--which reference` were deleted rather than left
as a trap.

## Component 3: the three statistics

All three are Delta-from-null in nats, on the same frequency grid.

| statistic | Fourier basis | amplitude prior | source |
|---|---|---|---|
| `Delta_H(nu; sigma_amp)` | yes, `H` harmonics | yes | `hp.periodogram` |
| `z0(nu)` | yes, `d=2` (RV) / `d=4` (Gaia) | no | `0.5 * chi20 * power` from `RvModel` / `AstroModel.periodogram` |
| `R(nu)` | no — true Keplerian | no | reference, MC over the orbital angles |

The reference is

```
Z(P) = E_{e, omega, M0} [ Z_lin(P, e, omega, M0) ]
```

where `Z_lin` marginalises `rv_semiamp` and `v_sys` analytically at fixed nonlinear
parameters, and the expectation is over the **science prior** — the same
`StandardRV().default_prior(...)` used for the analysis run. `R` is therefore the
ideal marginal-likelihood periodogram under harv's own priors, and `Delta` differs
from it by exactly two things: the Fourier basis and the `sigma_amp` choice.
Comparing `Delta` to `R` across `sigma_amp` is precisely the calibration question.

Because only the *shape* of `R` matters, use **common random numbers**: the same
`(e, omega, M0)` draws at every trial period. This removes most of the MC variance
in the shape.

The formula above is the RV case. For Gaia the Thiele-Innes constants are linear, so
the expectation is over `(e, M0)` only — see "The Gaia adapter".

The three-way structure is what separates the two approximations:

```
Delta   : Fourier basis + amplitude prior
z0      : Fourier basis (d=2 only), no prior
R       : neither
```

so `Delta - R` isolates the combined error, `z0 - R` isolates the basis error at
`d=2`, and the residual attributes the rest to the prior.

## Component 4: artifact schema

One HDF5 file, one row per `(sim_id, statistic, config)`, holding the full
frequency-grid arrays plus everything the reductions need:

- simulation parameters and `sim_id`, `cell_id`, `seed`
- `frequency` (shared per cell), `delta_ln_likelihood` or `z0` or `R`
- `occam(nu)`, `shrinkage(nu)` where applicable
- `n_terms` requested and effective (the cap), `sigma_amp`
- `chi20`, `ln_likelihood_base`
- integrated SNR, data RMS

## Component 5: the four reductions

- **`decompose`** — peak-location agreement across the grid, with disagreements
  attributed to `O` and `G`. Tests the prediction that agreement tracks how well
  the amplitude is constrained.
- **`roc`** — TPR at fixed FPR per cell, thresholds set on the null simulations.
  Threshold-free, so the different scales of the two statistics do not matter.
- **`prior_quality`** — wraps *both* `Delta` and `z0` in a `PeriodogramResult` (a
  plain `eqx.Module`, so kepmodel's statistic feeds `tempered_period_prior` with no
  new prior code), runs `RejectionSampler`, and records acceptance, whether `P_true`
  survives, and `acceptance_diagnostics()`. Per `docs/spec.md` ("Interpreting
  acceptance"), raw accept counts are **not** the metric — report
  `max_log_likelihood` and evidence ESS alongside, since a better prior can accept
  fewer samples against a correctly higher bar.
- **`calibrate`** — for each `sigma_amp`, how closely `Delta` reproduces `R` over the
  reference slice. Two metrics, both defined on the quantity that matters
  downstream: map `Delta` and `R` through `tempered_period_prior` at matched `beta`
  and `floor`, then report (a) total-variation distance between the two resulting
  densities in `ln P`, and (b) the signed peak-location error
  `ln P_peak(Delta) - ln P_peak(R)`, whose sign diagnoses the short-period bias.
  Reporting TV distance rather than a distance between raw nats curves means the
  frequency-independent part of `O` — which shifts `Delta` bodily but changes no
  inference — does not pollute the metric.

## Testing

Beyond the identity gate:

- **Noiseless single sinusoid.** All three statistics must peak at the injected
  period.
- **Empirical vs analytic FAP.** The false-alarm rate measured from the null
  simulations must match kepmodel's analytic `fap()`. This validates *our wiring of
  kepmodel* against its own published calibration — the cheapest available proof we
  are driving their code correctly.
- **Reference MC convergence.** `R` computed at two `n_mc` values must agree in peak
  location and in shape to a stated tolerance.

## The Gaia adapter

Implemented, gated and smoke-run end to end. What lines up, and what the original
design document got wrong about it.

**The bases are column-identical.** kepmodel's `astro.AstroModel._perio_phi` returns
`[cth*cos, sth*cos, cth*sin, sth*sin]`; harv's `FourierGaiaAstrometry(n_terms=1)`
frequency block is `[cos*cos_psi, cos*sin_psi, sin*cos_psi, sin*sin_psi]`. Same four
columns, same order, so `d = 4` and `H = 1` line up exactly as they do for RV.
`tests/test_gaia.py::test_bases_are_column_identical_to_kepmodel` pins this with a
two-way least-squares span check rather than trusting the ordering.

**kepmodel gets harv's own base columns.** `AstroModel` installs no linear columns of
its own, so the 5-parameter astrometric solution has to be added explicitly. The
adapter takes those five columns straight out of `_full_design_matrix` instead of
rebuilding them, which makes the proper-motion time origin, the along-scan sign
convention and the parallax factor structurally identical between the two codes rather
than coincidentally so. A mismatch there would look exactly like a scientific result.

**The identity gate is a stronger test here and it passes.** The base model has 5
columns instead of 1 and the trial model adds 4 instead of 2, so the block
factorisation being cancelled is much larger. Measured at four grid corners:
`ln Z vs harv` at `2e-10` absolute (`~2e-13` relative, the conditioning-independent
marginal path behaving exactly as for RV), and the cross-code identity residual at
`6e-9` to `1e-7` relative --- which is the `eps * cond**2` floor for `cond ~ 2e4`, not
a wiring bug. The conditioning is genuinely worse than RV's because the frequency
columns really do go degenerate with the parallax column near 1 yr; that is the
physics the grid is built to study, and the gate's cond-scaled tolerance absorbs it.

**Correction: the reference is cheaper, not more expensive.** The design document
assumed `R` would need a 5-dimensional MC over `(e, omega, Omega, cos i, M0)` and that
the reference slice would have to shrink. It does not.
`ThieleInnesGaiaAstrometry` makes `A, B, F, G` *linear*, so they marginalize
analytically along with the 5 astrometric parameters and only `(eccentricity,
phase_peri)` are integrated --- a **2-dimensional** MC, one dimension cheaper than
RV's. Common random numbers apply exactly as before.

**Correction: harv's default Thiele-Innes prior cannot be used for `R`.** It is
`PeriodDependentSemiMajorAxisPrior`, and the proper-motion prior is
`ParallaxDependentProperMotionPrior`. Those are the right priors for an analysis and
the wrong ones for the yardstick: `R` must carry no opinion about amplitudes, or the
`sigma_amp` sweep measures the reference's prior instead of harv's.
`GaiaAdapter.reference_prior` therefore overrides all nine linear priors with wide
zero-mean Normals --- including parallax, whose HalfNormal default is both an opinion
and a truncation that would break the analytic linear marginalization. `science_prior`
keeps the physical defaults, because `prior_quality` is asking what a real analysis
would do.

**Two unit traps in the science prior, both of which construct silently and fail
minutes into a sampler run.** `ThieleInnesGaiaAstrometry.default_prior` needs
`sigma_vtan` — its proper-motion prior is parameterized in *velocity*, not `mas/yr` —
and raises without it. And `sigma_a0` is a **physical length**, not an angle:
`PeriodDependentSemiMajorAxisPrior` is `sigma_a0 * (P/P0)**(2/3) * parallax`, so the
parallax does the conversion to mas. Passing mas builds a well-formed prior and then
raises `UnitConversionError` inside `_resolve_prior_to_mvn`. The adapter keeps these as
two separately named fields — `sigma_ti_reference` (mas, for `R`'s plain Normals) and
`sigma_a0_science` (AU, for the period-dependent science prior) — rather than one
number reused, and `tests/test_gaia.py::test_science_prior_builds` *evaluates* the
callables rather than only checking that the keys exist, because checking keys alone
passes with the wrong units.

**What is pinned and why.** `parallax = 10 mas` rather than the simulator's
`Exp(10 mas)` default: the parallax degeneracy is a headline axis, so its strength has
to be controlled rather than randomised, the same reasoning as pinning `v_sys` for RV.
Times, scan angles and parallax factors are constructed by the adapter rather than
left to the simulator, because `simulate_gaia_epoch_astrometry`'s default
`parallax_factor` is `U(-1, 1)` **white noise** --- uncorrelated with time, so the
parallax column would carry no 1-yr structure at all and the degeneracy axis would
test nothing.

**Known limitation, sharper than the RV one.** Scan angles are uniform. Real Gaia
scan angles come from the scan law and are strongly non-uniform, so alias structure is
under-represented more severely than in the RV case. `harv.simulate.scanlaw` has the
DR3/DR4/DR5 tables (they download on first use) and the swap is localised to
`GaiaAdapter.simulate`, marked with a `ponytail:` comment. Deferred; any result about
aliases carries this caveat rather than being generalised.

## Measured status (2026-09-01)

Implemented in `kepcmp/`; **41 tests pass** (31 RV, 10 Gaia). See
"Running the experiment" above for how to drive it.

### Porting this code

The directory is self-contained: copy it anywhere and `launch_full_grid.sh` resolves
its own path. The host environment must provide:

- **`harv` and `kepmodel` importable**, plus `h5py`, `numpy`, `jax`, `numpyro`,
  `equinox`, `unxt` (all already harv dependencies). In this repo both are git
  dependencies in the root `pyproject.toml` (`harv` from the `periodogram` branch,
  `kepmodel` from gitlab, which brings `spleaf`), so a plain `uv run` is enough and
  the old `requirements.txt` is gone. **kepmodel and spleaf are EUPL-1.2** and must
  still never enter a harv `pyproject.toml`; they belong to the experiment only.
- **float64.** `kepcmp/__init__.py` enables it at import. Do not bypass that import.
- Nothing else. `PY_CMD="python -m"` replaces the invocation if `uv` is not wanted.

**It reaches into harv internals**, which is the fragile coupling to watch if harv
moves. Deliberate — the identity gate must decompose *harv's own* matrices, not a
reimplementation that could drift:

| private API | used for |
|---|---|
| `AbstractComponentModel._build_marg_blocks` | `X`, `y`, `cov`, `prior_mu`, `prior_scale_tril` |
| `AbstractComponentModel._full_design_matrix` | batched design matrices over the grid |
| `AbstractComponentModel._auto_marginalized_names` | matching `log_prob`'s auto path |
| `AbstractComponentModel._all_linear_names` | column ordering |
| `harv.samplers._prior_resolution.effective_linear_prior_from_prior` | prior resolution |
| `harv.periodogram.core._effective_n_terms` | overfitting-cap test only |

If any of those change signature, `tests/test_identity.py` and
`tests/test_linalg.py` fail loudly rather than silently producing wrong numbers —
that is what the three-residual gate is for.

### Numerical behaviour of the two paths

Worth recording because it mirrors the small-`n` result in numerical form. Batched vs
scalar relative divergence, same data, varying only `n_terms`:

| `H` | `cond` | `chi2` (profile) | `ln_z` (marginal) |
|---|---|---|---|
| 1 | 2.4e3 | 4e-15 | 3e-16 |
| 2 | 7.8e6 | 4.7e-12 | 2.5e-16 |
| 3 | 2.4e10 | 4.2e-08 | **3.3e-16** |

The marginal quantities route through `M = I + B^T B >= I` and are accurate to ~3e-16
*regardless of conditioning*. Only the profile quantities degrade, linearly in `cond`,
because they need a pseudo-inverse of a near-singular matrix. Tolerances therefore
differ by path, and by *whose* code is involved:

- our code vs harv, and our own algebra: conditioning-independent, tight bar.
- our SVD vs our lstsq (batched vs scalar): error ~`eps * cond`.
- our code vs kepmodel: kepmodel forms the normal matrix explicitly, which *squares*
  the conditioning, so error ~`eps * cond**2`. Predicts 2.1e-9 at `cond = 3.1e3`
  against a measured 1.1e-11, and 1.3e-9 at `cond = 2.4e3` against a measured 1.1e-8.

A wiring bug is O(1) — the `2 pi` negative test lands far above every bar — so none of
this hides one.

**The reference-compute risk did not materialise.** The 1.4e9-evaluation estimate
assumed a naive loop; vmapping harv's analytic linear marginalization over draws and
`lax.map`-ing over periods makes it 6.5 s/sim at `n_mc = 4096`. Better, common random
numbers make the *shape* converge almost immediately: the curve range is stable at
~1126 nats from `n_mc = 256` to 4096, and shape is all `calibrate` reads. `n_mc` in
the low thousands is ample; the slice does not need shrinking.

### Findings so far

**RV only, and not from the full grid.** No Gaia science has been run --- the Gaia
adapter is gated and smoke-tested, nothing more. These come from a 36-simulation
RV calibration slice
(`P/T_span` 0.1–1.0, `SNR` 3–30, `e` 0 and 0.6, `n_obs = 40`, 2 seeds) plus targeted
sparse-`n` measurements. Both ends of both headline axes — `SNR = 1` and `100`,
`P/T_span = 0.02`, `3`, `10` — are unsampled, so nothing below is a regime map.

Sparse-data results are in "Component 2" above; the rest:

- *Prediction confirmed.* `Delta` and `z0` agree on peak location 98.9% of the time,
  and disagreement tracks amplitude constraint exactly as predicted:
  `ln_amp_constraint` is 3.40 nats where they agree vs 0.26 where they do not.
  Scored against the *injected* period, both are equally accurate (median `|d ln P|`
  0.0130 vs 0.0122, both below one grid spacing ~0.027).
- *`sigma_amp` guidance.* TV distance to the reference saturates at
  `sigma_amp >~ 1 x` data RMS (0.078 at 1x, 5.6x, 31.6x — identical) and degrades
  below it (0.133 at 0.18x, 0.463 at 0.03x). So the answer to the open question is a
  floor, not a value: at or above the data RMS, `Delta` is effectively
  amplitude-prior-free. Needs the full grid, and especially the partial-arc cells,
  before it goes into `docs/spec.md`.
- *The harmonics prediction holds on both halves.* TV to the reference, at
  `sigma_amp = 1x` RMS:

  | | e = 0.0 | e = 0.6 |
  |---|---|---|
  | H1 | **0.000** | 0.657 |
  | H2 | 0.282 | **0.111** |
  | H3 | 0.539 | **0.051** |
  | z0 (d=2) | 0.000 | 0.657 |

  H1 and `z0` are identical at both eccentricities, so at one harmonic the whole
  discrepancy is basis error and none of it is prior error — which is what the
  three-way decomposition was built to separate. Extra harmonics cost real accuracy
  on circular orbits and buy a lot on eccentric ones, and the marginal statistic
  prices that tradeoff automatically while `z0` is stuck at `d = 2`.
- *Shrinkage is not a small correction* across the `sigma_amp` sweep (median
  frequency-range 14.5 nats), contrary to this document's earlier expectation. It is
  negligible only in the wide-prior limit; at 0.03x RMS the prior dominates the
  likelihood. Worth keeping in the artifact rather than dropping as second-order.
- *The ROC is not yet informative.* At the reference slice's SNRs (3, 10, 30 with
  `n_obs = 40`) every statistic detects everything: TPR = 1.000 for all arms. It
  needs the `snr = 1` cells and >~100 nulls per `n_obs` — the reduction already warns
  when the null count cannot resolve the requested FPR.
- *Prior quality needs more prior samples to discriminate.* The periodogram arms find
  a mode ~50 nats better than the log-uniform control (`max_lnL` -45.3 vs -96.0), but
  at 50k prior samples every arm accepts ~1 sample and evidence ESS is ~2.7, i.e.
  under-resolved. This is exactly the regime `docs/spec.md` describes: the rejection
  stage is a mode-finder here, so `max_lnL` is the comparable metric and `n_acc` is
  not. A discriminating run needs far more prior samples or the MCMC continuation.

## Risks and open items

- ~~**Reference compute is the one real risk.**~~ Resolved by measurement — see
  "Measured status" above. Kept here for the record: the original estimate was
  ~1.4e9 small marginal-likelihood evaluations, which assumed a naive loop.
- **License.** kepmodel and spleaf are EUPL-1.2, unlike the rest of the stack. They
  enter as an optional dependency group used only by this experiment, never by
  harv's runtime, so there is no distribution concern — recorded here so it is not
  rediscovered later.
- **Version pinning.** Results are tied to a kepmodel/spleaf pair, and the artifact
  records both (`kepmodel_version`, `spleaf_version` root attrs). Currently kepmodel
  1.0.9 (gitlab `b003497`) and spleaf 2.1.17. The earlier `requirements.txt` pin of
  kepmodel 1.0.8 from PyPI is gone; it would have silently installed *over* the git
  checkout. The `power == 1 - chi2_K/chi2_H` assertion in `kepmodel_chi2ogram` is what
  catches a version whose statistic convention changed, for `RvModel` and `AstroModel`
  alike.

## Out of scope

- Any change to `harv.periodogram`'s public API. Findings feed `docs/spec.md` first.
- Correlated-noise comparison. harv's periodogram cannot marginalise a GP, so an
  spleaf-vs-harv noise comparison is a different experiment.
- Iterative multi-planet search. kepmodel's base model is the residuals after
  already-fitted Keplerians; harv has no equivalent. Noted as a capability gap, not
  tested here.
