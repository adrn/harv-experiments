# Handoff — start here

Written 2026-09-01 (supersedes the 2026-08-12 version). `README.md` next to this file
is the design document and the detailed measured status; this file is the shorter
"where we are and what to do next". Read this first, then the README sections it
points at.

## State of play

The harness is **built, tested, and ready to run, for both RV and Gaia epoch
astrometry**. **No full grid has been run for either.** 50 tests pass (31 RV, 10 Gaia,
9 MPI/deal).

Since the last handoff:

- The experiment moved out of the harv repo into `harv-experiments`. `harv` (from the
  `periodogram` branch on GitHub) and `kepmodel` (from gitlab, which brings `spleaf`)
  are now git dependencies in the root `pyproject.toml`. **`requirements.txt` is
  deleted** — it pinned kepmodel 1.0.8 from PyPI and would have installed over the git
  checkout. Every command is now a plain `uv run`.
- The harness is multi-adapter. `--adapter rv` (default) or `--adapter gaia` on
  `kepcmp.run` and `kepcmp.identity`; `ADAPTER=gaia` for the launcher. `Cell` is
  unitless and each adapter owns its own axes, units and grid geometry.
- **The Gaia adapter exists and its identity gate passes** at four grid corners. See
  README "The Gaia adapter" for the three corrections it forced on the design.
- The runner is **SPMD over mpi4py** (`mpirun`/`srun`), with work dealt
  longest-processing-time-first and one `gather` at the end. `launch_full_grid.sh` is
  gone; `slurm/run_grid.sh` replaces it. See README "Running the experiment".

What has actually been *measured* is unchanged from the last handoff: a
**36-simulation RV calibration slice** (`P/T_span` 0.1–1.0, `SNR` 3–30, `e` ∈ {0, 0.6},
`n_obs = 40`, 2 seeds) plus targeted sparse-`n` probes. **Nothing measured so far is a
regime map, and there is no Gaia science at all.** Do not present it as one.

## Next actions, in order

1. **Run the RV grid.** Smoke first (~15 s: `kepcmp.run --which smoke --stride 8
   --max-sims 8`), then `sbatch slurm/run_grid.sh`. ~14 core-hours for the signal
   phase, so minutes across nodes — but budget **~3 h** if you run it on one 16-core
   machine, not the ~26 min an earlier README claimed; see README "Timing" for why
   ranks on one host do not scale linearly. Commands in README "Running the
   experiment".
2. **Merge shards, then run `kepcmp.reduce.regime_map`.** That is the deliverable that
   answers "which periodogram is better in which regime".
3. **Run the Gaia grid** the same way into a *separate* directory. Never merge the two
   — the axes mean different things and `merge.py` will not stop you, because the
   frequency grids happen to be compatible in shape.
4. **The Gaia headline is the `P = 1 yr` column**, not a repeat of the RV comparison.
   `docs/spec.md` claims `Delta` suppresses parallax/proper-motion power by base-column
   cancellation; `z0` shares that cancellation, so agreement between the two *is* the
   result there. Read that column first.
5. **`prior_quality` separately, with far more prior samples.** It is not in the
   launcher. At 50k prior samples every arm was under-resolved (evidence ESS ~2.7, ~1
   accepted sample), so it could not discriminate.
6. **Only then** consider whether the `sigma_amp` finding is solid enough for
   `docs/spec.md` (it is currently a 36-sim, high-SNR, `n_obs=40`, RV-only result).

## Things a new instance will get wrong

These are the traps. Each cost real debugging time.

- **kepmodel is NOT the reference.** Truth is `R`, the Keplerian marginal-likelihood
  statistic, with a deliberately wide amplitude prior so it carries no opinion about
  amplitudes. kepmodel's `z0` is a *graded* statistic with its own row in every table.
  kepmodel is authoritative in exactly two places, neither scientific: the identity
  gate (as an independent implementation of the *profile* statistic) and the FAP test
  (against its own published calibration).
- **float64 is mandatory.** harv does not enable it; JAX defaults to float32 with
  `eps ~ 1e-7`. `kepcmp/__init__.py` enables it at import — never bypass that import.
- **Tolerances differ by code path, and that is not a fudge.** Marginal quantities
  route through `M = I + B^T B >= I` and are accurate to ~3e-16 *regardless of
  conditioning*. Profile quantities degrade as `eps * cond` (our SVD vs lstsq) and
  `eps * cond**2` (us vs kepmodel, who form the normal matrix explicitly). A single
  fixed tolerance fails on correct code. A real wiring bug is O(1) — the `2 pi`
  negative test proves the gate can still fail.
- **`z0` stops existing at `n_obs <= p + d`, and `p + d` differs by data type**: 3 for
  RV, **9 for Gaia**. `kepmodel_z0` returns `nan` plus a reason; **every reduction must
  gate on `ArtifactReader.z0_usable`**. Two reductions were bitten by this already.
- **`sigma_amp` is quoted against the base-model residual RMS, not the raw data RMS.**
  Identical for RV (the base is one constant column, so it is mean subtraction —
  verified bit-for-bit against the old harness). For Gaia they differ by ~50x, because
  the raw along-scan scatter is parallax and proper motion. Do not "simplify" this back
  to `np.std(y)`.
- **The Gaia simulator's default `parallax_factor` is `U(-1, 1)` white noise.**
  `GaiaAdapter.simulate` builds times, scan angles and parallax factors itself and
  passes them in explicitly. Take that out and the parallax column carries no 1-yr
  structure, and the headline Gaia axis silently tests nothing.
- **harv's default Thiele-Innes prior is period-dependent** and its proper-motion prior
  is parallax-dependent. Right for analysis, wrong for `R`. `reference_prior` overrides
  all nine linear priors with wide plain Normals; `science_prior` keeps the defaults.
- **`sigma_a0` is a physical length (AU), not an angle.** The period-dependent prior is
  `sigma_a0 * (P/P0)**(2/3) * parallax`, so the parallax converts to mas. Passing mas
  constructs a perfectly well-formed prior and then raises `UnitConversionError` deep
  inside the sampler — it cost a `prior_quality` run to find. `sigma_ti_reference` (mas)
  and `sigma_a0_science` (AU) are separate fields for exactly this reason; do not
  collapse them. `ThieleInnesGaiaAstrometry.default_prior` also *requires* `sigma_vtan`
  (a velocity), and raises without it.
- **Only `prior_quality` calls `science_prior`, and it is not in the launcher.** So
  nothing else exercises that path: both traps above survived a green test suite and a
  full smoke run. Run `reduce.prior_quality --limit 1` after touching an adapter.
- **`Delta` "winning" at `n_obs` below the profile threshold is not evidence it is
  right.** For RV it peaks ~30x short, on prior-and-window structure rather than
  signal. It wins those cells partly because `z0` is simply absent.
- **`peak_err` is meaningless past identifiability** (`P/T_span >= 3` for RV,
  `>= 2` for Gaia). `regime_map` flags it; only the TV metric is informative there.
- **`docs/spec.md` in the harv repo stays authoritative for harv itself.** Nothing in
  this experiment changes `harv.periodogram` semantics. If a result argues for a
  change, the change goes through `docs/spec.md` first.

## Decisions already made — do not relitigate

- **Verdict metric: all three, side by side** (`tv`, `peak_err`, `tpr`), not collapsed
  into one. They disagree in places and that is informative.
- **Head-to-head arms:** `delta_default` (harv `n_terms=2`) vs `z0`, plus `delta_h1` as
  a control whose columns match kepmodel's exactly. `delta_h1` vs `z0` isolates the
  *prior* effect; `delta_default` vs `delta_h1` isolates the *basis* effect.
  Best-config-per-cell was rejected as an oracle choice.
- **RV grid:** `P/T_span` ∈ {0.02, 0.1, 0.3, 1.0, 3.0, 10}, `SNR` ∈ {1, 3, 10, 30, 100},
  `e` ∈ {0, 0.3, 0.6}, `n_obs` ∈ {2, 4, 8, 16, 32}, 16 seeds = 450 cells, 7200 sims.
  Small `n_obs` is a primary harv use case, not an edge case.
- **Gaia grid:** `P/T_span` ∈ {0.05, 0.1, **0.2**, 0.4, 1.0, 2.0} at `T_span = 5 yr`
  (so `P` = 0.25, 0.5, **1**, 2, 5, 10 yr), `SNR` ∈ {1, 3, 10, 30, 100},
  `e` ∈ {0, 0.3, 0.6}, `n_obs` ∈ {10, 20, 40, 80}, 16 seeds = 360 cells, 5760 sims.
  `n_obs` does not go below 10 because `z0` needs `n_obs > 9` — below that both
  statistics are undefined and the comparison has no content.
- **The reference is computed on the whole grid, not a slice.** Measured at ~5 s/sim
  for RV and cheaper for Gaia (2-D MC), so the design's compute tiering is obsolete;
  `enumerate_reference_cells` and `--which reference` were deleted.
- **More ranks on one host is not faster past ~cores/2.** Each rank draws ~2.4 cores
  from XLA's CPU thread pool; `OMP_NUM_THREADS` does not control it, and
  `--intra_op_parallelism_threads` is **not a real XLA flag** — with `--` it aborts the
  process, without `--` it is silently skipped. Only
  `--xla_cpu_multi_thread_eigen=false` is real. Measured throughput on 16 cores: 0.54
  sims/s at 4 processes, 0.65 at 8, 0.64 at 16. Size `--ntasks-per-node` accordingly.
- **`import mpi4py.MPI` runs `MPI_Init`, and on a broken MPI it HANGS rather than
  raising.** So `kepcmp.mpi.mpi_context` imports it only when a launcher variable is
  set (or `--mpi`). Do not "simplify" that to a plain top-level import: it cost a
  10-minute hang on a *serial* run, and `tests/test_mpi.py` pins it.
- **On a cluster, build mpi4py against the site MPI** (`MPICC=$(which mpicc) uv pip
  install --no-binary mpi4py mpi4py`). The PyPI wheel vendors its own libmpi/libpmix,
  silently initialises every rank as a singleton under a system `mpirun`, and then
  segfaults in `PMIx_Finalize`.

## Ground rules from Adrian

- **This experiment stays out of git** while it is exploratory. Do not `git add` or
  commit it, including this file — ask first.
- Design docs live next to the code they describe, not in a `docs/superpowers/` tree.

## Where things are

| what | where |
|---|---|
| design + detailed status | `README.md` |
| the math (`Delta = z0 - Occam - Shrinkage`) | `README.md` "Background" |
| what the Gaia case changed | `README.md` "The Gaia adapter" |
| exact-identity gate | `kepcmp/identity.py` |
| marginal/profile decomposition | `kepcmp/linalg.py` |
| the adapter seam | `kepcmp/adapters/{base,common,rv,gaia}.py` |
| grid cells and shared knobs | `kepcmp/grid.py` |
| runner / artifact / shard merge | `kepcmp/{run,artifact,merge}.py` |
| the five reductions | `kepcmp/reduce/` |
| SPMD plumbing and the work deal | `kepcmp/mpi.py` |
| cluster job (both phases, merge, reduce) | `slurm/run_grid.sh` |
| tests (50) | `tests/` |
| standalone Gaia feasibility probe | `gaia_probe.py` |

## How to run anything

```bash
EXP=path/to/kepmodel-comparison
PYTHONPATH=$EXP uv run python -m kepcmp.identity --adapter gaia
PYTHONPATH=$EXP uv run python -m pytest $EXP/tests -q
```

The directory is location-independent. It needs `harv` and `kepmodel` importable; see
README "Porting this code" for the private harv APIs it depends on, which is the
coupling to watch.
