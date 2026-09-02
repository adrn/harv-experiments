#!/bin/zsh -l
#SBATCH -J ampcal
#SBATCH -o slurm/logs/ampcal-%j.o
#SBATCH -e slurm/logs/ampcal-%j.e
#SBATCH -N 6
# --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -p cca
#SBATCH --constraint=genoa
#
# One grid, as a single job.
#
#   sbatch experiments/amplitude-prior/slurm/run_grid.sh
#   AMPCAL_ADAPTER=gaia sbatch experiments/amplitude-prior/slurm/run_grid.sh
#
# Knobs, all AMPCAL_-prefixed (see the note by `set -euo pipefail` for why):
#   AMPCAL_ADAPTER        rv | gaia            (default rv)
#   AMPCAL_OUT_ROOT       artifacts go to $AMPCAL_OUT_ROOT/$ADAPTER
#   AMPCAL_SMOKE          1 to run the tiny end-to-end pass
#   AMPCAL_REFERENCE_N_MC reference MC draws   (default 2048)
#   AMPCAL_REPO           checkout to cd into
#
# `ampcal.run` is SPMD: each rank takes its own share of the simulation list and writes
# its own `<name>.<adapter>.rank<NNN>.h5`. Ranks never communicate during the work, so
# this needs no MPI-IO and no parallel HDF5 -- but it does need the ranks to share a
# filesystem for OUT, because the merge at the end reads all of them.
#
# The adapter is in the shard filename as well as the directory. Belt and braces: the
# directory split is now unoverridable (see AMPCAL_OUT_ROOT), and even if two runs did
# somehow share a directory they would no longer write the same files.
#
# Sizing. RV is 420 cells x 16 seeds = 6,720 simulations; Gaia is 336 x 16 = 5,376.
# Measured on the predecessor harness: ~0.65 sims/s saturating at 8 processes per node
# (16 bought nothing and made each 4x slower), 4.4-12 s/sim, dominated by the reference
# MC. This grid runs 24 arms per simulation where that one ran 15, so budget ~1.3x its
# per-simulation cost. Check the three numbers rank 0 prints at the end before scaling
# up: `balance` below ~85% means the deal is the limit rather than the compute, and
# `peak RSS` x ntasks-per-node has to fit in a node.
#
# AMPCAL_SMOKE=1 runs the whole chain on a handful of cells first. Do that before the real
# launch; it costs a minute and it is the only thing that exercises launch -> shards ->
# merge -> reduce -> report end to end.

set -euo pipefail

# Every knob is AMPCAL_-prefixed, and that prefix is load-bearing.
#
# This runs under `zsh -l`, a LOGIN shell, so each job sources ~/.zshenv, ~/.zprofile
# and ~/.zshrc; `sbatch` also defaults to --export=ALL, so the submitting environment
# arrives too. An earlier version read plain `OUT`, `REPO`, `ADAPTER`, `SMOKE`. Those
# are not defaults, they are fallbacks that any same-named variable anywhere in that
# environment silently wins -- and it did: `OUT` was already set to `output/rv`, which
# does not depend on $ADAPTER, so the RV and Gaia grids were both sent to the same
# directory. Same rank count, same `signal.rank000.h5`, one grid written over the other.
#
# Names nothing else plausibly uses, plus the provenance echo below, plus deriving OUT
# rather than accepting it whole (see AMPCAL_OUT_ROOT).
for _v in AMPCAL_REPO AMPCAL_ADAPTER AMPCAL_OUT_ROOT AMPCAL_SMOKE AMPCAL_REFERENCE_N_MC
do
    if [[ -n "${(P)_v:-}" ]]; then
        echo "config: $_v=${(P)_v}  <- INHERITED from the environment"
    else
        echo "config: $_v  <- unset, using the script default"
    fi
done

REPO="${AMPCAL_REPO:-$HOME/work/harv-experiments}"
cd "$REPO"
source .venv/bin/activate

EXP="experiments/amplitude-prior"
# PREPEND, never assign. mpi4py is the site build (a nix-store view on PYTHONPATH, not a
# venv package), which is what we want -- it is compiled against the same MPI this job's
# `mpirun` comes from, unlike a PyPI wheel, which vendors its own libmpi and would
# silently initialise every rank as a singleton. Overwriting PYTHONPATH here therefore
# removes mpi4py from every rank, and the job dies one traceback per rank in
# `mpi_context`. Absolute path so it survives any rank whose cwd differs.
export PYTHONPATH="$REPO/$EXP${PYTHONPATH:+:$PYTHONPATH}"

# Fail in one second, not after mpirun has started several hundred ranks.
python -c "import mpi4py; print(f'mpi4py {mpi4py.__version__} from {mpi4py.__file__}')"

ADAPTER="${AMPCAL_ADAPTER:-rv}"
SMOKE="${AMPCAL_SMOKE:-0}"
REFERENCE_N_MC="${AMPCAL_REFERENCE_N_MC:-2048}"

# OUT is DERIVED, never accepted whole: the adapter subdirectory is always appended, so
# no environment can put two grids in one directory even by accident. Override the root
# to move the artifacts somewhere roomier; you cannot override away the per-adapter
# split, which is the thing that broke.
#
# Absolute, so the merge globs and every rank agree regardless of cwd. The checkout is
# already on a filesystem every rank can see -- they import ampcal from it -- so this
# needs no configuration.
# Sizing: ~1.5 MB/simulation for RV and ~0.8 MB for Gaia at full grid density with 24
# arms, so the grid is ~10 GB (RV) or ~4 GB (Gaia).
OUT_ROOT="${AMPCAL_OUT_ROOT:-$REPO/$EXP/output}"
OUT="$OUT_ROOT/$ADAPTER"
mkdir -p "$OUT"

# One BLAS thread per rank: the design matrices are ~80 x 17, so BLAS threads buy
# nothing and with tens of ranks per node they oversubscribe the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
# XLA's CPU thread pool does NOT honour OMP_NUM_THREADS and takes ~2.4 cores per rank by
# default. This is the only flag that touches it; note that the frequently-copied
# `intra_op_parallelism_threads=1` companion is NOT a real XLA flag -- spelled with `--`
# it aborts the process, and spelled without it is silently skipped.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

date
echo "adapter=$ADAPTER  out=$OUT  smoke=$SMOKE"

# The correctness gate, on a *period-dependent* arm, before anything expensive. This is
# the check that the batched per-frequency prior agrees with harv.periodogram and with
# its own algebra; a grid run without it can produce a full artifact of wrong Occam
# terms and no error.
python -m ampcal.identity --adapter "$ADAPTER" --n-obs 40

if [[ "$SMOKE" == "1" ]]; then
    python -m ampcal.run --adapter "$ADAPTER" --which smoke --out "$OUT/signal.h5" \
        --stride 16 --n-seeds 2 --reference-n-mc 256
else
    # Clear THIS ADAPTER's shards from any earlier run before launching. The merge globs
    # and trusts what it finds, so a previous run at a different rank count leaves orphans
    # the glob picks up. Scoped to $ADAPTER: the shard names carry the adapter precisely
    # so two adapters sharing an OUT cannot collide, and a blanket rm would undo that.
    rm -f "$OUT"/signal."$ADAPTER".rank*.h5

    mpirun python -m ampcal.run \
        --adapter "$ADAPTER" --which signal --out "$OUT/signal.h5" \
        --reference-n-mc "$REFERENCE_N_MC"
    # Merge is serial and cheap -- one process, no mpirun. It names any shard it cannot
    # read rather than dying on an opaque h5py traceback; `--allow-partial` is the
    # deliberate override, never the default.
    python -m ampcal.merge --out "$OUT/signal.h5" "$OUT"/signal."$ADAPTER".rank*.h5
fi

python -m ampcal.reduce.calibrate \
    --artifact "$OUT/signal.h5" --csv "$OUT/calibrate.csv"

# The report bundle: the thing harv actually ingests. Runs the reduction, writes
# REPORT.md / findings.json / tables / figures. Minutes, single process.
python -m ampcal.report --adapter "$ADAPTER" --artifact-dir "$OUT" \
    --out "$REPO/$EXP/report/$ADAPTER"

# Once BOTH adapters have run, cross them and emit the harv-facing documents:
#   python -m ampcal.report.synthesis report/rv report/gaia --out report/
date
