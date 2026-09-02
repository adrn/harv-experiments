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
#   ADAPTER=gaia sbatch experiments/amplitude-prior/slurm/run_grid.sh
#
# `ampcal.run` is SPMD: each rank takes its own share of the simulation list and writes
# its own `<name>.rank<NNN>.h5`. Ranks never communicate during the work, so this needs
# no MPI-IO and no parallel HDF5 -- but it does need the ranks to share a filesystem for
# OUT, because the merge at the end reads all of them.
#
# Sizing. RV is 420 cells x 16 seeds = 6,720 simulations; Gaia is 336 x 16 = 5,376.
# Measured on the predecessor harness: ~0.65 sims/s saturating at 8 processes per node
# (16 bought nothing and made each 4x slower), 4.4-12 s/sim, dominated by the reference
# MC. This grid runs 24 arms per simulation where that one ran 15, so budget ~1.3x its
# per-simulation cost. Check the three numbers rank 0 prints at the end before scaling
# up: `balance` below ~85% means the deal is the limit rather than the compute, and
# `peak RSS` x ntasks-per-node has to fit in a node.
#
# SMOKE=1 runs the whole chain on a handful of cells first. Do that before the real
# launch; it costs a minute and it is the only thing that exercises launch -> shards ->
# merge -> reduce -> report end to end.

set -euo pipefail

REPO="${REPO:-$HOME/work/harv-experiments}"
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

ADAPTER="${ADAPTER:-rv}"
SMOKE="${SMOKE:-0}"
# Absolute, so the merge globs and every rank agree regardless of cwd. The checkout is
# already on a filesystem every rank can see -- they import ampcal from it -- so this
# needs no configuration; set OUT to point somewhere roomier if you'd rather.
# Sizing: ~1.5 MB/simulation for RV and ~0.8 MB for Gaia at full grid density with 24
# arms, so the grid is ~10 GB (RV) or ~4 GB (Gaia).
OUT="${OUT:-$REPO/$EXP/output/$ADAPTER}"
REFERENCE_N_MC="${REFERENCE_N_MC:-2048}"
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
    # Clear shards from any earlier run BEFORE launching. The merge globs
    # `signal.rank*.h5` and trusts what it finds, so a previous run at a different rank
    # count leaves orphans the glob happily picks up -- and an orphan of a killed run is
    # exactly the kind of file that is truncated. This run is about to overwrite the
    # ones it owns anyway; the only files this deletes are ones nothing else will.
    rm -f "$OUT"/signal.rank*.h5

    mpirun python -m ampcal.run \
        --adapter "$ADAPTER" --which signal --out "$OUT/signal.h5" \
        --reference-n-mc "$REFERENCE_N_MC"
    # Merge is serial and cheap -- one process, no mpirun. It names any shard it cannot
    # read rather than dying on an opaque h5py traceback; `--allow-partial` is the
    # deliberate override, never the default.
    python -m ampcal.merge --out "$OUT/signal.h5" "$OUT"/signal.rank*.h5
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
