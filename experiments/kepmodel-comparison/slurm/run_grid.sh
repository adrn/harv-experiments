#!/bin/zsh -l
#SBATCH -J kepcmp
#SBATCH -o slurm/logs/kepcmp.o
#SBATCH -e slurm/logs/kepcmp.e
#SBATCH -N 6
# --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -p cca
#SBATCH --constraint=genoa
#
# One grid, both phases, as a single job.
#
#   sbatch experiments/kepmodel-comparison/slurm/run_grid.sh
#   ADAPTER=gaia sbatch experiments/kepmodel-comparison/slurm/run_grid.sh
#
# `kepcmp.run` is SPMD: each rank takes its own share of the simulation list and
# writes its own `<name>.rank<NNN>.h5`. Ranks never communicate during the work, so
# this needs no MPI-IO and no parallel HDF5 -- but it does need the ranks to share a
# filesystem for OUT, because the merge at the end reads all of them.
#
# Sizing. RV is 450 cells x 16 seeds = 7,200 simulations at ~7 s each, so ~14
# core-hours for the signal phase; Gaia is 360 x 16 = 5,760. 128 ranks is therefore
# minutes of compute, and the walltime above is mostly slack for the null phase and
# for a slow filesystem. Check the three numbers rank 0 prints at the end before
# scaling up: `balance` below ~85% means the deal is the limit rather than the
# compute, and `peak RSS` x ntasks-per-node has to fit in a node.

set -euo pipefail

REPO="${REPO:-$HOME/work/harv-experiments}"
cd "$REPO"
source .venv/bin/activate

EXP="experiments/kepmodel-comparison"
# PREPEND, never assign. mpi4py is the site build (a nix-store view on PYTHONPATH,
# not a venv package), which is what we want -- it is compiled against the same MPI
# this job's `mpirun` comes from, unlike a PyPI wheel, which vendors its own libmpi
# and would silently initialise every rank as a singleton. Overwriting PYTHONPATH
# here therefore removes mpi4py from every rank, and the job dies one traceback per
# rank in `mpi_context`. Absolute path so it survives any rank whose cwd differs.
export PYTHONPATH="$REPO/$EXP${PYTHONPATH:+:$PYTHONPATH}"

# Fail in one second, not after mpirun has started several hundred ranks.
python -c "import mpi4py; print(f'mpi4py {mpi4py.__version__} from {mpi4py.__file__}')"

ADAPTER="${ADAPTER:-rv}"
# Absolute, so the merge globs and every rank agree regardless of cwd. The checkout is
# already on a filesystem every rank can see -- they import kepcmp from it -- so this
# needs no configuration; set OUT to point somewhere roomier if you'd rather.
# Sizing: ~1.0 MB/simulation for RV and ~0.5 MB for Gaia at full grid density, so the
# signal phase is ~7 GB (RV) or ~3 GB (Gaia).
OUT="${OUT:-$REPO/$EXP/output/$ADAPTER}"
REFERENCE_N_MC="${REFERENCE_N_MC:-2048}"
NULL_SEEDS="${NULL_SEEDS:-1000}"
mkdir -p "$OUT"

# One BLAS thread per rank: the design matrices are ~80 x 17, so BLAS threads buy
# nothing and with tens of ranks per node they oversubscribe the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
# XLA's CPU thread pool does NOT honour OMP_NUM_THREADS and takes ~2.4 cores per rank
# by default. This is the only flag that touches it; note that the frequently-copied
# `intra_op_parallelism_threads=1` companion is NOT a real XLA flag -- spelled with
# `--` it aborts the process, and spelled without it is silently skipped.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

date
echo "adapter=$ADAPTER  out=$OUT"

mpirun python -m kepcmp.run \
    --adapter "$ADAPTER" --which signal --out "$OUT/signal.h5" \
    --reference-n-mc "$REFERENCE_N_MC"

mpirun python -m kepcmp.run \
    --adapter "$ADAPTER" --which null --out "$OUT/null.h5" \
    --n-seeds "$NULL_SEEDS" --n-terms 1 2 --sigma-amp-mults 1.0

# Merge is serial and cheap -- one process, no mpirun.
python -m kepcmp.merge --out "$OUT/signal.h5" "$OUT"/signal.rank*.h5
python -m kepcmp.merge --out "$OUT/null.h5"   "$OUT"/null.rank*.h5

python -m kepcmp.reduce.regime_map \
    --artifact "$OUT/signal.h5" --null-artifact "$OUT/null.h5" \
    --csv "$OUT/regime_map.csv"
date
