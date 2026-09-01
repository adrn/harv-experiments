#!/usr/bin/env bash
# Launch a full grid, sharded across processes. Intended for a workstation.
#
# Two backends, same result --- N processes each writing `<name>.rank<NNN>.h5`, merged
# afterwards by `kepcmp.merge`:
#
#   MPI (default when `mpirun` is on PATH): one `mpirun -n N` per phase. The ranks
#   never communicate; `kepcmp.run` reads its rank from the launcher's environment
#   (OMPI_COMM_WORLD_RANK / PMI_RANK / PMIX_RANK / SLURM_PROCID), so nothing links
#   libmpi and a broken MPI runtime cannot take the run down at MPI_Init.
#
#   Background processes (fallback, or MPI=0): the same split via explicit
#   --shard/--n-shards.
#
# Each process is pinned to a single thread on purpose. The linear algebra is tiny
# (n_obs <= 80, k <= 17), so it does not parallelize within a process; N single-threaded
# shards beat N shards fighting over BLAS threads.
#
# Nulls run only the two configs the regime map calibrates against (H1_s1, H2_s1),
# which is what keeps thousands of them cheap. FPR=0.01 needs >~1000 nulls per n_obs.
#
# Usage
# -----
#   # 1. Validate the plumbing first (a handful of sims, ~1 min):
#   SMOKE=1 bash launch_full_grid.sh scratch/kepcmp/smoketest
#
#   # 2. Then the real thing:
#   NPROC=32 bash launch_full_grid.sh scratch/kepcmp/rv_full
#   ADAPTER=gaia NPROC=32 bash launch_full_grid.sh scratch/kepcmp/gaia_full
#
#   # 3. Merge shards and reduce:
#   export PYTHONPATH=<this directory>
#   uv run python -m kepcmp.merge --out scratch/kepcmp/rv_full/signal.h5 \
#        scratch/kepcmp/rv_full/signal.rank*.h5
#   uv run python -m kepcmp.merge --out scratch/kepcmp/rv_full/null.h5 \
#        scratch/kepcmp/rv_full/null.rank*.h5
#   uv run python -m kepcmp.reduce.regime_map --artifact scratch/kepcmp/rv_full/signal.h5 \
#        --null-artifact scratch/kepcmp/rv_full/null.h5 \
#        --csv scratch/kepcmp/rv_full/regime_map.csv
#
# Environment
# -----------
#   ADAPTER        rv | gaia                   (default: rv)
#   NPROC          processes per phase         (default: cores, capped at 32)
#   MPI            1 to force mpirun, 0 to force background processes
#   N_SEEDS        signal seeds                (default: 16)
#   NULL_SEEDS     null seeds per n_obs        (default: 1000)
#   REFERENCE_N_MC MC draws for the reference  (default: 2048; shape converges by ~256)
#   PY_CMD         replace the python invocation, e.g. PY_CMD="python -m"
#   SMOKE          if set, 2 sims per shard --- plumbing validation only
set -euo pipefail

OUTDIR="${1:-scratch/kepcmp/full}"
ADAPTER="${ADAPTER:-rv}"

# Resolve our own location rather than assuming a path inside any particular repo, so
# this directory can be copied elsewhere unchanged.
EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$EXPDIR/kepcmp" ]]; then
  echo "error: cannot find kepcmp/ next to this script ($EXPDIR)" >&2
  exit 1
fi

if command -v nproc >/dev/null 2>&1; then
  CORES="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
  CORES="$(sysctl -n hw.ncpu)"
else
  CORES=4
fi
NPROC="${NPROC:-$(( CORES < 32 ? CORES : 32 ))}"
[[ "$NPROC" -lt 1 ]] && NPROC=1

N_SEEDS="${N_SEEDS:-16}"
NULL_SEEDS="${NULL_SEEDS:-1000}"
REFERENCE_N_MC="${REFERENCE_N_MC:-2048}"

if [[ -z "${MPI:-}" ]]; then
  if command -v mpirun >/dev/null 2>&1; then MPI=1; else MPI=0; fi
fi

mkdir -p "$OUTDIR" "$OUTDIR/logs"

export PYTHONPATH="$EXPDIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

if [[ -n "${PY_CMD:-}" ]]; then
  read -ra RUN <<<"$PY_CMD"
else
  RUN=(uv run python -m)
fi

LIMIT_ARGS=()
if [[ -n "${SMOKE:-}" ]]; then
  LIMIT_ARGS=(--limit 2)
  N_SEEDS=1
  NULL_SEEDS="$NPROC"
  echo "SMOKE mode: 2 sims per shard, plumbing validation only"
fi

# run_phase <name> <extra args...>
run_phase() {
  local name="$1"; shift
  echo "--- $name: $NPROC processes ($([[ $MPI == 1 ]] && echo mpirun || echo background))"
  if [[ "$MPI" == "1" ]]; then
    mpirun -n "$NPROC" --oversubscribe \
      "${RUN[@]}" kepcmp.run --adapter "$ADAPTER" --out "$OUTDIR/$name.h5" "$@" \
      >"$OUTDIR/logs/$name.log" 2>&1
  else
    local pids=() i
    for i in $(seq 0 $((NPROC - 1))); do
      "${RUN[@]}" kepcmp.run --adapter "$ADAPTER" --out "$OUTDIR/$name.h5" \
        --shard "$i" --n-shards "$NPROC" "$@" \
        >"$OUTDIR/logs/$name.rank$i.log" 2>&1 &
      pids+=($!)
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" -eq 0 ]] || { echo "$name: a shard failed, see $OUTDIR/logs/" >&2; exit 1; }
  fi
}

echo "adapter=$ADAPTER  cores=$CORES  nproc=$NPROC  mpi=$MPI"
echo "signal seeds=$N_SEEDS  null seeds=$NULL_SEEDS  reference_n_mc=$REFERENCE_N_MC"
echo "output -> $OUTDIR"

run_phase signal --which signal --n-seeds "$N_SEEDS" \
  --reference-n-mc "$REFERENCE_N_MC" "${LIMIT_ARGS[@]}"

run_phase null --which null --n-seeds "$NULL_SEEDS" \
  --n-terms 1 2 --sigma-amp-mults 1.0 "${LIMIT_ARGS[@]}"

echo "all shards complete -> $OUTDIR"
echo "next: merge, then reduce --- see the header of this script"
