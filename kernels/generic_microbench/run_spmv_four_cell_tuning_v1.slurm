#!/bin/bash
#SBATCH -J spmv4-tune
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=02:00:00
#SBATCH -o spmv4-tune-%j.out
#SBATCH -e spmv4-tune-%j.err
set -euo pipefail

ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench}
ENV_DIR=${ENV_DIR:-/public/home/cnic_lsb/conda-envs/rsf-triton-v2}
RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root}
OUTPUT=$ROOT/evidence/revision-v1-spmv4-tuning-$SLURM_JOB_ID
TEMP_ROOT=$ROOT/revision-v1-tmp-$SLURM_JOB_ID
mkdir -p "$OUTPUT/raw/triton" "$OUTPUT/raw/hip" "$TEMP_ROOT/cache"

module purge
module load compiler/dtk/26.04
export PATH=$ENV_DIR/bin:$PATH ROCM_PATH=$RUNTIME_DIR TRITON_CACHE_DIR=$TEMP_ROOT/cache
export TMPDIR=$TEMP_ROOT TEMP=$TEMP_ROOT TMP=$TEMP_ROOT PYTHONPATH=$ROOT
export LD_LIBRARY_PATH=/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}

MATRICES=(
  "$ROOT/data/bcsstk17/bcsstk17.mtx"
  "$ROOT/data/poisson3Da/poisson3Da.mtx"
)
BLOCK_CANDIDATES=(64 128 256)
GROUP_CANDIDATES=(1 2 4 8)

{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-unset}"
  echo "visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  python --version
  hipcc --version
  python -c 'import torch,triton; print("torch="+torch.__version__); print("triton="+triton.__version__); print("device="+torch.cuda.get_device_name(0))'
} > "$OUTPUT/environment.txt" 2>&1

sha256sum "$ROOT/spmv_four_cell_triton_v1.py" "$ROOT/spmv_four_cell_native_v1.cpp" "$0" "${MATRICES[@]}" > "$OUTPUT/source.sha256"
hipcc -std=c++17 -O3 --offload-arch=gfx936 "$ROOT/spmv_four_cell_native_v1.cpp" -o "$OUTPUT/spmv_four_cell_native_v1"

for matrix in "${MATRICES[@]}"; do
  name=$(basename "$matrix" .mtx)
  for block in "${BLOCK_CANDIDATES[@]}"; do
    for groups in "${GROUP_CANDIDATES[@]}"; do
      for mapping in one_thread_per_row cooperative_row; do
        "$OUTPUT/spmv_four_cell_native_v1" "$matrix" "$mapping" "$block" "$groups" 10 30 \
          > "$OUTPUT/raw/hip/${name}-${mapping}-b${block}-g${groups}.txt"
      done
      for mapping in row_program block_row; do
        python "$ROOT/spmv_four_cell_triton_v1.py" --matrix "$matrix" --mapping "$mapping" \
          --block-size "$block" --num-warps "$groups" --warmup 10 --repeat 30 \
          --json "$OUTPUT/raw/triton/${name}-${mapping}-b${block}-g${groups}.json" \
          > "$OUTPUT/raw/triton/${name}-${mapping}-b${block}-g${groups}.stdout"
      done
    done
  done
done

python "$ROOT/summarize_spmv_four_cell_tuning.py" --root "$OUTPUT/raw" --output "$OUTPUT/tuning-ledger.json"
echo "SPMV_FOUR_CELL_TUNING_PASS output=$OUTPUT"
