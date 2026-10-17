#!/bin/bash
#SBATCH -J spmv4-smoke
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:20:00
#SBATCH -o spmv4-smoke-%j.out
#SBATCH -e spmv4-smoke-%j.err
set -euo pipefail

ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench}
ENV_DIR=${ENV_DIR:-/public/home/cnic_lsb/conda-envs/rsf-triton-v2}
RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root}
OUTPUT=$ROOT/evidence/revision-v1-spmv4-smoke-$SLURM_JOB_ID
TEMP_ROOT=$ROOT/revision-v1-tmp-$SLURM_JOB_ID
MATRIX=$ROOT/data/bcsstk17/bcsstk17.mtx
mkdir -p "$OUTPUT/triton" "$OUTPUT/native" "$TEMP_ROOT/cache"

module purge
module load compiler/dtk/26.04
export PATH=$ENV_DIR/bin:$PATH
export ROCM_PATH=$RUNTIME_DIR
export TRITON_CACHE_DIR=$TEMP_ROOT/cache
export TMPDIR=$TEMP_ROOT TEMP=$TEMP_ROOT TMP=$TEMP_ROOT
export PYTHONPATH=$ROOT
export LD_LIBRARY_PATH=/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}

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

sha256sum \
  "$ROOT/spmv_four_cell_triton_v1.py" \
  "$ROOT/spmv_four_cell_native_v1.cpp" \
  "$MATRIX" \
  "$0" > "$OUTPUT/source.sha256"

hipcc -std=c++17 -O3 --offload-arch=gfx936 \
  "$ROOT/spmv_four_cell_native_v1.cpp" -o "$OUTPUT/spmv_four_cell_native_v1"

# Smoke uses one registered configuration per cell. It establishes build and
# correctness only; its timing values are not formal performance evidence.
"$OUTPUT/spmv_four_cell_native_v1" "$MATRIX" one_thread_per_row 128 1 2 3 \
  > "$OUTPUT/native/one_thread_per_row.txt"
"$OUTPUT/spmv_four_cell_native_v1" "$MATRIX" cooperative_row 128 4 2 3 \
  > "$OUTPUT/native/cooperative_row.txt"
python "$ROOT/spmv_four_cell_triton_v1.py" --matrix "$MATRIX" --mapping row_program \
  --block-size 128 --num-warps 4 --warmup 2 --repeat 3 \
  --json "$OUTPUT/triton/row_program.json"
python "$ROOT/spmv_four_cell_triton_v1.py" --matrix "$MATRIX" --mapping block_row \
  --block-size 128 --num-warps 4 --warmup 2 --repeat 3 \
  --json "$OUTPUT/triton/block_row.json"

grep -q 'correctness=PASS' "$OUTPUT/native/one_thread_per_row.txt"
grep -q 'correctness=PASS' "$OUTPUT/native/cooperative_row.txt"
python - "$OUTPUT/triton/row_program.json" "$OUTPUT/triton/block_row.json" <<'PY'
import json, sys
for name in sys.argv[1:]:
    result = json.load(open(name, encoding="utf-8"))
    if result["correctness"] != "PASS":
        raise SystemExit(f"correctness failure: {name}")
print("TRITON_CORRECTNESS_PASS")
PY

echo "SPMV_FOUR_CELL_SMOKE_PASS output=$OUTPUT"
