#!/bin/bash
#SBATCH -J nonspmv-pair
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --array=0-4
#SBATCH --time=00:45:00
#SBATCH -o nonspmv-pair-%A_%a.out
#SBATCH -e nonspmv-pair-%A_%a.err
set -euo pipefail

ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench}
RSF_ROOT=${RSF_ROOT:-/public/home/cnic_lsb/rsf-triton-gate}
ENV_DIR=${ENV_DIR:-/public/home/cnic_lsb/conda-envs/rsf-triton-v2}
RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root}
RULE=$ROOT/frozen_rule_v1.json
PAIR_ID=${SLURM_ARRAY_TASK_ID:?array task required}
OUTPUT=$ROOT/evidence/revision-v1-nonspmv-paired-${SLURM_ARRAY_JOB_ID}_${PAIR_ID}
TEMP_ROOT=$ROOT/revision-v1-nonspmv-paired-tmp-${SLURM_ARRAY_JOB_ID}_${PAIR_ID}
mkdir -p "$OUTPUT" "$TEMP_ROOT/cache"

module purge
module load compiler/dtk/26.04
export PATH=$ENV_DIR/bin:$PATH ROCM_PATH=$RUNTIME_DIR TRITON_CACHE_DIR=$TEMP_ROOT/cache
export TMPDIR=$TEMP_ROOT TEMP=$TEMP_ROOT TMP=$TEMP_ROOT PYTHONPATH=$ROOT
export LD_LIBRARY_PATH=/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}

{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "slurm_array_job_id=$SLURM_ARRAY_JOB_ID"
  echo "slurm_array_task_id=$PAIR_ID"
  echo "visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  hipcc --version
  python --version
  python -c 'import torch,triton; print("torch="+torch.__version__); print("triton="+triton.__version__); print("device="+torch.cuda.get_device_name(0))'
} > "$OUTPUT/environment.txt" 2>&1

sha256sum "$ROOT/timing_decomposition_triton_v2.py" "$ROOT/timing_decomposition_native_v2.cpp" \
  "$ROOT/timing_decomposition_rsf_native_v2.cpp" "$ROOT/spmv_four_cell_triton_v1.py" \
  "$RSF_ROOT/probes/triton/rsf/benchmark_c3_native_compare_v1.py" \
  "$RSF_ROOT/probes/triton/rsf/aging_rsf_numerical_feasibility.py" "$RULE" "$0" > "$OUTPUT/source.sha256"

hipcc -std=c++17 -O3 --offload-arch=gfx936 "$ROOT/timing_decomposition_native_v2.cpp" -o "$OUTPUT/timing_native"
hipcc -std=c++17 -O3 --offload-arch=gfx936 "$ROOT/timing_decomposition_rsf_native_v2.cpp" -o "$OUTPUT/timing_rsf_native"

run_hip() {
  "$OUTPUT/timing_native" "$ROOT" 20 unused > "$OUTPUT/hip-generic.txt"
  "$OUTPUT/timing_rsf_native" 65537 20 > "$OUTPUT/hip-rsf.txt"
  grep -q TIMING_DECOMPOSITION_NATIVE_V2_PASS "$OUTPUT/hip-generic.txt"
  grep -q TIMING_DECOMPOSITION_RSF_NATIVE_V2_PASS "$OUTPUT/hip-rsf.txt"
}

run_triton() {
  python "$ROOT/timing_decomposition_triton_v2.py" --root "$ROOT" --rsf-root "$RSF_ROOT" \
    --rule "$RULE" --repeat 20 --batches 1 8 32 128 --json "$OUTPUT/triton.json" \
    > "$OUTPUT/triton.stdout"
  python - "$OUTPUT/triton.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert len(d["cases"]) == 7
assert all(x["correctness"]["status"] == "PASS" for x in d["cases"])
PY
}

if (( PAIR_ID % 2 == 0 )); then
  echo hip_then_triton > "$OUTPUT/order.txt"
  run_hip
  run_triton
else
  echo triton_then_hip > "$OUTPUT/order.txt"
  run_triton
  run_hip
fi

echo "NONSPMV_INDEPENDENT_PAIR_PASS pair=$PAIR_ID output=$OUTPUT"
