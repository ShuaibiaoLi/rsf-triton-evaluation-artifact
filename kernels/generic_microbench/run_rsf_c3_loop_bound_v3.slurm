#!/bin/bash
#SBATCH -J rsf-c3-bound-v3
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:30:00
#SBATCH -o /public/home/cnic_lsb/generic-microbench/evidence/rsf-c3-bound-v3-%j.out
#SBATCH -e /public/home/cnic_lsb/generic-microbench/evidence/rsf-c3-bound-v3-%j.err
set -euo pipefail

MODE=${1:-smoke}
if [[ "$MODE" != smoke && "$MODE" != formal ]]; then
  echo "mode must be smoke or formal" >&2
  exit 2
fi
ROOT=/public/home/cnic_lsb/generic-microbench
RSF_ROOT=/public/home/cnic_lsb/rsf-triton-gate
ENV_DIR=/public/home/cnic_lsb/conda-envs/rsf-triton-v2
RUNTIME_DIR=/public/home/cnic_lsb/dtk2604-runtime-root
OUTPUT=$ROOT/evidence/revision-v2-rsf-c3-loop-bound-${MODE}-${SLURM_JOB_ID}
TEMP_ROOT=$ROOT/revision-v2-rsf-c3-loop-bound-tmp-${SLURM_JOB_ID}
mkdir -p "$OUTPUT" "$TEMP_ROOT/cache"
module purge
module load compiler/dtk/26.04
export PATH="$ENV_DIR/bin:$PATH" ROCM_PATH="$RUNTIME_DIR"
export TRITON_CACHE_DIR="$TEMP_ROOT/cache" TMPDIR="$TEMP_ROOT" TEMP="$TEMP_ROOT" TMP="$TEMP_ROOT"
export LD_LIBRARY_PATH="/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}"
{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "slurm_array_task_id=${SLURM_ARRAY_TASK_ID:-none}"
  echo "mode=$MODE"
  hipcc --version
  python --version
  python -c 'import torch,triton;print("torch="+torch.__version__);print("triton="+triton.__version__);print("device="+torch.cuda.get_device_name(0))'
} > "$OUTPUT/environment.txt" 2>&1
sha256sum \
  "$ROOT/rsf_c3_fixed_replay_native_v2.cpp" \
  "$ROOT/rsf_c3_fixed_replay_triton_v2.py" \
  "$ROOT/rsf_c3_loop_bound_native_v3.cpp" \
  "$ROOT/rsf_c3_loop_bound_triton_v3.py" \
  "$ROOT/run_rsf_c3_loop_bound_v3.slurm" \
  "$RSF_ROOT/aging_rsf_fp64_reference.py" \
  "$RSF_ROOT/test-data/generated/src03-aging-rsf-newton-fp64-input-v1.tsv" \
  > "$OUTPUT/source.sha256"
hipcc -std=c++17 -O3 --offload-arch=gfx936 \
  "$ROOT/rsf_c3_loop_bound_native_v3.cpp" -o "$OUTPUT/native"

if [[ "$MODE" == smoke ]]; then
  SIZES=(1 9 65 257 4099 65537)
  WARMUP=1
  REPEAT=3
else
  SIZES=(65537)
  WARMUP=10
  REPEAT=30
fi
run_native() {
  local n=$1
  "$OUTPUT/native" "${NATIVE_ARGS[@]}" > "$OUTPUT/native-${n}.txt"
  grep -q RSF_C3_LOOP_BOUND_NATIVE_V3_PASS "$OUTPUT/native-${n}.txt"
}
run_triton() {
  local n=$1
  python "$ROOT/rsf_c3_loop_bound_triton_v3.py" \
    --rsf-root "$RSF_ROOT" --source-v2 "$ROOT/rsf_c3_fixed_replay_triton_v2.py" \
    --n "$n" --warmup "$WARMUP" --repeat "$REPEAT" \
    --output "$OUTPUT/triton-${n}.json" "${TRITON_ARGS[@]}" > "$OUTPUT/triton-${n}.stdout"
  grep -q '"status": "PASS"' "$OUTPUT/triton-${n}.stdout"
}
for N in "${SIZES[@]}"; do
  NATIVE_ARGS=("$N" "$WARMUP" "$REPEAT")
  TRITON_ARGS=()
  if [[ "$MODE" == smoke ]]; then
    NATIVE_ARGS+=(smoke)
    TRITON_ARGS+=(--smoke)
  fi
  if [[ "$MODE" == formal && $(( ${SLURM_ARRAY_TASK_ID:-0} % 2 )) -eq 1 ]]; then
    run_triton "$N"
    run_native "$N"
  else
    run_native "$N"
    run_triton "$N"
  fi
done
echo "RSF_C3_LOOP_BOUND_V3_${MODE^^}_PASS output=$OUTPUT"
