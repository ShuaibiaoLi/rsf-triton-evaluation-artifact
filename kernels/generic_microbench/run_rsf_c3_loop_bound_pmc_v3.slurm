#!/bin/bash
#SBATCH -J rsf-c3-bound-pmc
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:45:00
#SBATCH -o /public/home/cnic_lsb/generic-microbench/evidence/rsf-c3-bound-pmc-v3-%j.out
#SBATCH -e /public/home/cnic_lsb/generic-microbench/evidence/rsf-c3-bound-pmc-v3-%j.err
set -euo pipefail

ROOT=/public/home/cnic_lsb/generic-microbench
RSF_ROOT=/public/home/cnic_lsb/rsf-triton-gate
ENV_DIR=/public/home/cnic_lsb/conda-envs/rsf-triton-v2
RUNTIME_DIR=/public/home/cnic_lsb/dtk2604-runtime-root
OUTPUT=$ROOT/evidence/revision-v2-rsf-c3-loop-bound-pmc-${SLURM_JOB_ID}
TEMP_ROOT=$ROOT/revision-v2-rsf-c3-loop-bound-pmc-tmp-${SLURM_JOB_ID}
mkdir -p "$OUTPUT" "$TEMP_ROOT/cache"
module purge
module load compiler/dtk/26.04
export PATH="$ENV_DIR/bin:$PATH" ROCM_PATH="$RUNTIME_DIR"
export TRITON_CACHE_DIR="$TEMP_ROOT/cache" TMPDIR="$TEMP_ROOT" TEMP="$TEMP_ROOT" TMP="$TEMP_ROOT"
export LD_LIBRARY_PATH="/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}"
export HIPPROF_METRICS="$RUNTIME_DIR/share/profiler/counters/derived_counters.xml"
{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  command -v hipprof
  hipcc --version
  python --version
} > "$OUTPUT/environment.txt" 2>&1
sha256sum "$ROOT/rsf_c3_fixed_replay_native_v2.cpp" \
  "$ROOT/rsf_c3_fixed_replay_triton_v2.py" \
  "$ROOT/rsf_c3_loop_bound_native_v3.cpp" \
  "$ROOT/rsf_c3_loop_bound_triton_v3.py" \
  "$ROOT/run_rsf_c3_loop_bound_pmc_v3.slurm" \
  "$RSF_ROOT/aging_rsf_fp64_reference.py" \
  "$RSF_ROOT/test-data/generated/src03-aging-rsf-newton-fp64-input-v1.tsv" \
  > "$OUTPUT/source.sha256"
hipcc -std=c++17 -O3 --offload-arch=gfx936 \
  "$ROOT/rsf_c3_loop_bound_native_v3.cpp" -o "$OUTPUT/native"

# Correctness-only profile path avoids using profiler-distorted timing as a gate.
hipprof --pmc --pmc-type 3 --stats -d "$OUTPUT/native-data" -o "$OUTPUT/native-pmc" \
  "$OUTPUT/native" 65537 1 2 smoke > "$OUTPUT/native.stdout" 2> "$OUTPUT/native.stderr"
grep -q RSF_C3_LOOP_BOUND_NATIVE_V3_PASS "$OUTPUT/native.stdout"

# Populate JIT cache before profiling. Neither run contributes performance data.
python "$ROOT/rsf_c3_loop_bound_triton_v3.py" \
  --rsf-root "$RSF_ROOT" --source-v2 "$ROOT/rsf_c3_fixed_replay_triton_v2.py" \
  --n 65537 --warmup 1 --repeat 2 --smoke \
  --output "$OUTPUT/triton-prewarm.json" > "$OUTPUT/triton-prewarm.stdout"
hipprof --pmc --pmc-type 3 --stats -d "$OUTPUT/triton-data" -o "$OUTPUT/triton-pmc" \
  python "$ROOT/rsf_c3_loop_bound_triton_v3.py" \
  --rsf-root "$RSF_ROOT" --source-v2 "$ROOT/rsf_c3_fixed_replay_triton_v2.py" \
  --n 65537 --warmup 1 --repeat 2 --smoke \
  --output "$OUTPUT/triton-profile.json" > "$OUTPUT/triton.stdout" 2> "$OUTPUT/triton.stderr"
grep -q '"status": "PASS"' "$OUTPUT/triton.stdout"
find "$OUTPUT" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort > "$OUTPUT/file_manifest.txt"
echo "RSF_C3_LOOP_BOUND_PMC_V3_PASS output=$OUTPUT"
