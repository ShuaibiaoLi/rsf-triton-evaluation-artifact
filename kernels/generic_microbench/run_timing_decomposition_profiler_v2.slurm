#!/bin/bash
#SBATCH -J timing-v2-prof
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:45:00
#SBATCH -o timing-v2-prof-%j.out
#SBATCH -e timing-v2-prof-%j.err
set -euo pipefail

ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench}
RSF_ROOT=${RSF_ROOT:-/public/home/cnic_lsb/rsf-triton-gate}
ENV_DIR=${ENV_DIR:-/public/home/cnic_lsb/conda-envs/rsf-triton-v2}
RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root}
RULE=$ROOT/frozen_rule_v1.json
OUTPUT=$ROOT/evidence/revision-v1-timing-v2-profiler-$SLURM_JOB_ID
TEMP_ROOT=$ROOT/revision-v1-timing-v2-profiler-tmp-$SLURM_JOB_ID
mkdir -p "$OUTPUT" "$TEMP_ROOT/cache"

module purge
module load compiler/dtk/26.04
export PATH=$ENV_DIR/bin:$PATH ROCM_PATH=$RUNTIME_DIR TRITON_CACHE_DIR=$TEMP_ROOT/cache
export TMPDIR=$TEMP_ROOT TEMP=$TEMP_ROOT TMP=$TEMP_ROOT PYTHONPATH=$ROOT
export LD_LIBRARY_PATH=/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}

{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  command -v hipprof
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

# Profiler runs are deliberately separate from formal latency and are never used for speedup.
hipprof --hip-trace --stats -d "$OUTPUT/hip-generic-data" -o "$OUTPUT/hip-generic" \
  "$OUTPUT/timing_native" "$ROOT" 1 unused > "$OUTPUT/hip-generic.stdout" 2> "$OUTPUT/hip-generic.stderr"
grep -q TIMING_DECOMPOSITION_NATIVE_V2_PASS "$OUTPUT/hip-generic.stdout"

hipprof --hip-trace --stats -d "$OUTPUT/hip-rsf-data" -o "$OUTPUT/hip-rsf" \
  "$OUTPUT/timing_rsf_native" 65537 1 > "$OUTPUT/hip-rsf.stdout" 2> "$OUTPUT/hip-rsf.stderr"
grep -q TIMING_DECOMPOSITION_RSF_NATIVE_V2_PASS "$OUTPUT/hip-rsf.stdout"

# Populate JIT cache outside profiler; the profiled process still validates correctness.
python "$ROOT/timing_decomposition_triton_v2.py" --root "$ROOT" --rsf-root "$RSF_ROOT" \
  --rule "$RULE" --repeat 1 --batches 1 --json "$OUTPUT/triton-prewarm.json" > "$OUTPUT/triton-prewarm.stdout"

hipprof --hip-trace --stats -d "$OUTPUT/triton-data" -o "$OUTPUT/triton" \
  python "$ROOT/timing_decomposition_triton_v2.py" --root "$ROOT" --rsf-root "$RSF_ROOT" \
  --rule "$RULE" --repeat 1 --batches 1 --json "$OUTPUT/triton-profile.json" \
  > "$OUTPUT/triton.stdout" 2> "$OUTPUT/triton.stderr"

python - "$OUTPUT/triton-profile.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert len(d["cases"]) == 7
assert all(x["correctness"]["status"] == "PASS" for x in d["cases"])
PY

find "$OUTPUT" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort > "$OUTPUT/file_manifest.txt"
echo "TIMING_DECOMPOSITION_PROFILER_V2_PASS output=$OUTPUT"
