#!/bin/bash
#SBATCH -J timing-v2-hip
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:30:00
#SBATCH -o timing-v2-hip-%j.out
#SBATCH -e timing-v2-hip-%j.err
set -euo pipefail
ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench}
RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root}
OUTPUT=$ROOT/evidence/revision-v1-timing-v2-hip-$SLURM_JOB_ID
mkdir -p "$OUTPUT"
module purge
module load compiler/dtk/26.04
export ROCM_PATH=$RUNTIME_DIR LD_LIBRARY_PATH=/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}
{
 echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)";echo "hostname=$(hostname)";echo "slurm_job_id=$SLURM_JOB_ID";hipcc --version
} > "$OUTPUT/environment.txt" 2>&1
sha256sum "$ROOT/timing_decomposition_native_v2.cpp" "$0" \
 "$ROOT/data/suitesparse/expanded/dw8192/dw8192.mtx" "$ROOT/data/suitesparse/expanded/orani678/orani678.mtx" > "$OUTPUT/source.sha256"
hipcc -std=c++17 -O3 --offload-arch=gfx936 "$ROOT/timing_decomposition_native_v2.cpp" -o "$OUTPUT/timing_decomposition_native_v2"
"$OUTPUT/timing_decomposition_native_v2" "$ROOT" 20 unused > "$OUTPUT/result.txt"
grep -q TIMING_DECOMPOSITION_NATIVE_V2_PASS "$OUTPUT/result.txt"
echo "TIMING_DECOMPOSITION_NATIVE_V2_PASS output=$OUTPUT"
