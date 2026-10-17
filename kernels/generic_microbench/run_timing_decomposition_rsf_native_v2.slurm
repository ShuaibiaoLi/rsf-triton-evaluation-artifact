#!/bin/bash
#SBATCH -J timing-v2-rsf-hip
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:20:00
#SBATCH -o timing-v2-rsf-hip-%j.out
#SBATCH -e timing-v2-rsf-hip-%j.err
set -euo pipefail
ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench};OUTPUT=$ROOT/evidence/revision-v1-timing-v2-rsf-hip-$SLURM_JOB_ID
mkdir -p "$OUTPUT";module purge;module load compiler/dtk/26.04
sha256sum "$ROOT/timing_decomposition_rsf_native_v2.cpp" "$0" > "$OUTPUT/source.sha256"
hipcc -std=c++17 -O3 --offload-arch=gfx936 "$ROOT/timing_decomposition_rsf_native_v2.cpp" -o "$OUTPUT/timing_decomposition_rsf_native_v2"
"$OUTPUT/timing_decomposition_rsf_native_v2" 65537 20 > "$OUTPUT/result.txt"
grep -q TIMING_DECOMPOSITION_RSF_NATIVE_V2_PASS "$OUTPUT/result.txt"
echo "TIMING_DECOMPOSITION_RSF_NATIVE_V2_PASS output=$OUTPUT"
