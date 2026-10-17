#!/bin/bash
#SBATCH -J rsf-coeff-iso
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:20:00
#SBATCH -o rsf-coeff-iso-%j.out
#SBATCH -e rsf-coeff-iso-%j.err
set -euo pipefail
ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench};ENV_DIR=${ENV_DIR:-/public/home/cnic_lsb/conda-envs/rsf-triton-v2};RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root};OUT=$ROOT/evidence/revision-v1-rsf-coefficient-isolation-$SLURM_JOB_ID;TMP=$ROOT/revision-v1-rsf-coeff-tmp-$SLURM_JOB_ID;mkdir -p "$OUT" "$TMP/cache";module purge;module load compiler/dtk/26.04;export PATH=$ENV_DIR/bin:$PATH ROCM_PATH=$RUNTIME_DIR TRITON_CACHE_DIR=$TMP/cache TMPDIR=$TMP TEMP=$TMP;export LD_LIBRARY_PATH=/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}
{ echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)";echo "hostname=$(hostname)";echo "job=$SLURM_JOB_ID";hipcc --version;python --version;python -c 'import torch,triton;print(torch.__version__,triton.__version__,torch.cuda.get_device_name(0))'; } > "$OUT/environment.txt" 2>&1
sha256sum "$ROOT/rsf_coefficient_isolation_native_v1.cpp" "$ROOT/rsf_coefficient_isolation_triton_v1.py" "$0" > "$OUT/source.sha256"
hipcc -std=c++17 -O3 --offload-arch=gfx936 "$ROOT/rsf_coefficient_isolation_native_v1.cpp" -o "$OUT/native";"$OUT/native" 65537 10 30 128 > "$OUT/native.txt";grep -q RSF_COEFFICIENT_ISOLATION_NATIVE_V1_PASS "$OUT/native.txt"
python "$ROOT/rsf_coefficient_isolation_triton_v1.py" --n 65537 --warmup 10 --repeat 30 --batch 128 --output "$OUT/triton.json" > "$OUT/triton.stdout";grep -q '"status": "PASS"' "$OUT/triton.stdout"
echo "RSF_COEFFICIENT_ISOLATION_V1_PASS output=$OUT"
