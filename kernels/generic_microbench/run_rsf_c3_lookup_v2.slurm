#!/bin/bash
#SBATCH -J rsf-c3-lookup
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:35:00
#SBATCH -o rsf-c3-lookup-%j.out
#SBATCH -e rsf-c3-lookup-%j.err
set -euo pipefail
ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench};RSF_ROOT=${RSF_ROOT:-/public/home/cnic_lsb/rsf-triton-gate};ENV_DIR=${ENV_DIR:-/public/home/cnic_lsb/conda-envs/rsf-triton-v2};RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root};OUTPUT=$ROOT/evidence/revision-v1-rsf-c3-lookup-$SLURM_JOB_ID;TEMP_ROOT=$ROOT/revision-v1-rsf-c3-lookup-tmp-$SLURM_JOB_ID;mkdir -p "$OUTPUT" "$TEMP_ROOT/cache"
module purge;module load compiler/dtk/26.04;export PATH=$ENV_DIR/bin:$PATH ROCM_PATH=$RUNTIME_DIR TRITON_CACHE_DIR=$TEMP_ROOT/cache TMPDIR=$TEMP_ROOT TEMP=$TEMP_ROOT TMP=$TEMP_ROOT PYTHONPATH=$ROOT;export LD_LIBRARY_PATH=/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}
{ echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)";echo "hostname=$(hostname)";echo "slurm_job_id=$SLURM_JOB_ID";hipcc --version;python --version;python -c 'import torch,triton;print("torch="+torch.__version__);print("triton="+triton.__version__);print("device="+torch.cuda.get_device_name(0))'; } > "$OUTPUT/environment.txt" 2>&1
sha256sum "$ROOT/rsf_c3_lookup_native_v2.cpp" "$ROOT/rsf_c3_lookup_triton_v2.py" "$ROOT/rsf_c3_causal_ablation_v2.json" "$0" > "$OUTPUT/source.sha256"
hipcc -std=c++17 -O3 --offload-arch=gfx936 "$ROOT/rsf_c3_lookup_native_v2.cpp" -o "$OUTPUT/native"
"$OUTPUT/native" 65537 10 30 > "$OUTPUT/native.txt";grep -q RSF_C3_LOOKUP_NATIVE_V2_PASS "$OUTPUT/native.txt"
python "$ROOT/rsf_c3_lookup_triton_v2.py" --rsf-root "$RSF_ROOT" --n 65537 --warmup 10 --repeat 30 --output "$OUTPUT/triton.json" > "$OUTPUT/triton.stdout";grep -q '"status": "PASS"' "$OUTPUT/triton.stdout"
echo "RSF_C3_LOOKUP_V2_PASS output=$OUTPUT"
