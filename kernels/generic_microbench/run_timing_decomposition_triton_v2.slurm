#!/bin/bash
#SBATCH -J timing-v2-triton
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:45:00
#SBATCH -o timing-v2-triton-%j.out
#SBATCH -e timing-v2-triton-%j.err
set -euo pipefail

ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench}
RSF_ROOT=${RSF_ROOT:-/public/home/cnic_lsb/rsf-triton-gate}
ENV_DIR=${ENV_DIR:-/public/home/cnic_lsb/conda-envs/rsf-triton-v2}
RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root}
RULE=$ROOT/frozen_rule_v1.json
OUTPUT=$ROOT/evidence/revision-v1-timing-v2-triton-$SLURM_JOB_ID
TEMP_ROOT=$ROOT/revision-v1-timing-v2-tmp-$SLURM_JOB_ID
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
  python --version
  python -c 'import torch,triton; print("torch="+torch.__version__); print("triton="+triton.__version__); print("device="+torch.cuda.get_device_name(0))'
} > "$OUTPUT/environment.txt" 2>&1
sha256sum "$ROOT/timing_decomposition_triton_v2.py" "$ROOT/spmv_four_cell_triton_v1.py" \
  "$RSF_ROOT/probes/triton/rsf/benchmark_c3_native_compare_v1.py" \
  "$RSF_ROOT/probes/triton/rsf/aging_rsf_numerical_feasibility.py" "$RULE" "$0" > "$OUTPUT/source.sha256"
python "$ROOT/timing_decomposition_triton_v2.py" --root "$ROOT" --rsf-root "$RSF_ROOT" \
  --rule "$RULE" --repeat 20 --batches 1 8 32 128 --json "$OUTPUT/result.json"
echo "TIMING_DECOMPOSITION_TRITON_V2_PASS output=$OUTPUT"
