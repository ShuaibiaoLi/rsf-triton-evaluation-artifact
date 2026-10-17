#!/bin/bash
#SBATCH -J spmv4-formal
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=01:00:00
#SBATCH -o spmv4-formal-%j.out
#SBATCH -e spmv4-formal-%j.err
set -euo pipefail

ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench}
ENV_DIR=${ENV_DIR:-/public/home/cnic_lsb/conda-envs/rsf-triton-v2}
RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root}
RULE=$ROOT/frozen_rule_v1.json
OUTPUT=$ROOT/evidence/revision-v1-spmv4-formal-$SLURM_JOB_ID
TEMP_ROOT=$ROOT/revision-v1-tmp-$SLURM_JOB_ID
mkdir -p "$OUTPUT/raw/triton" "$OUTPUT/raw/hip" "$TEMP_ROOT/cache"

module purge
module load compiler/dtk/26.04
export PATH=$ENV_DIR/bin:$PATH ROCM_PATH=$RUNTIME_DIR TRITON_CACHE_DIR=$TEMP_ROOT/cache
export TMPDIR=$TEMP_ROOT TEMP=$TEMP_ROOT TMP=$TEMP_ROOT PYTHONPATH=$ROOT
export LD_LIBRARY_PATH=/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-}

MATRIX_PATHS=(
  "$ROOT/data/suitesparse/expanded/dw8192/dw8192.mtx"
  "$ROOT/data/suitesparse/expanded/apache2/apache2.mtx"
  "$ROOT/data/suitesparse/expanded/orani678/orani678.mtx"
)

test -f "$RULE"
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
sha256sum "$ROOT/spmv_four_cell_triton_v1.py" "$ROOT/spmv_four_cell_native_v1.cpp" \
  "$ROOT/summarize_spmv_four_cell_formal.py" "$RULE" "$0" "${MATRIX_PATHS[@]}" > "$OUTPUT/source.sha256"
hipcc -std=c++17 -O3 --offload-arch=gfx936 "$ROOT/spmv_four_cell_native_v1.cpp" -o "$OUTPUT/spmv_four_cell_native_v1"

python - "$RULE" > "$TEMP_ROOT/selected.tsv" <<'PY'
import json, sys
rule=json.load(open(sys.argv[1], encoding="utf-8"))
if rule.get("status") != "FROZEN": raise SystemExit("rule is not frozen")
for item in rule["selected_per_cell"]:
    print(item["backend"], item["mapping"], item["block_size"], item["num_warps_or_wave_groups"], sep="\t")
PY

while IFS=$'\t' read -r backend mapping block groups; do
  for matrix in "${MATRIX_PATHS[@]}"; do
    name=$(basename "$matrix" .mtx)
    if [ "$backend" = hip ]; then
      "$OUTPUT/spmv_four_cell_native_v1" "$matrix" "$mapping" "$block" "$groups" 10 30 \
        > "$OUTPUT/raw/hip/${name}-${mapping}-b${block}-g${groups}.txt"
    else
      python "$ROOT/spmv_four_cell_triton_v1.py" --matrix "$matrix" --mapping "$mapping" \
        --block-size "$block" --num-warps "$groups" --warmup 10 --repeat 30 \
        --json "$OUTPUT/raw/triton/${name}-${mapping}-b${block}-g${groups}.json" \
        > "$OUTPUT/raw/triton/${name}-${mapping}-b${block}-g${groups}.stdout"
    fi
  done
done < "$TEMP_ROOT/selected.tsv"

python "$ROOT/summarize_spmv_four_cell_formal.py" --root "$OUTPUT/raw" --output "$OUTPUT/formal-summary.json"
echo "SPMV_FOUR_CELL_FORMAL_PASS output=$OUTPUT"
