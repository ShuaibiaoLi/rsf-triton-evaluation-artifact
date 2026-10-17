#!/bin/bash
#SBATCH -J rsf-c3-lut-pmc
#SBATCH -p hx1hdnormal
#SBATCH --gres=dcu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:45:00
#SBATCH -o rsf-c3-lut-pmc-%j.out
#SBATCH -e rsf-c3-lut-pmc-%j.err
set -euo pipefail
ROOT=${ROOT:-/public/home/cnic_lsb/generic-microbench};RSF_ROOT=${RSF_ROOT:-/public/home/cnic_lsb/rsf-triton-gate};ENV_DIR=${ENV_DIR:-/public/home/cnic_lsb/conda-envs/rsf-triton-v2};RUNTIME_DIR=${RUNTIME_DIR:-/public/home/cnic_lsb/dtk2604-runtime-root};OUTPUT=$ROOT/evidence/revision-v1-rsf-c3-lookup-pmc-$SLURM_JOB_ID;TEMP_ROOT=$ROOT/revision-v1-rsf-c3-lookup-pmc-tmp-$SLURM_JOB_ID;mkdir -p "$OUTPUT" "$TEMP_ROOT/cache"
module purge;module load compiler/dtk/26.04;export PATH=$ENV_DIR/bin:$PATH ROCM_PATH=$RUNTIME_DIR TRITON_CACHE_DIR=$TEMP_ROOT/cache TMPDIR=$TEMP_ROOT TEMP=$TEMP_ROOT TMP=$TEMP_ROOT PYTHONPATH=$ROOT;export LD_LIBRARY_PATH=/public/home/cnic_lsb/rsf-triton-runtime/lib:${LD_LIBRARY_PATH:-};export HIPPROF_METRICS=$RUNTIME_DIR/share/profiler/counters/derived_counters.xml
{ echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)";echo "hostname=$(hostname)";echo "slurm_job_id=$SLURM_JOB_ID";command -v hipprof;hipcc --version;python --version; } > "$OUTPUT/environment.txt" 2>&1
sha256sum "$ROOT/rsf_c3_lookup_native_v2.cpp" "$ROOT/rsf_c3_lookup_triton_v2.py" "$ROOT/rsf_c3_causal_ablation_v2.json" "$0" > "$OUTPUT/source.sha256"
hipcc -std=c++17 -O3 --offload-arch=gfx936 "$ROOT/rsf_c3_lookup_native_v2.cpp" -o "$OUTPUT/native"
hipprof --pmc --pmc-type 3 --stats -d "$OUTPUT/native-data" -o "$OUTPUT/native-pmc" "$OUTPUT/native" 65537 1 2 > "$OUTPUT/native.stdout" 2> "$OUTPUT/native.stderr";grep -q RSF_C3_LOOKUP_NATIVE_V2_PASS "$OUTPUT/native.stdout"
python "$ROOT/rsf_c3_lookup_triton_v2.py" --rsf-root "$RSF_ROOT" --n 65537 --warmup 1 --repeat 1 --output "$OUTPUT/triton-prewarm.json" > "$OUTPUT/triton-prewarm.stdout"
hipprof --pmc --pmc-type 3 --stats -d "$OUTPUT/triton-data" -o "$OUTPUT/triton-pmc" python "$ROOT/rsf_c3_lookup_triton_v2.py" --rsf-root "$RSF_ROOT" --n 65537 --warmup 1 --repeat 2 --output "$OUTPUT/triton.json" > "$OUTPUT/triton.stdout" 2> "$OUTPUT/triton.stderr";grep -q '"status": "PASS"' "$OUTPUT/triton.stdout"
find "$OUTPUT" -maxdepth 2 -type f -printf '%P %s bytes\n' | sort > "$OUTPUT/file_manifest.txt"
echo "RSF_C3_LOOKUP_PMC_V2_PASS output=$OUTPUT"
