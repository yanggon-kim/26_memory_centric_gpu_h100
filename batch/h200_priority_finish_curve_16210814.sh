#!/bin/bash
# Existing allocation and original numerical/source cohort only.
set -euo pipefail
test "${SLURM_JOB_ID:-}" = 16210814
repo=/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100
cd "$repo"
unset PYTHONPATH PYTHONHOME LD_PRELOAD CUDA_INJECTION64_PATH NVBIT_INSTRUMENTATION_ENABLED
export PYTHONNOUSERSITE=1
exec /storage/scratch1/6/jkim4112/envs/h200_priority_20261125/bin/python -u scripts/h200_finish_eager_curve.py \
 --root "$repo/h200/20261127_job16210814_priority/gpu0" --previous-pid 2817337
