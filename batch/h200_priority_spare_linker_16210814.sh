#!/bin/bash
# Existing owned allocation only; validated idle-GPU1 transition in Python.
set -euo pipefail
test "${SLURM_JOB_ID:-}" = 16210814
repo=/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100
cd "$repo"
export CUDA_HOME=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/none-none/cuda-13.0.1-ne4hey4ce57lmd34nixeikyfribdqln3
export CUDA_INSTALL_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export HF_HUB_CACHE=/storage/scratch1/6/jkim4112/models/h200_priority_hub
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
unset PYTHONPATH PYTHONHOME LD_PRELOAD CUDA_INJECTION64_PATH NVBIT_INSTRUMENTATION_ENABLED
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH
exec /storage/scratch1/6/jkim4112/envs/h200_priority_20261125/bin/python -u scripts/h200_spare_linker_followup.py \
  --source-session "$repo/h200/20261127_job16210814_priority/gpu1/session.json" \
  --previous-pid 2817338 --root "$repo/h200/20261127_job16210814_priority/gpu1_native_linker"
