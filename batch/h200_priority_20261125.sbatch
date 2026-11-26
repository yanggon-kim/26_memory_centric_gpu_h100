#!/bin/bash
#SBATCH --job-name=yg-h200-priority
#SBATCH --account=gts-dmahajan7
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:2
#SBATCH --mem-per-gpu=196G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=h200-priority-%j.out
set -euo pipefail
repo=/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100
cd "$repo"
exec 9>.git/h200_priority_batch.lock
flock -n 9
# The owner authorized exactly one two-H200 submission. No self-submission/requeue.
test "$(git branch --show-current)" = codex/h200-phase01
git merge-base --is-ancestor 7843c845 HEAD
export CUDA_HOME=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/none-none/cuda-13.0.1-ne4hey4ce57lmd34nixeikyfribdqln3
export CUDA_INSTALL_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export HF_HUB_CACHE=/storage/scratch1/6/jkim4112/models/h200_priority_hub
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
unset PYTHONPATH LD_PRELOAD CUDA_INJECTION64_PATH NVBIT_INSTRUMENTATION_ENABLED
# Remove inherited toolkit search paths; wheel libraries and the explicit CUDA_HOME govern.
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH
srun --ntasks=1 --cpus-per-task=16 --gpus-per-task=2 --gpu-bind=none \
  /usr/local/pace-apps/manual/packages/anaconda3/2023.03/bin/python3 -u \
  scripts/h200_priority_campaign.py --bootstrap --cuda "$CUDA_HOME"
