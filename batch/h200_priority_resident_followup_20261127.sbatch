#!/bin/bash
# Owner-authorized sequential two-H200 follow-up; choose shape/conditioning from the published core disposition.
#SBATCH --job-name=yg-h200-followup
#SBATCH --account=gts-dmahajan7
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:2
#SBATCH --mem-per-gpu=196G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=h200-priority-%j.out
set -euo pipefail
test "$#" -gt 0 # require an explicit reviewed follow-up scope
repo=/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100
cd "$repo"
exec 9>.git/h200_priority_batch.lock
flock -n 9
test "$(git branch --show-current)" = codex/h200-phase01
git merge-base --is-ancestor 7843c845 HEAD
export CUDA_HOME=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/none-none/cuda-13.0.1-ne4hey4ce57lmd34nixeikyfribdqln3
export CUDA_INSTALL_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export HF_HUB_CACHE=/storage/scratch1/6/jkim4112/models/h200_priority_hub
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
unset PYTHONPATH PYTHONHOME LD_PRELOAD CUDA_INJECTION64_PATH NVBIT_INSTRUMENTATION_ENABLED
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH
h200_task_cpus=${SLURM_CPUS_PER_TASK:-16}
test "$h200_task_cpus" -ge 8 # two independent workers, OMP_NUM_THREADS=4 each
srun --ntasks=1 --cpus-per-task="$h200_task_cpus" --gpus-per-task=2 --gpu-bind=none \
  bash batch/h200_priority_allocated_step.sh --core-first --eager-first --resident-first \
  --prior-kv-pilot "$repo/h200/20261126_job16163464_priority/results/P0_real_hf_kv.json" "$@"
