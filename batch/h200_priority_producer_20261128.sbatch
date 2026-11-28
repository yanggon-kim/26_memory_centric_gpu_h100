#!/bin/bash
# Prepared producer pilot only. Do not submit while another campaign job is queued/running.
#SBATCH --job-name=yg-h200-producer
#SBATCH --account=gts-dmahajan7
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H200:2
#SBATCH --mem-per-gpu=196G
#SBATCH --time=02:00:00
#SBATCH --time-min=02:00:00
#SBATCH --exclude=atl1-1-02-014-9-0
#SBATCH --no-requeue
#SBATCH --output=h200-producer-%j.out
set -euo pipefail
test "$#" -gt 0 # explicit reviewed context/scope, never silently repeat C128
repo=/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100
cd "$repo"
exec 9>.git/h200_priority_batch.lock
flock -n 9
test "$(git branch --show-current)" = codex/h200-phase01
git merge-base --is-ancestor 7843c845 HEAD
git merge-base --is-ancestor f2bc4f90 HEAD
export CUDA_HOME=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/none-none/cuda-13.0.1-ne4hey4ce57lmd34nixeikyfribdqln3
export CUDA_INSTALL_PATH="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export HF_HUB_CACHE=/storage/scratch1/6/jkim4112/models/h200_priority_hub
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH LIBRARY_PATH CPATH CUDA_INJECTION64_PATH NVBIT_INSTRUMENTATION_ENABLED
h200_task_cpus=${SLURM_CPUS_PER_TASK:-16}
test "$h200_task_cpus" -ge 8
export H200_VISIBILITY_MANIFEST="/tmp/h200_native_visibility_${SLURM_JOB_ID}_${UID}.json"
srun --ntasks=1 --cpus-per-task="$h200_task_cpus" --gpus-per-task=2 --gpu-bind=none \
  bash -c 'set -euo pipefail
    /usr/local/pace-apps/manual/packages/anaconda3/2023.03/bin/python3 scripts/h200_allocated_visibility.py --output "$H200_VISIBILITY_MANIFEST"
    CUDA_VISIBLE_DEVICES=$(/usr/local/pace-apps/manual/packages/anaconda3/2023.03/bin/python3 -c '\''import json,os; print(json.load(open(os.environ["H200_VISIBILITY_MANIFEST"]))["selected_cuda_visible_devices"])'\'')
    export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
    exec /storage/scratch1/6/jkim4112/envs/h200_priority_20261125/bin/python -u scripts/h200_producer_campaign.py "$@"' h200-native "$@"
