#!/bin/bash
#SBATCH --job-name=h200-fail-v3
#SBATCH --account=gts-dmahajan7
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem-per-gpu=196G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=h200-shoreline-%j.out

# Submit this file instead of the allocation-and-sleep script. No Codex needed.
set -euo pipefail
repo=${H200_REPO:-/storage/home/hcoda1/6/jkim4112/scratch/gpullm/02_mem_centric/26_memory_centric_gpu_h100}
cd "$repo"
exec 9>.git/h200_batch.lock
flock -n 9 || { echo 'Another automatic campaign holds the repository lock'; exit 1; }
test "$(git branch --show-current)" = codex/h200-phase01
git diff --quiet HEAD -- scripts configs docs batch AGENTS.md
test -z "$(git diff --cached --name-only)"
export GIT_TERMINAL_PROMPT=0
timeout 120s git fetch origin codex/h200-phase01
git merge --ff-only origin/codex/h200-phase01
bootstrap_python=${H200_BOOTSTRAP_PYTHON:-/usr/local/pace-apps/manual/packages/anaconda3/2023.03/bin/python3}
# Keep the campaign in the batch lifetime, in a GPU-bound step. No backgrounding.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --gpu-bind=single:1 \
  "$bootstrap_python" -u scripts/bootstrap_h200_batch.py \
  --plan "${H200_BATCH_PLAN:-configs/h200_batch_failure_recovery_v3.json}"
