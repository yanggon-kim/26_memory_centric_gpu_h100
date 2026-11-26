#!/bin/bash
# Run only as an overlapping step of the existing allocation, after staging
# attempt0 fails/holds. This script never submits or cancels a Slurm job.
set -euo pipefail
test "${SLURM_JOB_ID:-}" = 16163464
repo=/storage/scratch1/6/jkim4112/gpullm/02_mem_centric/26_memory_centric_gpu_h100
cd "$repo"
bootstrap_python=/usr/local/pace-apps/manual/packages/anaconda3/2023.03/bin/python3
"$bootstrap_python" - <<'PY'
import json, os, socket, subprocess
expected={'GPU-951a11c2-40ee-3fcb-f71f-49849e744cb1','GPU-b51c6186-ac3b-5149-8b8a-f33943072c47'}
assert socket.gethostname().split('.')[0]=='atl1-1-03-019-14-0'
state=subprocess.check_output(['scontrol','show','job','-o','16163464'],text=True)
assert 'JobState=RUNNING' in state and 'gres/gpu:h200=2' in state
lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip().splitlines()
rows=[line.split(', ',1) for line in lines]
assert len(rows)==2 and {r[0] for r in rows}==expected and all(r[1]=='NVIDIA H200' for r in rows)
print(json.dumps({'job_id':os.environ['SLURM_JOB_ID'],'hostname':socket.gethostname(),
                  'inherited_visibility':os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'authorized_pair':sorted(expected),'driver_inventory':rows,
                  'scope':'explicit UUID visibility repair within the existing allocated pair'}),flush=True)
PY
export CUDA_VISIBLE_DEVICES=GPU-951a11c2-40ee-3fcb-f71f-49849e744cb1,GPU-b51c6186-ac3b-5149-8b8a-f33943072c47
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/none-none/cuda-13.0.1-ne4hey4ce57lmd34nixeikyfribdqln3
export CUDA_INSTALL_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export HF_HUB_CACHE=/storage/scratch1/6/jkim4112/models/h200_priority_hub
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 VLLM_NO_USAGE_STATS=1
unset PYTHONPATH LD_PRELOAD CUDA_INJECTION64_PATH NVBIT_INSTRUMENTATION_ENABLED
unset LD_LIBRARY_PATH LIBRARY_PATH CPATH
exec "$bootstrap_python" -u scripts/h200_priority_campaign.py --bootstrap --bootstrap-attempt 1 --cuda "$CUDA_HOME"
