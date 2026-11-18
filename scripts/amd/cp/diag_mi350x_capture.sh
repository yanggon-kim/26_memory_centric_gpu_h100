#!/bin/bash
#SBATCH -J cp-diag350
#SBATCH -o /work1/proj_blaisetine/yanggon/cp_dev/diag350_%j.out
cd /home1/yanggon/00_first_dir/26_memory_centric_gpu_h100
source scripts/amd/env.sh >/dev/null 2>&1
U=$($PY -c "import sys;sys.path.insert(0,'scripts');import amd_nvml as n;n.nvmlInit();print(n.nvmlDeviceGetUUID(n.nvmlDeviceGetHandleByIndex(0)))")
O=$WORK/cp_dev/diag350_$SLURM_JOB_ID; mkdir -p $O
run(){ name=$1; extra=$2; export APPTAINERENV_HIP_VISIBLE_DEVICES=0 AMD_EXTRA_ENV="PYTHONPATH=$PWD/scripts:$PWD/scripts/amd/cp,$extra"
  echo "$(date +%T) start $name [$extra]"; timeout 900 scripts/amd/vllm_in_container.sh scripts/amd/cp/cp_replay.py --config scripts/amd/cp/configs/qwen7b.json --gpu-uuid $U --contexts 1024 --widths 8 --chunks 0,64,256 --reps 2 --budget 4096 --output $O/$name > $O/$name.log 2>&1; echo "$(date +%T) end $name rc=$?"; }
run m128_cap512 "HSA_CU_MASK=0:0-127,CUBW_CU_COUNT_OVERRIDE=128"
run m192_cap512 "HSA_CU_MASK=0:0-191,CUBW_CU_COUNT_OVERRIDE=192"
run m128_cap32 "HSA_CU_MASK=0:0-127,CUBW_CU_COUNT_OVERRIDE=128,CP_MAX_CAPTURE=32"
run m128_cap256 "HSA_CU_MASK=0:0-127,CUBW_CU_COUNT_OVERRIDE=128,CP_MAX_CAPTURE=256"
run full_cap512 ""
