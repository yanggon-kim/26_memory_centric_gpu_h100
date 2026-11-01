#!/bin/bash
#SBATCH -J cuprobe-smoke
#SBATCH -o /work1/proj_blaisetine/yanggon/smoke/cuprobe_smoke.%j.out
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 00:15:00
#SBATCH -p devel
# MI210 smoke of svc_bench_v4 cuprobe: alone, beside a native read co-runner on disjoint CUs, and beside a
# positive-control cuprobe co-runner overlapping 8 probe CUs.
set -u
B=/work1/proj_blaisetine/yanggon/bin/svc_bench_gfx90a_v4
O=/work1/proj_blaisetine/yanggon/smoke/cuprobe_$SLURM_JOB_ID; mkdir -p $O
export HIP_VISIBLE_DEVICES=0
run() { # name probe_mask [corunner_mode corunner_mask]
  HSA_CU_MASK=0:$2 $B --mode cuprobe --seconds 12 --iters 2000 > $O/$1.jsonl 2> $O/$1.err & p=$!
  if [ -n "${3:-}" ]; then sleep 1; HSA_CU_MASK=0:$4 $B --mode $3 --seconds 10 --iters 2000 --bytes 1073741824 > $O/$1_co.jsonl 2>&1 & c=$!; wait $c; fi
  wait $p; echo "$1 rc=$?"
}
run alone 52-103
run read_disjoint 52-103 read 0-51
run fma_disjoint 52-103 fma 0-51
run cuprobe_overlap8 52-103 cuprobe 0-59
echo done
