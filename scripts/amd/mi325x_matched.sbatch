#!/bin/bash
#SBATCH -J mi325x-matched
#SBATCH -p mi3258x
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o /work1/proj_blaisetine/yanggon/mi325x_matched_%j.out
# Compute-matched MI325X vs MI300X (bandwidth effect at fixed compute, r > 1 at c ~ 1).
# MI325X is restricted to 264 CUs, the count that scales its measured register-resident BF16 MFMA rate
# (1230 TFLOP/s at 304 CUs, job 468949) to MI300X's 1071 TFLOP/s; the job re-measures that match.
# Full (304) and matched (264) conditions run in the same job on GPU 0 for a paired comparison.
# Remaining differences (HBM3E vs HBM3, power cap, clock behaviour) are recorded, not removed.
set -uo pipefail
repo=/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100
cd "$repo"; source scripts/amd/env.sh
root=$repo/results/amd_$(date +%Y%m%d)_mi325x_matched_job${SLURM_JOB_ID}; mkdir -p "$root/configs" "$root/m1"
uuid=$($PY -c "import sys; sys.path.insert(0,'scripts'); import amd_nvml as n; n.nvmlInit(); print(n.nvmlDeviceGetUUID(n.nvmlDeviceGetHandleByIndex(0)))")
echo "uuid=$uuid" > "$root/selected_gpu.txt"
$PY scripts/inventory.py --gpu-uuid "$uuid" --check-cuda --output "$root/environment.json"
$PY scripts/amd/gpu_monitor.py --gpu-uuid "$uuid" --interval 0.1 --output "$root/node_monitor.jsonl" &
mon=$!
cases="qwen7b_graph_b1_c4096 qwen7b_graph_b8_c4096 mistral7b_graph_b1_c4096 qwen14b_graph_b1_c4096 qwen32b_graph_b1_c4096"
for c in $cases; do cp "results/amd_20261018_mi300x/configs/$c.json" "$root/configs/"; done
python3 scripts/amd/service_sweep.py --binary "$WORK/bin/svc_bench_gfx942" --gpu-uuid "$uuid" --physical-index 0 \
  --output "$root/m4_service" --cus 304 264 > "$root/m4_service.log" 2>&1
for acq in 0 1 2; do for cus in 304 264; do
  mask=""; [ $cus -lt 304 ] && mask="0:0-$((cus-1))"
  for c in $cases; do
    out=$root/m1/${c}_cu${cus}_a$acq
    AMD_EXTRA_ENV=${mask:+HSA_CU_MASK=$mask} timeout -s KILL 2400 scripts/amd/vllm_in_container.sh \
      scripts/controlled_vllm_experiments.py --config "$root/configs/$c.json" --gpu-uuid "$uuid" --mode timing \
      --output "$out" > "$out.log" 2>&1
    echo "$(date +%FT%T) $c cu$cus a$acq rc=$?" >> "$root/m1_lane.log"
  done
done; done
kill $mon
python3 scripts/amd/summarize_service.py --input "$root/m4_service/service.jsonl" --output "$root/m4_service/summary.json" > "$root/m4_service/summary.txt"
python3 scripts/amd/summarize_timing.py --root "$root/m1" --monitor "$root/node_monitor.jsonl" --output "$root/m1/summary.json" > "$root/m1/summary.txt"
echo "done $(date)" > "$root/COMPLETE"
