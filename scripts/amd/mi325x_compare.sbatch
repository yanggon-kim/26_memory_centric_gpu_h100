#!/bin/bash
#SBATCH -J mi325x-compare
#SBATCH -p mi3258x
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o /work1/proj_blaisetine/yanggon/mi325x_compare_%j.out
# Self-contained MI325X comparison for the same-compute bandwidth pair (MI300X vs MI325X, c ~ 1, r > 1).
# Runs on GPU 0 only: full-CU service sweep (M4) and graph-mode focused baselines (M1, 3 acquisitions),
# with a GPU monitor. Cross-SKU confounds (power cap, HBM3E, clocks) are recorded, not removed.
set -uo pipefail
repo=/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100
cd "$repo"; source scripts/amd/env.sh
root=$repo/results/amd_$(date +%Y%m%d)_mi325x_job${SLURM_JOB_ID}; mkdir -p "$root/configs"
uuid=$($PY -c "import sys; sys.path.insert(0,'scripts'); import amd_nvml as n; n.nvmlInit(); print(n.nvmlDeviceGetUUID(n.nvmlDeviceGetHandleByIndex(0)))")
echo "uuid=$uuid" > "$root/selected_gpu.txt"
$PY scripts/inventory.py --gpu-uuid "$uuid" --check-cuda --output "$root/environment.json"
$PY scripts/amd/gpu_monitor.py --gpu-uuid "$uuid" --interval 0.1 --output "$root/node_monitor.jsonl" &
mon=$!
# Same focused graph configs as MI300X (same models, shapes and statistics).
for c in bridge_graph_b1_c1024 bridge_graph_b8_c4096 qwen7b_graph_b1_c4096 qwen7b_graph_b8_c4096 \
         mistral7b_graph_b1_c4096 mistral7b_graph_b8_c4096 qwen14b_graph_b1_c4096 qwen32b_graph_b1_c4096; do
  cp "results/amd_20261018_mi300x/configs/$c.json" "$root/configs/"
done
python3 -c "
import json, sys; from pathlib import Path
d = Path(sys.argv[1])
rows = [{'config': str(c), 'engine': 'graph', 'status': 'materialized'} for c in sorted(d.glob('*_graph_*.json'))]
(d / 'index.json').write_text(json.dumps(rows, indent=1))" "$root/configs"
python3 scripts/amd/service_sweep.py --binary "$WORK/bin/svc_bench_gfx942" --gpu-uuid "$uuid" --physical-index 0 \
  --output "$root/m4_service" --cus 304 > "$root/m4_service.log" 2>&1
scripts/amd/lane_m1_baselines.sh "$uuid" "$root" > "$root/m1_lane.log" 2>&1
kill $mon
python3 scripts/amd/summarize_service.py --input "$root/m4_service/service.jsonl" --output "$root/m4_service/summary.json" > "$root/m4_service/summary.txt"
python3 scripts/amd/summarize_timing.py --root "$root/m1" --monitor "$root/node_monitor.jsonl" --output "$root/m1/summary.json" > "$root/m1/summary.txt"
echo "done $(date)" > "$root/COMPLETE"
