#!/bin/bash
#SBATCH -J mi325x-full
#SBATCH -p mi3258x
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -o /work1/proj_blaisetine/yanggon/mi325x_full_%j.out
# Unattended repeat of the MI300X campaign (allocation 468885) on the 8x MI325X node.
#
# Phase 0  preflight (scripts/amd/preflight_node.py): aborts within minutes if GPUs, torch, the vLLM
#          container, models, counters or CU masking are not as expected.
# Phase 1  eight single-GPU lanes in parallel (same roles as MI300X); each lane runs twice so the second
#          pass fills runs that failed sporadically.
# Phase 2  TP4 (GPUs 0-3) + TP2 (4,5) full and compute-matched; same-GPU competitor lanes (6, 7).
# Phase 3  TP8 full and compute-matched.
# Phase 4  TP at ~160 total CUs (TP4 on 0-3, TP2 on 4,5 and 6,7).
# Before each phase the remaining SLURM time is checked; a phase that cannot finish is recorded as
# skipped. Progress is appended to <root>/RUN_STATUS.log.
#
# Dry run on the login node: DRY_RUN=1 bash scripts/amd/mi325x_full.sbatch
# (prints every lane command with placeholder GPU IDs, checks every referenced config/binary, no GPU use)
set -uo pipefail
repo=/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100
src=$repo/results/amd_20261018_mi300x
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
DRY=${DRY_RUN:-0}
cd "$repo"
if [ "$DRY" = 1 ]; then name=dryrun_mi325x_full_$(date +%Y%m%d_%H%M%S); else name=amd_$(date +%Y%m%d)_mi325x_full_job${SLURM_JOB_ID}; fi
root=$WORK/results_amd/$name
mkdir -p "$root"
[ "$DRY" = 1 ] || ln -sfn "$root" "$repo/results/$name"   # publish scripts read results/amd_*; raw data stays in $WORK
status() { echo "$(date +%FT%T) $*" | tee -a "$root/RUN_STATUS.log"; }

# Configs: same models, shapes and statistics as MI300X; absolute paths rewritten to this root.
for d in configs configs_profile configs_energy; do
  mkdir -p "$root/$d"; cp "$src/$d"/*.json "$root/$d/"
done
python3 - "$root" "$src" <<'EOF'
import json, sys
from pathlib import Path
root, src = Path(sys.argv[1]), sys.argv[2]
p = root / 'configs/index.json'
rows = json.loads(p.read_text())
for r in rows:  # MI300X index paths may be relative; point every entry at this root's copy
    if 'config' in r:
        r['config'] = str(root / 'configs' / Path(r['config']).name)
p.write_text(json.dumps(rows, indent=2) + '\n')
EOF

# GPU table: real on the node, placeholders for a dry run.
if [ "$DRY" = 1 ]; then
  for i in 0 1 2 3 4 5 6 7; do printf '%s\tGPU-DRYRUN%02d\t-\n' $i $i; done > "$root/gpus.tsv"
  status "dry run: placeholder GPU table"
else
  source scripts/amd/env.sh
  status "phase0 preflight start (job $SLURM_JOB_ID on $(hostname))"
  if ! $PY scripts/amd/preflight_node.py --root "$root" > "$root/preflight.out" 2>&1; then
    status "phase0 preflight FAILED: $(tail -n 1 "$root/preflight.out")"; exit 2
  fi
  status "phase0 preflight ok: $(tail -n 1 "$root/preflight.out")"
fi
U=(); while IFS=$'\t' read -r i u b; do U[$i]=$u; done < "$root/gpus.tsv"
ALL=$(IFS=,; echo "${U[*]}")

# Every config a lane will read must exist before any GPU time is spent.
need=(); for c in bridge_graph_b1_c1024 bridge_eager_b1_c1024 bridge_graph_b8_c4096 bridge_eager_b8_c4096 qwen7b_graph_b1_c4096 \
  qwen7b_eager_b1_c4096 qwen7b_graph_b8_c4096 qwen7b_eager_b8_c4096 mistral7b_graph_b1_c4096 mistral7b_eager_b1_c4096 \
  mistral7b_graph_b8_c4096 mistral7b_eager_b8_c4096 qwen14b_graph_b1_c4096 qwen14b_eager_b1_c4096 qwen32b_graph_b1_c4096 \
  qwen32b_eager_b1_c4096 qwen7b_graph_b1_c1024 qwen7b_graph_b4_c1024 qwen7b_graph_b16_c1024 qwen7b_graph_b32_c1024 \
  qwen7b_graph_b64_c1024 bridge_graph_text_b1_c128 bridge_graph_text_b4_c128 bridge_base qwen7b_base qwen14b_base \
  qwen32b_base mistral7b_base; do need+=("$root/configs/$c.json"); done
for c in bridge_eager_b1_c1024 bridge_graph_b1_c1024 bridge_eager_b8_c4096 bridge_graph_b8_c4096 qwen7b_eager_b1_c4096 \
  qwen7b_graph_b1_c4096 qwen7b_eager_b8_c4096 mistral7b_eager_b1_c4096 mistral7b_graph_b1_c4096 mistral7b_eager_b8_c4096 \
  qwen14b_eager_b1_c4096 qwen32b_eager_b1_c4096; do need+=("$root/configs_profile/$c.json"); done
for c in bridge_graph_b1_c1024 bridge_graph_b8_c4096 qwen7b_graph_b1_c4096 qwen7b_graph_b8_c4096 mistral7b_graph_b1_c4096 \
  mistral7b_graph_b8_c4096 qwen14b_graph_b1_c4096 qwen32b_graph_b1_c4096; do need+=("$root/configs_energy/$c.json"); done
need+=("$WORK/bin/svc_bench_gfx942" "$WORK/bin/svc_bench_gfx942_v2" "$WORK/containers/vllm_rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0.sif")
miss=0; for f in "${need[@]}"; do [ -e "$f" ] || { status "MISSING $f"; miss=1; }; done
[ $miss = 0 ] || { status "aborting: missing inputs"; exit 3; }
status "inputs ok (${#need[@]} files)"

lane() {  # lane <label> <command...>: run (or print in a dry run) one lane step, logging start/end
  local label=$1; shift
  if [ "$DRY" = 1 ]; then echo "[$label] $*" >> "$root/DRYRUN_COMMANDS.txt"; return 0; fi
  status "$label start: $*"; "$@" > "$root/lane_${label}.log" 2>&1; status "$label end rc=$?"
}
remaining() {  # seconds left in this allocation (large in a dry run)
  [ "$DRY" = 1 ] && { echo 99999; return; }
  local end; end=$(squeue -h -j "$SLURM_JOB_ID" -o %e); echo $(( $(date -d "$end" +%s) - $(date +%s) ))
}
phase_ok() {  # phase_ok <name> <needed-seconds>
  local left; left=$(remaining)
  if [ "$left" -lt "$2" ]; then status "$1 SKIPPED: ${left}s left < ${2}s needed"; return 1; fi
  status "$1 start (${left}s left)"; return 0
}

if [ "$DRY" != 1 ]; then
  $PY scripts/amd/gpu_monitor.py $(for u in "${U[@]}"; do printf -- '--gpu-uuid %s ' "$u"; done) --interval 0.1 \
    --output "$root/node_monitor.jsonl" > "$root/monitor.log" 2>&1 &
  MON=$!
fi
A=scripts/amd

# ---- Phase 1: single-GPU lanes (same GPU roles as MI300X) ----
if phase_ok phase1 14400; then
  ( for p in 1 2; do lane g0_m1_p$p $A/lane_m1_baselines.sh ${U[0]} $root; done ) &
  ( for p in 1 2; do lane g1_counters_p$p $A/lane_counters.sh ${U[1]} $root bridge_eager_b1_c1024 bridge_graph_b1_c1024 \
      bridge_eager_b8_c4096 qwen7b_eager_b1_c4096 qwen7b_eager_b8_c4096; done ) &
  ( lane g2_service $A/service_sweep_lane.sh ${U[2]} 2 $root
    for p in 1 2; do lane g2_batch_p$p $A/lane_cumask.sh ${U[2]} 2 $root qwen7b_graph_b1_c1024 qwen7b_graph_b4_c1024 \
      qwen7b_graph_b16_c1024 qwen7b_graph_b32_c1024 qwen7b_graph_b64_c1024; done ) &
  ( for p in 1 2; do lane g3_cumask_p$p $A/lane_cumask.sh ${U[3]} 3 $root bridge_graph_b1_c1024 bridge_eager_b1_c1024 \
      bridge_graph_b8_c4096 qwen7b_graph_b1_c4096 qwen7b_graph_b8_c4096; done
    for p in 1 2; do lane g3_prefill_p$p $A/lane_prefill.sh ${U[3]} 3 $root qwen7b_graph_b1_c4096 qwen7b_graph_b8_c4096 \
      bridge_graph_b8_c4096 qwen32b_graph_b1_c4096; done ) &
  ( for p in 1 2; do lane g4_energy_p$p $A/lane_energy.sh ${U[4]} $root bridge_graph_b1_c1024 bridge_graph_b8_c4096 \
      qwen7b_graph_b1_c4096 qwen7b_graph_b8_c4096 mistral7b_graph_b1_c4096 mistral7b_graph_b8_c4096 qwen14b_graph_b1_c4096 \
      qwen32b_graph_b1_c4096; done
    lane g4_energy_cu $A/lane_energy_cu.sh ${U[4]} 4 $root qwen7b_graph_b1_c4096 qwen32b_graph_b1_c4096 ) &
  ( for p in 1 2; do lane g5_counters_p$p $A/lane_counters.sh ${U[5]} $root bridge_graph_b8_c4096 mistral7b_eager_b1_c4096 \
      mistral7b_graph_b1_c4096 mistral7b_eager_b8_c4096 qwen14b_eager_b1_c4096 qwen32b_eager_b1_c4096 qwen7b_graph_b1_c4096; done ) &
  ( lane g6_rocprof_compute $A/lane_rocprof_compute.sh 6 $root
    for p in 1 2; do lane g6_repro_p$p $A/lane_cases.sh ${U[6]} $root m1_gpu6 bridge_graph_b1_c1024 bridge_graph_b8_c4096 \
      qwen7b_graph_b1_c4096 qwen7b_graph_b8_c4096 mistral7b_graph_b1_c4096 mistral7b_graph_b8_c4096 qwen14b_graph_b1_c4096 \
      qwen32b_graph_b1_c4096; done ) &
  ( lane g7_pressure $A/lane_pressure.sh ${U[7]} $root 'bridge 1 1024' 'bridge 1 2048' 'bridge 1 4096' 'bridge 8 4096' 'qwen7b 1 4096'
    for p in 1 2; do lane g7_cumask_p$p $A/lane_cumask.sh ${U[7]} 7 $root mistral7b_graph_b1_c4096 mistral7b_graph_b8_c4096 \
      qwen14b_graph_b1_c4096 qwen32b_graph_b1_c4096; done
    lane g7_text $A/lane_cases.sh ${U[7]} $root m1_text bridge_graph_text_b1_c128 bridge_graph_text_b4_c128 ) &
  wait
  status "phase1 end"
fi

# ---- Phase 2: TP4 / TP2 and same-GPU competitor ----
if phase_ok phase2 7200; then
  lane tp4 $A/lane_tp.sh 4 ${U[0]},${U[1]},${U[2]},${U[3]} 0,1,2,3 $root bridge_graph_b1_c1024 qwen7b_graph_b1_c4096 \
    qwen7b_graph_b8_c4096 mistral7b_graph_b1_c4096 qwen14b_graph_b1_c4096 qwen32b_graph_b1_c4096 &
  lane tp2 $A/lane_tp.sh 2 ${U[4]},${U[5]} 4,5 $root bridge_graph_b1_c1024 qwen7b_graph_b1_c4096 qwen7b_graph_b8_c4096 \
    mistral7b_graph_b1_c4096 qwen14b_graph_b1_c4096 qwen32b_graph_b1_c4096 &
  lane j6_g6 $A/lane_competitor.sh ${U[6]} 6 $root bridge_graph_b1_c1024 qwen7b_graph_b1_c4096 &
  lane j6_g7 $A/lane_competitor.sh ${U[7]} 7 $root qwen32b_graph_b1_c4096 qwen7b_graph_b8_c4096 &
  wait
  status "phase2 end"
fi

# ---- Phase 3: TP8 ----
if phase_ok phase3 3600; then
  lane tp8 $A/lane_tp.sh 8 $ALL 0,1,2,3,4,5,6,7 $root mistral7b_graph_b1_c4096 qwen14b_graph_b1_c4096 qwen32b_graph_b1_c4096
  status "phase3 end"
fi

# ---- Phase 4: TP at ~160 total CUs (compare with TP1 at 152 CUs from M6) ----
if phase_ok phase4 3600; then
  lane tp4_152 env TP_VARIANTS=matched152 TP_MATCH_TOTAL=152 $A/lane_tp.sh 4 ${U[0]},${U[1]},${U[2]},${U[3]} 0,1,2,3 $root \
    qwen7b_graph_b1_c4096 qwen14b_graph_b1_c4096 qwen32b_graph_b1_c4096 &
  lane tp2_152a env TP_VARIANTS=matched152 TP_MATCH_TOTAL=152 $A/lane_tp.sh 2 ${U[4]},${U[5]} 4,5 $root \
    qwen7b_graph_b1_c4096 qwen14b_graph_b1_c4096 &
  lane tp2_152b env TP_VARIANTS=matched152 TP_MATCH_TOTAL=152 $A/lane_tp.sh 2 ${U[6]},${U[7]} 6,7 $root qwen32b_graph_b1_c4096 &
  wait
  status "phase4 end"
fi

[ "$DRY" = 1 ] || kill $MON 2>/dev/null
status "COMPLETE"
