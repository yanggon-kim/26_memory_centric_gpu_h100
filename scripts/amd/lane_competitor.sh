#!/usr/bin/env bash
# Fixed-compute bandwidth reduction (contract J6, r < 1 at constant c): the decoder runs on CUs 0-231 of one
# GPU; an owned competitor process runs on the disjoint CUs 232-303 (both XCD-balanced) and streams HBM
# at a requested duty cycle, lowering the bandwidth left for the decoder without changing its CUs.
# An arithmetic-only competitor at full duty separates power/clock interference from bandwidth stealing.
# The competitor logs its achieved bandwidth in 0.25 s windows for joining to decoder trials.
# This is shared-service sensitivity, not a physical change of memory interfaces.
# Usage: scripts/amd/lane_competitor.sh <gpu-uuid> <physical-index> <result-root> <case>...
set -uo pipefail
uuid=$1; idx=$2; root=$3; shift 3
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
B=$WORK/bin/svc_bench_gfx942_v2
out_root=$root/j6_competitor; mkdir -p "$out_root"; log=$out_root/lane_${idx}.log
treatments=(solo read25 read50 read75 read100 fma100)
for acq in 0 1 2; do
  # rotate treatment order per acquisition so drift does not align with one treatment
  for k in 0 1 2 3 4 5; do t=${treatments[$(( (k + acq * 2) % 6 ))]}
    for name in "$@"; do
      out=$out_root/${name}_${t}_a$acq
      if python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/run.json'))['status']=='complete' else 1)" 2>/dev/null; then continue; fi
      [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%s)"
      cpid=""  # the runner creates $out itself; competitor logs sit beside it
      case $t in
        read*) duty=$(python3 -c "print(${t#read}/100)")
               setsid env HIP_VISIBLE_DEVICES=$idx HSA_CU_MASK=$idx:232-303 "$B" --mode read --bytes 1073741824 \
                 --seconds 900 --duty $duty > "$out.competitor.jsonl" 2> "$out.competitor.err" & cpid=$! ;;
        fma*)  setsid env HIP_VISIBLE_DEVICES=$idx HSA_CU_MASK=$idx:232-303 "$B" --mode fma --iters 2000 \
                 --seconds 900 > "$out.competitor.jsonl" 2> "$out.competitor.err" & cpid=$! ;;
      esac
      [ -n "$cpid" ] && sleep 3
      echo "$(date +%FT%T) start $name $t a$acq competitor_pid=${cpid:-none}" >> "$log"
      extra="HSA_CU_MASK=$idx:0-231"; [ -n "$cpid" ] && extra="$extra,RESEARCH_ALLOWED_PIDS=$cpid"
      AMD_EXTRA_ENV=$extra timeout -s KILL 2400 scripts/amd/vllm_in_container.sh scripts/controlled_vllm_experiments.py \
        --config "$root/configs/$name.json" --gpu-uuid "$uuid" --mode timing --output "$out" > "$out.log" 2>&1
      rc=$?
      [ -n "$cpid" ] && { kill $cpid 2>/dev/null; sleep 1; kill -9 $cpid 2>/dev/null; }
      echo "$(date +%FT%T) end $name $t a$acq rc=$rc" >> "$log"
    done
  done
done
echo "$(date +%FT%T) lane complete" >> "$log"
