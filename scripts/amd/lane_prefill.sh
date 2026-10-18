#!/usr/bin/env bash
# M8 counterexample: prefill (all-request prompt processing + first token) wall time vs CU restriction.
# Each run generates 2 tokens (one decode tick) so the per-trial prefill_wall_ms is the measurement.
# Usage: scripts/amd/lane_prefill.sh <gpu-uuid> <physical-index> <result-root> <case>...
set -uo pipefail
uuid=$1; idx=$2; root=$3; shift 3
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
out_root=$root/m8_prefill; mkdir -p "$out_root"; log=$out_root/lane.log
for acq in 0 1 2; do for cus in 304 232 152 96; do
  mask=""; [ "$cus" -lt 304 ] && mask="$idx:0-$((cus-1))"
  for name in "$@"; do
    out=$out_root/${name}_cu${cus}_a$acq
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/run.json'))['status']=='complete' else 1)" 2>/dev/null; then continue; fi
    [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%s)"
    echo "$(date +%FT%T) start $name cu$cus a$acq" >> "$log"
    AMD_EXTRA_ENV=${mask:+HSA_CU_MASK=$mask} timeout -s KILL 2400 scripts/amd/vllm_in_container.sh \
      scripts/controlled_vllm_experiments.py --config "$root/configs/$name.json" --gpu-uuid "$uuid" \
      --mode timing --generated-tokens 2 --output "$out" > "$out.log" 2>&1
    echo "$(date +%FT%T) end $name cu$cus a$acq rc=$?" >> "$log"
  done
done; done
echo "$(date +%FT%T) lane complete" >> "$log"
