#!/usr/bin/env bash
# M6: vLLM decode under process-wide CU restriction (HSA_CU_MASK=<physical index>:0-(k-1)).
# Contiguous CU indices interleave across the 8 XCDs on MI300X, so k a multiple of 8 trims every XCD
# equally. Each CU level first records an FMA throughput control under the same mask as proof.
# Usage: scripts/amd/lane_cumask.sh <gpu-uuid> <physical-index> <result-root> <case>...
set -uo pipefail
uuid=$1; idx=$2; root=$3; shift 3
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
out_root=$root/m6_cumask; mkdir -p "$out_root"; log=$out_root/lane.log
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
for acq in 0 1 2; do
  for cus in 304 232 152 96; do
    mask=""; [ "$cus" -lt 304 ] && mask="$idx:0-$((cus-1))"
    proof=$out_root/proof_cu${cus}_a$acq.json
    [ -s "$proof" ] || HIP_VISIBLE_DEVICES=$idx HSA_CU_MASK=$mask timeout -s KILL 60 \
      "$WORK/bin/svc_bench_gfx942" --mode fma --iters 20000 --trials 3 --reps 1 > "$proof" 2>&1
    for name in "$@"; do
      out=$out_root/${name}_cu${cus}_a$acq
      if python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/run.json'))['status']=='complete' else 1)" 2>/dev/null; then continue; fi
      [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%s)"
      echo "$(date +%FT%T) start $name cu$cus a$acq mask=$mask" >> "$log"
      AMD_EXTRA_ENV=${mask:+HSA_CU_MASK=$mask} timeout -s KILL 3600 scripts/amd/vllm_in_container.sh \
        scripts/controlled_vllm_experiments.py --config "$root/configs/$name.json" --gpu-uuid "$uuid" \
        --mode timing --output "$out" > "$out.log" 2>&1
      echo "$(date +%FT%T) end $name cu$cus a$acq rc=$?" >> "$log"
    done
  done
done
echo "$(date +%FT%T) lane complete" >> "$log"
