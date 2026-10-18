#!/usr/bin/env bash
# M1 timelines and M2/M3 counters for focused vLLM cases, inside the ROCm vLLM container.
# Usage: scripts/amd/lane_profile.sh <gpu-uuid> <result-root> <case-name>...   (case = e.g. bridge_eager_b1_c1024)
# One counter group per rocprofv3 run: FETCH_SIZE+WRITE_SIZE together exceed the gfx942 counter budget.
# Every profiled run is instrumented (--external-profiler) and never a baseline timing.
set -uo pipefail
uuid=$1; root=$2; shift 2
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
mkdir -p "$root/m2_profile"; log=$root/m2_profile/lane_$(echo $uuid | cut -c5-12).log
groups=("trace" "FETCH_SIZE" "WRITE_SIZE" "TCC_HIT TCC_MISS" "SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_WAVES GRBM_GUI_ACTIVE")
for name in "$@"; do
  cfg=$root/configs_profile/$name.json
  for g in "${groups[@]}"; do
    tag=$(echo "$g" | tr ' ' '+'); out=$root/m2_profile/$name/$tag
    [ -f "$out/done" ] && continue
    rm -rf "$out"; mkdir -p "$out"
    if [ "$g" = trace ]; then prof=(--kernel-trace --marker-trace --memory-copy-trace --stats)
    else prof=(--kernel-trace --marker-trace --pmc $g); fi
    echo "$(date +%FT%T) start $name $tag" >> "$log"
    AMD_ENTRY=rocprofv3 timeout -s KILL 2400 scripts/amd/vllm_in_container.sh "${prof[@]}" --output-format csv -d "$out/rocprof" -o p -- \
      python3 scripts/controlled_vllm_experiments.py --config "$cfg" --gpu-uuid "$uuid" --mode timing \
      --external-profiler --generated-tokens 16 --output "$out/run" > "$out/log.txt" 2>&1
    rc=$?; echo "$(date +%FT%T) end $name $tag rc=$rc" >> "$log"
    [ $rc -eq 0 ] && touch "$out/done"
  done
done
echo "$(date +%FT%T) lane complete" >> "$log"
