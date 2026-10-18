#!/usr/bin/env bash
# M2/M3 per-dispatch counters for vLLM cases (inside the ROCm container), one counter group per run.
# Runtime-loaded Triton kernels (names starting "_" or "triton") are excluded from counter collection:
# collecting on them hung or aborted rocprofv3 ("AQL packet is malformed"). Their dispatches remain in
# the kernel trace, so per-step coverage of the excluded kernels is measurable.
# Usage: scripts/amd/lane_counters.sh <gpu-uuid> <result-root> <case>...
set -uo pipefail
uuid=$1; root=$2; shift 2
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
mkdir -p "$root/m2_counters"; log=$root/m2_counters/lane_$(echo $uuid | cut -c5-12).log
groups=("FETCH_SIZE" "WRITE_SIZE" "TCC_HIT TCC_MISS" "SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_WAVES GRBM_GUI_ACTIVE")
for name in "$@"; do
  for g in "${groups[@]}"; do
    tag=$(echo "$g" | tr ' ' '+'); out=$root/m2_counters/$name/$tag
    [ -f "$out/done" ] && continue
    rm -rf "$out"; mkdir -p "$out"
    echo "$(date +%FT%T) start $name $tag" >> "$log"
    AMD_ENTRY=rocprofv3 timeout -s KILL 900 scripts/amd/vllm_in_container.sh --kernel-trace --marker-trace --pmc $g \
      --kernel-exclude-regex '^(_|triton)' --output-format csv -d "$out/rocprof" -o p -- \
      python3 scripts/controlled_vllm_experiments.py --config "$root/configs_profile/$name.json" --gpu-uuid "$uuid" \
      --mode timing --external-profiler --generated-tokens 16 --output "$out/run" > "$out/log.txt" 2>&1
    rc=$?; echo "$(date +%FT%T) end $name $tag rc=$rc" >> "$log"
    [ $rc -eq 0 ] && touch "$out/done"
  done
done
echo "$(date +%FT%T) lane complete" >> "$log"
