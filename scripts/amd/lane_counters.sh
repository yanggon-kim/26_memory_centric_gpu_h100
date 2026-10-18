#!/usr/bin/env bash
# M2/M3 per-dispatch counters for vLLM cases (inside the ROCm container), one counter group per run.
# Runtime-loaded Triton kernels (names starting "_" or "triton") are excluded from counter collection;
# collecting on them hung or aborted rocprofv3 ("AQL packet is malformed"). Runs still abort
# intermittently right after those code objects load, so each group gets up to 4 attempts (all kept).
# Each attempt runs in its own process group. A hang in rocprofv3's crash handler ("will wait for")
# or the time limit kills the whole group, and the next attempt waits until the GPU lock is free.
# Usage: scripts/amd/lane_counters.sh <gpu-uuid> <result-root> <case>...
set -uo pipefail
uuid=$1; root=$2; shift 2
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
mkdir -p "$root/m2_counters"; log=$root/m2_counters/lane_$(echo $uuid | cut -c5-12).log
lock=/tmp/h100_research_$uuid.lock
groups=("FETCH_SIZE" "WRITE_SIZE" "TCC_HIT TCC_MISS" "SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_WAVES GRBM_GUI_ACTIVE")
for name in "$@"; do
  for g in "${groups[@]}"; do
    tag=$(echo "$g" | tr ' ' '+'); out=$root/m2_counters/$name/$tag
    for attempt in 1 2 3 4; do
      [ -f "$out/done" ] && break
      [ -d "$out" ] && mv "$out" "${out}_attempt_failed_$(date +%s%N)"
      mkdir -p "$out"
      for i in $(seq 120); do flock -n "$lock" true 2>/dev/null && break; sleep 5; done
      echo "$(date +%FT%T) start $name $tag attempt $attempt" >> "$log"
      setsid env AMD_ENTRY=rocprofv3 scripts/amd/vllm_in_container.sh --kernel-trace --marker-trace --pmc $g \
        --kernel-exclude-regex '^(_|triton)' --output-format csv -d "$out/rocprof" -o p -- \
        python3 scripts/controlled_vllm_experiments.py --config "$root/configs_profile/$name.json" --gpu-uuid "$uuid" \
        --mode timing --external-profiler --generated-tokens 16 --output "$out/run" > "$out/log.txt" 2>&1 &
      pid=$!; start=$(date +%s); verdict=""
      while kill -0 $pid 2>/dev/null; do
        sleep 5
        if grep -q "rocprofv3 will wait for" "$out/log.txt"; then verdict=crash-handler-hang; sleep 10; fi
        [ $(( $(date +%s) - start )) -gt 900 ] && verdict=timeout
        if [ -n "$verdict" ]; then kill -9 -- -$pid 2>/dev/null; sleep 3; break; fi
      done
      wait $pid 2>/dev/null; rc=$?
      ok=no; python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/run/run.json'))['status']=='complete' else 1)" 2>/dev/null && ok=yes
      echo "$(date +%FT%T) end $name $tag rc=$rc complete=$ok ${verdict}" >> "$log"
      [ $ok = yes ] && touch "$out/done"
    done
  done
done
echo "$(date +%FT%T) lane complete" >> "$log"
