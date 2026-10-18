#!/usr/bin/env bash
# Owner-authorized multi-GPU extension: TP-N vLLM decode, full CUs and compute-matched (each GPU
# restricted to ~1/N of its CUs via HSA_CU_MASK, so total active CUs stay near one GPU's while HBM
# stacks, bandwidth and Infinity Cache scale with N). Three process acquisitions per condition.
# Usage: scripts/amd/lane_tp.sh <tp> <comma-uuids> <comma-physical-indices> <result-root> <case>...
set -uo pipefail
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
tp=$1; uuids=$2; idxs=$3; root=$4; shift 4
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
out_root=$root/tp; mkdir -p "$out_root"; log=$out_root/lane_tp${tp}_$(echo $idxs | tr , _).log
per=$(( (304 / tp + 7) / 8 * 8 ))   # ~1/N of 304 CUs, rounded up to a multiple of 8 (XCD-balanced)
mask=$(echo $idxs | tr , '\n' | sed "s/\$/:0-$((per-1))/" | paste -sd';')
for acq in 0 1 2; do
  for variant in full matched; do
    for name in "$@"; do
      out=$out_root/tp${tp}_${name}_${variant}_a$acq
      for attempt in 1 2 3; do
      if python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/run.json'))['status']=='complete' else 1)" 2>/dev/null; then break; fi
      [ -d "$out" ] && { mv "$out" "${out}_failed_$(date +%s)"; mv "$out.log" "${out}_failed_$(date +%s).log" 2>/dev/null; }
      extra=""; [ $variant = matched ] && extra="HSA_CU_MASK=$mask"
      echo "$(date +%FT%T) start $name $variant a$acq attempt $attempt ${extra}" >> "$log"
      # Persistent per-case compile cache: workers of one TP run raced when compiling into an empty cache
      # ("Cubin file saved by TritonBundler not found"), leaving the other ranks hung.
      AMD_CACHE_DIR=$WORK/cache/tp/tp${tp}_${name} AMD_EXTRA_ENV=$extra setsid timeout -s KILL 1500 \
        scripts/amd/vllm_in_container.sh scripts/amd/tp_vllm_experiments.py \
        --config "$root/configs/$name.json" --gpu-uuids "$uuids" --tp $tp --output "$out" > "$out.log" 2>&1 &
      pid=$!; t=0
      while kill -0 $pid 2>/dev/null; do sleep 10; t=$((t+10))
        if grep -q "Cubin file saved by TritonBundler not found\|unspecified launch failure" "$out.log" && [ $t -gt 60 ]; then
          sleep 20; kill -9 -- -$pid 2>/dev/null; echo "$(date +%FT%T) killed hung/failed rank" >> "$log"; break; fi
      done
      wait $pid 2>/dev/null
      echo "$(date +%FT%T) end $name $variant a$acq rc=$?" >> "$log"
      done
    done
  done
done
echo "$(date +%FT%T) lane complete" >> "$log"
