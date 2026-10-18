#!/usr/bin/env bash
# M1: three independent process acquisitions of every focused vLLM case, graph and eager interleaved.
# Usage (on the allocation): scripts/amd/lane_m1_baselines.sh <gpu-uuid> <result-root>
# Completed acquisitions are skipped; failed ones keep their directory and log and the lane continues.
set -uo pipefail
uuid=$1; root=$2
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"
log=$root/m1/lane.log; mkdir -p "$root/m1"
cases=$(python3 -c "import json;[print(e['config']) for e in json.load(open('$root/configs/index.json')) if e['status']=='materialized' and 'b1_c128' not in e['config']]")
for acq in 0 1 2; do
  for cfg in $cases; do
    name=$(basename "$cfg" .json); out=$root/m1/${name}_a$acq
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/run.json'))['status']=='complete' else 1)" 2>/dev/null; then continue; fi
    [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%s)"
    echo "$(date +%FT%T) start $name a$acq" >> "$log"
    timeout -s KILL 3600 scripts/amd/vllm_in_container.sh scripts/controlled_vllm_experiments.py --config "$cfg" \
      --gpu-uuid "$uuid" --mode timing --output "$out" > "$out.log" 2>&1
    echo "$(date +%FT%T) end $name a$acq rc=$?" >> "$log"
  done
done
echo "$(date +%FT%T) lane complete" >> "$log"
