#!/usr/bin/env bash
# Generic three-acquisition vLLM timing lane for explicitly named configs (no CU mask).
# Usage: scripts/amd/lane_cases.sh <gpu-uuid> <result-root> <subdir> <config-name>...
set -uo pipefail
uuid=$1; root=$2; sub=$3; shift 3
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
out_root=$root/$sub; mkdir -p "$out_root"; log=$out_root/lane.log
for acq in 0 1 2; do for name in "$@"; do
  out=$out_root/${name}_a$acq
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/run.json'))['status']=='complete' else 1)" 2>/dev/null; then continue; fi
  [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%s)"
  echo "$(date +%FT%T) start $name a$acq" >> "$log"
  timeout -s KILL 2400 scripts/amd/vllm_in_container.sh scripts/controlled_vllm_experiments.py --config "$root/configs/$name.json" \
    --gpu-uuid "$uuid" --mode timing --output "$out" > "$out.log" 2>&1
  echo "$(date +%FT%T) end $name a$acq rc=$?" >> "$log"
done; done
echo "$(date +%FT%T) lane complete" >> "$log"
