#!/usr/bin/env bash
# M8: long-window decode energy (amdsmi energy accumulator vs sampled socket power), graph vLLM.
# Usage: scripts/amd/lane_energy.sh <gpu-uuid> <result-root> <case>...
set -uo pipefail
uuid=$1; root=$2; shift 2
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
out_root=$root/m8_energy; mkdir -p "$out_root"; log=$out_root/lane.log
for acq in 0 1 2; do
  for name in "$@"; do
    cfg=$root/configs_energy/$name.json; out=$out_root/${name}_a$acq
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/run.json'))['status']=='complete' else 1)" 2>/dev/null; then continue; fi
    [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%s)"
    n=$(python3 -c "import json;print(json.load(open('$cfg'))['energy_generated_tokens'])")
    echo "$(date +%FT%T) start $name a$acq tokens=$n" >> "$log"
    timeout -s KILL 3600 scripts/amd/vllm_in_container.sh scripts/controlled_vllm_experiments.py --config "$cfg" \
      --gpu-uuid "$uuid" --mode energy --generated-tokens "$n" --output "$out" > "$out.log" 2>&1
    echo "$(date +%FT%T) end $name a$acq rc=$?" >> "$log"
  done
done
echo "$(date +%FT%T) lane complete" >> "$log"
