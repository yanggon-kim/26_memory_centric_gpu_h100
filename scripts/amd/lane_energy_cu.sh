#!/usr/bin/env bash
# M8 x M6: decode energy under CU restriction (software proxy: masked CUs still exist and leak;
# this is not removed silicon). Same GPU for every condition. Usage: <gpu-uuid> <physical-index> <root> <case>...
set -uo pipefail
uuid=$1; idx=$2; root=$3; shift 3
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
out_root=$root/m8_energy_cu; mkdir -p "$out_root"; log=$out_root/lane.log
for acq in 0 1 2; do for cus in 304 152 96; do
  mask=""; [ "$cus" -lt 304 ] && mask="$idx:0-$((cus-1))"
  for name in "$@"; do
    cfg=$root/configs_energy/$name.json; out=$out_root/${name}_cu${cus}_a$acq
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$out/run.json'))['status']=='complete' else 1)" 2>/dev/null; then continue; fi
    [ -d "$out" ] && mv "$out" "${out}_failed_$(date +%s)"
    n=$(python3 -c "import json;print(json.load(open('$cfg'))['energy_generated_tokens'])")
    echo "$(date +%FT%T) start $name cu$cus a$acq" >> "$log"
    AMD_EXTRA_ENV=${mask:+HSA_CU_MASK=$mask} timeout -s KILL 3600 scripts/amd/vllm_in_container.sh \
      scripts/controlled_vllm_experiments.py --config "$cfg" --gpu-uuid "$uuid" --mode energy --generated-tokens "$n" \
      --output "$out" > "$out.log" 2>&1
    echo "$(date +%FT%T) end $name cu$cus a$acq rc=$?" >> "$log"
  done
done; done
echo "$(date +%FT%T) lane complete" >> "$log"
