#!/usr/bin/env bash
# M5: three acquisitions of normal-vs-pressure for selected HF compiled cases (host module env).
# Usage: scripts/amd/lane_pressure.sh <gpu-uuid> <result-root> "<model> <batch> <context>"...
set -uo pipefail
uuid=$1; root=$2; shift 2
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"; source scripts/amd/env.sh
out=$root/m5_pressure; mkdir -p "$out"
cases=("$@")
for acq in 0 1 2; do for c in "${cases[@]}"; do read -r m b x <<< "$c"; f=$out/${m}_b${b}_c${x}_a$acq.json
  [ -s "$f" ] && continue
  echo "$(date +%FT%T) start $m b$b c$x a$acq" >> "$out/lane.log"
  timeout -s KILL 2400 $PY scripts/amd/pressure_bench.py --config "$root/configs/${m}_base.json" --gpu-uuid "$uuid" \
    --batch $b --context $x --acquisition $acq --output "$f" > "$out/${m}_b${b}_c${x}_a$acq.log" 2>&1
  echo "$(date +%FT%T) end $m b$b c$x a$acq rc=$?" >> "$out/lane.log"
done; done
echo "$(date +%FT%T) lane complete" >> "$out/lane.log"
