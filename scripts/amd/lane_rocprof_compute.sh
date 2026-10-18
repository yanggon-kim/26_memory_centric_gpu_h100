#!/usr/bin/env bash
# rocprof-compute (Omniperf) on the short HF decode workload, one single-pass metric set at a time.
# The default full profile crashed the workload (illegal address) in its first 100+ counter pass,
# so each curated set and the roofline run separately; failures are kept.
# Usage: scripts/amd/lane_rocprof_compute.sh <rocr-index> <result-root> <batch> <context>
set -uo pipefail
idx=$1; root=$2; b=$3; c=$4
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"; source scripts/amd/env.sh
export ROCR_VISIBLE_DEVICES=$idx PATH=$WORK/bin/rpc_shim:$PATH
out=$root/m3_rocprof_compute; mkdir -p "$out"; cd "$out"
RPC=(env PYTHONPATH=/opt/rocm-7.1.0/share/amd_smi "$WORK/venvs/rocprof_compute_py312/bin/python" /opt/rocm-7.1.0/bin/rocprof-compute)
APP=(env PYTHONPATH=$PYTHONPATH "$PY" "$repo/scripts/amd/decode_profile_bench.py" --config "$root/configs/bridge_base.json"
     --batch "$b" --context "$c" --tokens 8 --warmups 1 --output "$out/replay_ids_b${b}_c${c}.jsonl")
for s in launch_stats compute_thruput_util compute_thruput_flops mem_thruput roof; do
  name=hf_bridge_b${b}_c${c}_$s
  [ -f "$out/$name.done" ] && continue
  if [ $s = roof ]; then extra=(--roof-only); else extra=(--set $s); fi
  echo "$(date +%FT%T) start $name" >> lane_sets.log
  timeout -s KILL 1800 "${RPC[@]}" profile -n "$name" "${extra[@]}" -- "${APP[@]}" > "$name.log" 2>&1
  rc=$?; echo "$(date +%FT%T) end $name rc=$rc" >> lane_sets.log
  grep -q "illegal memory access\|AQL packet\|rocprofv3 caught signal" "$name.log" && rc=99
  [ $rc -eq 0 ] && touch "$out/$name.done"
done
