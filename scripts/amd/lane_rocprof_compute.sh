#!/usr/bin/env bash
# rocprof-compute (Omniperf) detailed analysis.
# Findings that shaped this lane: the default full profile (first pass: 100+ counters including
# per-channel TCC) crashes real LLM kernels (illegal address / launch failure) even with -k filters,
# while service microbenchmarks profile fine. The HF decode workload therefore runs only the curated
# single-pass sets and the roofline, filtered to Tensile GEMV/GEMM kernels (-k Cijk) with eager
# attention (avoids lazily loaded Triton attention). Each run has a hard limit and its own process group.
# Usage: scripts/amd/lane_rocprof_compute.sh <rocr-index> <result-root>
set -uo pipefail
idx=$1; root=$2
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"; source scripts/amd/env.sh
export ROCR_VISIBLE_DEVICES=$idx PATH=$WORK/bin/rpc_shim:$PATH
out=$root/m3_rocprof_compute; mkdir -p "$out"; cd "$out"
RPC=(env PYTHONPATH=/opt/rocm-7.1.0/share/amd_smi "$WORK/venvs/rocprof_compute_py312/bin/python" /opt/rocm-7.1.0/bin/rocprof-compute)
B=$WORK/bin/svc_bench_gfx942
run() { local name=$1; shift; [ -f "$name.done" ] && return
  echo "$(date +%FT%T) start $name" >> lane_sets.log
  setsid "${RPC[@]}" profile -n "$name" "$@" > "$name.log" 2>&1 & local pid=$! t=0
  while kill -0 $pid 2>/dev/null && [ $t -lt 600 ]; do sleep 5; t=$((t+5)); done
  kill -0 $pid 2>/dev/null && { kill -9 -- -$pid; echo "$(date +%FT%T) killed $name at limit" >> lane_sets.log; }
  wait $pid 2>/dev/null; local rc=$?
  grep -q "illegal memory access\|launch failure\|AQL packet\|caught signal" "$name.log" && rc=99
  echo "$(date +%FT%T) end $name rc=$rc" >> lane_sets.log; [ $rc -eq 0 ] && touch "$name.done"; }
run svc_read4g -- "$B" --mode read --bytes 4294967296 --reps 5 --trials 3 --warmups 1
run svc_mixed16 -- "$B" --mode mixed --bytes 4294967296 --intensity 16 --reps 3 --trials 3 --warmups 1
run svc_gather1g -- "$B" --mode gather --bytes 1073741824 --reps 3 --trials 3 --warmups 1
APP=(env PYTHONPATH=$PYTHONPATH "$PY" "$repo/scripts/amd/decode_profile_bench.py" --config "$root/configs/bridge_base.json"
     --tokens 8 --warmups 1 --attn eager)
for c in "1 1024" "8 4096"; do read -r b x <<< "$c"
  for s in launch_stats mem_thruput compute_thruput_util compute_thruput_flops roof; do
    if [ $s = roof ]; then extra=(--roof-only); else extra=(--set $s); fi
    run hf_bridge_b${b}_c${x}_${s} "${extra[@]}" -k Cijk -- "${APP[@]}" --batch $b --context $x \
      --output "$out/replay_ids_b${b}_c${x}.jsonl"
  done
done
echo "$(date +%FT%T) lane complete" >> lane_sets.log
