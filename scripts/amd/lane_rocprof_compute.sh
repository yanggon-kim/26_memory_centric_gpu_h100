#!/usr/bin/env bash
# rocprof-compute (Omniperf) detailed analysis. The HF decode workload is filtered to Tensile GEMV/GEMM
# kernels (-k Cijk): collecting on its runtime-loaded Triton attention (attn_fwd) hung rocprofv3.
# Service microbenchmarks are profiled in full, including the roofline.
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
  timeout -s KILL 2400 "${RPC[@]}" profile -n "$name" "$@" > "$name.log" 2>&1; local rc=$?
  grep -q "illegal memory access\|AQL packet\|caught signal" "$name.log" && rc=99
  echo "$(date +%FT%T) end $name rc=$rc" >> lane_sets.log; [ $rc -eq 0 ] && touch "$name.done"; }
run svc_read4g -- "$B" --mode read --bytes 4294967296 --reps 5 --trials 3 --warmups 1
run svc_mixed16 -- "$B" --mode mixed --bytes 4294967296 --intensity 16 --reps 3 --trials 3 --warmups 1
for c in "1 1024" "8 4096"; do read -r b x <<< "$c"
  run hf_bridge_b${b}_c${x}_cijk_eagerattn -k Cijk -- env PYTHONPATH=$PYTHONPATH "$PY" "$repo/scripts/amd/decode_profile_bench.py" \
    --config "$root/configs/bridge_base.json" --batch $b --context $x --tokens 8 --warmups 1 --attn eager --output "$out/replay_ids_b${b}_c${x}.jsonl"
done
echo "$(date +%FT%T) lane complete" >> lane_sets.log
