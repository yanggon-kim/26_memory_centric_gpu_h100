#!/usr/bin/env bash
# Run a repo script inside the AMD rocm/vllm container with a clean, recorded environment.
# Usage: scripts/amd/vllm_in_container.sh <script.py> [args...]
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
SIF=${AMD_VLLM_SIF:-$WORK/containers/vllm_rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0.sif}
# Per-process compile caches: concurrent lanes sharing one Triton cache produced
# "hip_utils.so could not be loaded" and launch failures (cache write race). AMD_CACHE_DIR may name a
# persistent per-case cache instead (TP: workers of one run race when compiling into an empty cache).
CACHE=${AMD_CACHE_DIR:-$WORK/cache/proc/$(date +%s%N)_$$}; mkdir -p "$CACHE"
SHA=$(awk -v f="$(basename "$SIF")" '$2==f{print $1}' "$WORK/containers/sha256.txt" 2>/dev/null || true)
exec apptainer exec --rocm --cleanenv --home $WORK/container_home -B /work1 -B /home1 \
  --env AMD_VLLM_IMAGE="$(basename "$SIF")",AMD_VLLM_IMAGE_SHA256="${SHA:-unrecorded}" \
  --env HF_HOME=$WORK/hf,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONPATH="$repo/scripts" \
  --env VLLM_CACHE_ROOT=$CACHE/vllm,TRITON_CACHE_DIR=$CACHE/triton,TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
  ${AMD_EXTRA_ENV:+--env "$AMD_EXTRA_ENV"} \
  --pwd "$repo" "$SIF" ${AMD_ENTRY:-python3} "$@"
