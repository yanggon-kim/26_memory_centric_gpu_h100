#!/usr/bin/env bash
# M4/A-D/M7 service sweep lane on one GPU (all CU levels). Usage: <gpu-uuid> <physical-index> <result-root>
set -uo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"; source scripts/amd/env.sh
exec python3 scripts/amd/service_sweep.py --binary "$WORK/bin/svc_bench_gfx942" --gpu-uuid "$1" --physical-index "$2" --output "$3/m4_service"
