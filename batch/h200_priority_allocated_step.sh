#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
bootstrap_python=/usr/local/pace-apps/manual/packages/anaconda3/2023.03/bin/python3
export H200_VISIBILITY_MANIFEST="/tmp/h200_visibility_${SLURM_JOB_ID}_${UID}.json"
"$bootstrap_python" scripts/h200_allocated_visibility.py --output "$H200_VISIBILITY_MANIFEST"
CUDA_VISIBLE_DEVICES=$("$bootstrap_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_cuda_visible_devices"])' "$H200_VISIBILITY_MANIFEST")
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$bootstrap_python" -u scripts/h200_priority_campaign.py --bootstrap --cuda "$CUDA_HOME" "$@"
