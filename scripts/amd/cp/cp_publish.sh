#!/usr/bin/env bash
# Publish a campaign run root's compact evidence: analysis tables, plan, journals, status, topology/GPU lists, raw
# checksums and the raw-root location, into results_published/<root name>/. Raw data stay on $WORK.
# Usage: scripts/amd/cp/cp_publish.sh <run root>
set -euo pipefail
R=$(readlink -f "$1"); repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
P=$repo/results_published/$(basename "$R")
mkdir -p "$P/journal"
cp "$R"/launch_plan.json "$R"/RUN_STATUS.log "$P/"
[ -f "$R/PENDING.json" ] && cp "$R/PENDING.json" "$P/"
cp "$R"/journal/*.jsonl "$P/journal/"
cp "$R"/analysis/*.json "$P/" 2>/dev/null || true
cp "$R"/topology_job*.txt "$R"/gpus_job*.tsv "$R"/ras_*.txt "$P/" 2>/dev/null || true
cat "$R"/SHA256SUMS_job* > "$P/raw_SHA256SUMS" 2>/dev/null || true
echo "$R" > "$P/raw_root.txt"
du -sh "$P"
