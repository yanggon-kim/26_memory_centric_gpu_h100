#!/usr/bin/env bash
# Rebuild every summary/analysis for the AMD result roots, publish compact evidence, commit and push.
# Run on the login node (CPU only). Usage: scripts/amd/refresh_and_publish.sh "<commit message>"
set -uo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$repo"
msg=${1:-"Refresh intermediate AMD results"}
d=results/amd_20261018_mi300x; mon=$d/monitor/node_monitor.jsonl
for sub in m1 m6_cumask m1_gpu6 m1_text j6_competitor; do
  [ -d $d/$sub ] && python3 scripts/amd/summarize_timing.py --root $d/$sub --monitor $mon --output $d/$sub/summary.json > $d/$sub/summary.txt 2>&1
done
python3 scripts/amd/summarize_energy.py $d/m8_energy > $d/m8_energy/summary.txt 2>&1
python3 scripts/amd/summarize_pressure.py $d/m5_pressure > $d/m5_pressure/summary.txt 2>&1
python3 scripts/amd/summarize_service.py --input $d/m4_service/service.jsonl --output $d/m4_service/summary.json > $d/m4_service/summary.txt 2>&1
python3 scripts/amd/analyze_counters.py --root $d/m2_counters --configs $d/configs_profile --output $d/m2_counters/summary.json > $d/m2_counters/summary.txt 2>&1
python3 scripts/amd/summarize_tp.py $d > $d/tp/summary.txt 2>&1
python3 scripts/amd/summarize_m6.py $d > $d/analysis/m6_prefill_summary.txt 2>&1
[ -f scripts/amd/summarize_competitor.py ] && [ -d $d/j6_competitor ] && python3 scripts/amd/summarize_competitor.py $d > $d/j6_competitor/summary_competitor.txt 2>&1
python3 scripts/amd/latency_bounds.py --m1 $d/m1/summary.json --service $d/m4_service/summary.json --counters $d/m2_counters/summary.json \
  --configs $d/configs --output $d/analysis/shoreline_latency_bounds.json > $d/analysis/shoreline_latency_bounds.txt 2>&1
python3 scripts/amd/consolidate_shoreline.py --root $d > /dev/null
roots=$(ls -d results/amd_2026101*_* | tr '\n' ' ')
for r in $roots; do [ -d $r/m1 ] && [ -f $r/node_monitor.jsonl ] && \
  python3 scripts/amd/summarize_timing.py --root $r/m1 --monitor $r/node_monitor.jsonl --output $r/m1/summary.json > $r/m1/summary.txt 2>&1; done
python3 scripts/amd/publish_results.py $roots > /dev/null
if grep -rlE "hf_[A-Za-z0-9]{30,}|gh[op]_[A-Za-z0-9]{30,}|BEGIN (RSA|OPENSSH)" results_published >/dev/null; then echo "secret-like content found; not committing"; exit 1; fi
git add results_published docs/amd scripts/amd
git diff --cached --quiet && { echo "no changes"; exit 0; }
git commit -q -m "$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin amd/mi300x-phase0 && git log --oneline -1
