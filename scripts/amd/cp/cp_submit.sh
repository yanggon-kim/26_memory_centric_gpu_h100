#!/usr/bin/env bash
# Chunked-prefill campaign: freeze a release snapshot of the committed scripts and campaign docs, generate the plan inside
# it (scripts/amd/cp/cp_plan.py), and submit scripts/amd/cp/cp.sbatch with a persistent run root. Refuses uncommitted
# changes under scripts/. Resubmitting with --root <existing root> resumes that plan (the root's plan must match).
# Usage: scripts/amd/cp/cp_submit.sh <plan> <cohort> <HH:MM:SS> [--smoke] [--params P.json] [--root ROOT] [-- sbatch args]
#   e.g. scripts/amd/cp/cp_submit.sh s1 mi300x 03:00:00
set -euo pipefail
which=$1; cohort=$2; tlimit=$3; shift 3
smoke=; root=; params=; mcap=; low=; sb=()
while [ $# -gt 0 ]; do
  case $1 in --smoke) smoke=--smoke; shift;; --root) root=$2; shift 2;; --params) params=$(readlink -f $2); shift 2;; --max-capture) mcap=$2; shift 2;; --low) low=$2; shift 2;; --) shift; sb=("$@"); break;; *) echo "bad arg $1" >&2; exit 1;; esac
done
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$repo"
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
if ! git diff --quiet HEAD -- scripts docs/amd/cp_campaign_design_20261118.md; then echo "uncommitted changes; commit first" >&2; exit 1; fi
sha=$(git rev-parse HEAD)
rel=$WORK/cp_release/$sha
if [ ! -d "$rel" ]; then
  mkdir -p "$rel"
  git archive "$sha" scripts docs/amd/chunked_prefill_campaign_20261118.md docs/amd/cp_campaign_design_20261118.md | tar -x -C "$rel"
  echo "$sha" > "$rel/RELEASE_COMMIT"; chmod -R a-w "$rel/scripts"
fi
if [ -n "$root" ]; then
  plan=$root/launch_plan.json
else
  mkdir -p "$WORK/cp_plans"
  tag=${which}${smoke:+_smoke}${mcap:+_cap$mcap}${low:+_low$low}
  plan=$WORK/cp_plans/cp_${cohort}_${tag}_$(date +%Y%m%d_%H%M%S)_${sha:0:8}.json
  python3.12 "$rel/scripts/amd/cp/cp_plan.py" "$which" --cohort "$cohort" $smoke ${params:+--params "$params"} ${mcap:+--max-capture $mcap} ${low:+--low $low} --out "$plan"
  chmod a-w "$plan"
  root=$WORK/results_amd/amd_$(date +%Y%m%d)_${cohort}_cp_${tag}
  [ -e "$root" ] && root=${root}_$(date +%H%M%S)
fi
part=$(python3 -c "import json;print(json.load(open('$plan'))['partition'])")
sbatch -p "$part" -N 1 -n 8 -c 16 -t "$tlimit" -J "cp-$cohort-$which" \
  --export=ALL,CP_PLAN=$plan,CP_RELEASE=$rel,CP_ROOT=$root "${sb[@]}" "$rel/scripts/amd/cp/cp.sbatch"
echo "plan=$plan release=$rel root=$root"
