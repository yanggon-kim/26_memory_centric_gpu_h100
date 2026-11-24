#!/usr/bin/env bash
# Long-context campaign: freeze a release snapshot of the committed scripts and campaign docs, generate the plan inside it
# (scripts/amd/lc/lc_plan.py), and submit scripts/amd/cp/cp.sbatch (shared with the November 18 campaign) with a persistent
# run root. Refuses uncommitted changes under scripts/ or the design. Resubmitting with --root <root> resumes that plan.
# Usage: scripts/amd/lc/lc_submit.sh <plan> <cohort> <HH:MM:SS> [--root ROOT] [-- sbatch args]
set -euo pipefail
which=$1; cohort=$2; tlimit=$3; shift 3
root=; sb=()
while [ $# -gt 0 ]; do
  case $1 in --root) root=$2; shift 2;; --) shift; sb=("$@"); break;; *) echo "bad arg $1" >&2; exit 1;; esac
done
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$repo"
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
if ! git diff --quiet HEAD -- scripts docs/amd/lc_campaign_design_20261124.md; then echo "uncommitted changes; commit first" >&2; exit 1; fi
sha=$(git rev-parse HEAD)
rel=$WORK/cp_release/$sha
if [ ! -d "$rel" ]; then
  mkdir -p "$rel"
  git archive "$sha" scripts docs/amd/long_context_chunked_prefill_20261124.md docs/amd/lc_campaign_design_20261124.md \
    docs/amd/chunked_prefill_campaign_20261118.md docs/amd/cp_campaign_design_20261118.md | tar -x -C "$rel"
  echo "$sha" > "$rel/RELEASE_COMMIT"; chmod -R a-w "$rel/scripts"
fi
if [ -n "$root" ]; then
  plan=$root/launch_plan.json
else
  mkdir -p "$WORK/cp_plans"
  plan=$WORK/cp_plans/lc_${cohort}_${which}_$(date +%Y%m%d_%H%M%S)_${sha:0:8}.json
  python3.12 "$rel/scripts/amd/lc/lc_plan.py" "$which" --cohort "$cohort" --out "$plan"
  chmod a-w "$plan"
  root=$WORK/results_amd/amd_$(date +%Y%m%d)_${cohort}_lc_${which}
  [ -e "$root" ] && root=${root}_$(date +%H%M%S)
fi
read part ngpu < <(python3 -c "import json;p=json.load(open('$plan'));print(p['partition'], p['gpus'])")
if [ "$ngpu" = 1 ]; then res=(-n 1 -c 16); else res=(-n 8 -c 16); fi
sbatch -p "$part" -N 1 "${res[@]}" -t "$tlimit" -J "lc-$cohort-$which" \
  --export=ALL,CP_PLAN=$plan,CP_RELEASE=$rel,CP_ROOT=$root "${sb[@]}" "$rel/scripts/amd/cp/cp.sbatch"
echo "plan=$plan release=$rel root=$root"
