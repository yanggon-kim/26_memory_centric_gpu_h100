#!/usr/bin/env bash
# goal_v1: freeze a release snapshot of the committed scripts and goal docs, generate the plan inside it
# (scripts/amd/goal/plan_goal.py), dry-run every lane (no GPU), then submit goal.sbatch. Refuses uncommitted scripts/.
# Usage: scripts/amd/goal/submit_goal.sh <smoke|auto|g5x_smoke|g5x> <mi300x|mi325x|mi350x> <time HH:MM:SS> [-- sbatch args]
#   g5x / g5x_smoke: G5 extension on gfx942 (scripts/amd/goal/plan_g5x.py, docs/amd/goal_g5x_plan_20261107.md)
#   e.g. submit_goal.sh smoke mi300x 03:00:00
#        submit_goal.sh auto mi300x 09:00:00 -- --hold
set -euo pipefail
kind=$1; cohort=$2; tlimit=$3; shift 3
smoke=; { [ "$kind" = smoke ] || [ "$kind" = g5x_smoke ]; } && smoke=--smoke
gen=plan_goal.py; case $kind in g5x|g5x_smoke) gen=plan_g5x.py;; esac
plan_args=(); sb_args=()
while [ $# -gt 0 ]; do [ "$1" = -- ] && { shift; sb_args=("$@"); break; }; plan_args+=("$1"); shift; done
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$repo"
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
if ! git diff --quiet HEAD -- scripts; then echo "uncommitted changes under scripts/; commit first" >&2; exit 1; fi
sha=$(git rev-parse HEAD)
rel=$WORK/cubw_release/$sha
if [ ! -d "$rel" ]; then
  mkdir -p "$rel"; git archive "$sha" scripts docs/amd/compute_bandwidth_followup_20261026.md docs/amd/goal_based_followup_20261105.md docs/amd/goal_followup_execution_plan_20261105.md docs/amd/goal $(git ls-files docs/amd/goal_g5x_plan_20261107.md) | tar -x -C "$rel"
  echo "$sha" > "$rel/RELEASE_COMMIT"; chmod -R a-w "$rel/scripts"
fi
mkdir -p "$WORK/cubw_plans"
plan=$WORK/cubw_plans/goal_${cohort}_${kind}_$(date +%Y%m%d_%H%M%S)_${sha:0:8}.json
python3.12 "$rel/scripts/amd/goal/$gen" --cohort "$cohort" $smoke --out "$plan"
chmod a-w "$plan"
DRY_RUN=1 CUBW_PLAN=$plan CUBW_RELEASE=$rel bash "$rel/scripts/amd/goal/goal.sbatch" | tail -3
part=$(python3 -c "import json;print(json.load(open('$plan'))['partition'])")
nn=$(python3 -c "import json;print(len(json.load(open('$plan'))['lanes']))")
sbatch -p $part -N 1 -n $nn -t "$tlimit" -J "goal-$cohort-$kind" --export=ALL,CUBW_PLAN=$plan,CUBW_RELEASE=$rel "${sb_args[@]}" \
  "$rel/scripts/amd/goal/goal.sbatch"
echo "plan=$plan release=$rel"
