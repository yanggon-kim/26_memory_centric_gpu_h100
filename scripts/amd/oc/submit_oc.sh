#!/usr/bin/env bash
# OC-v0: freeze a release snapshot of the committed scripts and OC docs, generate the plan inside it
# (scripts/amd/oc/plan_oc.py), dry-run every lane (no GPU), then submit oc.sbatch. Refuses uncommitted scripts/.
# Usage: scripts/amd/oc/submit_oc.sh <pilot|dev|heldout> <mi300x|mi350x> <time HH:MM:SS> [-- sbatch args]
set -euo pipefail
kind=$1; cohort=$2; tlimit=$3; shift 3
plan_args=(); sb_args=()
while [ $# -gt 0 ]; do [ "$1" = -- ] && { shift; sb_args=("$@"); break; }; plan_args+=("$1"); shift; done
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$repo"
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
if ! git diff --quiet HEAD -- scripts; then echo "uncommitted changes under scripts/; commit first" >&2; exit 1; fi
sha=$(git rev-parse HEAD)
rel=$WORK/cubw_release/$sha
if [ ! -d "$rel" ]; then
  mkdir -p "$rel"; git archive "$sha" scripts docs/amd/compute_bandwidth_followup_20261026.md docs/amd/goal_based_followup_20261105.md docs/amd/goal_followup_execution_plan_20261105.md docs/amd/goal docs/amd/overlap_calibration_followup_20261105.md docs/amd/overlap_calibration_execution_plan_20261105.md docs/amd/oc docs/compute_memory_overlap_calibration_framework_v0_20261105.md | tar -x -C "$rel"
  echo "$sha" > "$rel/RELEASE_COMMIT"; chmod -R a-w "$rel/scripts"
fi
mkdir -p "$WORK/cubw_plans"
plan=$WORK/cubw_plans/oc_${cohort}_${kind}_$(date +%Y%m%d_%H%M%S)_${sha:0:8}.json
python3.12 "$rel/scripts/amd/oc/plan_oc.py" --cohort "$cohort" --kind "$kind" "${plan_args[@]}" --out "$plan"
chmod a-w "$plan"
DRY_RUN=1 CUBW_PLAN=$plan CUBW_RELEASE=$rel bash "$rel/scripts/amd/oc/oc.sbatch" | tail -3
part=$(python3 -c "import json;print(json.load(open('$plan'))['partition'])")
nn=$(python3 -c "import json;print(len(json.load(open('$plan'))['lanes']))")
sbatch -p $part -N 1 -n $nn -t "$tlimit" -J "oc-$cohort-$kind" --export=ALL,CUBW_PLAN=$plan,CUBW_RELEASE=$rel "${sb_args[@]}" \
  "$rel/scripts/amd/oc/oc.sbatch"
echo "plan=$plan release=$rel"
