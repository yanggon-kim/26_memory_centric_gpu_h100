#!/usr/bin/env bash
# Freeze a release snapshot of the committed scripts, generate the launch plan inside it, dry-run every
# lane (no GPU), then submit cubw.sbatch. Refuses to run with uncommitted changes under scripts/.
# Usage: scripts/amd/cubw/submit.sh <pilot|production|auto> <mi300x|mi325x> <time HH:MM:SS> [plan.py args] [-- sbatch args]
#   e.g. submit.sh pilot mi300x 01:45:00
#        CUBW_GATES=/path/gates.json submit.sh production mi300x 03:30:00 --boundary 152
#        submit.sh auto mi325x 03:45:00 -- --dependency=afterany:481072
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
  mkdir -p "$rel"; git archive "$sha" scripts docs/amd/compute_bandwidth_followup_20261026.md | tar -x -C "$rel"
  echo "$sha" > "$rel/RELEASE_COMMIT"; chmod -R a-w "$rel/scripts"
fi
mkdir -p "$WORK/cubw_plans"
plan=$WORK/cubw_plans/${cohort}_${kind}_$(date +%Y%m%d_%H%M%S)_${sha:0:8}.json
python3 "$rel/scripts/amd/cubw/plan.py" --kind "$kind" --cohort "$cohort" --out "$plan" "${plan_args[@]}"
chmod a-w "$plan"
DRY_RUN=1 CUBW_PLAN=$plan CUBW_RELEASE=$rel bash "$rel/scripts/amd/cubw/cubw.sbatch" | tail -3
part=mi3008x; [ "$cohort" = mi325x ] && part=mi3258x
sbatch -p $part -t "$tlimit" -J "cubw-$cohort-$kind" --export=ALL,CUBW_PLAN=$plan,CUBW_RELEASE=$rel${CUBW_GATES:+,CUBW_GATES=$CUBW_GATES} "${sb_args[@]}" \
  "$rel/scripts/amd/cubw/cubw.sbatch"
echo "plan=$plan release=$rel"
