#!/bin/bash
#SBATCH -J cp
#SBATCH -N 1
#SBATCH -o /work1/proj_blaisetine/yanggon/cp_%j.out
# Chunked-prefill campaign job (docs/amd/chunked_prefill_campaign_20261118.md; plan scripts/amd/cp/cp_plan.py; submitted by
# scripts/amd/cp/cp_submit.sh). Runs from a frozen release snapshot (CP_RELEASE = git archive of one commit).
# The run root CP_ROOT is persistent: a resubmitted job with the same plan resumes from the journals (completed steps are
# skipped; a step gets at most two attempts overall).
#   1  preflight (GPUs, container, CU mask), GPU inventory and topology into the root
#   2  for each plan stage: one lane per GPU set in parallel (scripts/amd/cp/cp_run.py); steps that would cross the
#      deadline (job end - reserve) stay pending
#   3  SHA-256 manifest of the root
set -uo pipefail
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
REL=${CP_RELEASE:?release snapshot}
PLAN=${CP_PLAN:?plan}
ROOT=${CP_ROOT:?run root}
cd "$REL"
mkdir -p "$ROOT"
name=$(basename "$ROOT")
ln -sfn "$ROOT" "/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/$name"
status() { echo "$(date +%FT%T) $*" | tee -a "$ROOT/RUN_STATUS.log"; }
status "job ${SLURM_JOB_ID} release=$(cat RELEASE_COMMIT) plan=$PLAN host=$(hostname)"
[ -f "$ROOT/launch_plan.json" ] || cp "$PLAN" "$ROOT/launch_plan.json"
cmp -s "$PLAN" "$ROOT/launch_plan.json" || { status "aborting: root holds a different plan"; exit 3; }
source scripts/amd/env.sh
arch=$(python3 -c "import json;print(json.load(open('$PLAN'))['arch'])")
full=$(python3 -c "import json;print(json.load(open('$PLAN'))['full_cus'])")
gname=$(python3 -c "import json;print(json.load(open('$PLAN'))['gpu_name'])")
ngpu=$(python3 -c "import json;print(json.load(open('$PLAN'))['gpus'])")
export CUBW_BENCH=$WORK/bin/svc_bench_${arch}_v3 CUBW_BENCH4=$WORK/bin/svc_bench_${arch}_v4
pf=$ROOT/preflight_job${SLURM_JOB_ID}
mkdir -p "$pf"
if ! $PY scripts/amd/preflight_node.py --root "$pf" --expect-gpus $ngpu --expect-name $gname --expect-arch $arch --expect-cus $full > "$pf/preflight.out" 2>&1; then
  status "preflight FAILED: $(tail -n 1 "$pf/preflight.out")"; exit 2
fi
$PY - "$ROOT" "$ngpu" > "$ROOT/gpus_job${SLURM_JOB_ID}.tsv" <<'EOF'
import sys
sys.path.insert(0, 'scripts')
import amd_nvml as n
n.nvmlInit()
for i in range(int(sys.argv[2])):
    h = n.nvmlDeviceGetHandleByIndex(i)
    print(f'{i}\t{n.nvmlDeviceGetUUID(h)}')
EOF
amd-smi topology > "$ROOT/topology_job${SLURM_JOB_ID}.txt" 2>&1 || true
status "preflight ok; GPUs: $(cut -f2 "$ROOT/gpus_job${SLURM_JOB_ID}.tsv" | paste -sd' ')"
end=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
reserve=$(python3 -c "import json;print(json.load(open('$PLAN')).get('reserve_s', 900))")
export CP_DEADLINE_UNIX=$(( $(date -d "$end" +%s) - reserve ))
status "deadline for new steps: $(date -d @$CP_DEADLINE_UNIX +%FT%T)"
uuid_of() { awk -v i="$1" '$1==i{print $2}' "$ROOT/gpus_job${SLURM_JOB_ID}.tsv"; }
for stage in $(python3 -c "import json;print(' '.join(json.load(open('$PLAN'))['stages']))"); do
  status "stage $stage start"
  pids=()
  while IFS=$'\t' read -r lane gpus; do
    uu=$(for g in ${gpus//,/ }; do uuid_of $g; done | paste -sd,)
    python3 scripts/amd/cp/cp_run.py lane --plan "$ROOT/launch_plan.json" --root "$ROOT" --stage "$stage" --lane "$lane" \
      --gpus "$gpus" --uuids "$uu" > "$ROOT/lane_${lane}_${stage}_job${SLURM_JOB_ID}.log" 2>&1 &
    pids+=($!)
  done < <(python3 -c "import json;p=json.load(open('$PLAN'));[print(f\"{l}\t{','.join(map(str,g))}\") for l,g in p['lanes'][\"$stage\"].items()]")
  for p in "${pids[@]}"; do wait $p; done
  status "stage $stage end"
done
python3 - "$ROOT" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); plan = json.loads((root / 'launch_plan.json').read_text())
done = set()
for j in (root / 'journal').glob('*.jsonl'):
    for line in j.read_text().splitlines():
        r = json.loads(line)
        if r.get('command_completed'):
            done.add(r['step'])
todo = [s['id'] for s in plan['steps'] if s['id'] not in done]
(root / 'PENDING.json').write_text(json.dumps({'completed': len(done), 'pending': todo}, indent=1) + '\n')
print(f'completed {len(done)} / {len(plan["steps"])}; pending {len(todo)}')
EOF
status "$(python3 -c "import json;d=json.load(open('$ROOT/PENDING.json'));print(f\"steps completed {d['completed']}, pending {len(d['pending'])}\")")"
(cd "$ROOT" && find . -type f ! -name SHA256SUMS -newer launch_plan.json -print0 | sort -z | xargs -0 sha256sum > "SHA256SUMS_job${SLURM_JOB_ID}") || true
status "COMPLETE job ${SLURM_JOB_ID}"
