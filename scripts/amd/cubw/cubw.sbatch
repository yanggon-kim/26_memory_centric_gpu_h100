#!/bin/bash
#SBATCH -J cubw
#SBATCH -N 1
#SBATCH -o /work1/proj_blaisetine/yanggon/cubw_%j.out
# cubw_v1 campaign job for docs/amd/compute_bandwidth_followup_20261026.md (partition/time set by submit.sh).
#
# Runs from a frozen release snapshot (CUBW_RELEASE = git archive of one commit), never from the live
# repo, so later edits cannot change a queued job. Steps:
#   0  fresh root, configs (+ derived configs from the plan), preflight (GPUs, torch, container, models,
#      counters, CU mask), environment/source/case manifests
#   1  node monitor (observation only; never a lane-completion prerequisite)
#   2  for each plan stage: one lane per GPU in parallel; an "auto" plan computes gates.json from its
#      pilot stage before releasing production steps; a production plan uses CUBW_GATES from the pilot
#   3  closeout reserve (plan reserve_s): stop the monitor, analysis report, SHA-256 manifest of the root
#
# Dry run (login node, no GPU, no scheduler job):
#   DRY_RUN=1 CUBW_PLAN=<plan.json> CUBW_RELEASE=<dir> bash scripts/amd/cubw/cubw.sbatch
set -uo pipefail
WORK=${WORK:-/work1/proj_blaisetine/yanggon}
DRY=${DRY_RUN:-0}
REL=${CUBW_RELEASE:?release snapshot directory}
PLAN=${CUBW_PLAN:?launch plan}
cd "$REL"
kind=$(python3 -c "import json;print(json.load(open('$PLAN'))['kind'])")
cohort=$(python3 -c "import json;print(json.load(open('$PLAN'))['cohort'])")
if [ "$DRY" = 1 ]; then name=dryrun_cubw_${cohort}_${kind}_$(date +%Y%m%d_%H%M%S); else name=amd_$(date +%Y%m%d)_${cohort}_cubw_v1_${kind}_job${SLURM_JOB_ID}; fi
root=$WORK/results_amd/$name
mkdir -p "$root"
[ "$DRY" = 1 ] || ln -sfn "$root" "/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/$name"
status() { echo "$(date +%FT%T) $*" | tee -a "$root/RUN_STATUS.log"; }
status "cubw_v1 $kind/$cohort release=$(cat RELEASE_COMMIT 2>/dev/null) plan=$PLAN"

cp "$PLAN" "$root/launch_plan.json"
[ -n "${CUBW_GATES:-}" ] && cp "$CUBW_GATES" "$root/gates.json" && status "gates from $CUBW_GATES: $(python3 -c "import json;d=json.load(open('$root/gates.json'));print({k:v for k,v in d.items() if k.startswith('release')})")"
python3 - "$root" <<'EOF'
import json, shutil, sys
from pathlib import Path
root = Path(sys.argv[1]); plan = json.loads((root / 'launch_plan.json').read_text())
src = Path(plan['config_source'])
for d in ('configs', 'configs_energy'):
    (root / d).mkdir(exist_ok=True)
    for f in (src / d).glob('*.json'):
        if f.name != 'index.json':
            shutil.copy(f, root / d / f.name)
for rel, cfg in plan['derived_configs'].items():
    (root / rel).write_text(json.dumps(cfg, indent=2) + '\n')
missing = sorted({s['config'] for s in plan['steps'] if s.get('config') and not (root / s['config']).exists()})
print('missing configs:', missing)
sys.exit(1 if missing else 0)
EOF
[ $? = 0 ] || { status "aborting: missing configs"; exit 3; }
for b in svc_bench_gfx942 svc_bench_gfx942_v2 svc_bench_gfx942_v3; do
  [ -x "$WORK/bin/$b" ] || { status "aborting: missing $WORK/bin/$b"; exit 3; }
done

if [ "$DRY" = 1 ]; then
  for i in 0 1 2 3 4 5 6 7; do printf '%s\tGPU-DRYRUN%02d\t-\n' $i $i; done > "$root/gpus.tsv"
  for stage in $(python3 -c "import json;print(' '.join(json.load(open('$PLAN'))['stages']))"); do
    for i in 0 1 2 3 4 5 6 7; do
      python3 scripts/amd/cubw/cubw.py lane --plan "$root/launch_plan.json" --root "$root" --lane L$i --stage $stage --dry-run
    done
  done > "$root/DRYRUN_COMMANDS.txt"
  status "dry run: $(wc -l < "$root/DRYRUN_COMMANDS.txt") step commands in $root/DRYRUN_COMMANDS.txt"
  exit 0
fi

source scripts/amd/env.sh
export CUBW_BENCH=$WORK/bin/svc_bench_gfx942_v3
expect=MI300X; [ "$cohort" = mi325x ] && expect=MI325X
status "preflight start on $(hostname)"
if ! $PY scripts/amd/preflight_node.py --root "$root" --expect-name $expect > "$root/preflight.out" 2>&1; then
  status "preflight FAILED: $(tail -n 1 "$root/preflight.out")"; exit 2
fi
status "preflight ok: $(tail -n 1 "$root/preflight.out")"
$PY scripts/amd/cubw/manifest.py --root "$root" --release "$REL" >> "$root/RUN_STATUS.log" 2>&1

end=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
reserve=$(python3 -c "import json;print(json.load(open('$PLAN'))['reserve_s'])")
export CUBW_DEADLINE=$(( $(date -d "$end" +%s) - reserve ))
status "deadline for new steps: $(date -d @$CUBW_DEADLINE +%FT%T) (reserve ${reserve}s)"

U=(); while IFS=$'\t' read -r i u b; do U[$i]=$u; done < "$root/gpus.tsv"
$PY scripts/amd/gpu_monitor.py $(for u in "${U[@]}"; do printf -- '--gpu-uuid %s ' "$u"; done) --interval 0.05 \
  --output "$root/node_monitor.jsonl" > "$root/monitor.log" 2>&1 &
MON=$!
wait_lanes() {  # lanes only; the monitor never exits on its own
  local p; for p in $(jobs -p); do [ "$p" = "$MON" ] || wait "$p"; done
}

for stage in $(python3 -c "import json;print(' '.join(json.load(open('$PLAN'))['stages']))"); do
  if [ "$kind" = auto ] && [ "$stage" != pilot ]; then
    $PY scripts/amd/cubw/analyze.py gate --root "$root" --out "$root/gates.json" >> "$root/RUN_STATUS.log" 2>&1
    status "gates: $(python3 -c "import json;d=json.load(open('$root/gates.json'));print({k:v for k,v in d.items() if k.endswith('_ok') or k.startswith('release')})")"
  fi
  status "stage $stage start"
  for i in 0 1 2 3 4 5 6 7; do
    $PY scripts/amd/cubw/cubw.py lane --plan "$root/launch_plan.json" --root "$root" --lane L$i --stage $stage \
      > "$root/lane_L${i}_$stage.log" 2>&1 &
  done
  wait_lanes
  status "stage $stage end"
done

kill $MON 2>/dev/null; wait $MON 2>/dev/null
status "analysis start"
$PY scripts/amd/cubw/analyze.py report --root "$root" > "$root/analysis.log" 2>&1
status "analysis rc=$?"
(cd "$root" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
status "COMPLETE ($(wc -l < "$root/SHA256SUMS") files hashed)"
