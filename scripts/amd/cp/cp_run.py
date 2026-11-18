#!/usr/bin/env python3
"""Chunked-prefill campaign step runner (docs/amd/chunked_prefill_campaign_20261118.md). Executes one lane of a campaign
plan (scripts/amd/cp/cp_plan.py) on its GPU set, one step at a time, journaling every attempt to
<root>/journal/<lane>.jsonl. The run root is persistent across jobs: a step whose journal shows a completed attempt is
skipped, so a resubmitted job resumes where an interrupted one stopped.

Step kinds:
  engine   a harness script (cp_replay.py, cp_serve.py, cp_disagg.py, cp_link.py) in the vLLM container
           (scripts/amd/vllm_in_container.sh) with the lane's GPUs visible; optional decoder CU mask (HSA_CU_MASK on
           0..cus-1 of a single-GPU lane), tuned grid (CUBW_CU_COUNT_OVERRIDE, scripts/sitecustomize.py), and an owned
           competitor process on a disjoint CU set (experiment 4: cubw treatments, scripts/amd/cubw/cubw.py)
  native   svc_bench_v3 placement proof of a CU mask (cubw run_native)
Lifecycle, competitor start/stop, orphan sweep and residual-process checks reuse scripts/amd/cubw/cubw.py. A step gets at
most two attempts in total (across jobs); a second attempt is made only after an execution failure, never after a
correctness failure. A step is not started when its estimate would cross the job deadline (CP_DEADLINE_UNIX); it stays
pending for the next job.

  cp_run.py lane --plan plan.json --root ROOT --stage STAGE --lane L0 --gpus 0 --uuids GPU-...
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/amd/cubw'))
import cubw as B  # noqa: E402

MAX_ATTEMPTS = 2
CORRECTNESS = ('composition mismatch', 'barrier not reached', 'incomplete', 'prompt changed')


def history(root, lane):
    p = root / 'journal' / f'{lane}.jsonl'
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if 'step' in r and 'attempt' in r:
                out.setdefault(r['step'], []).append(r)
    return out


def run_engine(step, plan, root, lane, gpus, uuids, adir):
    tag = f"{os.getpid()}-{step['id']}-{adir.name}-{int(time.time() * 1e6)}"
    rec = {'step': step['id'], 'kind': 'engine', 'attempt_dir': str(adir), 'step_tag': tag}
    comp = None
    if step.get('treatment', 'absent') != 'absent':
        g = {'index': gpus[0], 'uuid': uuids[0]}
        cs = dict(step, plan_competitor_mask=plan['competitor_mask'], _root=root, lane=lane)
        comp, crec = B.start_competitor(cs, g, adir, step['timeout_s'] + 120, tag)
        rec['competitor'] = crec
        if 'start_failed' in crec:
            crec['stop'] = B.stop_group(comp)
            rec.update(command_completed=False, failure_class='execution:competitor_start')
            return rec
    extra = [f'CUBW_STEP_TAG={tag}', f"PYTHONPATH={REPO / 'scripts'}:{REPO / 'scripts/amd/cp'}"]
    if step.get('cus') and step['cus'] < plan['full_cus']:
        assert len(gpus) == 1, 'a CU mask applies to single-GPU lanes (cp_disagg.py masks its own workers)'
        extra.append(f"HSA_CU_MASK={gpus[0]}:{B.mask_range(0, step['cus'] - 1)}")
    if step.get('grid'):
        extra.append(f"CUBW_CU_COUNT_OVERRIDE={step['grid']}")
    if comp:
        extra.append(f'RESEARCH_ALLOWED_PIDS={comp.pid}')
    for k, v in step.get('env', {}).items():
        extra.append(f'{k}={v}')
    args = [str(x).replace('@UUIDS', ','.join(uuids)).replace('@UUID', uuids[0]).replace('@IDX', ','.join(map(str, gpus)))
            for x in step['args']]
    argv = [str(REPO / 'scripts/amd/vllm_in_container.sh'), step['script']] + args + ['--output', str(adir / 'run')]
    env = dict(os.environ, AMD_EXTRA_ENV=','.join(extra), APPTAINERENV_HIP_VISIBLE_DEVICES=','.join(map(str, gpus)))
    if step.get('profile'):
        env['AMD_ENTRY'] = 'rocprofv3'
        argv = argv[:1] + ['--kernel-trace', '--marker-trace', '--output-format', 'csv', '-d', str(adir / 'rocprof'), '-o', 'p',
                           '--', 'python3'] + argv[1:]
    rec.update(argv=argv, amd_extra_env=env['AMD_EXTRA_ENV'], gpus=gpus, uuids=uuids, t_start=time.time())
    with open(adir / 'runner.log', 'w') as log:
        proc = subprocess.Popen(argv, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rec['runner_pid'] = proc.pid
        try:
            proc.wait(step['timeout_s'])
            rec['timed_out'] = False
        except subprocess.TimeoutExpired:
            rec['timed_out'] = True
            rec['runner_stop'] = B.stop_group(proc, grace=10)
    rec.update(t_end=time.time(), returncode=proc.returncode)
    if comp:
        rec['competitor']['stop'] = B.stop_group(comp)
    rec['orphans'] = B.sweep_tag(tag)
    rec['residual_gpu_pids'] = {u: B.gpu_processes(u) for u in uuids}
    run = adir / 'run' / 'run.json'
    status = json.loads(run.read_text()).get('status') if run.exists() else 'missing'
    rec['run_status'] = status
    rec['command_completed'] = proc.returncode == 0 and not rec['timed_out'] and status == 'complete'
    if not rec['command_completed']:
        fail = adir / 'run' / 'failure.json'
        msg = json.loads(fail.read_text())['message'] if fail.exists() else ''
        rec['failure_message'] = msg[:800]
        rec['failure_class'] = ('correctness' if any(k in msg for k in CORRECTNESS)
                                else 'execution:timeout' if rec['timed_out'] else 'execution')
    return rec


def lane(a):
    plan = json.loads(Path(a.plan).read_text())
    root = Path(a.root)
    gpus = [int(x) for x in a.gpus.split(',')]
    uuids = a.uuids.split(',')
    journal = B.Journal(root, a.lane)
    hist = history(root, a.lane)
    deadline = float(os.environ.get('CP_DEADLINE_UNIX', 'inf'))
    journal.write(event='lane_start', lane=a.lane, stage=a.stage, gpus=gpus, uuids=uuids, job=os.environ.get('SLURM_JOB_ID'))
    for step in plan['steps']:
        if step['stage'] != a.stage or step['lane'] != a.lane:
            continue
        prev = hist.get(step['id'], [])
        if any(r.get('command_completed') for r in prev):
            continue
        if any(r.get('failure_class') == 'correctness' for r in prev) or len(prev) >= MAX_ATTEMPTS:
            continue
        while len(prev) < MAX_ATTEMPTS:
            if time.time() + step['est_s'] > deadline:
                journal.write(event='deferred', step=step['id'], reason='job deadline')
                break
            k = len(prev) + 1
            adir = root / step['experiment'] / step['id'] / f'attempt{k}'
            adir.mkdir(parents=True, exist_ok=True)
            if step['kind'] == 'native':
                rec = B.run_native(dict(step, args=step['args']), {'index': gpus[0], 'uuid': uuids[0]}, adir)
                rec['step'] = step['id']
            else:
                rec = run_engine(step, plan, root, a.lane, gpus, uuids, adir)
            rec.update(attempt=k, lane=a.lane, stage=a.stage, experiment=step['experiment'], job=os.environ.get('SLURM_JOB_ID'))
            journal.write(**rec)
            prev.append(rec)
            if rec.get('command_completed') or rec.get('failure_class') == 'correctness':
                break
        else:
            pass
        if time.time() + 60 > deadline:
            break
    journal.write(event='lane_end', lane=a.lane, stage=a.stage)


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = q.add_subparsers(dest='cmd', required=True)
    r = s.add_parser('lane')
    for k in ('plan', 'root', 'stage', 'lane', 'gpus', 'uuids'):
        r.add_argument('--' + k, required=True)
    lane(q.parse_args())
