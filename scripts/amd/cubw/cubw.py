#!/usr/bin/env python3
"""Compute/bandwidth follow-up (docs/amd/compute_bandwidth_followup_20261026.md) step runner, cubw_v1.

Executes one lane of a launch plan (scripts/amd/cubw/plan.py) on one GPU, one step at a time, and
appends every attempt to <root>/journal/<lane>.jsonl. Step kinds:

  vllm     one decoder acquisition (scripts/controlled_vllm_experiments.py in the vLLM container) under a
           decoder CU mask, optionally beside an owned competitor process on a disjoint CU mask, optionally
           under rocprofv3 (kernel trace or one counter group).
  native   one svc_bench_v3 process (one acquisition of a service control or a placement probe).
  shell    a recorded capability command (counter lists, version strings).
  competitor_only  the competitor alone for a fixed time (probe-alone reference, lifecycle checks), or
           beside a placement-proven native co-runner on the decoder mask (v1.1 leak reference).

Command completion is recorded separately from scientific status; nothing here decides qualification
(scripts/amd/cubw/analyze.py does). A step is retried once (two attempts total) only for a diagnosed
execution/tool failure, never for a correctness failure or to obtain different timings. Each attempt
has its own directory; nothing is overwritten. Owned processes run in their own process groups and are
stopped with SIGTERM, then SIGKILL after a bounded wait; their exit is verified by PID, never by name.

v1.1 (after the MI300X pilot, job 481167): the residual-process check applies the existing
scripts/gpu_target.py rule (VRAM >= 64 MiB or GFX engine time on the target GPU) instead of listing every
KFD process on the node; every owned process carries a per-attempt CUBW_STEP_TAG and processes that
escaped their process group (rocprofv3 left PPID=1 orphans) are found by that tag, killed and recorded.

  cubw.py lane --plan <launch_plan.json> --root <root> --lane <name> [--dry-run]
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
WORK = Path(os.environ.get('WORK', '/work1/proj_blaisetine/yanggon'))
BIN = Path(os.environ.get('CUBW_BENCH', WORK / 'bin/svc_bench_gfx942_v3'))
COMPETITOR_BYTES = 1 << 30  # 1 GiB read footprint: far above the 256 MiB MALL
TREATMENTS = {  # name -> (competitor mode, requested duty); None = competitor absent
    'absent': None, 'inactive': ('idle', 1.0),
    'mem25': ('read', 0.25), 'mem50': ('read', 0.50), 'mem75': ('read', 0.75), 'mem100': ('read', 1.00),
    'arith100': ('fma', 1.00), 'leakprobe': ('fma', 1.00),
}


def mask_range(lo, hi):
    return f'{lo}-{hi}'


def decoder_mask(n, full):
    """Contiguous physical CU indices 0..n-1 (interleaved across XCDs); None = unrestricted."""
    return None if n >= full else mask_range(0, n - 1)


def now():
    return time.time()


class Journal:
    def __init__(self, root, lane):
        self.path = root / 'journal' / f'{lane}.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **row):
        row.setdefault('t_unix', now())
        with self.path.open('a') as f:
            f.write(json.dumps(row, default=str) + '\n')


def gpu_processes(uuid):
    """PIDs holding VRAM (>= 64 MiB) or GFX engine time on the target GPU, the scripts/gpu_target.py rule.
    amdsmi lists every KFD process on the node for every GPU, so the unfiltered list is not per GPU."""
    try:
        import amd_nvml as n
        n.nvmlInit()
        h = n.nvmlDeviceGetHandleByUUID(uuid)
        return sorted({p['pid'] for p in n.processes_detail(h) if p['vram_bytes'] >= 64 << 20 or p['gfx_engine_ns'] > 0})
    except Exception as error:  # recorded, not fatal
        return {'error': str(error)[:200]}


def sweep_tag(tag, wait=30):
    """Kill owned processes that escaped their process group, found by CUBW_STEP_TAG in their environment."""
    needle = f'CUBW_STEP_TAG={tag}'.encode()
    found = []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if needle in (d / 'environ').read_bytes().split(b'\0'):
                found.append({'pid': int(d.name), 'cmd': (d / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')[:200]})
        except OSError:
            continue
    for f in found:
        try:
            os.kill(f['pid'], signal.SIGKILL)
        except ProcessLookupError:
            pass
    end = now() + wait
    for f in found:
        while now() < end and Path(f'/proc/{f["pid"]}').exists():
            time.sleep(0.2)
        f['gone'] = not Path(f'/proc/{f["pid"]}').exists()
    return found


def stop_group(proc, grace=15):
    """SIGTERM the owned process group, wait, then SIGKILL; return what happened."""
    out = {'pid': proc.pid}
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(grace)
            out['stopped_by'] = 'SIGTERM'
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(30)
            out['stopped_by'] = 'SIGKILL'
    else:
        out['stopped_by'] = 'exited_before_stop'
    out['returncode'] = proc.returncode
    try:
        os.kill(proc.pid, 0)
        out['gone'] = False
    except ProcessLookupError:
        out['gone'] = True
    return out


def start_competitor(step, gpu, adir, seconds, tag, name='competitor'):
    mode, duty = TREATMENTS[step['treatment']]
    mask = step.get('competitor_mask') or step['plan_competitor_mask']
    args = [str(BIN), '--mode', mode, '--seconds', str(seconds), '--duty', str(duty)]
    if mode == 'read':
        args += ['--bytes', str(COMPETITOR_BYTES)]
    elif mode == 'fma':
        args += ['--iters', '2000']
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu['index']), HSA_CU_MASK=f"{gpu['index']}:{mask}", CUBW_STEP_TAG=tag)
    out = open(adir / f'{name}.jsonl', 'w')
    err = open(adir / f'{name}.err', 'w')
    proc = subprocess.Popen(args, env=env, stdout=out, stderr=err, start_new_session=True)
    rec = {'argv': args, 'hsa_cu_mask': env['HSA_CU_MASK'], 'pid': proc.pid, 'mode': mode, 'duty_requested': duty}
    deadline = now() + 90
    while now() < deadline:  # activity proof: the "started" event follows the first completed launch
        if proc.poll() is not None:
            rec['start_failed'] = f'exited rc={proc.returncode}'
            return proc, rec
        txt = (adir / f'{name}.jsonl').read_text()
        if '"event":"started"' in txt:
            rec['started_unix'] = now()
            return proc, rec
        time.sleep(0.2)
    rec['start_failed'] = 'no started event within 90 s'
    return proc, rec


def run_attempt(step, gpu, root, adir, plan):
    """One attempt of a vllm or competitor_only step. Returns the journal record."""
    tag = f"{os.getpid()}-{adir.parent.name}-{adir.name}-{int(now() * 1e6)}"
    rec = {'step': step['id'], 'attempt_dir': str(adir), 'kind': step['kind'], 'step_tag': tag}
    comp = None
    if step['kind'] == 'competitor_only':
        proc, crec = start_competitor(step, gpu, adir, step['seconds'] + 60, tag)
        rec['competitor'] = crec
        co = step.get('corunner')
        if co and 'start_failed' not in crec:  # placement-proven native co-runner on the decoder mask
            cstep = dict(step, treatment=co['treatment'], competitor_mask=co['mask'])
            cproc, corec = start_competitor(cstep, gpu, adir, step['seconds'] + 60, tag, name='corunner')
            rec['corunner'] = corec
        if 'start_failed' not in crec and 'start_failed' not in rec.get('corunner', {}):
            time.sleep(step['seconds'])
        if co and 'corunner' in rec:
            rec['corunner']['stop'] = stop_group(cproc)
            rec['corunner']['stopped_unix'] = now()
        crec['stop'] = stop_group(proc)
        rec['orphans'] = sweep_tag(tag)
        rec['residual_gpu_pids'] = gpu_processes(gpu['uuid'])
        rec['command_completed'] = ('start_failed' not in crec and crec['stop']['gone']
                                    and 'start_failed' not in rec.get('corunner', {})
                                    and rec.get('corunner', {}).get('stop', {}).get('gone', True))
        return rec
    t = TREATMENTS[step.get('treatment', 'absent')]
    if t is not None:
        comp, crec = start_competitor(step, gpu, adir, step['timeout_s'] + 120, tag)
        rec['competitor'] = crec
        if 'start_failed' in crec:
            crec['stop'] = stop_group(comp)
            rec['command_completed'] = False
            rec['failure_class'] = 'execution:competitor_start'
            return rec
    extra = [f'CUBW_STEP_TAG={tag}']
    mask = decoder_mask(step['decoder_cus'], plan['full_cus'])
    if mask:
        extra.append(f"HSA_CU_MASK={gpu['index']}:{mask}")
    if comp:
        extra.append(f'RESEARCH_ALLOWED_PIDS={comp.pid}')
    runner = ['scripts/controlled_vllm_experiments.py', '--config', str(root / step['config']), '--gpu-uuid', gpu['uuid'],
              '--mode', step.get('mode', 'timing'), '--output', str(adir / 'run')]
    if step.get('generated_tokens'):
        runner += ['--generated-tokens', str(step['generated_tokens'])]
    env = dict(os.environ)
    prof = step.get('profile')
    if prof:
        env['AMD_ENTRY'] = 'rocprofv3'
        pre = ['--kernel-trace', '--marker-trace', '--output-format', 'csv', '-d', str(adir / 'rocprof'), '-o', 'p']
        if prof.get('counters'):
            pre += ['--pmc'] + prof['counters']
        if prof.get('exclude_regex'):
            pre += ['--kernel-exclude-regex', prof['exclude_regex']]
        argv = [str(REPO / 'scripts/amd/vllm_in_container.sh')] + pre + ['--', 'python3'] + runner + ['--external-profiler']
    else:
        argv = [str(REPO / 'scripts/amd/vllm_in_container.sh')] + runner
    if extra:
        env['AMD_EXTRA_ENV'] = ','.join(extra)
    rec.update(argv=argv, amd_extra_env=env.get('AMD_EXTRA_ENV', ''), decoder_mask=mask,
               timeout_s=step['timeout_s'], t_start=now())
    with open(adir / 'runner.log', 'w') as log:
        proc = subprocess.Popen(argv, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        rec['runner_pid'] = proc.pid
        try:
            proc.wait(step['timeout_s'])
            rec['timed_out'] = False
        except subprocess.TimeoutExpired:
            rec['timed_out'] = True
            rec['runner_stop'] = stop_group(proc, grace=5)
    rec.update(t_end=now(), returncode=proc.returncode)
    if comp:
        rec['competitor']['stop'] = stop_group(comp)
        txt = (adir / 'competitor.jsonl').read_text()
        rec['competitor']['stopped_event'] = '"event":"stopped"' in txt
    rec['orphans'] = sweep_tag(tag)
    rec['residual_gpu_pids'] = gpu_processes(gpu['uuid'])
    run = adir / 'run' / 'run.json'
    status = json.loads(run.read_text())['status'] if run.exists() else 'missing'
    rec['run_status'] = status
    rec['command_completed'] = proc.returncode == 0 and not rec['timed_out'] and status == 'complete'
    if not rec['command_completed']:
        fail = adir / 'run' / 'failure.json'
        msg = json.loads(fail.read_text())['message'] if fail.exists() else ''
        rec['failure_message'] = msg[:500]
        # Output-identity/correctness assertions are scientific failures and are never retried.
        correctness = any(k in msg for k in ('differ', 'barrier', 'decode did not progress'))
        rec['failure_class'] = 'correctness' if correctness else ('execution:timeout' if rec['timed_out'] else 'execution')
    return rec


def run_native(step, gpu, adir):
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu['index']))
    if step.get('mask'):
        env['HSA_CU_MASK'] = f"{gpu['index']}:{step['mask']}"
    elif step.get('mask_other_index'):  # names another GPU's index: historically did not restrict this GPU
        env['HSA_CU_MASK'] = f"{(gpu['index'] + 1) % 8}:{step['mask_other_index']}"
    else:
        env.pop('HSA_CU_MASK', None)
    argv = [str(BIN)] + [str(a) for a in step['args']]
    rec = {'step': step['id'], 'attempt_dir': str(adir), 'kind': 'native', 'argv': argv,
           'hsa_cu_mask': env.get('HSA_CU_MASK', ''), 't_start': now()}
    with open(adir / 'out.json', 'w') as out, open(adir / 'err.txt', 'w') as err:
        proc = subprocess.Popen(argv, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            proc.wait(step['timeout_s'])
            rec['timed_out'] = False
        except subprocess.TimeoutExpired:
            rec['timed_out'] = True
            rec['stop'] = stop_group(proc, grace=2)
    rec.update(t_end=now(), returncode=proc.returncode)
    line = (adir / 'out.json').read_text().strip().splitlines()
    try:
        res = json.loads(line[-1]) if line else None
    except json.JSONDecodeError:
        res = None
    rec['check_passed'] = bool(res and res.get('check_passed'))
    # A probe that is expected to be rejected (e.g. an unsupported mask form) completes as a recorded outcome.
    rec['command_completed'] = (proc.returncode == 0 and not rec['timed_out']) or step.get('expect_failure', False)
    if not rec['command_completed']:
        rec['failure_class'] = 'execution:timeout' if rec['timed_out'] else 'execution'
    return rec


def run_shell(step, gpu, adir):
    env = dict(os.environ, HIP_VISIBLE_DEVICES=str(gpu['index']), CUBW_STEP_DIR=str(adir))
    rec = {'step': step['id'], 'attempt_dir': str(adir), 'kind': 'shell', 'cmd': step['cmd'], 't_start': now()}
    with open(adir / 'out.txt', 'w') as out:
        proc = subprocess.Popen(step['cmd'], shell=True, env=env, cwd=REPO, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(step['timeout_s'])
            rec['timed_out'] = False
        except subprocess.TimeoutExpired:
            rec['timed_out'] = True
            rec['stop'] = stop_group(proc, grace=2)
    rec.update(t_end=now(), returncode=proc.returncode, command_completed=proc.returncode == 0 and not rec['timed_out'])
    return rec


def describe(step, gpu, plan):
    """Dry-run rendering of the commands a step would execute."""
    if step['kind'] == 'native':
        m = f"HSA_CU_MASK={gpu['index']}:{step['mask']} " if step.get('mask') else ''
        return f"HIP_VISIBLE_DEVICES={gpu['index']} {m}{BIN.name} {' '.join(map(str, step['args']))}"
    if step['kind'] == 'shell':
        return step['cmd']
    parts = []
    t = TREATMENTS[step.get('treatment', 'absent')]
    if t:
        parts.append(f"competitor[{t[0]} duty={t[1]} mask={step.get('competitor_mask') or step['plan_competitor_mask']}]")
    if step['kind'] == 'competitor_only':
        if step.get('corunner'):
            c = step['corunner']
            parts.append(f"corunner[{TREATMENTS[c['treatment']][0]} mask={c['mask']}]")
        return ' + '.join(parts) + f" for {step['seconds']} s"
    mask = decoder_mask(step['decoder_cus'], plan['full_cus'])
    prof = step.get('profile')
    parts.append(f"{'rocprofv3 ' + json.dumps(prof) + ' ' if prof else ''}decoder[{Path(step['config']).stem} "
                 f"mask={mask or 'none'} mode={step.get('mode', 'timing')} tokens={step.get('generated_tokens', 'default')}]")
    return ' + '.join(parts)


def lane(args):
    plan = json.loads(args.plan.read_text())
    steps = [s for s in plan['steps'] if s['lane'] == args.lane and (args.stage is None or s['stage'] == args.stage)]
    gpus = {}
    for line in (args.root / 'gpus.tsv').read_text().splitlines():
        i, u, *_ = line.split('\t')
        gpus[int(i)] = {'index': int(i), 'uuid': u}
    gpu = gpus[plan['lanes'][args.lane]['gpu_index']]
    journal = Journal(args.root, args.lane)
    deadline = float(os.environ.get('CUBW_DEADLINE', '0')) or None
    gates = None
    for step in steps:
        step = dict(step, plan_competitor_mask=plan['competitor_mask'])
        if args.dry_run:
            print(f"[{args.lane} gpu{gpu['index']}] {step['id']}: {describe(step, gpu, plan)}")
            continue
        if step.get('requires'):
            gates = json.loads((args.root / 'gates.json').read_text()) if (args.root / 'gates.json').exists() else {}
            if not gates.get(step['requires']):
                journal.write(step=step['id'], disposition='not_run', reason=f"gate {step['requires']} not passed")
                continue
        if deadline and now() + step['est_s'] > deadline:
            journal.write(step=step['id'], disposition='not_run',
                          reason=f"deadline: {int(deadline - now())} s left < est {step['est_s']} s")
            continue
        base = args.root / step['out']
        for attempt in (1, 2):
            adir = base / f'attempt{attempt}'
            adir.mkdir(parents=True, exist_ok=False)
            if step['kind'] == 'native':
                rec = run_native(step, gpu, adir)
            elif step['kind'] == 'shell':
                rec = run_shell(step, gpu, adir)
            else:
                rec = run_attempt(step, gpu, args.root, adir, plan)
            rec.update(attempt=attempt, lane=args.lane, gpu_index=gpu['index'], gpu_uuid=gpu['uuid'],
                       experiment=step['experiment'], cell=step.get('cell'), acquisition=step.get('acquisition'))
            journal.write(**rec)
            retryable = not rec['command_completed'] and str(rec.get('failure_class', '')).startswith('execution')
            if not retryable or step.get('no_retry'):
                break
            if deadline and now() + step['est_s'] > deadline:
                break
    if not args.dry_run:
        journal.write(event='lane_complete', lane=args.lane)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('lane')
    q.add_argument('--plan', type=Path, required=True)
    q.add_argument('--root', type=Path, required=True)
    q.add_argument('--lane', required=True)
    q.add_argument('--stage')
    q.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    lane(a)
