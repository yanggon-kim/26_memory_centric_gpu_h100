#!/usr/bin/env python3
"""Long-context campaign, families L2 and L4 (step level): decode attention vs prefill attention at long resident
contexts (docs/amd/lc_campaign_design_20261124.md sections 4 and 6). Extends the controlled admission/barrier replay of
scripts/amd/cp/cp_replay.py (same Run class, barrier, paired reference and composition checks) with a fixed prefill
prefix:

  for each cell (S, N), order shuffled with the seed:
    1. admit N requests of S fresh prompt tokens; step to the decode barrier (outside measurement)
    2. if P > 0: prefill a P-token prefix once with max_tokens 1 (prefix caching on; untimed, recorded)
    3. warmup + reps x shuffled C: a decode-only reference step T(N,0,S); then a request of prefix + C new tokens
       (max_tokens 1) is added and the next step is timed: N decodes + one C-token chunk whose attention covers P + C
       positions (the P positions come from the cache, verified per step: cached_at_admit == P, executed == C).
       C = 0 gives the null pair (two decode steps).
    4. abort the N requests and drain.
  N = 0 cells are prefill-alone anchors (the same C-token chunk after the same P-token prefix, no decoders).
  The prefix of a cell is fixed; at the control cell several P are swept (--prefix-sweep) at one (S, N).

Recorded per timed step: actual decode positions (decode_ctx_sum / max), chunk start and tokens, cached tokens at
admission, kind, exec_s, sched_s, harness wall. Graph path is derived in the analysis from total tokens and the capture
sizes in run.json. --markers: roctx range 'cp|S{S}_N{N}_P{P}_C{C}_r{rep}_{role}' per timed step (rocprofv3 kernel-trace
or --pmc runs, instrumented). --eager for counter runs (recorded).

  lc_replay.py --config cfg.json --gpu-uuid U --output DIR --contexts 1024,8192 --widths 1,8 --chunks 0,256,1024
               --prefix 4096 [--prefix-sweep 0,4096,16384,28672 --sweep-cell 8192:8] [--anchors] [--reps 10]
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'cp'))
import cp_common as C  # noqa: E402
from cp_replay import Run  # noqa: E402


def build_prefix(run, tag, ids):
    run.add(f'{tag}_pfx', ids, 1)
    t = time.perf_counter()
    n = 0
    while any(k == f'{tag}_pfx' or k.startswith(f'{tag}_pfx-') for k in run.sched.requests):  # internal ids carry a suffix
        run.step()
        n += 1
        assert n < 1000
    return {'prefix_build_s': time.perf_counter() - t, 'prefix_build_steps': n}


def cell(run, S, N, P, rng):
    a = run.a
    tag = f'S{S}_N{N}_P{P}'
    meta = {'S': S, 'N': N, 'P': P}
    ids, info = run.resident(S, N, tag) if N else ([], {'barrier_steps': 0})
    prefix = run.tok(P) if P else []
    if P:
        info.update(build_prefix(run, tag, prefix))
    run.write(event='barrier', **meta, **info)
    bad = 0
    for w in range(1 + a.reps):
        order = list(a.chunks)
        rng.shuffle(order)
        for Cn in order:
            rep = w - 1
            m = dict(meta, C=Cn, rep=rep)
            if N:
                r0, _ = run.timed(f'{tag}_C{Cn}_r{rep}_ref', dict(m, role='ref'))
                bad += not (r0.get('n_decode') == N and not r0.get('prefill'))
            if Cn == 0 and not N:
                continue
            fresh = (f'{tag}_C{Cn}_r{rep}_f', prefix + run.tok(Cn)) if Cn else None
            r1, _ = run.timed(f'{tag}_C{Cn}_r{rep}_mix', dict(m, role='mix'), fresh)
            pf = r1.get('prefill') or []
            if Cn:
                cached = pf[0][4] if pf else None
                ok = r1.get('n_decode') == N and len(pf) == 1 and pf[0][2] == Cn and (cached or 0) == P
                run.write(event='chunk_check', **m, cached_at_admit=cached, executed=pf[0][2] if pf else None,
                          start=pf[0][1] if pf else None, ok=ok)
            else:
                ok = r1.get('n_decode') == N and not pf
            bad += not ok
            if fresh and not ok:
                run.drain()
                if N:
                    raise RuntimeError(f'composition mismatch at {m}: {r1}')
    run.drain(ids)
    return bad


def main(a):
    cfg = json.loads(Path(a.config).read_text())
    C.setup_env()
    nvml, handle, physical, torch = C.open_gpu(a.gpu_uuid)
    need_pc = bool(a.prefix or a.prefix_sweep)
    over = dict(cfg.get('engine', {}), max_num_batched_tokens=a.budget, max_num_seqs=max(a.widths) + 4,
                max_model_len=a.max_model_len, enable_prefix_caching=need_pc, enable_chunked_prefill=True,
                gpu_memory_utilization=a.gpu_mem)
    if a.eager:
        over['enforce_eager'] = True
    data = {'status': 'running', 'classification': 'measured-real-GPU (controlled admission/barrier step replay, fixed prefix)',
            'scope': __doc__.strip(), 'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            'config': cfg, 'layout': C.model_layout(cfg), 'hardware': {'uuid': a.gpu_uuid, 'physical_index': physical,
                                                                       **C.hw(nvml, handle, torch)},
            'environment': C.env_record(), 'script_sha256': C.sha_file(__file__),
            'replay_sha256': C.sha_file(HERE.parent / 'cp' / 'cp_replay.py'),
            'sched_sha256': C.sha_file(HERE.parent / 'cp' / 'cp_sched.py')}
    C.dump(a.output / 'run.json', data)
    import vllm
    llm, engine, sched, settings = C.build(cfg, over, 'mixed', a.output / 'steplog.jsonl')
    run = Run(a, cfg)
    run.vllm, run.engine, run.sched, run.settings, run.torch = vllm, engine, sched, settings, torch
    sched.cp_listeners.append(run.listen)
    run.tok = C.Tokens(data['layout']['vocab'], cfg['seed'] + a.seed_offset)
    data.update(engine_settings=settings, engine_info=C.engine_info(engine, llm), versions=C.versions(torch))
    C.dump(a.output / 'run.json', data)
    rng = random.Random(cfg['seed'] + a.seed_offset)
    cells = [(S, N, a.prefix) for S in a.contexts for N in a.widths if not a.max_tokens or S * N <= a.max_tokens]
    if a.prefix_sweep:
        S0, N0 = (int(x) for x in a.sweep_cell.split(':'))
        cells += [(S0, N0, P) for P in a.prefix_sweep if (S0, N0, P) not in cells]
    rng.shuffle(cells)
    if a.anchors:
        cells += sorted({(0, 0, P) for P in [a.prefix] + list(a.prefix_sweep or [])})
    data['cells'] = cells
    data['skipped_cells'] = [(S, N) for S in a.contexts for N in a.widths if a.max_tokens and S * N > a.max_tokens]
    data['hardware_before'] = C.hw(nvml, handle, torch)
    sensor = C.Sensor(nvml, handle, 0.05)
    sensor.start()
    t0 = time.perf_counter()
    data['composition_mismatches'] = 0
    for S, N, P in cells:
        data['composition_mismatches'] += cell(run, S, N, P, rng)
        C.dump(a.output / 'run.json', data)
        print(f'cell S={S} N={N} P={P} done', flush=True)
    t1 = time.perf_counter()
    tel = sensor.finish(t0, t1, sensor.samples[0], sensor.read(), 1, 5)
    data['telemetry'] = {k: v for k, v in tel.items() if k != 'samples'} if isinstance(tel, dict) else None
    data['telemetry_samples'] = [{k: s.get(k) for k in ('t', 'power_w', 'sm_clock_mhz', 'memory_clock_mhz', 'temperature_c')}
                                 for s in sensor.samples]
    data['hardware_after'] = C.hw(nvml, handle, torch)
    data['status'] = 'complete'
    C.dump(a.output / 'run.json', data)
    try:
        engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ints = lambda v: [int(x) for x in v.split(',') if x != '']  # noqa: E731
    p.add_argument('--config', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contexts', type=ints, default=[1024, 8192])
    p.add_argument('--widths', type=ints, default=[1, 8])
    p.add_argument('--chunks', type=ints, default=[0, 256, 1024])
    p.add_argument('--prefix', type=int, default=4096)
    p.add_argument('--prefix-sweep', type=ints, default=[])
    p.add_argument('--sweep-cell', default='8192:8')
    p.add_argument('--anchors', action='store_true')
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--budget', type=int, default=4096)
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--max-tokens', type=int, default=0, help='skip cells with S x N above this many resident tokens')
    p.add_argument('--gpu-mem', type=float, default=0.9)
    p.add_argument('--markers', action='store_true')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--seed-offset', type=int, default=12)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        main(a)
    except Exception as error:
        if (a.output / 'run.json').exists():
            d = json.loads((a.output / 'run.json').read_text())
            d['status'] = 'failed'
            C.dump(a.output / 'run.json', d)
        C.dump(a.output / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                                           'traceback': traceback.format_exc()})
        raise
