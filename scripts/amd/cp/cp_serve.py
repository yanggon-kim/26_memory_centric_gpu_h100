#!/usr/bin/env python3
"""Chunked-prefill campaign, experiments 2 and 3 (serving): open-loop replay of deterministic traces
(scripts/amd/cp/cp_trace.py) through one in-process vLLM 0.23 engine under a scheduling policy
(scripts/amd/cp/cp_sched.py) and token budget (docs/amd/chunked_prefill_campaign_20261118.md sections 2-3).

Open loop: a request is added to the engine at the first step boundary at or after its timestamp (the engine core
also admits requests only between steps); when the engine is idle the driver sleeps until the next arrival. TTFT is
measured from the scheduled arrival time, so queueing before admission counts. Token times are the end of the step that
produced them.

Windows (per trace): warmup = the first 10 % of requests by arrival (served, not scored); measured = the rest; drain
deadline = last arrival + --drain-s. Requests still unfinished at the deadline are aborted and scored as incomplete
(SLO failures), so overload cannot raise goodput. Every request is recorded with its arrival, admission, token times
(compressed: first, last, ITL list), completion and output digest; SLO compliance is computed by the analysis against the
pre-registered limits, not here.

Per trace also: offered vs achieved, queue depth over time (scheduler records: n_waiting, n_running), step composition
fractions, preemptions, energy (board counter + sampled power over [first arrival, last completion or deadline]; scope:
one board, whole-request service incl. idle gaps, no idle subtraction).

Experiment 3 (--workload prefix --prefix-len P --prefix-mode warm|cold): every request has its own P-token prefix;
'warm' first runs each prefix alone (max_tokens 1, closed loop, outside timing; its cost is recorded) so the trace hits
the prefix cache; 'cold' runs the same trace with prefix caching on but nothing warmed. Hits are verified per request
(RequestOutput.num_cached_tokens and the scheduler's num_computed_tokens at admission).

  cp_serve.py --config cfg.json --gpu-uuid U --output DIR --policy mixed|separate --budget 2048 [--no-chunk]
              --workload balanced --arrivals poisson,bursty --rates 0.5,1,2 --seeds 101 --requests 300 [--energy]
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cp_common as C  # noqa: E402
import cp_trace as T  # noqa: E402

WARMUP_FRACTION = 0.10


def serve_trace(run, reqs, drain_s, sensor=None, label=''):
    eng, vllm, seed = run.engine, run.vllm, run.cfg['seed']
    assert not eng.has_unfinished_requests()
    steps = []
    run.sched.cp_listeners[:] = [lambda r: steps.append(r)]
    pending = list(reqs)
    recs = {}
    if sensor:
        sensor.start()
        before = sensor.read()
    t0 = time.perf_counter()
    deadline = t0 + reqs[-1]['arrival_s'] + drain_s
    aborted = []
    while True:
        now = time.perf_counter()
        while pending and t0 + pending[0]['arrival_s'] <= now:
            q = pending.pop(0)
            eng.add_request(q['id'], {'prompt_token_ids': q['prompt_ids']}, C.sampling(vllm, q['max_tokens'], seed))
            recs[q['id']] = {'id': q['id'], 'arrival_s': q['arrival_s'], 'admit_s': now - t0, 'prompt_len': len(q['prompt_ids']),
                             'max_tokens': q['max_tokens'], 'prefix_len': q['prefix_len'], 'times': []}
        if eng.has_unfinished_requests():
            if now > deadline:
                live = [k for k, r in recs.items() if 'done_s' not in r]
                eng.abort_request(live)
                aborted = live
                while eng.has_unfinished_requests():
                    eng.step()
                break
            outs = eng.step()
            tn = time.perf_counter() - t0
            for o in outs:
                r = recs.get(o.request_id)
                if r is None:
                    continue
                got = len(o.outputs[0].token_ids)
                while len(r['times']) < got:
                    r['times'].append(tn)
                if o.finished:
                    r['done_s'] = tn
                    r['finish_reason'] = o.outputs[0].finish_reason
                    r['output_sha256'] = C.digest([list(o.outputs[0].token_ids)])
                    r['num_cached_tokens'] = getattr(o, 'num_cached_tokens', None)
        elif pending:
            time.sleep(max(0.0, min(t0 + pending[0]['arrival_s'] - time.perf_counter(), 0.05)))
        else:
            break
    t1 = time.perf_counter()
    tel = None
    if sensor:
        after = sensor.read()
        tel = sensor.finish(t0, t1, before, after, sum(len(r['times']) for r in recs.values()), 5)
    for r in recs.values():
        ts = r.pop('times')
        r['tokens'] = len(ts)
        r['complete'] = r.get('finish_reason') == 'length' and len(ts) == r['max_tokens']
        r['ttft_s'] = ts[0] - r['arrival_s'] if ts else None
        r['itl_ms'] = [round((b - a) * 1e3, 3) for a, b in zip(ts, ts[1:])]
        r['last_token_s'] = ts[-1] if ts else None
    order = sorted(recs.values(), key=lambda r: r['arrival_s'])
    nw = int(len(order) * WARMUP_FRACTION)
    for i, r in enumerate(order):
        r['window'] = 'warmup' if i < nw else 'measured'
    res = {'label': label, 'span_s': t1 - t0, 'aborted': len(aborted), 'steps': len(steps),
           'requests': order, 'step_records': steps}
    if tel:
        res['telemetry'] = {k: v for k, v in tel.items() if k != 'samples'}
        res['power_samples'] = [{k: s.get(k) for k in ('t', 'power_w', 'energy_mj', 'sm_clock_mhz', 'memory_clock_mhz')}
                                for s in sensor.samples]
    return res


def warm_prefixes(run, reqs, P):
    """Prefill every request's own prefix once (max_tokens 1), closed loop, outside timing."""
    eng = run.engine
    t = time.perf_counter()
    for q in reqs:
        eng.add_request(q['id'] + '_warm', {'prompt_token_ids': q['prompt_ids'][:P]}, C.sampling(run.vllm, 1, run.cfg['seed']))
    n = 0
    while eng.has_unfinished_requests():
        eng.step()
        n += 1
    return {'warm_s': time.perf_counter() - t, 'warm_steps': n, 'warm_tokens': P * len(reqs)}


def main(a):
    cfg = json.loads(Path(a.config).read_text())
    C.setup_env()
    nvml, handle, physical, torch = C.open_gpu(a.gpu_uuid)
    prefix = a.workload == 'prefix'
    over = dict(cfg.get('engine', {}), max_num_batched_tokens=a.budget, max_num_seqs=a.max_seqs,
                max_model_len=min(a.max_model_len, a.budget) if a.no_chunk else a.max_model_len,
                enable_chunked_prefill=not a.no_chunk, enable_prefix_caching=prefix and a.prefix_mode != 'none',
                gpu_memory_utilization=a.gpu_mem)
    layout = C.model_layout(cfg)
    data = {'status': 'running', 'classification': 'measured-real-GPU (open-loop serving replay)', 'scope': __doc__.strip(),
            'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, 'config': cfg, 'layout': layout,
            'hardware': {'uuid': a.gpu_uuid, 'physical_index': physical, **C.hw(nvml, handle, torch)},
            'environment': C.env_record(), 'script_sha256': C.sha_file(__file__),
            'sched_sha256': C.sha_file(Path(__file__).parent / 'cp_sched.py'),
            'trace_sha256': C.sha_file(Path(__file__).parent / 'cp_trace.py'), 'traces': []}
    C.dump(a.output / 'run.json', data)
    import vllm
    llm, engine, sched, settings = C.build(cfg, over, a.policy, None)

    class R:
        pass
    run = R()
    run.engine, run.vllm, run.sched, run.cfg = engine, vllm, sched, cfg
    data.update(engine_settings=settings, engine_info=C.engine_info(engine, llm), versions=C.versions(torch), policy=a.policy)
    C.dump(a.output / 'run.json', data)
    # engine warmup outside timing: a short closed burst exercising prompt chunks and decodes
    wr, _ = T.make('balanced' if not prefix else 'prefix', 'poisson', 1000.0, 8, 999, layout['vocab'],
                   prefix_len=a.prefix_len, suffix=a.suffix, output=16)
    for q in wr:
        q['max_tokens'] = 16
        q['arrival_s'] = 0.0
        q['id'] += '_engine_warmup'
    serve_trace(run, wr, 600, label='engine_warmup')
    if prefix and a.prefix_mode != 'none':
        llm.reset_prefix_cache()
    k = 0
    for seed in a.seeds:
        for arrival in a.arrivals:
            for rate in a.rates:
                reqs, h = T.make(a.workload, arrival, rate, a.requests, seed, layout['vocab'],
                                 prefix_len=a.prefix_len, suffix=a.suffix, output=a.output_len)
                tag = f'{a.workload}_{arrival}_r{rate}_s{seed}'
                if prefix and a.prefix_mode == 'warm':  # unique ids per trace so cache entries never cross traces
                    warm = warm_prefixes(run, reqs, a.prefix_len)
                else:
                    warm = None
                if prefix and a.prefix_mode == 'cold':
                    llm.reset_prefix_cache()
                hw0 = C.hw(nvml, handle, torch)
                sensor = C.Sensor(nvml, handle, 0.05) if a.energy else None
                res = serve_trace(run, reqs, a.drain_s, sensor, tag)
                res.update(workload=a.workload, arrival=arrival, rate=rate, seed=seed, trace_sha256=h,
                           content_sha256=T.content_digest(reqs), demand=T.demand(reqs), warm=warm,
                           prefix_len=a.prefix_len, prefix_mode=a.prefix_mode, hardware_before=hw0,
                           hardware_after=C.hw(nvml, handle, torch))
                (a.output / f'trace_{k:02d}_{tag}.json').write_text(json.dumps(res, separators=(',', ':')) + '\n')
                done = sum(r['complete'] for r in res['requests'])
                data['traces'].append({'file': f'trace_{k:02d}_{tag}.json', 'tag': tag, 'trace_sha256': h,
                                       'complete': done, 'requests': len(reqs), 'aborted': res['aborted'], 'span_s': res['span_s']})
                C.dump(a.output / 'run.json', data)
                print(f'{tag}: {done}/{len(reqs)} complete, span {res["span_s"]:.1f} s', flush=True)
                k += 1
                if prefix and a.prefix_mode != 'none':
                    llm.reset_prefix_cache()
    data['status'] = 'complete'
    C.dump(a.output / 'run.json', data)
    try:
        engine.engine_core.shutdown()
    except Exception:  # noqa: BLE001
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ints = lambda v: [int(x) for x in v.split(',') if x != '']  # noqa: E731
    floats = lambda v: [float(x) for x in v.split(',') if x != '']  # noqa: E731
    p.add_argument('--config', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', choices=('mixed', 'separate'), required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--no-chunk', action='store_true', help='enable_chunked_prefill=False: whole prompts only')
    p.add_argument('--workload', choices=sorted(T.WORKLOADS) + ['prefix'], required=True)
    p.add_argument('--arrivals', type=lambda v: v.split(','), default=['poisson'])
    p.add_argument('--rates', type=floats, required=True)
    p.add_argument('--seeds', type=ints, default=[T.DEV_SEEDS[0]])
    p.add_argument('--requests', type=int, default=300)
    p.add_argument('--drain-s', type=float, default=60.0)
    p.add_argument('--max-seqs', type=int, default=64)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--gpu-mem', type=float, default=0.9)
    p.add_argument('--energy', action='store_true')
    p.add_argument('--prefix-len', type=int, default=0)
    p.add_argument('--prefix-mode', choices=('warm', 'cold', 'none'), default='none')
    p.add_argument('--suffix', type=int, default=256)
    p.add_argument('--output-len', type=int, default=128)
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
