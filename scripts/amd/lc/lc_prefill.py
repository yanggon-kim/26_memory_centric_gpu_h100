#!/usr/bin/env python3
"""Long-context campaign, family L1: cost of chunking one long prompt (docs/amd/lc_campaign_design_20261124.md section 3).

One engine per arm (the token budget is an engine setting): 'whole' (enable_chunked_prefill=False, budget =
max_model_len, so the prompt runs in one step) or 'mixed' with budget B (stock V1 chunking, decode-first mixing; with no
other request in the engine every step is a prompt chunk). All arms share max_model_len, an explicit KV budget
(kv_cache_memory_bytes), prefix caching off (cold prefix state), BF16, the graph policy of cp_common and the token ids:
the generator is seeded identically and the (rep, length) order is drawn from the same seed, so rep r of length L is the
same prompt in every arm.

Per prompt (max_tokens 1, alone in the engine): every engine step until the request finishes, with the scheduler record
(chunk start = earlier prefix positions, chunk tokens), exec_s (synchronous model execution incl. sampling) and the
harness wall of engine.step(); TTFT = add_request -> output carrying the token (harness wall); GPU event span on the
current stream from just before the first step to just after the last (includes host gaps between chunk steps); the
first output token id (numerical agreement across arms is reported, never asserted). rep -1 is an untimed warmup of
every length. --markers wraps each step in a roctx range 'lc|L{L}_r{rep}_k{chunk}' for rocprofv3 kernel-trace or
counter runs (instrumented; their timings are not the clean timings).

  lc_prefill.py --config cfg.json --gpu-uuid U --output DIR --arm whole|mixed --budget B
                [--lengths 1024,8192,16384,24576,32000 --reps 5 --max-model-len 32768 --kv-gib 40 --markers --eager]
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

GIB = 1 << 30


def main(a):
    cfg = json.loads(Path(a.config).read_text())
    C.setup_env()
    nvml, handle, physical, torch = C.open_gpu(a.gpu_uuid)
    whole = a.arm == 'whole'
    budget = max(a.budget, a.max_model_len) if whole else a.budget
    over = dict(cfg.get('engine', {}), max_num_batched_tokens=budget, max_num_seqs=4, max_model_len=a.max_model_len,
                enable_chunked_prefill=not whole, enable_prefix_caching=False, kv_cache_memory_bytes=int(a.kv_gib * GIB))
    if a.eager:
        over['enforce_eager'] = True
    layout = C.model_layout(cfg)
    assert max(a.lengths) + 1 <= a.max_model_len, 'prompt + one output token must fit max_model_len'
    data = {'status': 'running', 'classification': 'measured-real-GPU (single long prompt, chunked vs whole)',
            'scope': __doc__.strip(), 'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            'config': cfg, 'layout': layout, 'hardware': {'uuid': a.gpu_uuid, 'physical_index': physical,
                                                          **C.hw(nvml, handle, torch)},
            'environment': C.env_record(), 'script_sha256': C.sha_file(__file__),
            'sched_sha256': C.sha_file(HERE.parent / 'cp' / 'cp_sched.py'), 'prompts': []}
    C.dump(a.output / 'run.json', data)
    import vllm
    llm, engine, sched, settings = C.build(cfg, over, 'mixed', a.output / 'steplog.jsonl')
    data.update(engine_settings=settings, engine_info=C.engine_info(engine, llm), versions=C.versions(torch))
    C.dump(a.output / 'run.json', data)
    recs = []
    sched.cp_listeners.append(recs.append)
    tok = C.Tokens(layout['vocab'], cfg['seed'] + a.seed_offset)
    rng = random.Random(cfg['seed'] + a.seed_offset)
    rows = open(a.output / 'prompts.jsonl', 'w', buffering=1)
    data['hardware_before'] = C.hw(nvml, handle, torch)
    sensor = C.Sensor(nvml, handle, 0.05)
    sensor.start()
    t_all = time.perf_counter()
    bad = 0
    for rep in range(-1, a.reps):
        order = list(a.lengths)
        rng.shuffle(order)
        for L in order:
            ids = tok(L)
            rid = f'L{L}_r{rep}'
            recs.clear()
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            t0 = time.perf_counter()
            engine.add_request(rid, {'prompt_token_ids': ids}, C.sampling(vllm, 1, cfg['seed']))
            walls, ttft, first, k = [], None, None, 0
            while engine.has_unfinished_requests():
                if a.markers:
                    torch.cuda.nvtx.range_push(f'lc|L{L}_r{rep}_k{k}')
                tw = time.perf_counter()
                outs = engine.step()
                walls.append(time.perf_counter() - tw)
                if a.markers:
                    torch.cuda.nvtx.range_pop()
                k += 1
                for o in outs:
                    if o.request_id == rid and o.outputs and o.outputs[0].token_ids and ttft is None:
                        ttft = time.perf_counter() - t0
                        first = int(o.outputs[0].token_ids[0])
                assert k < 10000
            ev1.record()
            torch.cuda.synchronize()
            steps = [{'kind': r['kind'], 'exec_s': r.get('exec_s'), 'sched_s': r.get('sched_s'), 'prefill': r['prefill'],
                      'total_tokens': r['total_tokens'], 'n_decode': r['n_decode']} for r in recs]
            chunks = [c for s in steps for c in s['prefill'] if c[0] == rid]
            ok = sum(c[2] for c in chunks) == L and all(s['n_decode'] == 0 for s in steps)
            if whole:
                ok = ok and len(chunks) == 1
            bad += not ok
            row = {'L': L, 'rep': rep, 'arm': a.arm, 'budget': budget, 'n_steps': len(steps), 'n_chunks': len(chunks),
                   'chunk_tokens': [c[2] for c in chunks], 'chunk_start': [c[1] for c in chunks], 'ttft_s': ttft,
                   'exec_s_sum': sum(s['exec_s'] or 0 for s in steps), 'exec_s': [s['exec_s'] for s in steps],
                   'wall_s': walls, 'gpu_event_ms': ev0.elapsed_time(ev1), 'first_token': first, 'composition_ok': ok,
                   'prompt_sha256': C.digest([ids])}
            rows.write(json.dumps(row, separators=(',', ':')) + '\n')
            data['prompts'].append({k_: row[k_] for k_ in ('L', 'rep', 'n_chunks', 'ttft_s', 'exec_s_sum', 'gpu_event_ms',
                                                           'first_token', 'composition_ok')})
            if not ok:
                raise RuntimeError(f'composition mismatch at L={L} rep={rep}: {chunks}')
        C.dump(a.output / 'run.json', data)
        print(f'rep {rep} done', flush=True)
    t1 = time.perf_counter()
    tel = sensor.finish(t_all, t1, sensor.samples[0], sensor.read(), 1, 5)
    data['telemetry'] = {k: v for k, v in tel.items() if k != 'samples'} if isinstance(tel, dict) else None
    data['telemetry_samples'] = [{k: s.get(k) for k in ('t', 'power_w', 'sm_clock_mhz', 'memory_clock_mhz', 'temperature_c')}
                                 for s in sensor.samples]
    data['hardware_after'] = C.hw(nvml, handle, torch)
    data['composition_mismatches'] = bad
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
    p.add_argument('--arm', choices=('whole', 'mixed'), required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--lengths', type=ints, default=[1024, 8192, 16384, 24576, 32000])
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--kv-gib', type=float, default=40)
    p.add_argument('--markers', action='store_true')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--seed-offset', type=int, default=11)
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
