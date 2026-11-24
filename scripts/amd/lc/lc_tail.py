#!/usr/bin/env python3
"""Long-context campaign, family L3 part A: where the pure-decode tail begins (docs/amd/lc_campaign_design_20261124.md
section 5). One engine per scheduling policy (scripts/amd/cp/cp_sched.py; budget and chunking are engine settings).

Per case (O, I), order shuffled with the seed, reps x:
  request A: --prompt fresh tokens, max_tokens O (greedy, EOS ignored); stepped alone until A has its first output token;
  then, if I > 0, request B with I fresh tokens and max_tokens --inject-output is added at that step boundary;
  the engine is stepped until both finish.
Every executed step is recorded from the scheduler (kind decode|prefill|mixed, decode count and positions, chunks with
start/tokens/recompute flag, waiting/running, preemptions, exec_s) plus the harness wall, the A/B token times and the
output digests. rep -1 is an untimed warmup of every case. --markers wraps each step in a roctx range
'lc|O{O}_I{I}_r{rep}_s{i}' (rocprofv3 kernel-trace runs: GPU-active time per step type on profiled windows).

  lc_tail.py --config cfg.json --gpu-uuid U --output DIR --policy mixed|separate --budget B [--no-chunk]
             [--outputs 512,2048,4096 --injects 0,128,8192 --prompt 128 --inject-output 256 --reps 3]
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


def main(a):
    cfg = json.loads(Path(a.config).read_text())
    C.setup_env()
    nvml, handle, physical, torch = C.open_gpu(a.gpu_uuid)
    over = dict(cfg.get('engine', {}), max_num_batched_tokens=a.budget, max_num_seqs=8,
                max_model_len=min(a.max_model_len, a.budget) if a.no_chunk else a.max_model_len,
                enable_chunked_prefill=not a.no_chunk, enable_prefix_caching=False)
    layout = C.model_layout(cfg)
    assert a.prompt + max(a.outputs) <= over['max_model_len'] and max(a.injects) + a.inject_output <= over['max_model_len']
    data = {'status': 'running', 'classification': 'measured-real-GPU (two-request long-output tail)',
            'scope': __doc__.strip(), 'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            'config': cfg, 'layout': layout, 'hardware': {'uuid': a.gpu_uuid, 'physical_index': physical,
                                                          **C.hw(nvml, handle, torch)},
            'environment': C.env_record(), 'script_sha256': C.sha_file(__file__),
            'sched_sha256': C.sha_file(HERE.parent / 'cp' / 'cp_sched.py'), 'cases': []}
    C.dump(a.output / 'run.json', data)
    import vllm
    llm, engine, sched, settings = C.build(cfg, over, a.policy, None)
    data.update(engine_settings=settings, engine_info=C.engine_info(engine, llm), versions=C.versions(torch),
                policy=a.policy)
    C.dump(a.output / 'run.json', data)
    recs = []
    sched.cp_listeners.append(recs.append)
    tok = C.Tokens(layout['vocab'], cfg['seed'] + a.seed_offset)
    rng = random.Random(cfg['seed'] + a.seed_offset)
    seed = cfg['seed']
    cases = [(O, I) for O in a.outputs for I in a.injects]
    data['hardware_before'] = C.hw(nvml, handle, torch)
    sensor = C.Sensor(nvml, handle, 0.05)
    sensor.start()
    t_all = time.perf_counter()
    k = 0
    for rep in range(-1, a.reps):
        order = cases[:]
        rng.shuffle(order)
        for O, I in order:
            ida, idb = f'A_O{O}_I{I}_r{rep}', f'B_O{O}_I{I}_r{rep}'
            recs.clear()
            times = {ida: [], idb: []}
            digests = {}
            walls = []
            t0 = time.perf_counter()
            engine.add_request(ida, {'prompt_token_ids': tok(a.prompt)}, C.sampling(vllm, O, seed))
            injected = I == 0
            i = 0
            while engine.has_unfinished_requests():
                if a.markers:
                    torch.cuda.nvtx.range_push(f'lc|O{O}_I{I}_r{rep}_s{i}')
                tw = time.perf_counter()
                outs = engine.step()
                walls.append(time.perf_counter() - tw)
                if a.markers:
                    torch.cuda.nvtx.range_pop()
                tn = time.perf_counter() - t0
                i += 1
                for o in outs:
                    got = len(o.outputs[0].token_ids)
                    ts = times[o.request_id]
                    while len(ts) < got:
                        ts.append(tn)
                    if o.finished:
                        digests[o.request_id] = C.digest([list(o.outputs[0].token_ids)])
                if not injected and times[ida]:
                    engine.add_request(idb, {'prompt_token_ids': tok(I)}, C.sampling(vllm, a.inject_output, seed))
                    injected = True
                    t_inj = tn
                assert i < 50000
            steps = [{k_: r.get(k_) for k_ in ('i', 'kind', 'mode', 'n_decode', 'decode_ctx_sum', 'decode_ctx_max', 'prefill',
                                               'n_running', 'n_waiting', 'preempted', 'total_tokens', 'exec_s', 'sched_s')}
                     for r in recs]
            for s, w in zip(steps, walls):
                s['wall_s'] = w
            res = {'O': O, 'I': I, 'rep': rep, 'policy': a.policy, 'budget': a.budget, 'no_chunk': a.no_chunk,
                   'inject_at_s': t_inj if I else None, 'a_tokens': len(times[ida]), 'b_tokens': len(times[idb]),
                   'a_times': times[ida], 'b_times': times[idb], 'digests': digests, 'steps': steps,
                   'span_s': time.perf_counter() - t0}
            fn = f'case_{k:03d}_O{O}_I{I}_r{rep}.json'
            (a.output / fn).write_text(json.dumps(res, separators=(',', ':')) + '\n')
            complete = len(times[ida]) == O and (I == 0 or len(times[idb]) == a.inject_output)
            data['cases'].append({'file': fn, 'O': O, 'I': I, 'rep': rep, 'steps': len(steps), 'complete': complete,
                                  'span_s': res['span_s']})
            k += 1
            if not complete:
                raise RuntimeError(f'incomplete case O={O} I={I} rep={rep}')
        C.dump(a.output / 'run.json', data)
        print(f'rep {rep} done', flush=True)
    t1 = time.perf_counter()
    tel = sensor.finish(t_all, t1, sensor.samples[0], sensor.read(), 1, 5)
    data['telemetry'] = {k_: v for k_, v in tel.items() if k_ != 'samples'} if isinstance(tel, dict) else None
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
    p.add_argument('--policy', choices=('mixed', 'separate'), required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--no-chunk', action='store_true')
    p.add_argument('--outputs', type=ints, default=[512, 2048, 4096])
    p.add_argument('--injects', type=ints, default=[0, 128, 8192])
    p.add_argument('--prompt', type=int, default=128)
    p.add_argument('--inject-output', type=int, default=256)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--markers', action='store_true')
    p.add_argument('--seed-offset', type=int, default=13)
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
