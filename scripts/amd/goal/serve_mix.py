#!/usr/bin/env python3
"""G2/G3 mixed-serving runner (goal follow-up, docs/amd/goal_based_followup_20261105.md §4-5).

One in-process vLLM engine (same pinned model, BF16, TP1, chunked prefill, no prefix caching, greedy, EOS ignored
as scripts/controlled_vllm_experiments.py) serves a fixed, seeded request mix under a closed-loop concurrency cap C:
C requests are admitted at t0, and every time a request finishes the next one is admitted. Prompts are seeded
synthetic token IDs; prompt and output lengths cycle through predeclared lists, so prefill and decode interleave
(chunked prefill) as in serving.

Per request: admission time, first-token time (TTFT), every later token time (inter-token latency, ITL), completion
time, output digest. Per trial: makespan, output tokens, output/request throughput, TTFT/ITL/E2E percentiles and the
hardware state before/after. In energy mode a Sensor (board energy accumulator + sampled power, the existing
scripts/software_decode_experiments.py route) covers each timed trial: J per output token over the whole-request
scope (prefill + decode + scheduling), no idle subtraction.

Warmups: 3 runs of the first 2C requests of the trace (all prompt lengths compile and run) outside timing.
Outputs: greedy outputs can change with batch composition, which depends on timing in a closed loop. Output identity
is therefore recorded (per-request digests, agreement with trial 0 and with the warmup), not asserted; every request
must still finish with exactly its max_tokens tokens and unchanged prompt.

  serve_mix.py --config <cfg.json> --gpu-uuid <uuid> --mode timing|energy --output <dir>
               [--concurrency 8 --requests 32 --trials 10 --prompt-lens 256,1024,4096,512,2048 --output-lens 64,256,128]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics as st
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/
from software_decode_experiments import Sensor, digest, dump  # noqa: E402


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary(xs):
    return {'n': len(xs), 'mean': st.mean(xs) if xs else None, 'p50': pct(xs, .5), 'p90': pct(xs, .9),
            'p99': pct(xs, .99), 'max': max(xs) if xs else None}


def trace(n, prompt_lens, output_lens, vocab, seed):
    import random
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        p, o = prompt_lens[i % len(prompt_lens)], output_lens[i % len(output_lens)]
        reqs.append({'index': i, 'prompt_ids': [rng.randrange(vocab) for _ in range(p)], 'max_tokens': o})
    return reqs


def run_trace(engine, vllm, reqs, conc, prefix, seed, sensor=None):
    """Closed-loop serving of reqs with at most conc in flight. Returns per-request records and trial totals."""
    from vllm.sampling_params import RequestOutputKind
    assert not engine.has_unfinished_requests(), 'previous requests not drained'
    pending = list(reqs)
    live, recs = {}, {}

    def admit():
        r = pending.pop(0)
        rid = f'{prefix}_{r["index"]}'
        sp = vllm.SamplingParams(temperature=0, repetition_penalty=1.0, ignore_eos=True, detokenize=False,
                                 max_tokens=r['max_tokens'], seed=seed, output_kind=RequestOutputKind.CUMULATIVE)
        engine.add_request(rid, {'prompt_token_ids': r['prompt_ids']}, sp)
        live[rid] = r
        recs[rid] = {'index': r['index'], 'prompt_len': len(r['prompt_ids']), 'max_tokens': r['max_tokens'],
                     't_admit': time.perf_counter(), 'token_times': []}

    if sensor:
        sensor.start()
        before = sensor.read()
    t0 = time.perf_counter()
    wall0 = time.time_ns()
    for _ in range(min(conc, len(pending))):
        admit()
    steps = 0
    while engine.has_unfinished_requests():
        outs = engine.step()
        now = time.perf_counter()
        steps += 1
        for o in outs:
            rec = recs[o.request_id]
            got = len(o.outputs[0].token_ids)
            while len(rec['token_times']) < got:
                rec['token_times'].append(now)
            if o.finished:
                rec['t_done'] = now
                rec['finish_reason'] = o.outputs[0].finish_reason
                rec['output_ids'] = list(o.outputs[0].token_ids)
                rec['prompt_ok'] = list(o.prompt_token_ids) == live[o.request_id]['prompt_ids']
                del live[o.request_id]
                if pending:
                    admit()
        assert steps < 10_000_000
    t1 = time.perf_counter()
    wall1 = time.time_ns()
    tel = None
    if sensor:
        after = sensor.read()
        tel = sensor.finish(t0, t1, before, after, sum(len(r['output_ids']) for r in recs.values()), 5)
    for rid, r in recs.items():
        assert r.get('finish_reason') == 'length' and len(r['output_ids']) == r['max_tokens'], f'{rid} incomplete'
        assert r['prompt_ok'], f'{rid} prompt changed'
        r['ttft_ms'] = (r['token_times'][0] - r['t_admit']) * 1e3
        r['itl_ms'] = [(b - a) * 1e3 for a, b in zip(r['token_times'], r['token_times'][1:])]
        r['e2e_ms'] = (r['t_done'] - r['t_admit']) * 1e3
        r['output_sha256'] = digest([r['output_ids']])
        r['t_admit_rel_ms'] = (r['t_admit'] - t0) * 1e3
        del r['token_times'], r['t_admit'], r['t_done']
    out_tokens = sum(r['max_tokens'] for r in recs.values())
    span = t1 - t0
    res = {'prefix': prefix, 'makespan_s': span, 'engine_steps': steps, 'requests': len(recs),
           'output_tokens': out_tokens, 'prompt_tokens': sum(r['prompt_len'] for r in recs.values()),
           'output_tokens_per_s': out_tokens / span, 'requests_per_s': len(recs) / span,
           'ttft_ms': summary([r['ttft_ms'] for r in recs.values()]),
           'itl_ms': summary([x for r in recs.values() for x in r['itl_ms']]),
           'e2e_ms': summary([r['e2e_ms'] for r in recs.values()]),
           'wall_start_unix_ns': wall0, 'wall_end_unix_ns': wall1,
           'per_request': sorted(recs.values(), key=lambda r: r['index'])}
    if tel:
        res['telemetry'] = tel
    return res


def main(a):
    cfg = json.loads(Path(a.config).read_text())
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0')
    from gpu_target import load_nvml, select, verify_cuda, describe
    nvml = load_nvml()
    nvml.nvmlInit()
    handle, physical = select(nvml, a.gpu_uuid)
    import torch
    verify_cuda(torch, a.gpu_uuid)
    import vllm
    path = Path(cfg['model_path'])
    assert path.is_dir() and path.name == cfg['model_revision']
    settings = {'model': str(path), 'dtype': 'bfloat16', 'seed': cfg['seed'], 'generation_config': 'vllm',
                'enable_prefix_caching': False, 'enable_chunked_prefill': True, 'disable_log_stats': True,
                'async_scheduling': False, 'distributed_executor_backend': 'uni', 'tensor_parallel_size': 1,
                'max_model_len': 8192, 'max_num_seqs': 8, 'max_num_batched_tokens': 8192,
                'gpu_memory_utilization': 0.85, 'enforce_eager': False}
    settings.update({k: v for k, v in cfg.get('engine', {}).items()
                     if k in ('max_model_len', 'max_num_seqs', 'max_num_batched_tokens', 'gpu_memory_utilization')})
    assert settings['max_num_seqs'] >= a.concurrency, 'engine max_num_seqs below the concurrency cap'
    data = {'status': 'running', 'classification': 'measured-real-GPU (mixed serving, closed loop)', 'mode': a.mode,
            'scope': __doc__.strip(), 'config': cfg, 'engine_settings': settings,
            'workload': {'concurrency': a.concurrency, 'requests': a.requests, 'prompt_lens': a.prompt_lens,
                         'output_lens': a.output_lens, 'seed': cfg['seed'], 'prompt_kind': 'seeded synthetic uniform ids',
                         'arrival': 'closed loop: admit the next request when one finishes'},
            'hardware': {'uuid': a.gpu_uuid, 'physical_index': physical, **describe(nvml, handle, torch)},
            'environment': {k: os.environ.get(k, 'unset') for k in ('HIP_VISIBLE_DEVICES', 'HSA_CU_MASK',
                            'CUBW_CU_COUNT_OVERRIDE', 'RESEARCH_ALLOWED_PIDS', 'VLLM_ENABLE_V1_MULTIPROCESSING')},
            'versions': {'vllm': vllm.__version__, 'torch': torch.__version__},
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'warmups': [], 'trials': []}
    dump(a.output / 'run.json', data)
    llm = vllm.LLM(**settings)
    engine = llm.llm_engine
    vocab = json.loads((path / 'config.json').read_text())['vocab_size']
    reqs = trace(a.requests, a.prompt_lens, a.output_lens, vocab, cfg['seed'])
    data['workload']['trace_sha256'] = digest([[r['max_tokens']] + r['prompt_ids'] for r in reqs])
    warm = reqs[:2 * a.concurrency]
    for w in range(3):
        r = run_trace(engine, vllm, warm, a.concurrency, f'warmup{w}', cfg['seed'])
        data['warmups'].append({k: v for k, v in r.items() if k != 'per_request'}
                               | {'output_sha256': [q['output_sha256'] for q in r['per_request']]})
        dump(a.output / 'run.json', data)
    for t in range(a.trials):
        sensor = Sensor(nvml, handle, 0.02) if a.mode == 'energy' else None
        hw0 = describe(nvml, handle, torch)
        r = run_trace(engine, vllm, reqs, a.concurrency, f'trial{t}', cfg['seed'], sensor)
        r['hardware_before'], r['hardware_after'] = hw0, describe(nvml, handle, torch)
        if a.mode == 'energy':
            tel = r['telemetry']
            r['joules_per_output_token'] = (tel['counter_j'] / r['output_tokens']
                                            if isinstance(tel.get('counter_j'), (int, float)) else 'NA')
        data['trials'].append(r)
        dump(a.output / 'run.json', data)
        print(f"trial {t}: {r['makespan_s']:.2f} s, {r['output_tokens_per_s']:.1f} tok/s, "
              f"TTFT p50 {r['ttft_ms']['p50']:.1f} ms, ITL p50 {r['itl_ms']['p50']:.2f} ms", flush=True)
    ref = [q['output_sha256'] for q in data['trials'][0]['per_request']]
    wref = data['warmups'][0]['output_sha256']
    data['output_agreement'] = {
        'requests_matching_trial0_per_trial': [sum(q['output_sha256'] == h for q, h in zip(t['per_request'], ref))
                                               for t in data['trials']],
        'warmup_requests_matching_trial0': sum(a_ == b for a_, b in zip(wref, ref)),
        'note': 'recorded, not asserted: closed-loop batch composition can change greedy numerics'}
    data['status'] = 'complete'
    dump(a.output / 'run.json', data)
    try:
        engine.engine_core.shutdown()
    except Exception:
        dump(a.output / 'cleanup.json', {'status': 'failed', 'traceback': traceback.format_exc()})
    nvml.nvmlShutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--mode', choices=['timing', 'energy'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--requests', type=int, default=32)
    p.add_argument('--trials', type=int, default=10)
    ints = lambda v: [int(x) for x in v.split(',') if x]
    p.add_argument('--prompt-lens', type=ints, default=[256, 1024, 4096, 512, 2048])
    p.add_argument('--output-lens', type=ints, default=[64, 256, 128])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        main(a)
    except Exception as error:
        if (a.output / 'run.json').exists():
            d = json.loads((a.output / 'run.json').read_text())
            d['status'] = 'failed'
            dump(a.output / 'run.json', d)
        dump(a.output / 'failure.json', {'type': type(error).__name__, 'message': str(error),
                                         'traceback': traceback.format_exc()})
        raise
