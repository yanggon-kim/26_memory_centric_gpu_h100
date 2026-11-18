#!/usr/bin/env python3
"""Chunked-prefill campaign, experiments 1, 3 (step level) and 4 (decoder side): controlled admission/barrier replay of
engine iterations (docs/amd/chunked_prefill_campaign_20261118.md section 1).

The installed engine (vLLM 0.23 V1) has no iteration replay API, so this is the handoff's fallback, a controlled
admission/barrier harness, with its fidelity measured and reported:

  for each resident context S and decode width N (order shuffled with the seed):
    1. admit N requests with S fresh prompt tokens each (independent random ids), step until every one has produced
       its first token (barrier: the last step was decode-only with exactly N decodes)
    2. warm: one untimed pair per prefill size
    3. reps x shuffled prefill sizes C: a decode-only step (the paired reference T(N,0,S)), then a fresh request of C
       prompt tokens and max_tokens 1 is added and the next step is timed: N decodes + one C-token prompt in one
       iteration (T(N,C,S)); the request finishes in that step and frees its blocks. C = 0 gives two decode steps
       (the null pair). Composition is verified from the scheduler record of every timed step.
    4. abort the N requests and drain.
  Decode positions advance by one per step: the drift over a sweep (2 x reps x |C| steps) is recorded per step
  (decode_ctx_sum) and pairs are adjacent, so the drift is shared within a pair.
  N = 0 rows are pure-prefill anchors (one C-token prompt alone).

--prefix (experiment 3, needs prefix caching on): at each S, N the fresh request is prefix P + suffix F. Treatments per
P: 'warm' (P cached by an untimed request with the same P tokens and max_tokens 1; executes F, attends P+F), 'cold'
(same P+F tokens, nothing cached; executes P+F), 'fresh' (F tokens, no prefix). Cached tokens are verified from the
scheduler record (num_computed_tokens at admission). Prefixes are independent across requests (new ids every rep).

Timing per step: exec_s (scheduler return -> update_from_output, synchronous model execution incl. sampling), sched_s,
and the harness wall of engine.step(). --markers wraps every timed step in a roctx range (torch.cuda.nvtx) for
rocprofv3 --marker-trace runs (instrumented; their timings are not used for headroom).

  cp_replay.py --config cfg.json --gpu-uuid U --output DIR [--contexts 1024,8192 --widths 1,8,32
               --chunks 0,64,256,512,1024 --reps 10 --budget 4096 --anchors] [--prefix 0,1024,4096 --suffix 256]
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cp_common as C  # noqa: E402


class Run:
    def __init__(self, a, cfg):
        self.a, self.cfg = a, cfg
        self.out = a.output
        self.rows = open(self.out / 'steps.jsonl', 'w', buffering=1)
        self.last = []

    def listen(self, rec):
        self.last.append(rec)

    def step(self, tag=None):
        """One engine step; returns (outputs, [scheduler records of the step], wall_s)."""
        self.last = []
        nv = self.a.markers and tag is not None
        if nv:
            self.torch.cuda.nvtx.range_push(f'cp|{tag}')
        t = time.perf_counter()
        outs = self.engine.step()
        w = time.perf_counter() - t
        if nv:
            self.torch.cuda.nvtx.range_pop()
        return outs, self.last, w

    def write(self, **row):
        self.rows.write(json.dumps(row, separators=(',', ':')) + '\n')

    def add(self, rid, ids, max_tokens):
        self.engine.add_request(rid, {'prompt_token_ids': ids}, C.sampling(self.vllm, max_tokens, self.cfg['seed']))

    def drain(self, ids=()):
        if ids:
            self.engine.abort_request(list(ids))
        n = 0
        while self.engine.has_unfinished_requests():
            self.engine.step()
            n += 1
            assert n < 100000
        return n

    def resident(self, S, N, tag):
        """Admit N requests of S prompt tokens; step to the decode barrier. Returns their ids and barrier info."""
        ids = [f'{tag}_d{i}' for i in range(N)]
        max_tok = self.settings['max_model_len'] - S - 64
        for rid in ids:
            self.add(rid, self.tok(S), max_tok)
        t = time.perf_counter()
        steps = 0
        while True:
            _, recs, _ = self.step()
            steps += 1
            r = recs[-1] if recs else None
            if r and r['kind'] == 'decode' and r['n_decode'] == N and r['n_waiting'] == 0:
                break
            assert steps < 20000, 'barrier not reached'
        return ids, {'barrier_steps': steps, 'barrier_s': time.perf_counter() - t}

    def timed(self, tag, meta, fresh=None):
        """A timed step. fresh = (rid, ids) of a one-token request added just before the step."""
        if fresh:
            self.add(fresh[0], fresh[1], 1)
        outs, recs, wall = self.step(tag)
        r = recs[-1] if recs else {}
        self.write(**meta, wall_s=wall, **{k: r.get(k) for k in ('i', 'kind', 'n_decode', 'decode_ctx_sum', 'decode_ctx_max',
                                                                   'prefill', 'exec_s', 'sched_s', 'total_tokens', 'preempted')})
        return r, outs


def composition_ok(r, N, exec_tokens):
    pf = r.get('prefill') or []
    return r.get('n_decode') == N and (exec_tokens == 0 and not pf or len(pf) == 1 and pf[0][2] == exec_tokens)


def sweep(run, S, N, rng):
    a = run.a
    tag = f'S{S}_N{N}'
    meta = {'S': S, 'N': N}
    ids, info = run.resident(S, N, tag) if N else ([], {'barrier_steps': 0})
    run.write(event='barrier', **meta, **info)
    chunks = list(a.chunks)
    bad = 0
    for w in range(1 + a.reps):
        order = chunks[:]
        rng.shuffle(order)
        for Cn in order:
            rep = w - 1  # rep -1 = warmup
            m = dict(meta, C=Cn, rep=rep)
            if N:
                r0, _ = run.timed(f'{tag}_C{Cn}_r{rep}_ref', dict(m, role='ref'))
                bad += not composition_ok(r0, N, 0)
            if Cn == 0 and not N:
                continue
            fresh = (f'{tag}_C{Cn}_r{rep}_f', run.tok(Cn)) if Cn else None
            r1, _ = run.timed(f'{tag}_C{Cn}_r{rep}_mix', dict(m, role='mix'), fresh)
            ok = composition_ok(r1, N, Cn)
            bad += not ok
            if fresh and not ok:
                run.drain()  # never leave an unfinished fresh request behind
                if N:
                    raise RuntimeError(f'composition mismatch at {m}: {r1}')
    run.drain(ids)
    return bad


def prefix_sweep(run, S, N, rng):
    a = run.a
    tag = f'P_S{S}_N{N}'
    meta = {'S': S, 'N': N, 'F': a.suffix}
    ids, info = run.resident(S, N, tag) if N else ([], {'barrier_steps': 0})
    run.write(event='barrier', **meta, **info)
    bad = 0
    for w in range(1 + a.reps):
        order = [(P, t) for P in a.prefix for t in (('warm', 'cold', 'fresh') if P else ('fresh',))]
        rng.shuffle(order)
        for P, t in order:
            rep = w - 1
            m = dict(meta, P=P, treatment=t, rep=rep)
            prefix, suffix = run.tok(P), run.tok(a.suffix)
            if t == 'warm':  # build the cache entry outside timing (the prefix alone, one output token)
                run.add(f'{tag}_P{P}_r{rep}_warm', prefix, 1)
                tb = time.perf_counter()
                while any('_warm' in k for k in run.sched.requests):
                    run.step()
                run.write(event='warm_build', **m, s=time.perf_counter() - tb)
            if N:
                r0, _ = run.timed(f'{tag}_P{P}_{t}_r{rep}_ref', dict(m, role='ref'))
                bad += not composition_ok(r0, N, 0)
            ids_ = prefix + suffix if t != 'fresh' else suffix
            r1, _ = run.timed(f'{tag}_P{P}_{t}_r{rep}_mix', dict(m, role='mix'), (f'{tag}_P{P}_{t}_r{rep}_f', ids_))
            pf = r1.get('prefill') or []
            cached = pf[0][4] if pf else None
            want_exec = a.suffix if t == 'warm' else len(ids_)
            ok = r1.get('n_decode') == N and len(pf) == 1 and pf[0][2] == want_exec
            run.write(event='prefix_check', **m, cached_at_admit=cached, executed=pf[0][2] if pf else None, ok=ok)
            bad += not ok
            if not ok:
                run.drain()
    run.drain(ids)
    return bad


def main(a):
    cfg = json.loads(Path(a.config).read_text())
    C.setup_env()
    nvml, handle, physical, torch = C.open_gpu(a.gpu_uuid)
    over = dict(cfg.get('engine', {}), max_num_batched_tokens=a.budget, max_num_seqs=max(a.widths) + 4,
                max_model_len=a.max_model_len, enable_prefix_caching=bool(a.prefix), enable_chunked_prefill=True)
    data = {'status': 'running', 'classification': 'measured-real-GPU (controlled admission/barrier step replay)',
            'scope': __doc__.strip(), 'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            'config': cfg, 'layout': C.model_layout(cfg), 'hardware': {'uuid': a.gpu_uuid, 'physical_index': physical,
                                                                       **C.hw(nvml, handle, torch)},
            'environment': C.env_record(), 'script_sha256': C.sha_file(__file__),
            'sched_sha256': C.sha_file(Path(__file__).parent / 'cp_sched.py')}
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
    cells = [(S, N) for S in a.contexts for N in a.widths]
    rng.shuffle(cells)
    if a.anchors:
        cells.append((0, 0))
    data['cells'] = cells
    data['hardware_before'] = C.hw(nvml, handle, torch)
    sensor = C.Sensor(nvml, handle, 0.05)
    sensor.start()
    t0 = time.perf_counter()
    data['composition_mismatches'] = 0
    for S, N in cells:
        fn = prefix_sweep if a.prefix else sweep
        data['composition_mismatches'] += fn(run, S, N, rng)
        C.dump(a.output / 'run.json', data)
        print(f'cell S={S} N={N} done', flush=True)
    t1 = time.perf_counter()
    before, after = sensor.samples[0], sensor.read()
    tel = sensor.finish(t0, t1, before, after, 1, 5)
    data['telemetry_samples'] = [{k: s.get(k) for k in ('t', 'power_w', 'sm_clock_mhz', 'memory_clock_mhz', 'temperature_c')}
                                 for s in sensor.samples]
    data['telemetry'] = {k: v for k, v in tel.items() if k != 'samples'} if isinstance(tel, dict) else None
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
    p.add_argument('--widths', type=ints, default=[1, 8, 32])
    p.add_argument('--chunks', type=ints, default=[0, 64, 256, 512, 1024])
    p.add_argument('--reps', type=int, default=10)
    p.add_argument('--budget', type=int, default=4096)
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--anchors', action='store_true')
    p.add_argument('--prefix', type=ints, default=[])
    p.add_argument('--suffix', type=int, default=256)
    p.add_argument('--markers', action='store_true')
    p.add_argument('--seed-offset', type=int, default=0)
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
