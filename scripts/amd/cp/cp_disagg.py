#!/usr/bin/env python3
"""Chunked-prefill campaign, experiment 5: colocated replicas vs disaggregated prefill/decode workers on two allocated
GPUs (docs/amd/chunked_prefill_campaign_20261118.md section 5). Runs inside the vLLM container (one process): starts the
OpenAI servers (`vllm serve`, same pinned model, BF16, TP1, greedy, EOS ignored), replays deterministic open-loop traces
(scripts/amd/cp/cp_trace.py) through an asyncio client, then stops the servers.

Deployments (--deployment):
  single     one server on GPU A (the single-GPU colocated reference; GPU B idle but metered)
  colocated  one server per GPU; router = least outstanding work (prompt tokens not yet prefilled + output tokens not yet
             generated, by client accounting), ties to the lower index
  disagg     GPU A = prefill worker (kv_producer), GPU B = decode worker (kv_consumer); real KV transfer with vLLM's
             P2pNcclConnector (RCCL over the GPUs' peer link, PUT_ASYNC). Per request the client sends the prompt to the
             prefill worker with max_tokens 1 (X-Request-Id carries both workers' KV addresses, the connector's protocol),
             waits for it, then streams the full request from the decode worker, which loads the transferred KV and
             generates every output token. --decode-cus/--decode-grid restrict the decode worker (HSA_CU_MASK on 0..n-1,
             CUBW_CU_COUNT_OVERRIDE) for the restricted-decode variant.
Each server can carry a scheduler policy and token budget (scripts/amd/cp/cp_sched.py via --scheduler-cls, CP_POLICY).

Recorded per request: arrival, route, prefill-worker response time (disagg), first-token and per-token times (streamed
chunks), completion, output digest. Per trace: both workers' queue depth (/metrics, every 0.25 s), energy of both boards
(board counters and sampled power through amd-smi, scope = two boards over the whole trace incl. idle capacity), offered
vs completed. Logical KV bytes per transferred request = prompt tokens x kv_bytes_per_token (block-granular bytes are
rounded up to the block size in the analysis).

Multi-GPU generalization (design section 14, plan s4): with k GPUs in --gpu-indices, colocated runs k servers (same
least-outstanding-work router) and disagg runs --prefill-workers P producers on the first P GPUs and the remaining k - P
GPUs as consumers. Per request the client picks the producer with the fewest outstanding prompt tokens and the consumer
with the least outstanding work (output tokens not yet generated), both at send time, because the connector's request id
names both KV addresses. --decode-kv-bytes sets --kv-cache-memory-bytes on consumers (explicit decode KV capacity).

  cp_disagg.py --config cfg.json --gpu-uuids UA,UB --gpu-indices 0,1 --output DIR --deployment disagg
               --workload balanced --arrivals poisson --rates 1,2 --seeds 201 --requests 200 [--decode-cus 128 --decode-grid 128]
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cp_common as C  # noqa: E402
import cp_trace as T  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HTTP = [18101, 18102]  # worker k: 18101 + k + 10 x first GPU index (concurrent lanes on one node, set in main)
KV_PORT = [21101, 22101]  # worker k: 21101 + 1000 k + 10 x first GPU index


def mask_range(n):
    return f'0-{n - 1}'


def server_cmd(a, cfg, port, role=None):
    seqs = a.decode_max_seqs if (role == 'kv_consumer' and a.decode_max_seqs) else a.max_seqs
    k = HTTP.index(port)
    s = C.engine_settings(cfg, dict(cfg.get('engine', {}), max_num_batched_tokens=a.budget, max_num_seqs=seqs,
                                    max_model_len=min(a.max_model_len, a.budget) if a.no_chunk else a.max_model_len,
                                    gpu_memory_utilization=a.gpu_mem))
    cmd = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', s['model'], '--port', str(port), '--host', '127.0.0.1',
           '--dtype', 'bfloat16', '--seed', str(s['seed']), '--generation-config', 'vllm', '--max-model-len', str(s['max_model_len']),
           '--max-num-seqs', str(s['max_num_seqs']), '--max-num-batched-tokens', str(s['max_num_batched_tokens']),
           '--gpu-memory-utilization', str(s['gpu_memory_utilization']), '--no-enable-prefix-caching',
           '--no-enable-chunked-prefill' if a.no_chunk else '--enable-chunked-prefill', '--scheduler-cls', 'cp_sched.PolicyScheduler', '--served-model-name', 'm',
           '--disable-uvicorn-access-log', '--tensor-parallel-size', '1', '--no-async-scheduling',
           '--compilation-config', json.dumps(C.COMPILATION)]
    if role:
        kv = {'kv_connector': 'P2pNcclConnector', 'kv_role': role, 'kv_port': str(KV_PORT[k]),
              'kv_buffer_size': '1e1' if role == 'kv_producer' else '8e9',
              'kv_connector_extra_config': {'send_type': 'PUT_ASYNC', 'nccl_num_channels': '16', 'http_port': str(port)}}
        cmd += ['--kv-transfer-config', json.dumps(kv)]
        if role == 'kv_consumer' and a.decode_kv_bytes:
            cmd += ['--kv-cache-memory-bytes', str(a.decode_kv_bytes)]
    return cmd


def start_servers(a, cfg, out, uuids, idx):
    procs = []
    env0 = dict(os.environ, PYTHONPATH=f"{REPO / 'scripts'}:{REPO / 'scripts/amd/cp'}", CP_POLICY=a.policy, CP_P2P_CUMEM='0',
                VLLM_NO_USAGE_STATS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env0.pop('VLLM_ENABLE_V1_MULTIPROCESSING', None)
    if a.deployment == 'disagg':
        # P2pNcclConnector keys transferred tensors by request id; vLLM 0.23 appends a random suffix per engine, so the
        # decode worker waited forever for KV under a different id (dev run 2026-11-18). The client makes every request id
        # unique (uuid4 in X-Request-Id), so vLLM's documented switch is safe here.
        env0['VLLM_DISABLE_REQUEST_ID_RANDOMIZATION'] = '1'
        if a.p2p_none_blocks_fix:
            env0['CP_P2P_NONE_BLOCKS'] = '1'  # cp_sched._p2p_none_blocks (connector assertion on chunked prompts)
    env0.pop('HSA_CU_MASK', None)
    env0.pop('CUBW_CU_COUNT_OVERRIDE', None)
    n = len(idx)
    plan = {'single': [(0, None)], 'colocated': [(k, None) for k in range(n)],
            'disagg': [(k, 'kv_producer' if k < a.prefill_workers else 'kv_consumer') for k in range(n)]}[a.deployment]
    for k, role in plan:
        # disagg: all lane GPUs visible, own GPU first (vLLM uses device 0), so RCCL can open the peers for the KV transfer
        # (with only its own GPU visible, ncclCommInitRank failed with 'unhandled cuda error', dev run 2026-11-18)
        vis = [idx[k]] + [g for g in idx if g != idx[k]] if a.deployment == 'disagg' else [idx[k]]
        env = dict(env0, HIP_VISIBLE_DEVICES=','.join(map(str, vis)), CP_STEPLOG=str(out / f'steplog_w{k}.jsonl'))
        if a.nccl_debug:
            env['NCCL_DEBUG'] = 'INFO'
        cus, grid = (a.decode_cus, a.decode_grid) if role == 'kv_consumer' else (a.cus, a.grid)
        if cus:
            env['HSA_CU_MASK'] = f'{idx[k]}:{mask_range(cus)}'
        if grid:
            env['CUBW_CU_COUNT_OVERRIDE'] = str(grid)
        cmd = server_cmd(a, cfg, HTTP[k], role)
        log = open(out / f'server_w{k}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append({'worker': k, 'role': role or 'colocated', 'gpu_index': idx[k], 'uuid': uuids[k], 'pid': p.pid,
                      'proc': p, 'port': HTTP[k], 'cmd': cmd, 'hsa_cu_mask': env.get('HSA_CU_MASK'),
                      'cu_override': env.get('CUBW_CU_COUNT_OVERRIDE')})
    return procs


async def wait_ready(session, procs, timeout):
    t = time.time() + timeout
    for p in procs:
        while True:
            if p['proc'].poll() is not None:
                raise RuntimeError(f"server w{p['worker']} exited rc={p['proc'].returncode}")
            try:
                async with session.get(f"http://127.0.0.1:{p['port']}/health") as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.time() > t:
                raise RuntimeError(f"server w{p['worker']} not ready in {timeout} s")
            await asyncio.sleep(2)


async def stream(session, port, body, headers, rec, key):
    """POST a streaming completion; record the arrival time of every token chunk (perf_counter)."""
    times, toks = [], []
    async with session.post(f'http://127.0.0.1:{port}/v1/completions', json=body, headers=headers) as r:
        if r.status != 200:
            rec[key + '_error'] = f'HTTP {r.status}: {(await r.text())[:300]}'
            return times, toks
        async for line in r.content:
            line = line.strip()
            if not line.startswith(b'data:'):
                continue
            d = line[5:].strip()
            if d == b'[DONE]':
                break
            j = json.loads(d)
            ch = j.get('choices') or []
            if ch and ch[0].get('token_ids') is not None:
                n = len(ch[0]['token_ids'])
                toks += ch[0]['token_ids']
            elif ch and ch[0].get('text') is not None:
                n = 1
            else:
                continue
            t = time.perf_counter()
            times += [t] * n
    return times, toks


async def one(session, a, q, t0, route, rec, out_load):
    body = {'model': 'm', 'prompt': q['prompt_ids'], 'max_tokens': q['max_tokens'], 'temperature': 0, 'ignore_eos': True,
            'stream': True, 'return_token_ids': True, 'detokenize': False, 'seed': 20261013}
    await asyncio.sleep(max(0.0, t0 + q['arrival_s'] - time.perf_counter()))
    rec['send_s'] = time.perf_counter() - t0
    try:
        if a.deployment == 'disagg':
            i, j = route(q)
            rec['route'] = f'p{i}->d{j}'
            rid = f"___prefill_addr_{a.kv_ip}:{KV_PORT[i]}___decode_addr_{a.kv_ip}:{KV_PORT[j]}_{uuid.uuid4().hex}"
            hdr = {'X-Request-Id': rid}
            pb = dict(body, max_tokens=1, stream=False)
            P, O = len(q['prompt_ids']), q['max_tokens']
            out_load[i] += P
            out_load[j] += O
            try:
                tp = time.perf_counter()
                try:
                    async with session.post(f'http://127.0.0.1:{HTTP[i]}/v1/completions', json=pb, headers=hdr) as r:
                        txt = await r.text()
                        if r.status != 200:
                            rec['prefill_error'] = f'HTTP {r.status}: {txt[:300]}'
                finally:
                    out_load[i] -= P
                rec['prefill_done_s'] = time.perf_counter() - t0
                rec['prefill_s'] = time.perf_counter() - tp
                times, toks = await stream(session, HTTP[j], body, hdr, rec, 'decode')
            finally:
                out_load[j] -= O
        else:
            k = route(q)
            rec['route'] = f'w{k}'
            out_load[k] += len(q['prompt_ids']) + q['max_tokens']
            try:
                times, toks = await stream(session, HTTP[k], body, {}, rec, 'serve')
            finally:
                out_load[k] -= len(q['prompt_ids']) + q['max_tokens']
    except Exception as e:  # noqa: BLE001
        rec['error'] = repr(e)[:300]
        times, toks = [], []
    rec['token_s'] = [round(t - t0, 6) for t in times]
    rec['output_sha256'] = C.digest([toks]) if toks else None
    rec['tokens'] = len(times)


async def poll_metrics(session, procs, t0, stop, sink):
    names = (b'vllm:num_requests_running', b'vllm:num_requests_waiting', b'vllm:kv_cache_usage_perc',
             b'vllm:num_preemptions_total')
    while not stop.is_set():
        row = {'t': time.perf_counter() - t0}
        for p in procs:
            try:
                async with session.get(f"http://127.0.0.1:{p['port']}/metrics") as r:
                    body = await r.read()
                for line in body.split(b'\n'):
                    for nm in names:
                        if line.startswith(nm):
                            row[f"w{p['worker']}_{nm.decode().split(':')[1]}"] = float(line.rsplit(b' ', 1)[1])
            except Exception:  # noqa: BLE001
                pass
        sink.append(row)
        try:
            await asyncio.wait_for(stop.wait(), 0.25)
        except asyncio.TimeoutError:
            pass


async def run_trace(session, a, reqs, procs, deadline_s):
    n = len(procs)
    out_load = [0] * n
    npf = a.prefill_workers

    def route(q):
        if a.deployment == 'disagg':  # (producer, consumer)
            return (min(range(npf), key=lambda k: (out_load[k], k)), min(range(npf, n), key=lambda k: (out_load[k], k)))
        return min(range(n), key=lambda k: (out_load[k], k)) if a.deployment == 'colocated' else 0
    recs = [{'id': q['id'], 'arrival_s': q['arrival_s'], 'prompt_len': len(q['prompt_ids']), 'max_tokens': q['max_tokens'],
             'tokens': 0}
            for q in reqs]
    stop, metrics = asyncio.Event(), []
    t0 = time.perf_counter()
    mon = asyncio.create_task(poll_metrics(session, procs, t0, stop, metrics))
    tasks = [asyncio.create_task(one(session, a, q, t0, route, r, out_load)) for q, r in zip(reqs, recs)]
    done, pending = await asyncio.wait(tasks, timeout=reqs[-1]['arrival_s'] + deadline_s)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    stop.set()
    await mon
    span = time.perf_counter() - t0
    return recs, metrics, span, len(pending)


def main(a):
    import aiohttp
    cfg = json.loads(Path(a.config).read_text())
    C.setup_env()
    os.environ.pop('VLLM_ENABLE_V1_MULTIPROCESSING', None)
    uuids, idx = a.gpu_uuids.split(','), [int(x) for x in a.gpu_indices.split(',')]
    from vllm.utils.network_utils import get_ip
    off = 10 * idx[0]
    HTTP[:] = [18101 + k + off for k in range(len(idx))]
    KV_PORT[:] = [21101 + 1000 * k + off for k in range(len(idx))]
    if a.deployment == 'disagg' and not 1 <= a.prefill_workers < len(idx):
        raise SystemExit('disagg needs 1 <= --prefill-workers < number of GPUs')
    a.kv_ip = get_ip()  # the P2pNccl engine binds its ZMQ router to this address (kv_ip is not used for binding)
    from gpu_target import load_nvml
    nvml = load_nvml()
    nvml.nvmlInit()
    handles = [nvml.nvmlDeviceGetHandleByUUID(u) for u in uuids]
    layout = C.model_layout(cfg)
    topo = {}
    try:
        topo['amd_smi_topology'] = subprocess.run(['amd-smi', 'topology'], capture_output=True, text=True, timeout=60).stdout[-4000:]
    except Exception as e:  # noqa: BLE001
        topo['error'] = repr(e)
    data = {'status': 'running', 'classification': 'measured-real-GPU (two-GPU open-loop serving)', 'scope': __doc__.strip(),
            'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, 'config': cfg, 'layout': layout,
            'gpus': [{'uuid': u, 'index': i} for u, i in zip(uuids, idx)], 'topology': topo,
            'script_sha256': C.sha_file(__file__), 'sched_sha256': C.sha_file(Path(__file__).parent / 'cp_sched.py'),
            'trace_sha256': C.sha_file(Path(__file__).parent / 'cp_trace.py'), 'traces': []}
    C.dump(a.output / 'run.json', data)
    procs = start_servers(a, cfg, a.output, uuids, idx)
    data['servers'] = [{k: v for k, v in p.items() if k != 'proc'} for p in procs]
    C.dump(a.output / 'run.json', data)

    async def go():
        timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
        async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
            await wait_ready(session, procs, a.ready_s)
            data['ready_unix'] = time.time()
            wr, _ = T.make('balanced', 'poisson', 4.0, 12, 999, layout['vocab'])
            for q in wr:
                q['max_tokens'] = 16
            await run_trace(session, a, wr, procs, 600)  # warmup, not scored
            k = 0
            for seed in a.seeds:
                for arrival in a.arrivals:
                    for rate in a.rates:
                        reqs, h = T.make(a.workload, arrival, rate, a.requests, seed, layout['vocab'])
                        tag = f'{a.workload}_{arrival}_r{rate}_s{seed}'
                        sens = [C.Sensor(nvml, hd, 0.05) for hd in handles]
                        b = []
                        for s in sens:
                            s.start()
                            b.append(s.read())
                        tt0 = time.perf_counter()
                        recs, metrics, span, unfinished = await run_trace(session, a, reqs, procs, a.drain_s)
                        tt1 = time.perf_counter()
                        tel = []
                        for s, b0 in zip(sens, b):
                            t_ = s.finish(tt0, tt1, b0, s.read(), max(1, sum(r['tokens'] for r in recs)), 5)
                            tel.append({k_: v for k_, v in t_.items() if k_ != 'samples'})
                        res = {'tag': tag, 'deployment': a.deployment, 'workload': a.workload, 'arrival': arrival, 'rate': rate,
                               'seed': seed, 'trace_sha256': h, 'content_sha256': T.content_digest(reqs),
                               'demand': T.demand(reqs), 'span_s': span, 'unfinished': unfinished, 'requests': recs,
                               'metrics': metrics, 'telemetry': tel,
                               'power_samples': [[{k_: x.get(k_) for k_ in ('t', 'power_w', 'energy_mj', 'sm_clock_mhz')}
                                                  for x in s.samples] for s in sens]}
                        f = f'trace_{k:02d}_{tag}.json'
                        (a.output / f).write_text(json.dumps(res, separators=(',', ':')) + '\n')
                        ok = sum(r['tokens'] == r['max_tokens'] for r in recs)
                        data['traces'].append({'file': f, 'tag': tag, 'complete': ok, 'requests': len(recs), 'span_s': span})
                        C.dump(a.output / 'run.json', data)
                        print(f'{tag}: {ok}/{len(recs)} complete, span {span:.1f} s', flush=True)
                        k += 1
    try:
        asyncio.run(go())
        data['status'] = 'complete'
    finally:
        for p in procs:
            pr = p['proc']
            if pr.poll() is None:
                try:
                    os.killpg(pr.pid, 15)
                    pr.wait(60)
                except Exception:  # noqa: BLE001
                    os.killpg(pr.pid, 9)
        data['server_returncodes'] = [p['proc'].returncode for p in procs]
        C.dump(a.output / 'run.json', data)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ints = lambda v: [int(x) for x in v.split(',') if x != '']  # noqa: E731
    floats = lambda v: [float(x) for x in v.split(',') if x != '']  # noqa: E731
    p.add_argument('--config', required=True)
    p.add_argument('--gpu-uuids', required=True)
    p.add_argument('--gpu-indices', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deployment', choices=('single', 'colocated', 'disagg'), required=True)
    p.add_argument('--policy', choices=('mixed', 'separate'), default='mixed')
    p.add_argument('--budget', type=int, default=2048)
    p.add_argument('--no-chunk', action='store_true', help='whole prompts only (chunked prefill off)')
    p.add_argument('--workload', choices=sorted(T.WORKLOADS), required=True)
    p.add_argument('--arrivals', type=lambda v: v.split(','), default=['poisson'])
    p.add_argument('--rates', type=floats, required=True)
    p.add_argument('--seeds', type=ints, default=[T.HELDOUT_SEEDS[0]])
    p.add_argument('--requests', type=int, default=200)
    p.add_argument('--drain-s', type=float, default=60.0)
    p.add_argument('--max-seqs', type=int, default=64)
    p.add_argument('--decode-max-seqs', type=int, default=None, help='disagg: max_num_seqs of the decode worker only')
    p.add_argument('--prefill-workers', type=int, default=1, help='disagg: producers on the first P GPUs, consumers on the rest')
    p.add_argument('--decode-kv-bytes', type=int, default=None, help='disagg: --kv-cache-memory-bytes of every consumer')
    p.add_argument('--p2p-none-blocks-fix', action='store_true', help='disagg: cp_sched._p2p_none_blocks connector fix')
    p.add_argument('--max-model-len', type=int, default=16384)
    p.add_argument('--gpu-mem', type=float, default=0.85)
    p.add_argument('--cus', type=int, default=None, help='CU mask 0..n-1 for colocated/single workers')
    p.add_argument('--grid', type=int, default=None)
    p.add_argument('--decode-cus', type=int, default=None)
    p.add_argument('--decode-grid', type=int, default=None)
    p.add_argument('--ready-s', type=float, default=1200)
    p.add_argument('--nccl-debug', action='store_true')
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
