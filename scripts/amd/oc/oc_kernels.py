#!/usr/bin/env python3
"""OC-v0 projection-family kernel controls (inside the vLLM container, one GPU, under the step's HSA_CU_MASK).

Times vLLM's bf16 skinny GEMM (ops.wvSplitK, the decode B1 projection route on gfx942/gfx950) for the weight shapes of
the named models, at each wvSplitK grid of --grids ('dev' = device CU count, the stock launch), in two cache histories:

  stream    weights rotated over copies totalling >= 1 GiB (each call reads its weights cold from HBM, as in decode)
  same      one weight called repeatedly (MALL/L2 history differs; a history control, not a compute reference)

plus, with --overhead, a tiny wvSplitK (m=64, k=256) at each grid: per-call time of the same route with negligible work
(launch/dispatch/reduction floor O_ref), and, with --prefill N, torch.nn.functional.linear for an N-token activation
(hipBLASLt route, prefill reference; weights rotated as in 'stream').

Each (shape, history, grid) runs --warmups untimed calls, then --trials timed trials of --reps calls each (HIP events);
outputs are checked against an fp32 reference (max |diff| / max |ref|). Useful FLOPs 2*m*k*N and logical bytes
(weights + activation + output) are recorded; physical traffic needs rocprofv3 counters separately. A kernel control
is not full-model latency.

  oc_kernels.py --models qwen32b,qwen1p5b --grids dev,192 --out k.json [--overhead] [--prefill 4096] [--light]
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import time

import torch

HF = '/work1/proj_blaisetine/yanggon/hf/hub'
MODELS = {'qwen32b': 'Qwen--Qwen2.5-32B-Instruct', 'qwen14b': 'Qwen--Qwen2.5-14B-Instruct', 'qwen7b': 'Qwen--Qwen2.5-7B-Instruct',
          'qwen1p5b': 'Qwen--Qwen2.5-1.5B-Instruct', 'mistral7b': 'mistralai--Mistral-7B-Instruct-v0.3'}


def model_shapes(name):
    cfg = sorted(glob.glob(f'{HF}/models--{MODELS[name]}/snapshots/*/config.json'))[0]
    c = json.loads(Path(cfg).read_text())
    h, i, nh, kv = c['hidden_size'], c['intermediate_size'], c['num_attention_heads'], c['num_key_value_heads']
    hd = c.get('head_dim') or h // nh
    return {'qkv': (nh * hd + 2 * kv * hd, h), 'o_proj': (h, nh * hd), 'gate_up': (2 * i, h), 'down': (h, i),
            'lm_head': (c['vocab_size'], h)}, cfg


def timed(f, args_iter, warmups, trials, reps):
    for j in range(warmups):
        f(*args_iter(j))
    torch.cuda.synchronize()
    us = []
    for t in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for j in range(reps):
            f(*args_iter(j))
        e1.record(); e1.synchronize()
        us.append(e0.elapsed_time(e1) * 1e3 / reps)
    return us


def med(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2


def main(a):
    from vllm import _custom_ops as ops
    dev = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(dev)
    cus = props.multi_processor_count
    grids = [cus if g == 'dev' else int(g) for g in a.grids]
    torch.manual_seed(20261105)
    warm, trials, reps = (1, 1, 3) if a.light else (a.warmups, a.trials, a.reps)
    res = {'classification': 'measured-real-GPU (kernel-shape control)', 'scope': __doc__.strip(), 'device': props.name,
           'device_cus': cus, 'hsa_cu_mask': os.environ.get('HSA_CU_MASK', ''), 'torch': torch.__version__,
           'grids': grids, 'warmups': warm, 'trials': trials, 'reps': reps, 'light': a.light, 'rows': []}
    if a.overhead:
        w = torch.randn(64, 256, device=dev, dtype=torch.bfloat16)
        x = torch.randn(1, 256, device=dev, dtype=torch.bfloat16)
        for g in grids:
            us = timed(lambda: ops.wvSplitK(w, x, g), lambda j: (), warm, trials, max(reps, 200) if not a.light else reps)
            res['rows'].append({'model': None, 'shape': 'tiny', 'm': 64, 'k': 256, 'n_tokens': 1, 'route': 'wvSplitK',
                                'history': 'same', 'grid': g, 'us_per_call': us, 'median_us': med(us)})
    for model in a.models:
        shapes, cfg = model_shapes(model)
        res.setdefault('model_configs', {})[model] = cfg
        for name, (m, k) in shapes.items():
            wbytes = m * k * 2
            copies = max(2, math.ceil((1 << 30) / wbytes))
            ws = [torch.randn(m, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
            x = torch.randn(1, k, device=dev, dtype=torch.bfloat16)
            ref = torch.nn.functional.linear(x.float(), ws[0].float())
            for g in grids:
                base = {'model': model, 'shape': name, 'm': m, 'k': k, 'n_tokens': 1, 'route': 'wvSplitK', 'grid': g,
                        'weight_bytes': wbytes, 'logical_bytes': wbytes + 2 * k + 2 * m, 'useful_flops': 2 * m * k}
                try:
                    out = ops.wvSplitK(ws[0], x, g)
                    torch.cuda.synchronize()
                    err = float((out.float() - ref).abs().max() / ref.abs().max())
                    for hist in ('stream', 'same'):
                        it = (lambda j: (ws[j % copies],)) if hist == 'stream' else (lambda j: (ws[0],))
                        us = timed(lambda w: ops.wvSplitK(w, x, g), it, warm, trials, reps)
                        res['rows'].append(dict(base, history=hist, copies=copies if hist == 'stream' else 1,
                                                max_rel_diff_vs_fp32=err, us_per_call=us, median_us=med(us)))
                except Exception as error:  # recorded per cell; other cells still run
                    res['rows'].append(dict(base, error=f'{type(error).__name__}: {str(error)[:300]}'))
            if a.prefill:
                xp = torch.randn(a.prefill, k, device=dev, dtype=torch.bfloat16)
                try:
                    us = timed(lambda w: torch.nn.functional.linear(xp, w), lambda j: (ws[j % copies],), warm, trials,
                               max(3, reps // 10))
                    res['rows'].append({'model': model, 'shape': name, 'm': m, 'k': k, 'n_tokens': a.prefill, 'route': 'linear',
                                        'history': 'stream', 'grid': None, 'weight_bytes': wbytes,
                                        'logical_bytes': wbytes + 2 * k * a.prefill + 2 * m * a.prefill,
                                        'useful_flops': 2 * m * k * a.prefill, 'us_per_call': us, 'median_us': med(us)})
                except Exception as error:
                    res['rows'].append({'model': model, 'shape': name, 'route': 'linear', 'n_tokens': a.prefill,
                                        'error': f'{type(error).__name__}: {str(error)[:300]}'})
            del ws
            torch.cuda.empty_cache()
    res['t_end_unix'] = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    for r in res['rows']:
        print(r.get('model'), r['shape'], r.get('route'), r.get('history'), r.get('grid'), round(r.get('median_us', -1), 2), r.get('error', ''))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', type=lambda v: v.split(','), required=True)
    p.add_argument('--grids', type=lambda v: v.split(','), default=['dev'])
    p.add_argument('--out', required=True)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--reps', type=int, default=50)
    p.add_argument('--overhead', action='store_true')
    p.add_argument('--prefill', type=int, default=0)
    p.add_argument('--light', action='store_true', help='few calls (under rocprofv3 counter collection)')
    main(p.parse_args())
