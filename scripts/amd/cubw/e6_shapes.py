#!/usr/bin/env python3
"""E6 kernel-shape controls (runs inside the vLLM container, one GPU, under the step's HSA_CU_MASK).

The graph-mode kernel traces of jobs 487341/489594 (scripts/amd/cubw/e6_kernels.py) put ~84% of Qwen32B B1
decode kernel time in vLLM's ROCm skinny GEMM `wvSplitK`, which vLLM sizes with the device CU count
(num_compute_units() = 304) even when HSA_CU_MASK exposes fewer CUs. This benchmark times the Qwen32B
weight shapes at batch 1 with supported kernels:

  wvsplitk_dev    ops.wvSplitK(weight, x, device CU count)   what vLLM decode launches
  wvsplitk_n      ops.wvSplitK(weight, x, active CU count)   the same kernel sized for the mask
  linear          torch.nn.functional.linear (hipBLASLt/rocBLAS)
  llmm1           ops.LLMM1(weight, x, 4) where vLLM allows it (k <= 8192)
  wvsplitk_c<C>   ops.wvSplitK(weight, x, C) for each C of --cu-counts (v1.5: e.g. the largest count <= the
                  active CUs that leaves every shader engine the same number of workgroups)

Weights are rotated over copies totalling >= 1 GiB so every call streams from HBM (MALL is 256 MiB).
Time is HIP-event time per call; GB/s is weight bytes per call / time (useful bytes, not physical traffic).
Outputs are compared with linear (max |diff| / max |ref|); a kernel-shape benchmark is a control, not a
substitute for full-model latency.

  e6_shapes.py --n-active 200 --out e6.json [--trials 5 --reps 50] [--light] [--cu-counts 192,160]
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

MODEL = '/work1/proj_blaisetine/yanggon/hf/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'


def shapes():
    c = json.loads((Path(MODEL) / 'config.json').read_text())
    h, i, nh, kv = c['hidden_size'], c['intermediate_size'], c['num_attention_heads'], c['num_key_value_heads']
    hd = h // nh
    return {'qkv': (h + 2 * kv * hd, h), 'o_proj': (h, h), 'gate_up': (2 * i, h), 'down': (h, i),
            'lm_head': (c['vocab_size'], h)}


def main(a):
    from vllm import _custom_ops as ops
    dev = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(dev)
    torch.manual_seed(20261101)
    res = {'classification': 'measured-real-GPU (kernel-shape control)', 'scope': __doc__.strip(),
           'device': props.name, 'device_cus': props.multi_processor_count, 'n_active': a.n_active,
           'hsa_cu_mask': os.environ.get('HSA_CU_MASK', ''), 'hip_visible_devices': os.environ.get('HIP_VISIBLE_DEVICES', ''),
           'torch': torch.__version__, 'trials': a.trials, 'reps': a.reps, 'cu_counts': a.cu_counts, 'shapes': {}}
    for name, (m, k) in shapes().items():
        wbytes = m * k * 2
        copies = max(2, math.ceil((1 << 30) / wbytes))
        ws = [torch.randn(m, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        x = torch.randn(1, k, device=dev, dtype=torch.bfloat16)
        variants = {'wvsplitk_dev': lambda w: ops.wvSplitK(w, x, props.multi_processor_count),
                    'wvsplitk_n': lambda w: ops.wvSplitK(w, x, a.n_active),
                    'linear': lambda w: torch.nn.functional.linear(x, w)}
        for c in a.cu_counts:
            variants[f'wvsplitk_c{c}'] = lambda w, c=c: ops.wvSplitK(w, x, c)
        if k <= 8192 and m % 4 == 0:
            variants['llmm1'] = lambda w: ops.LLMM1(w, x, 4)
        ref = torch.nn.functional.linear(x.float(), ws[0].float())
        row = {'m': m, 'k': k, 'weight_bytes': wbytes, 'copies': copies, 'variants': {}}
        for vn, f in variants.items():
            v = {}
            try:
                out = f(ws[0])
                torch.cuda.synchronize()
                v['max_rel_diff_vs_fp32'] = float((out.float() - ref).abs().max() / ref.abs().max())
                reps = 3 if a.light else a.reps
                for j in range(10 if not a.light else 1):
                    f(ws[j % copies])
                torch.cuda.synchronize()
                us = []
                for t in range(1 if a.light else a.trials):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for j in range(reps):
                        f(ws[j % copies])
                    e1.record(); e1.synchronize()
                    us.append(e0.elapsed_time(e1) * 1e3 / reps)
                v['us_per_call'] = us
                v['median_us'] = sorted(us)[len(us) // 2]
                v['useful_gb_per_s'] = wbytes / v['median_us'] / 1e3
            except Exception as error:  # recorded per variant; other variants still run
                v['error'] = f'{type(error).__name__}: {str(error)[:300]}'
            row['variants'][vn] = v
        res['shapes'][name] = row
        del ws
        torch.cuda.empty_cache()
    res['t_end_unix'] = time.time()
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    for name, row in res['shapes'].items():
        print(name, row['m'], row['k'], {vn: round(v.get('median_us', -1), 2) for vn, v in row['variants'].items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-active', type=int, required=True, help='CUs exposed by the step mask (304 = unrestricted)')
    p.add_argument('--out', required=True)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--reps', type=int, default=50)
    p.add_argument('--light', action='store_true', help='few calls (under rocprofv3 counter collection)')
    p.add_argument('--cu-counts', type=lambda v: [int(x) for x in v.split(',') if x], default=[],
                   help='extra wvSplitK sizes to time under the same mask (comma list)')
    main(p.parse_args())
