#!/usr/bin/env python3
"""Measure BF16 GEMM and useful-byte device-copy ceilings on one allocated GPU."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time


def measure(torch, work, repetitions, warmups, trials):
    for _ in range(warmups):
        for _ in range(repetitions):
            work()
        torch.cuda.synchronize()
    values = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter()
        for _ in range(repetitions):
            work()
        stop.record()
        torch.cuda.synchronize()
        values.append({'gpu_event_ms': start.elapsed_time(stop),
                       'synchronized_wall_ms': (time.perf_counter() - wall_start) * 1000})
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gemm-dimension', type=int, default=8192)
    p.add_argument('--gemm-repetitions', type=int, default=16)
    p.add_argument('--copy-bytes', type=int, default=1073741824)
    p.add_argument('--copy-repetitions', type=int, default=64)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--trials', type=int, default=3)
    a = p.parse_args()
    assert a.warmups >= 3 and a.trials >= 3
    assert a.gemm_dimension > 0 and a.gemm_repetitions > 0 and a.copy_repetitions > 0
    import pynvml
    pynvml.nvmlInit()
    from gpu_target import select, verify_cuda, describe
    handle, _ = select(pynvml, a.gpu_uuid)
    import torch
    props = verify_cuda(torch, a.gpu_uuid)
    l2_bytes = getattr(props, 'L2_cache_size', 0)
    assert a.copy_bytes > 2 * l2_bytes, 'copy buffer must exceed twice queried L2'
    assert a.copy_bytes % 2 == 0
    hardware = describe(pynvml, handle, torch)
    n = a.gemm_dimension
    left = torch.randn((n, n), device='cuda', dtype=torch.bfloat16)
    right = torch.randn((n, n), device='cuda', dtype=torch.bfloat16)
    output = torch.empty((n, n), device='cuda', dtype=torch.bfloat16)
    gemm = measure(torch, lambda: torch.mm(left, right, out=output),
                   a.gemm_repetitions, a.warmups, a.trials)
    assert torch.isfinite(output).all().item()
    gemm_flops = 2 * n**3 * a.gemm_repetitions
    for trial in gemm:
        trial['bf16_flops'] = gemm_flops
        trial['achieved_tflops'] = gemm_flops / (trial['gpu_event_ms'] / 1000) / 1e12
    del left, right, output
    torch.cuda.empty_cache()
    length = a.copy_bytes // 2
    source = torch.empty(length, device='cuda', dtype=torch.bfloat16)
    target = torch.empty_like(source)
    source.fill_(1)
    copies = measure(torch, lambda: target.copy_(source),
                     a.copy_repetitions, a.warmups, a.trials)
    assert target[0].item() == 1 and target[-1].item() == 1
    useful_bytes = 2 * a.copy_bytes * a.copy_repetitions
    for trial in copies:
        trial['useful_bytes'] = useful_bytes
        trial['useful_gb_per_s'] = useful_bytes / (trial['gpu_event_ms'] / 1000) / 1e9
    result = {'classification': 'measured-real-GPU-control',
              'note': 'GEMM operations and copy useful bytes are control ceilings; copy traffic is not measured HBM traffic and neither is an achieved decode roofline point.',
              'gpu_uuid': a.gpu_uuid, 'hardware': hardware,
              'versions': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda},
              'environment': {'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'CUDA_HOME': os.environ.get('CUDA_HOME', 'unset')},
              'warmups': a.warmups, 'trials': a.trials,
              'gemm': {'dimension': n, 'dtype': 'bfloat16', 'repetitions': a.gemm_repetitions,
                       'trials': gemm, 'median_tflops': statistics.median(x['achieved_tflops'] for x in gemm)},
              'copy': {'buffer_bytes': a.copy_bytes, 'repetitions': a.copy_repetitions,
                       'trials': copies, 'median_useful_gb_per_s': statistics.median(x['useful_gb_per_s'] for x in copies),
                       'dram_bytes_measured': 'NA'}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    pynvml.nvmlShutdown()
    print(f"BF16 GEMM {result['gemm']['median_tflops']:.2f} TFLOP/s; device copy {result['copy']['median_useful_gb_per_s']:.2f} useful GB/s")


if __name__ == '__main__':
    main()
