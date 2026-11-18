#!/usr/bin/env python3
"""Chunked-prefill campaign, experiment 5 support: KV-sized transfers between two allocated GPUs on the available link.

Two transports, both real data movement between the two boards:
  copy   torch peer copy (dst.copy_(src), non_blocking, device-to-device; HIP routes it over xGMI where available)
  rccl   torch.distributed send/recv with the RCCL backend, two processes (one per GPU), the library vLLM's
         P2pNcclConnector uses
Sizes = prompt tokens x kv_bytes_per_token of the model (one contiguous buffer; the connector sends per layer, so a
per-layer variant splits the same bytes into `layers` messages). Timed with device synchronization, warmup 3, reps 20;
reports median and p90 seconds and GB/s.

  cp_link.py --config cfg.json --gpu-indices 0,1 --output DIR [--tokens 256,1024,4096,8192]
"""
import argparse
import json
import os
from pathlib import Path
import statistics as st
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cp_common as C  # noqa: E402


def timeit(fn, sync, reps=20, warm=3):
    for _ in range(warm):
        fn()
    sync()
    xs = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        xs.append(time.perf_counter() - t)
    xs.sort()
    return {'median_s': st.median(xs), 'p90_s': xs[int(0.9 * (len(xs) - 1))], 'min_s': xs[0]}


def rccl_worker(rank, nbytes_list, layers, port, q):
    import torch
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', rank=rank, world_size=2)
    res = {}
    for nb in nbytes_list:
        for split in (1, layers):
            per = nb // split
            bufs = [torch.empty(per, dtype=torch.uint8, device=f'cuda:{rank}') for _ in range(split)]

            def f():
                for b in bufs:
                    (dist.send if rank == 0 else dist.recv)(b, 1 - rank)
            r = timeit(f, torch.cuda.synchronize)
            dist.barrier()
            res[f'{nb}:{split}'] = r
    if rank == 1:
        q.put(res)
    dist.destroy_process_group()


def main(a):
    cfg = json.loads(Path(a.config).read_text())
    lay = C.model_layout(cfg)
    import torch
    import torch.multiprocessing as mp
    out = {'classification': 'measured-real-GPU (link microbenchmark)', 'scope': __doc__.strip(), 'layout': lay,
           'gpu_indices': a.gpu_indices, 'visible': os.environ.get('HIP_VISIBLE_DEVICES'), 'rows': []}
    sizes = [(t, t * lay['kv_bytes_per_token']) for t in a.tokens]
    for t, nb in sizes:
        for split in (1, lay['layers']):
            per = nb // split
            src = [torch.empty(per, dtype=torch.uint8, device='cuda:0') for _ in range(split)]
            dst = [torch.empty(per, dtype=torch.uint8, device='cuda:1') for _ in range(split)]

            def f():
                for s, d in zip(src, dst):
                    d.copy_(s, non_blocking=True)

            def sync():
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
            r = timeit(f, sync)
            out['rows'].append({'transport': 'copy', 'tokens': t, 'bytes': nb, 'messages': split, **r, 'GB_s': nb / r['median_s'] / 1e9})
            del src, dst
    torch.cuda.empty_cache()
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    ps = [ctx.Process(target=rccl_worker, args=(r, [nb for _, nb in sizes], lay['layers'], 29531 + int(os.environ.get("HIP_VISIBLE_DEVICES", "0").split(",")[0]), q)) for r in range(2)]
    for p in ps:
        p.start()
    res = q.get(timeout=900)
    for p in ps:
        p.join(120)
    for t, nb in sizes:
        for split in (1, lay['layers']):
            r = res[f'{nb}:{split}']
            out['rows'].append({'transport': 'rccl', 'tokens': t, 'bytes': nb, 'messages': split, **r, 'GB_s': nb / r['median_s'] / 1e9})
    out['status'] = 'complete'
    C.dump(a.output / 'run.json', out)
    for r in out['rows']:
        print(r['transport'], r['tokens'], r['messages'], f"{r['median_s'] * 1e3:.3f} ms", f"{r['GB_s']:.1f} GB/s")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--gpu-indices', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=lambda v: [int(x) for x in v.split(',')], default=[256, 1024, 4096, 8192])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    main(a)
