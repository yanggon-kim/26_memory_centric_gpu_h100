#!/usr/bin/env python3
"""M5 cache-history pressure on the HF StaticCache decode path (one acquisition per process).

Treatments run interleaved per trial in one process: 'normal' and 'pressure'. Pressure reads
a buffer larger than L2 + Infinity Cache after each decode step completes and before the next
step's timing events, so it evicts cross-token cache contents without being inside any timed
interval. It is a cache-history intervention, not a guaranteed flush and not an emulation of
added SRAM. The decode step uses the package's compiled (CUDA/HIP graph) StaticCache path.
Per-step HIP-event times, output IDs and the pressure cost are recorded separately.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(args):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    import transformers
    from gpu_target import load_nvml, select, verify_cuda
    from software_decode_experiments import make_decode, digest
    nvml = load_nvml(); nvml.nvmlInit()
    handle, physical = select(nvml, args.gpu_uuid)
    props = verify_cuda(torch, args.gpu_uuid)
    config = json.loads(Path(args.config).read_text())
    path = Path(config['model_path'])
    torch.manual_seed(config['seed'])
    model = transformers.AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16, local_files_only=True,
                                                              attn_implementation='sdpa').eval().to('cuda')
    generator = torch.Generator().manual_seed(config['seed'])
    prompt = torch.randint(0, model.config.vocab_size, (args.batch, args.context), generator=generator).to('cuda')
    maximum = args.context + args.tokens
    pressure = torch.ones(args.pressure_bytes // 2, dtype=torch.bfloat16, device='cuda')
    sink = torch.zeros((), dtype=torch.float32, device='cuda')
    record = {'classification': 'measured-real-GPU', 'experiment': 'M5 cross-token cache-history pressure',
              'gpu_uuid': args.gpu_uuid, 'name': props.name, 'batch': args.batch, 'context': args.context,
              'generated_tokens': args.tokens, 'pressure_bytes': args.pressure_bytes, 'acquisition': args.acquisition,
              'model_revision': config['model_revision'], 'seed': config['seed'], 'compiled': True,
              'placement': 'pressure read after step k synchronizes and before step k+1 start event; outside timed steps',
              'trials': []}
    with torch.inference_mode():
        cache = transformers.StaticCache(config=model.config, max_cache_len=maximum)
        decode = make_decode(model, cache, maximum, torch, True)

        def trial(treatment):
            cache.reset()
            pre = model(input_ids=prompt, past_key_values=cache, cache_position=torch.arange(prompt.shape[1], device='cuda'),
                        use_cache=True, logits_to_keep=1)
            current = pre.logits[:, -1:].argmax(-1)
            ids = [current.clone()]
            position = torch.empty((1,), dtype=torch.long, device='cuda')
            step_ms, pressure_ms = [], []
            torch.cuda.synchronize()
            for index in range(args.tokens - 1):
                if treatment == 'pressure':
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); sink.add_(pressure.float().sum() * 0); b.record(); torch.cuda.synchronize()
                    pressure_ms.append(a.elapsed_time(b))
                position.fill_(prompt.shape[1] + index)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.compiler.cudagraph_mark_step_begin()
                s.record()
                nxt = decode(current, position)
                e.record()
                torch.cuda.synchronize()
                step_ms.append(s.elapsed_time(e))
                current.copy_(nxt)
                ids.append(current.clone())
            out = torch.cat(ids, dim=1).cpu().tolist()
            return {'treatment': treatment, 'step_ms': step_ms, 'pressure_ms': pressure_ms,
                    'output_sha256': digest(out), 'start_unix_ns': None}

        for w in range(args.warmups):
            for t in ('normal', 'pressure'):
                trial(t)
        order = []
        for i in range(args.trials):
            pair = ('normal', 'pressure') if (i + args.acquisition) % 2 == 0 else ('pressure', 'normal')
            for t in pair:
                t0 = time.time_ns()
                r = trial(t)
                r.update(start_unix_ns=t0, end_unix_ns=time.time_ns(), index=i)
                record['trials'].append(r)
                order.append(t)
    record['treatment_order'] = order
    record['outputs_identical'] = len({t['output_sha256'] for t in record['trials']}) == 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record) + '\n')
    nvml.nvmlShutdown()
    assert record['outputs_identical'], 'pressure changed output IDs'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--context', type=int, default=1024)
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--warmups', type=int, default=3)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--pressure-bytes', type=int, default=1 << 30)
    p.add_argument('--acquisition', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
