#!/usr/bin/env python3
"""Short, repeatable HF decode workload for rocprofv3 / rocprof-compute runs.

Uses the package's own StaticCache prefill/decode path and seeded synthetic prompt, so profiled
kernels match the timing runner. Profiler replays rerun this whole script; output IDs are
written each run so replays can be checked for identical work. Profiled timings are not
baseline timings.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(args):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    import transformers
    from software_decode_experiments import controlled_trial, make_decode
    config = json.loads(Path(args.config).read_text())
    path = Path(config['model_path'])
    assert path.name == config['model_revision'] and path.is_dir()
    torch.manual_seed(config['seed'])
    model = transformers.AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='sdpa').eval().to('cuda')
    generator = torch.Generator().manual_seed(config['seed'])
    prompt = torch.randint(0, model.config.vocab_size, (args.batch, args.context), generator=generator).to('cuda')
    maximum = args.context + args.tokens
    with torch.inference_mode():
        cache = transformers.StaticCache(config=model.config, max_cache_len=maximum)
        decode = make_decode(model, cache, maximum, torch, False)
        for _ in range(args.warmups):
            controlled_trial(model, cache, decode, prompt, args.tokens, torch)
        result = controlled_trial(model, cache, decode, prompt, args.tokens, torch)
    record = {'classification': 'profiling-workload', 'batch': args.batch, 'context': args.context,
              'generated_tokens': args.tokens, 'warmups': args.warmups, 'compiled': False,
              'model_revision': config['model_revision'], 'seed': config['seed'],
              'output_ids': result['output_ids'],
              'scope': 'eager StaticCache decode under external profiler; timings perturbed, not baseline'}
    print(json.dumps(record))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('a') as handle:
            handle.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--tokens', type=int, default=8, help='first token from prefill plus tokens-1 decode steps')
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--output', type=Path, help='append one JSON line per run (profiler replays add lines)')
    main(p.parse_args())
