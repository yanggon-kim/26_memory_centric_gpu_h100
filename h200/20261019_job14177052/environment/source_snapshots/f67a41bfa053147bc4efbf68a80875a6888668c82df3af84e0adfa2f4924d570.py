#!/usr/bin/env python3
"""Bounded PyTorch kernel inventory for a selected H100 eager decode route."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()
    assert 1 <= args.steps <= 3
    assert not args.output.exists(), 'output directory must be fresh'
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    assert config['precision'] == 'BF16' and config['warmup_trials'] >= 3
    assert config['stage'] in ('smoke', 'baseline')
    import os
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import pynvml as nvml
    from gpu_target import select, verify_cuda
    nvml.nvmlInit()
    handle, _ = select(nvml, args.gpu_uuid)
    import torch
    import transformers
    from software_decode_experiments import controlled_trial, digest, make_decode
    props = verify_cuda(torch, args.gpu_uuid)
    path = Path(config['model_path'])
    assert path.name == config['model_revision']
    torch.manual_seed(config['seed'])
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa'
    ).eval().to('cuda')
    batch = context = 1
    context = 128
    maximum = context + config['generated_tokens']
    prompt = torch.randint(0, model.config.vocab_size, (batch, context),
                           generator=torch.Generator().manual_seed(config['seed'])).to('cuda')
    cache = transformers.StaticCache(config=model.config, max_cache_len=maximum)
    decode = make_decode(model, cache, maximum, torch, False)
    try:
        with torch.inference_mode():
            warmups = [controlled_trial(model, cache, decode, prompt,
                                       config['generated_tokens'], torch)['output_sha256']
                       for _ in range(config['warmup_trials'])]
            assert len(set(warmups)) == 1, 'warmup outputs differ'
            cache.reset()
            prefill = model(input_ids=prompt, past_key_values=cache,
                            cache_position=torch.arange(context, device='cuda'),
                            use_cache=True, logits_to_keep=1)
            current = prefill.logits[:, -1:].argmax(-1)
            first_ids = current.reshape(-1).cpu().tolist()
            del prefill
            assert cache.get_seq_length().item() == context
            torch.cuda.synchronize()
            positions = torch.empty((1,), dtype=torch.long, device='cuda')
            output_ids = [first_ids]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=False, with_stack=False) as profile:
                for index in range(args.steps):
                    positions.fill_(context + index)
                    current = decode(current, positions)
                    output_ids.append(current.reshape(-1).cpu().tolist())
                torch.cuda.synchronize()
            events = [event for event in profile.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA]
            assert events, 'profiler produced no CUDA kernel events'
            groups = {}
            for event in events:
                row = groups.setdefault(event.name, {'kernel': event.name, 'launches': 0,
                                                     'instrumented_device_time_us': 0.0})
                row['launches'] += 1
                row['instrumented_device_time_us'] += event.self_device_time_total
            inventory = sorted(groups.values(), key=lambda row: -row['instrumented_device_time_us'])
            result = {
                'classification': 'measured-real-GPU-instrumented',
                'scope': 'PyTorch profiler, eager HF StaticCache, post-prefill decode only; not uninstrumented latency or hardware counters',
                'gpu_uuid': args.gpu_uuid, 'gpu_name': props.name,
                'model_id': config['model_id'], 'model_revision': config['model_revision'],
                'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
                'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                'batch_size': batch, 'context_length': context, 'seed': config['seed'],
                'warmup_trials': config['warmup_trials'], 'warmup_output_sha256': warmups,
                'prefill_first_selected_ids': first_ids, 'decode_steps': args.steps,
                'selected_token_ids': output_ids, 'selected_token_sha256': digest(output_ids),
                'cuda_kernel_events': len(events), 'kernel_inventory': inventory,
                'counter_status': 'NA: Nsight Compute ERR_NVGPUCTRPERM'
            }
            (args.output/'kernel_inventory.json').write_text(json.dumps(result, indent=2)+'\n')
            profile.export_chrome_trace(str(args.output/'decode_timeline.json'))
            print(json.dumps({'kernel_events': len(events), 'unique_kernels': len(inventory)}))
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps({
            'type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()
        }, indent=2)+'\n')
        raise
    finally:
        nvml.nvmlShutdown()


if __name__ == '__main__':
    main()
