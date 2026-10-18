#!/usr/bin/env python3
"""Tensor-parallel (TP4/TP8) vLLM decode on declared MI300X GPUs: the owner-authorized multi-GPU extension.

Reuses the TP1 runner's trial loop (prefill barrier, N-1 decode ticks, exact output IDs) with
vLLM's multiprocess executor. Workers run in separate processes, so the driver's HIP events
do not bracket GPU work: synchronized wall time per trial is the primary decode timing here.
Energy is derived afterwards from the node monitor (sum over participating GPUs). Every
participating GPU is locked and checked for foreign use; TP1 and TP-N results stay separately labelled.
"""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from software_decode_experiments import digest, dump


def main(args):
    config = json.loads(Path(args.config).read_text())
    uuids = args.gpu_uuids.split(',')
    assert len(uuids) == args.tp and args.tp in (2, 4, 8)
    assert config['warmup_trials'] >= 3 and config['trials'] >= 3
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0')
    import gpu_target
    nvml = gpu_target.load_nvml(); nvml.nvmlInit()
    selected, physical = [], []
    for u in uuids:
        handle, idx = gpu_target.select(nvml, u)  # lock + foreign-use check per GPU
        selected.append({'uuid': u, 'physical_index': idx, 'pci': nvml.nvmlDeviceGetPciInfo(handle).busId,
                         'observed_processes': list(gpu_target.OBSERVED_PROCESSES)})
        physical.append(idx)
    visible = ','.join(str(i) for i in sorted(physical))
    os.environ['HIP_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'] = visible
    import torch
    assert torch.cuda.device_count() == args.tp, 'visible device count differs from TP size'
    buses = sorted(int(s['pci'].split(':')[1], 16) for s in selected)
    assert sorted(torch.cuda.get_device_properties(i).pci_bus_id for i in range(args.tp)) == buses, 'HIP/amdsmi PCI mismatch'
    import vllm
    from vllm.sampling_params import RequestOutputKind
    from vllm_identity import verify as verify_vllm
    from controlled_vllm_experiments import run_trial, cleanup_owned_engine
    path = Path(config['model_path'])
    settings = {'model': str(path), 'dtype': 'bfloat16', 'seed': config['seed'], 'generation_config': 'vllm',
                'enable_prefix_caching': False, 'enable_chunked_prefill': True, 'disable_log_stats': False,
                'async_scheduling': False, 'distributed_executor_backend': 'mp', 'tensor_parallel_size': args.tp,
                'max_model_len': 8192, 'max_num_seqs': 8, 'max_num_batched_tokens': 32768,
                'gpu_memory_utilization': 0.70, 'enforce_eager': False}
    settings.update({k: v for k, v in config.get('engine', {}).items()
                     if k in ('max_model_len', 'max_num_seqs', 'max_num_batched_tokens', 'gpu_memory_utilization', 'enforce_eager')})
    data = {'status': 'running', 'classification': 'measured-real-GPU', 'extension': 'owner-authorized TP multi-GPU (2026-10-18)',
            'mode': 'timing', 'config': config, 'gpus': selected, 'hip_visible_devices': visible, 'engine_settings': settings,
            'versions': {'vllm': vllm.__version__, 'torch': torch.__version__}, 'vllm_identity': verify_vllm(vllm),
            'timing_scope': 'synchronized wall per decode interval is primary; driver HIP events do not bracket worker GPU work',
            'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parent.glob('*.py')},
            'cases': []}
    dump(args.output / 'run.json', data)
    llm = vllm.LLM(**settings)
    engine = llm.llm_engine
    cleanup = lambda: cleanup_owned_engine(engine, args.output)
    atexit.register(cleanup)
    data['engine_client_class'] = type(engine.engine_core).__name__
    count = args.generated_tokens
    sampling = vllm.SamplingParams(temperature=0, repetition_penalty=1.0, ignore_eos=True, detokenize=False,
                                   max_tokens=count, seed=config['seed'], output_kind=RequestOutputKind.CUMULATIVE)
    vocabulary = json.loads((path / 'config.json').read_text())['vocab_size']
    batch, context = config['batch_sizes'][0], config['context_lengths'][0]
    generator = torch.Generator().manual_seed(config['seed'])
    prompts = torch.randint(0, vocabulary, (batch, context), generator=generator).tolist()
    case = {'batch_size': batch, 'context_length': context, 'generated_tokens': count, 'prompt_sha256': digest(prompts),
            'trials': [], 'warmup_output_sha256': []}
    data['cases'].append(case)
    for i in range(config['warmup_trials']):
        r = run_trial(engine, prompts, sampling, f'B{batch}C{context}_warmup{i}', torch, args.output)
        case['warmup_output_sha256'].append(r['output_sha256'])
        case['warmup_output_ids'] = r['output_ids']
    for i in range(config['trials']):
        r = run_trial(engine, prompts, sampling, f'B{batch}C{context}_measured{i}', torch, args.output)
        r['ms_per_step_wall'] = r['wall_ms'] / r['decode_iterations']
        case['trials'].append(r)
        dump(args.output / 'run.json', data)
    digests = set(case['warmup_output_sha256']) | {t['output_sha256'] for t in case['trials']}
    case['output_digests_identical'] = len(digests) == 1
    case['output_sha256'] = sorted(digests)
    assert all(t['decode_iterations'] == count - 1 and t['prefill_barrier']['passed'] for t in case['trials'])
    data['status'] = 'complete' if case['output_digests_identical'] else 'outputs_differ'
    dump(args.output / 'run.json', data)
    print(f"TP{args.tp} B={batch} C={context}: {[round(t['ms_per_step_wall'], 4) for t in case['trials']]} ms/step (wall)", flush=True)
    cleanup(); atexit.unregister(cleanup)
    assert case['output_digests_identical'], 'same-shape greedy outputs differ across warmups/trials'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--gpu-uuids', required=True, help='comma-separated declared GPU IDs')
    p.add_argument('--tp', type=int, required=True)
    p.add_argument('--generated-tokens', type=int, default=64)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
