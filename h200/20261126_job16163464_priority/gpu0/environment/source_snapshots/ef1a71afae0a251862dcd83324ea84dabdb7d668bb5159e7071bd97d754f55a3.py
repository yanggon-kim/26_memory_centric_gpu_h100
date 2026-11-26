#!/usr/bin/env python3
"""Controlled real-GPU software decode, board-energy and copy-bandwidth studies."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import threading
import time
import traceback


def digest(rows):
    tokens = [t for row in rows for t in row]
    return hashlib.sha256(struct.pack('<' + 'q' * len(tokens), *tokens)).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def validate(path):
    data = json.loads((Path(path) / 'run.json').read_text())
    assert data['status'] == 'complete' and data['classification'] == 'measured-real-GPU'
    assert data['hardware']['uuid'] and data['versions']['torch']
    if data['mode'] == 'bandwidth':
        assert data['buffer_bytes'] > 0
        if data['hardware']['l2_bytes'] != 'NA':
            assert data['buffer_bytes'] > 2*data['hardware']['l2_bytes'], 'buffer not comfortably above queried L2'
        for trial in data['trials']:
            assert trial['cuda_event_ms'] > 0
            expected = 2 * data['buffer_bytes'] * data['config']['copy_repetitions']
            assert trial['useful_bytes'] == expected
            assert math.isclose(trial['useful_gb_per_s'], expected / trial['cuda_event_ms'] / 1e6)
        assert data['dram_bytes_measured'] == 'NA'
    else:
        assert data['correctness']['passed'], 'compiled/static greedy correctness failed'
        assert data['config']['warmup_trials'] >= 3 and data['config']['trials'] >= 3
        for case in data['cases']:
            assert case['compiled_correctness']['passed'] and case['outputs_identical']
            assert len(case['trials']) == data['config']['trials']
            for trial in case['trials']:
                assert trial['timed_decode_steps'] == case['generated_tokens'] - 1
                assert trial['timed_generated_tokens'] == case['batch_size'] * trial['timed_decode_steps']
                assert len(trial['output_ids']) == case['batch_size']
                assert all(len(row) == case['generated_tokens'] for row in trial['output_ids'])
                assert digest(trial['output_ids']) == trial['output_sha256']
                assert trial['cache_length'] == case['context_length'] + trial['timed_decode_steps']
                assert all(math.isfinite(trial[k]) and trial[k] > 0 for k in ['wall_ms', 'cuda_event_ms', 'tokens_per_s'])
                assert math.isclose(trial['tokens_per_s'], trial['timed_generated_tokens'] / (trial['wall_ms'] / 1000))
                if data['mode'] == 'energy':
                    assert trial['telemetry']['energy_scope'] == 'whole GPU board; decode only; no idle subtraction'
                    if trial['telemetry']['joules_per_token'] != 'NA':
                        assert trial['telemetry']['window_s'] >= data['config']['minimum_energy_window_s']
                        assert trial['telemetry']['counter_j'] > 0
                        assert trial['telemetry']['counter_power_relative_difference'] <= trial['telemetry']['counter_power_agreement_threshold']
                        assert math.isclose(trial['telemetry']['joules_per_token'], trial['telemetry']['counter_j'] / trial['timed_generated_tokens'])
    return data['mode']


if __name__ == '__main__' and '--validate' in __import__('sys').argv:
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', required=True)
    args = parser.parse_args()
    try:
        print('Validated ' + validate(args.validate))
    except (OSError, KeyError, ValueError, AssertionError) as error:
        parser.exit(1, f'INVALID EXPERIMENT: {error}\n')
    raise SystemExit(0)


class Sensor:
    def __init__(self, nvml, handle, interval):
        self.nvml, self.handle, self.interval = nvml, handle, interval
        self.samples, self.errors = [], []
        self.stop = threading.Event()

    def read(self):
        n, h = self.nvml, self.handle
        sample = {'t': time.perf_counter()}
        for key, call in {'energy_mj': lambda: n.nvmlDeviceGetTotalEnergyConsumption(h),
                          'power_w': lambda: n.nvmlDeviceGetPowerUsage(h) / 1000,
                          'temperature_c': lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                          'sm_clock_mhz': lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                          'memory_clock_mhz': lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM)}.items():
            try:
                sample[key] = call()
            except n.NVMLError as error:
                sample[key] = None
                self.errors.append(f'{key}: {error}')
        self.samples.append(sample)
        return sample

    def loop(self):
        while not self.stop.wait(self.interval):
            self.read()

    def start(self):
        self.read()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def finish(self, start, end, before, after, tokens, minimum):
        self.stop.set()
        self.thread.join()
        self.read()
        self.samples.sort(key=lambda sample: sample['t'])
        powers = [s for s in self.samples if s['power_w'] is not None]
        energy_updates = []
        last = None
        for sample in self.samples:
            if sample['energy_mj'] is not None and sample['energy_mj'] != last:
                energy_updates.append(sample)
                last = sample['energy_mj']
        power_changes = [s for i, s in enumerate(powers) if i == 0 or s['power_w'] != powers[i-1]['power_w']]
        def cadence(changes):
            return statistics.median(b['t'] - a['t'] for a, b in zip(changes, changes[1:])) if len(changes) >= 2 else 'NA'
        counter_j = (after['energy_mj'] - before['energy_mj']) / 1000 if before['energy_mj'] is not None and after['energy_mj'] is not None else 'NA'
        integrated_j = 'NA'
        if len(powers) >= 2 and powers[0]['t'] <= start and powers[-1]['t'] >= end:
            # Linear boundary interpolation and trapezoids over the exact wall window.
            def at(t):
                for a, b in zip(powers, powers[1:]):
                    if a['t'] <= t <= b['t']:
                        return a['power_w'] + (b['power_w']-a['power_w']) * (t-a['t']) / (b['t']-a['t'])
                raise ValueError('missing bracket power sample')
            nodes = [(start, at(start))] + [(s['t'], s['power_w']) for s in powers if start < s['t'] < end] + [(end, at(end))]
            integrated_j = sum((b[0]-a[0]) * (a[1]+b[1]) / 2 for a, b in zip(nodes, nodes[1:]))
        relative_difference = (abs(counter_j-integrated_j)/counter_j
                               if counter_j != 'NA' and counter_j > 0 and integrated_j != 'NA' else 'NA')
        sensor_adequate = (end-start >= minimum and counter_j != 'NA' and counter_j > 0 and len(energy_updates) >= 5)
        agreement = relative_difference != 'NA' and relative_difference <= 0.10
        qualified = sensor_adequate and agreement
        unavailable = ('cumulative counter and sampled power disagree; absolute energy unresolved'
                       if relative_difference != 'NA' and not agreement else
                       'window too short, energy sensor unsupported/nonpositive, fewer than 5 observed counter changes, or no bracketed power estimate')
        return {'samples': self.samples, 'errors': sorted(set(self.errors)), 'window_s': end-start,
                'energy_scope': 'whole GPU board; decode only; no idle subtraction',
                'sample_interval_s': self.interval, 'distinct_power_readings': len({s['power_w'] for s in powers}),
                'power_observed_change_interval_s': cadence(power_changes),
                'counter_observed_change_interval_s': cadence(energy_updates),
                'counter_distinct_readings': len(energy_updates), 'counter_j': counter_j,
                'counter_before': before, 'counter_after': after,
                'counter_window_s': after['t'] - before['t'],
                'integrated_power_j': integrated_j,
                'unvalidated_counter_joules_per_token': counter_j/tokens if counter_j != 'NA' and counter_j > 0 else 'NA',
                'sampled_power_integral_joules_per_token': integrated_j/tokens if integrated_j != 'NA' else 'NA',
                'estimator_scope_note': 'raw estimators may differ in sensor scope/resolution; neither is independently externally calibrated',
                'counter_power_agreement_threshold': 0.10,
                'power_integral_window': {'start': start, 'end': end},
                'joules_per_token': counter_j/tokens if qualified else 'NA',
                'method': 'NVML cumulative energy counter; board scope' if qualified else 'NA',
                'qualification': 'sensor cadence observed; absolute accuracy and fast transient response not independently calibrated',
                'unavailable_reason': '' if qualified else unavailable,
                'counter_power_relative_difference': relative_difference}


def controlled_trial(model, cache, decode, prompt, count, torch, sensor=None, minimum=5, capture=False):
    cache.reset()
    torch.cuda.nvtx.range_push('prefill')
    prefill = model(input_ids=prompt, past_key_values=cache,
                    cache_position=torch.arange(prompt.shape[1], device='cuda'),
                    use_cache=True, logits_to_keep=1)
    current = prefill.logits[:, -1:].argmax(-1)
    del prefill
    torch.cuda.nvtx.range_pop()
    output = torch.empty((prompt.shape[0], count), dtype=torch.long, device='cuda')
    output[:, :1].copy_(current)
    position = torch.empty((1,), dtype=torch.long, device='cuda')
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    if sensor:
        sensor.start()
        before = sensor.read()
    if capture:
        torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push('decode')
    start = time.perf_counter()
    start_event.record()
    for index in range(count - 1):
        position.fill_(prompt.shape[1] + index)
        torch.compiler.cudagraph_mark_step_begin()
        next_token = decode(current, position)
        current.copy_(next_token)
        output[:, index+1:index+2].copy_(current)
        del next_token
    end_event.record()
    torch.cuda.synchronize()
    end = time.perf_counter()
    torch.cuda.nvtx.range_pop()
    if capture:
        torch.cuda.cudart().cudaProfilerStop()
    tokens = prompt.shape[0] * (count - 1)
    telemetry = None
    if sensor:
        after = sensor.read()
        telemetry = sensor.finish(start, end, before, after, tokens, minimum)
    ids = output.cpu().tolist()
    result = {'timed_decode_steps': count-1, 'timed_generated_tokens': tokens,
              'wall_ms': (end-start)*1000, 'cuda_event_ms': start_event.elapsed_time(end_event),
              'tokens_per_s': tokens/(end-start), 'output_ids': ids, 'output_sha256': digest(ids),
              'cache_length': int(cache.get_seq_length().item())}
    if telemetry:
        result['telemetry'] = telemetry
    return result


def make_decode(model, cache, maximum, torch, compile_enabled, return_logits=False, mask_kind=None):
    cache_indices = torch.arange(maximum, device='cuda')
    # Transformers eager attention ADDS its 4D mask; SDPA accepts Boolean keep masks.
    # Preserve the original SDPA route; fix only the previously invalid eager diagnostic.
    if mask_kind is None:
        mask_kind = 'additive' if model.config._attn_implementation == 'eager' else 'boolean'
    assert mask_kind in ('boolean', 'additive', 'builtin')
    def forward(token, position):
        # Explicit 4D causal mask prevents Python mask construction / graph breaks.
        mask = (cache_indices <= position.reshape(1, 1)).reshape(1, 1, 1, maximum)
        if mask_kind == 'additive':
            mask = torch.zeros(mask.shape, device=mask.device, dtype=model.dtype).masked_fill(~mask, torch.finfo(model.dtype).min)
        elif mask_kind == 'builtin':
            mask = None
        logits = model(input_ids=token, past_key_values=cache, cache_position=position,
                       position_ids=position.unsqueeze(0), attention_mask=mask,
                       use_cache=True, logits_to_keep=1).logits[:, -1:]
        selected = logits.argmax(-1)
        return (selected, logits) if return_logits else selected
    return torch.compile(forward, mode='reduce-overhead', dynamic=False, fullgraph=True) if compile_enabled else forward


def differing_positions(eager_ids, compiled_ids):
    return [{'batch_index': batch, 'token_index': step, 'eager_token': eager, 'compiled_token': compiled}
            for batch, (eager_row, compiled_row) in enumerate(zip(eager_ids, compiled_ids))
            for step, (eager, compiled) in enumerate(zip(eager_row, compiled_row)) if eager != compiled]


def diagnostic_logits(model, cache, maximum, prompt, token_index, torch, compiled):
    """Replay to a differing token outside timing, returning logits for diagnosis."""
    cache.reset()
    prefill = model(input_ids=prompt, past_key_values=cache,
                    cache_position=torch.arange(prompt.shape[1], device='cuda'),
                    use_cache=True, logits_to_keep=1)
    if token_index == 0:
        return prefill.logits[:, -1].float().cpu()
    current = prefill.logits[:, -1:].argmax(-1)
    del prefill
    position = torch.empty((1,), dtype=torch.long, device='cuda')
    forward = make_decode(model, cache, maximum, torch, compiled, return_logits=True)
    for index in range(token_index):
        position.fill_(prompt.shape[1] + index)
        torch.compiler.cudagraph_mark_step_begin()
        selected, logits = forward(current, position)
        current.copy_(selected)
        if index == token_index-1:
            result = logits[:, -1].float().cpu()
        del selected, logits
    return result


def main(args):
    config = json.loads(Path(args.config).read_text())
    assert config['warmup_trials'] >= 3 and config['trials'] >= 3
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    import transformers
    import pynvml as nvml
    nvml.nvmlInit()
    from gpu_target import select, verify_cuda, query_l2
    handle,physical = select(nvml,args.gpu_uuid)
    props = verify_cuda(torch,args.gpu_uuid)
    hardware = {'uuid': args.gpu_uuid, 'name': props.name, 'sm_count': props.multi_processor_count,
                'driver': nvml.nvmlSystemGetDriverVersion(), 'compute_capability': list(torch.cuda.get_device_capability()),
                'graphics_processes': [{'pid': p.pid} for p in nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)],
                'preflight_scope': 'startup compute-process snapshot; no runtime exclusivity guarantee'}
    hardware.update(query_l2(torch))
    data = {'status': 'running', 'mode': args.mode, 'classification': 'measured-real-GPU',
            'config': config, 'hardware': hardware,
            'versions': {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'transformers': transformers.__version__},
            'environment': {k: os.environ.get(k, 'unset') for k in ['CUDA_HOME', 'TORCHINDUCTOR_COMPILE_THREADS', 'MAX_JOBS', 'OMP_NUM_THREADS']},
            'timing_scope': 'synchronized controlled decode after prefill; first token selected by prefill excluded',
            'baseline_scope': 'software StaticCache and optional compile; no physical GPU modifications',
            'external_instrumentation': args.mode == 'profile', 'baseline_comparison_eligible': args.mode != 'profile',
            'cases': []}
    dump(args.output/'run.json', data)
    torch.manual_seed(config['seed'])
    if args.mode == 'bandwidth':
        size = config['copy_buffer_mib'] * 1024**2
        assert hardware['l2_bytes'] != 'NA' and size > 2*hardware['l2_bytes'], 'bandwidth control requires buffer larger than independently queried L2'
        source = torch.randn(size//4, device='cuda')
        destination = torch.empty_like(source)
        for _ in range(3):
            destination.copy_(source)
        torch.cuda.synchronize()
        trials = []
        for _ in range(config['trials']):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push('bandwidth_copy')
            start = time.perf_counter()
            a.record()
            for _ in range(config['copy_repetitions']):
                destination.copy_(source)
            b.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            torch.cuda.nvtx.range_pop()
            useful = 2*size*config['copy_repetitions']
            trials.append({'cuda_event_ms': a.elapsed_time(b), 'wall_ms': elapsed*1000,
                           'useful_bytes': useful, 'useful_gb_per_s': useful/a.elapsed_time(b)/1e6})
        assert torch.equal(source, destination)
        data.update(buffer_bytes=size, trials=trials, dram_bytes_measured='NA',
                    traffic_scope='useful source read + destination write bytes; actual DRAM traffic requires profiler counters',
                    bandwidth_scope='contiguous FP32 copy; not a universal decode bandwidth ceiling')
    else:
        path = Path(config['model_path'])
        assert path.name == config['model_revision'] and path.is_dir()
        torch.manual_seed(config['seed'])
        model = transformers.AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.bfloat16,
                local_files_only=True, attn_implementation='sdpa').eval().to('cuda')
        manifest = [{'name': name, 'shape': list(p.shape), 'dtype': str(p.dtype),
                     'bytes': p.numel()*p.element_size(), 'gpu_address': p.data_ptr()} for name,p in model.named_parameters()]
        dump(args.output/'tensor_manifest.json', {'classification': 'measured-real-GPU', 'tensors': manifest,
                                                'scope': 'unique named parameter tensors; addresses run-specific; logical storage not measured traffic'})
        with torch.inference_mode():
            tiny = torch.tensor([[42,71,93,104]], device='cuda')
            check_cache = transformers.StaticCache(config=model.config, max_cache_len=8)
            check_decode = make_decode(model, check_cache, 8, torch, False)
            check = controlled_trial(model, check_cache, check_decode, tiny, 4, torch)
            reference = model.generate(input_ids=tiny, max_new_tokens=4, do_sample=False,
                                       eos_token_id=None, repetition_penalty=1.0, logits_to_keep=1)
            reference_ids = reference[:,4:].cpu().tolist()
            data['correctness'] = {'passed': reference_ids == check['output_ids'],
                                   'reference_ids': reference_ids, 'static_ids': check['output_ids'],
                                   'scope': 'tiny uncompiled static-cache versus model.generate plain greedy'}
            dump(args.output/'run.json', data)
            assert data['correctness']['passed'], 'static mask/cache token mismatch against dynamic greedy'
            pairs = config['energy_cases'] if args.mode == 'energy' else [[b,c] for b in config['batch_sizes'] for c in config['context_lengths']]
            if args.batch is not None or args.context is not None:
                pairs = [[b,c] for b,c in pairs if (args.batch is None or b==args.batch) and (args.context is None or c==args.context)]
            assert pairs, 'no matching configured cases'
            if args.mode == 'energy':
                idle = Sensor(nvml, handle, config['sensor_interval_s'])
                idle.start()
                time.sleep(config['idle_calibration_s'])
                idle.stop.set()
                idle.thread.join()
                idle.read()
                dump(args.output/'idle_sensor_calibration.json', {'samples': idle.samples,
                        'errors': idle.errors, 'scope': 'model resident, no decode; cadence diagnostic only; not subtracted'})
            for batch, context in pairs:
                count = config['energy_generated_tokens'] if args.mode == 'energy' else config['generated_tokens']
                maximum = context+count
                assert maximum <= config['maximum_cache_length']
                torch._dynamo.reset()
                cache = transformers.StaticCache(config=model.config, max_cache_len=maximum)
                decode = make_decode(model, cache, maximum, torch, config['compile'])
                generator = torch.Generator().manual_seed(config['seed'])
                prompt = torch.randint(0, model.config.vocab_size, (batch,context), generator=generator).to('cuda')
                case = {'batch_size': batch, 'context_length': context, 'generated_tokens': count,
                        'cache_capacity_tokens': maximum, 'prompt_sha256': digest(prompt.cpu().tolist()), 'trials': []}
                warmup_hashes = []
                compile_start = time.perf_counter()
                for _ in range(config['warmup_trials']):
                    warmup_hashes.append(controlled_trial(model, cache, decode, prompt, count, torch)['output_sha256'])
                case['warmup_wall_s_including_compile'] = time.perf_counter()-compile_start
                case['kv_tensor_manifest'] = [{'layer': index, 'kind': kind, 'shape': list(tensor.shape),
                                               'dtype': str(tensor.dtype), 'bytes': tensor.numel()*tensor.element_size(),
                                               'gpu_address': tensor.data_ptr()}
                                              for index, layer in enumerate(cache.layers)
                                              for kind, tensor in [('key', layer.keys), ('value', layer.values)]]
                case['kv_manifest_scope'] = 'allocated StaticCache storage including unused future slots; not measured accessed bytes'
                # Same-shape independent eager path checks compiled greedy behavior.
                eager = make_decode(model, cache, maximum, torch, False)
                eager_check = controlled_trial(model, cache, eager, prompt, min(count,4), torch)
                compiled_check = controlled_trial(model, cache, decode, prompt, min(count,4), torch)
                case['compiled_correctness'] = {'passed': eager_check['output_ids']==compiled_check['output_ids'],
                                               'eager_ids': eager_check['output_ids'], 'compiled_ids': compiled_check['output_ids']}
                if not case['compiled_correctness']['passed']:
                    differences = differing_positions(eager_check['output_ids'], compiled_check['output_ids'])
                    diagnostic = {'batch_size': batch, 'context_length': context,
                                  'cache_capacity_tokens': maximum, 'prompt_sha256': case['prompt_sha256'],
                                  'outside_measurement': True, 'hypothesis_status': 'unresolved',
                                  'narrow_eager_ids': eager_check['output_ids'], 'narrow_compiled_ids': compiled_check['output_ids'],
                                  'narrow_eager_sha256': eager_check['output_sha256'],
                                  'narrow_compiled_sha256': compiled_check['output_sha256'],
                                  'differing_positions': differences}
                    dump(args.output/'diagnostic_failure.json', diagnostic)
                    # Preserve the narrow evidence before any optional slower diagnostic.
                    if config.get('diagnose_full_on_failure', True):
                        diagnostic_count = min(count, 64)
                        full_eager = controlled_trial(model, cache, eager, prompt, diagnostic_count, torch)
                        full_compiled = controlled_trial(model, cache, decode, prompt, diagnostic_count, torch)
                        full_differences = differing_positions(full_eager['output_ids'], full_compiled['output_ids'])
                        diagnostic['full_comparison'] = {'generated_tokens': diagnostic_count,
                                                        'eager_ids': full_eager['output_ids'], 'compiled_ids': full_compiled['output_ids'],
                                                        'eager_sha256': full_eager['output_sha256'], 'compiled_sha256': full_compiled['output_sha256'],
                                                        'differing_positions': full_differences}
                        dump(args.output/'diagnostic_failure.json', diagnostic)
                        first_step = min(d['token_index'] for d in differences)
                        eager_logits = diagnostic_logits(model, cache, maximum, prompt, first_step, torch, False)
                        compiled_logits = diagnostic_logits(model, cache, maximum, prompt, first_step, torch, config['compile'])
                        logit_records = []
                        for difference in differences:
                            if difference['token_index'] != first_step:
                                continue
                            row = difference['batch_index']
                            record = dict(difference)
                            for name, logits in [('eager', eager_logits), ('compiled', compiled_logits)]:
                                top = logits[row].topk(8)
                                record[name+'_top8'] = [{'token': int(token), 'logit': float(logit)} for token, logit in zip(top.indices, top.values)]
                                record[name+'_top2_margin'] = float(top.values[0]-top.values[1])
                                record[name+'_chosen_logits'] = {str(token): float(logits[row, token]) for token in [difference['eager_token'], difference['compiled_token']]}
                            delta = compiled_logits[row]-eager_logits[row]
                            record['max_abs_logit_difference'] = float(delta.abs().max())
                            record['mean_abs_logit_difference'] = float(delta.abs().mean())
                            logit_records.append(record)
                        diagnostic['first_difference_logits'] = logit_records
                        diagnostic['logits_diagnostic_scope'] = 'independent eager/compiled replays returning logits at first differing token; adding logits output may change compiler fusion, so this is diagnostic evidence'
                        dump(args.output/'diagnostic_failure.json', diagnostic)
                assert case['compiled_correctness']['passed'], 'compiled/eager same-shape greedy mismatch; see diagnostic_failure.json'
                for _ in range(config['trials']):
                    sensor = Sensor(nvml, handle, config['sensor_interval_s']) if args.mode=='energy' else None
                    case['trials'].append(controlled_trial(model, cache, decode, prompt, count, torch,
                            sensor, config['minimum_energy_window_s'], args.mode=='profile'))
                hashes = {t['output_sha256'] for t in case['trials']} | set(warmup_hashes)
                case['warmup_output_sha256'] = warmup_hashes
                case['outputs_identical'] = len(hashes)==1
                data['cases'].append(case)
                dump(args.output/'run.json', data)
                assert case['outputs_identical'], 'same-shape greedy outputs differ across runs'
                print(f'{args.mode} B={batch} C={context}: {[round(t["wall_ms"],2) for t in case["trials"]]} ms', flush=True)
                del decode, eager, cache
                torch._dynamo.reset()
                torch.cuda.empty_cache()
    data['status'] = 'complete'
    dump(args.output/'run.json', data)
    print('Validated '+validate(args.output), flush=True)
    nvml.nvmlShutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--mode', choices=['timing','energy','bandwidth','profile'], required=True)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--context', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        main(args)
    except Exception as error:
        if (args.output/'run.json').exists():
            data = json.loads((args.output/'run.json').read_text())
            data['status']='failed'
            dump(args.output/'run.json',data)
        dump(args.output/'failure.json', {'type':type(error).__name__, 'error':str(error), 'traceback':traceback.format_exc()})
        raise
