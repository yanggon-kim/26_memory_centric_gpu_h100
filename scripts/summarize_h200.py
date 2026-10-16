#!/usr/bin/env python3
"""Consolidate only the supplied H200 session; retain failed and partial gates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from h200_session import dump, DEFAULT_ROOT
from consolidate_h100_results import SPEC_COLUMNS, EXTRA_COLUMNS


def median(values):
    return statistics.median(values) if values else 'NA'


def summarize(root):
    root = Path(root).resolve()
    rows, manifest = [], []
    for source in sorted((root/'runs').glob('*/run.json')):
        data = json.loads(source.read_text())
        config = data['config']
        vllm = 'engine_settings' in data
        external = data.get('external_instrumentation', False)
        cases = data.get('cases', [])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        entry = {'id': source.parent.name, 'status': data['status'], 'source':
                 str(source.relative_to(root)), 'source_sha256': digest,
                 'external_instrumentation': external, 'cases': []}
        for case in cases:
            trials = case.get('trials', [])
            identical = case.get('output_digests_identical' if vllm else 'outputs_identical', False)
            correctness = identical and (vllm or (data.get('correctness', {}).get('passed', False)
                                                 and case.get('compiled_correctness', {}).get('passed', False)))
            valid = data['status'] == 'complete' and correctness and len(trials) >= 3
            steps = case['generated_tokens']-1
            entry['cases'].append({'batch': case['batch_size'], 'context': case['context_length'],
                                   'valid': valid, 'trials': len(trials)})
            row = {key: 'NA' for key in SPEC_COLUMNS+EXTRA_COLUMNS}
            eager = data.get('engine_settings', {}).get('enforce_eager', False)
            design = ('vllm_eager' if eager else 'vllm_graph') if vllm else (
                'hf_staticcache_compiled' if config.get('compile') else 'hf_staticcache_eager')
            row.update(run_id=source.parent.name, design=design,
                       classification='measured-real-GPU-instrumented' if external else 'measured-real-GPU',
                       hardware_or_simulator_version=json.dumps({'hardware': data['hardware']['name'],
                                                                 'versions': data['versions']}),
                       model=config['model_id'], model_revision=config['model_revision'],
                       kernel='end_to_end_decode_'+data['mode'], batch_size=case['batch_size'],
                       context_length=case['context_length'], precision='BF16', kv_format='BF16',
                       attention_type=data.get('backend_policy', 'HF SDPA'),
                       sm_count=data['hardware']['sm_count'], ms_sm_count=0,
                       seed=config['seed'], workload_pattern=case.get('workload_version', 'seeded_synthetic_uniform_ids'),
                       selected_case_status='complete' if valid else data['status'] if data['status'] != 'complete' else 'failed',
                       generated_tokens_per_sequence=case['generated_tokens'],
                       timed_generated_tokens=case['batch_size']*steps,
                       source_run_json=str(source.relative_to(root)), source_sha256=digest,
                       external_instrumentation=external, timing_scope=data['timing_scope'],
                       energy_scope='whole GPU board; decode only; no idle subtraction' if data['mode']=='energy' else 'NA')
            if valid:
                row.update(latency_ms=median([t['wall_ms']/steps for t in trials]),
                           decode_tokens_per_s=median([t['tokens_per_s'] for t in trials]),
                           cuda_event_ms=median([t['cuda_event_ms'] for t in trials]),
                           decode_wall_ms=median([t['wall_ms'] for t in trials]))
                if data['mode'] == 'energy' and all(isinstance(t.get('telemetry', {}).get('joules_per_token'), (int, float)) for t in trials):
                    row['joules_per_token'] = median([t['telemetry']['joules_per_token'] for t in trials])
                    row['gpu_avg_power_w'] = median([t['telemetry']['integrated_power_j']/t['telemetry']['window_s'] for t in trials])
            row['notes'] = json.dumps({'correctness_passed': bool(correctness),
                'performance_eligible': bool(valid and not external),
                'missing_metric_reason': 'traffic/utilization require separately linked counters; architecture/PPA/NoC and component energy remain unmeasured',
                'energy_qualification': [t.get('telemetry', {}).get('unavailable_reason') for t in trials] if data['mode']=='energy' else 'separate energy run required'})
            rows.append(row)
        manifest.append(entry)
    for failure in sorted((root/'runs').glob('*/failure.json')):
        if not (failure.parent/'run.json').exists():
            manifest.append({'id': failure.parent.name, 'status': 'failed',
                             'source': str(failure.relative_to(root)),
                             'source_sha256': hashlib.sha256(failure.read_bytes()).hexdigest(), 'cases': []})
    result = root/'results'
    with (result/'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=SPEC_COLUMNS+EXTRA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    dump(result/'case_manifest.json', {'schema_version': 1, 'hardware_scope': 'H200 only', 'cases': manifest})
    expected = []
    registry = json.loads((Path(__file__).resolve().parents[1]/'configs/models.json').read_text())['models']
    for model in ('bridge', 'qwen7b', 'mistral7b', 'qwen14b', 'qwen32b'):
        for batch in registry[model]['stages']['baseline']['batch_sizes']:
            for context in registry[model]['stages']['baseline']['context_lengths']:
                expected.append({'id': f'{model}_graph_b{batch}_c{context}', 'model': model,
                                 'batch': batch, 'context': context})
    by_id = {x['id']: x for x in manifest}
    for case in expected:
        actual = by_id.get(case['id'])
        case['status'] = ('complete' if actual and actual['status']=='complete' and actual['cases']
                          and all(x['valid'] for x in actual['cases']) else actual['status'] if actual else 'pending')
    dump(result/'fixed_bandwidth_case_manifest.json', {'cases': expected, 'attempts': manifest})
    measurement_path=result/'fixed_bandwidth_measurements.json'
    if not measurement_path.exists():
        dump(measurement_path, {'measurements': [], 'status': 'pending decode profiles',
             'missing_value_policy': 'null with reason; never substitute logical storage for DRAM traffic'})
    launch_path=result/'attention_launch_inventory.json'
    if not launch_path.exists():
        dump(launch_path, {'kernels': [], 'status': 'pending decode timelines and kernel metadata',
                          'tiles_and_splits': None, 'reason': 'not yet collected'})
    completed = sum(x['status']=='complete' for x in expected)
    smoke_gate=root/'results/bridge_graph_hf_smoke_tokens.json'
    native=root/'environment/native_trace_micro.json'
    phase0_ready=(smoke_gate.exists() and json.loads(smoke_gate.read_text())['status']=='passed'
                  and native.exists() and isinstance(json.loads(native.read_text()).get('replay'),dict)
                  and (root/'environment/runtime_environment.json').exists())
    status = {'phase0': 'complete' if phase0_ready else 'partial', 'phase1': 'partial' if manifest else 'pending',
              'dense_matrix_completed': completed, 'dense_matrix_total': len(expected),
              'source_scope': 'this H200 directory only; historical H100/RTX rows excluded',
              'phase2': 'not started', 'architecture_modifications': 'not started',
              'phase0_scope':'environment/runtime/profiler/native-parser feasibility; H200 timing, async semantics and power calibration remain pending',
              'remaining': [x for x in expected if x['status'] != 'complete']}
    dump(root/'STATUS.json', status)
    lines = ['# H200 intermediate results', '',
             f'{completed} of {len(expected)} planned dense graph-baseline cases have passed the recorded timing and output gates.', '',
             'All measurements use the allocated H200. Prefill and the first selected token are excluded from pure decode. Latency is milliseconds per batch decode step; throughput counts all requests.', '',
             '| Run | Status | Step ms | Tokens/s | Board J/token |', '|---|---|---:|---:|---:|']
    def fmt(x):
        return f'{x:.4f}' if isinstance(x, (float, int)) else str(x)
    for row in rows:
        lines.append('| '+ ' | '.join([row['run_id'], row['selected_case_status'], fmt(row['latency_ms']), fmt(row['decode_tokens_per_s']), fmt(row['joules_per_token'])])+' |')
    lines += ['', '## Interpretation', '',
              'Graph, eager, HF compiled, energy, and profiled executions have separate identities. Differences between these programs require software and correctness analysis before any architectural interpretation.', '',
              'Qualified board energy uses raw cumulative-counter and sampled-power estimates with at least five seconds, five counter changes, and at most ten percent disagreement. Agreement provides an internal crosscheck; component energy and absolute accuracy remain uncalibrated.', '',
              '## Remaining work', '',
              'See ../STATUS.json and ../results/fixed_bandwidth_case_manifest.json for completed, failed, and pending cases. Hardware-counter attribution, representative kernel coverage, and native trace-tool validation retain their own evidence gates.']
    (root/'analysis/intermediate_results.md').write_text('\n'.join(lines)+'\n')
    for name, title, content in [
        ('bottleneck_analysis.md', 'H200 decode bottleneck analysis', 'A bottleneck classification requires observed decode kernel counters and timelines. Timing alone cannot distinguish HBM saturation, latency/MLP, cache/issue, compute, and scheduler overhead. Kernel-counter findings will be added with their sampling and replay scope.'),
        ('traffic_attribution_report.md', 'H200 decode traffic attribution', 'Measured DRAM transactions, requested logical bytes, and allocated tensor storage are separate quantities. Sampled kernels cannot establish a measured full-step total. Weight, KV, scratch, and unknown attribution require actual kernel/range evidence.'),
        ('residency_opportunity_report.md', 'H200 on chip residency opportunity', 'Phase 1 characterizes delivery and compute limits. Incremental residency benefits remain pending Phase 2 full-model access analysis and calibration. No added-SRAM, fixed-area, or architectural energy benefit is established by these baselines.')]:
        path=root/'analysis'/name
        if not path.exists():
            path.write_text('# '+title+'\n\n'+content+'\n')
    return status


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
