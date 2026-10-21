#!/usr/bin/env python3
"""Consolidate real H100 software runs into the specification's measured schema."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


SPEC_COLUMNS = """run_id design classification hardware_or_simulator_version model kernel batch_size context_length precision kv_format attention_type latency_ms decode_tokens_per_s joules_per_token gpu_avg_power_w hbm_read_bytes_per_token hbm_write_bytes_per_token l2_hit_rate added_sram_hit_rate local_bytes_per_token remote_bytes_per_token noc_hop_bytes_per_token noc_avg_latency_ns noc_p99_latency_ns compute_utilization tensor_pipe_utilization lsu_utilization agu_req_per_cycle outstanding_mem_reqs sm_count ms_sm_count compute_area_mm2 sram_area_mm2 noc_area_mm2 total_area_mm2 dynamic_energy_j static_energy_j notes""".split()
EXTRA_COLUMNS = """seed workload_pattern model_revision selected_case_status generated_tokens_per_sequence first_selected_token_ids prefill_ms timed_generated_tokens cuda_event_ms decode_wall_ms source_run_json source_sha256 external_instrumentation timing_scope energy_scope""".split()
UNKNOWN_COLUMNS = set(SPEC_COLUMNS) - {
    'run_id', 'design', 'classification', 'hardware_or_simulator_version', 'model',
    'kernel', 'batch_size', 'context_length', 'precision', 'kv_format',
    'attention_type', 'latency_ms', 'decode_tokens_per_s', 'joules_per_token',
    'gpu_avg_power_w', 'sm_count', 'ms_sm_count', 'notes'
}


def median(values):
    return statistics.median(values) if values else 'NA'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    rows = []
    for folder in args.runs:
        source = folder/'run.json'
        data = json.loads(source.read_text())
        config = data['config']
        assert data['classification'] == 'measured-real-GPU'
        assert config['precision'] == 'BF16' and config['seed'] == 20261013
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for case in data.get('cases', []):
            valid = (data['status'] == 'complete' and data['correctness']['passed']
                     and case['compiled_correctness']['passed'] and case['outputs_identical'])
            trials = case['trials']
            reasons = {key: 'not measured; requires H100 counter, calibrated simulator or physical cost evidence'
                       for key in UNKNOWN_COLUMNS}
            reasons['prefill_ms'] = 'first-token prefill excluded from this pure-decode timing run'
            if not valid:
                reasons.update(latency_ms='correctness or run-completion gate failed',
                               decode_tokens_per_s='correctness or run-completion gate failed',
                               joules_per_token='correctness or run-completion gate failed',
                               gpu_avg_power_w='correctness or run-completion gate failed')
            elif data['mode'] != 'energy':
                reasons.update(joules_per_token='no qualified decode energy window in this timing run',
                               gpu_avg_power_w='no qualified decode energy window in this timing run')
            else:
                for key in ['joules_per_token', 'gpu_avg_power_w']:
                    reasons[key] = 'energy sensor qualification failed; inspect raw trial telemetry'
            layout = config['layout']
            attention = 'SDPA GQA' if isinstance(layout['num_key_value_heads'], int) and layout['num_key_value_heads'] < layout['num_attention_heads'] else 'SDPA'
            row = {key: 'NA' for key in SPEC_COLUMNS + EXTRA_COLUMNS}
            row.update(run_id=folder.name, design='hf_staticcache_compiled' if config['compile'] else 'hf_staticcache_eager',
                       classification='measured-real-GPU',
                       hardware_or_simulator_version=f"{data['hardware']['name']}; driver {data['hardware']['driver']}; torch {data['versions']['torch']}; CUDA {data['versions']['cuda_runtime']}",
                       model=config['model_id']+'@'+config['model_revision'],
                       kernel='end_to_end_decode_'+data['mode'], batch_size=case['batch_size'],
                       context_length=case['context_length'], precision='BF16',
                       kv_format='BF16 StaticCache', attention_type=attention,
                       sm_count=data['hardware']['sm_count'], ms_sm_count=0,
                       seed=config['seed'], workload_pattern='seeded synthetic uniform token IDs',
                       model_revision=config['model_revision'],
                       selected_case_status='complete' if valid else 'failed',
                       generated_tokens_per_sequence=case['generated_tokens'],
                       first_selected_token_ids=json.dumps([ids[0] for ids in trials[0]['output_ids']]) if trials else 'NA',
                       timed_generated_tokens=case['batch_size'] * (case['generated_tokens']-1),
                       source_run_json=str(source), source_sha256=digest,
                       external_instrumentation=data['external_instrumentation'],
                       timing_scope=data['timing_scope'],
                       energy_scope='whole GPU board; decode only; no idle subtraction' if data['mode']=='energy' else 'NA')
            if valid:
                steps = case['generated_tokens']-1
                row['latency_ms'] = median([t['wall_ms']/steps for t in trials])
                row['decode_tokens_per_s'] = median([t['tokens_per_s'] for t in trials])
                row['cuda_event_ms'] = median([t['cuda_event_ms'] for t in trials])
                row['decode_wall_ms'] = median([t['wall_ms'] for t in trials])
                if data['mode'] == 'energy':
                    telemetries = [t['telemetry'] for t in trials]
                    if all(isinstance(t['joules_per_token'], (int, float)) for t in telemetries):
                        row['joules_per_token'] = median([t['joules_per_token'] for t in telemetries])
                        row['gpu_avg_power_w'] = median([t['integrated_power_j']/t['window_s'] for t in telemetries])
                        reasons.pop('joules_per_token')
                        reasons.pop('gpu_avg_power_w')
            row['notes'] = json.dumps({'source_mode': data['mode'], 'source_status': data['status'],
                                       'unavailable_reasons': reasons,
                                       'raw_energy_estimators': [{key: t['telemetry'].get(key) for key in
                                          ('counter_j', 'integrated_power_j', 'unvalidated_counter_joules_per_token',
                                           'sampled_power_integral_joules_per_token', 'counter_power_relative_difference',
                                           'window_s', 'qualification')}
                                          for t in trials] if data['mode']=='energy' else [],
                                       'warmup_output_sha256': case.get('warmup_output_sha256', []),
                                       'trial_output_sha256': [t['output_sha256'] for t in trials],
                                       'calibration': 'board-sensor agreement is internal only; no absolute calibration'},
                                      separators=(',', ':'))
            rows.append(row)
    assert rows, 'no measured cases found'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=SPEC_COLUMNS + EXTRA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} measured H100 rows to {args.output}')


if __name__ == '__main__':
    main()
