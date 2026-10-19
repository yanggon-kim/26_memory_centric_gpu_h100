#!/usr/bin/env python3
"""Combine measured decode time with explicitly analytical FLOP/read proxies."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ceilings', type=Path, required=True)
    p.add_argument('--model-config', type=Path, required=True)
    p.add_argument('--tensor-manifest', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    ceilings = json.loads(a.ceilings.read_text())
    config = json.loads(a.model_config.read_text())
    tensors = json.loads(a.tensor_manifest.read_text())['tensors']
    linear_weight_bytes = sum(t['bytes'] for t in tensors
                              if len(t['shape']) == 2 and 'embed_tokens' not in t['name'])
    hidden = config['hidden_size']
    layers = config['num_hidden_layers']
    kv_heads = config['num_key_value_heads']
    head_dim = config.get('head_dim', hidden // config['num_attention_heads'])
    compute_tflops = ceilings['gemm']['median_tflops']
    copy_gb_s = ceilings['copy']['median_useful_gb_per_s']
    points = []
    for path in a.runs:
        run = json.loads(path.read_text())
        assert run['status'] == 'complete', path
        for case in run['cases']:
            if not case.get('outputs_identical', case.get('output_digests_identical', False)):
                continue
            batch = case['batch_size']
            context = case['context_length']
            timed_steps = case['generated_tokens'] - 1
            average_attended_tokens = context + 1 + (timed_steps - 1) / 2
            # Every non-embedding dense matrix is counted once per decode tick.
            # Treat all of its BF16 bytes as a read once per tick. Actual HBM/L2
            # traffic and executed arithmetic can differ; neither is measured.
            linear_flops_per_step = batch * linear_weight_bytes
            attention_flops_per_step = 4 * layers * hidden * average_attended_tokens * batch
            proxy_flops_per_step = linear_flops_per_step + attention_flops_per_step
            kv_read_bytes_per_step = 2 * layers * kv_heads * head_dim * 2 * average_attended_tokens * batch
            analytical_read_bytes_per_step = linear_weight_bytes + kv_read_bytes_per_step
            intensity = proxy_flops_per_step / analytical_read_bytes_per_step
            timings = [t['wall_ms'] / timed_steps / 1000 for t in case['trials']]
            achieved_tflops_proxy = proxy_flops_per_step / statistics.median(timings) / 1e12
            points.append({'run': str(path), 'software_mode': run.get('mode'),
                           'batch_size': batch, 'context_length': context,
                           'timed_steps': timed_steps, 'average_attended_tokens': average_attended_tokens,
                           'median_wall_s_per_batch_step': statistics.median(timings),
                           'proxy_flops_per_batch_step': proxy_flops_per_step,
                           'analytical_weight_read_bytes_per_batch_step': linear_weight_bytes,
                           'analytical_kv_read_bytes_per_batch_step': kv_read_bytes_per_step,
                           'analytical_intensity_flops_per_read_byte': intensity,
                           'achieved_tflops_from_proxy_opcount': achieved_tflops_proxy,
                           'control_envelope_tflops': min(compute_tflops, copy_gb_s * intensity / 1000),
                           'measured_hbm_bytes': 'NA', 'measured_flops': 'NA',
                           'classification': 'hybrid-measured-time-analytical-opcount-and-reads'})
    result = {'classification': 'measured-control-ceilings-with-analytical-decode-proxies',
              'limitations': 'BF16 GEMM and useful-byte copy controls are measured, but decode FLOPs and weight/KV reads are analytical. No HBM traffic, cache hit rate, bandwidth saturation, or actual decode arithmetic intensity is established.',
              'ceiling_source': str(a.ceilings), 'compute_ceiling_tflops': compute_tflops,
              'useful_copy_ceiling_gb_s': copy_gb_s,
              'model_config': str(a.model_config), 'tensor_manifest': str(a.tensor_manifest),
              'linear_weight_bytes_counted': linear_weight_bytes, 'points': points}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {len(points)} analytical-proxy decode points')


if __name__ == '__main__':
    main()
