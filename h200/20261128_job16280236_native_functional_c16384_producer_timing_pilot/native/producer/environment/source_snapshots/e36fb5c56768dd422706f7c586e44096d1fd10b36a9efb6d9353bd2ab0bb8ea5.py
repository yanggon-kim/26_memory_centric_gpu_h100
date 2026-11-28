#!/usr/bin/env python3
"""Portable AMD request inputs for future H200 serving, without AMD outcomes.

The adjacent generator is byte-identical to cp_trace.py at AMD_COMMIT. Rates
are floats, as in AMD's argparse interface; their string representation seeds
the arrival RNG. This adapter does not drive a GPU or qualify a measurement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

AMD_COMMIT = '8c4cc5b9026342c084365be4edc1c33d36221106'
AMD_PATH = 'scripts/amd/cp/cp_trace.py'
GENERATOR_SHA = '75a1e42657498055d49a6e1e5b56bb3bf8891931cfa20d6eb5c557a26c5f1821'
VOCAB = 152064
MAX_POSITIONS = 32768
MODEL_REVISION = '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'


def generator():
    path = Path(__file__).with_name('h200_amd_trace_source.py')
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == GENERATOR_SHA, 'portable generator source changed'
    # Execute the exact checked bytes, rather than a possibly stale .pyc file.
    namespace = {'__name__': 'h200_pinned_amd_generator', '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace


def digest(record):
    return hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != 'sha256'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_trace(workload, arrival, rate, requests, seed, split):
    src = generator()
    assert workload in src['WORKLOADS'], 'prefix/history requests need a separately declared cache protocol'
    assert arrival in ('poisson', 'bursty')
    assert split in ('development', 'judging')
    allowed = src['DEV_SEEDS'] if split == 'development' else src['HELDOUT_SEEDS']
    assert type(seed) is int and seed in allowed, 'development/judging seed leakage'
    assert type(requests) is int and requests >= 30, 'first ten percent must contain at least three warmups'
    assert not isinstance(rate, bool) and math.isfinite(rate) and rate > 0
    rate = float(rate)
    rows, amd_digest = src['make'](workload, arrival, rate, requests, seed, VOCAB)
    assert len(rows) == requests and rows[0]['arrival_s'] == 0
    assert len({r['id'] for r in rows}) == requests
    # A gamma CV4 generator can round consecutive timestamps to the same float.
    # Preserve these ties and original request order; do not add epsilon gaps.
    assert all(math.isfinite(r['arrival_s']) and r['arrival_s'] >= 0 for r in rows)
    assert all(a['arrival_s'] <= b['arrival_s'] for a, b in zip(rows, rows[1:]))
    for row in rows:
        assert row['prefix_len'] == 0 and row['prompt_ids'] and row['max_tokens'] >= 2
        assert len(row['prompt_ids'])+row['max_tokens'] <= MAX_POSITIONS
    trace = {'version': 'H200_portable_AMD_requests_v1',
        'classification': 'deterministic workload input; no measurement or prediction',
        'AMD_reference_commit': AMD_COMMIT, 'AMD_generator_path': AMD_PATH,
        'AMD_generator_sha256': GENERATOR_SHA,
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': sys.version, 'workload': workload, 'arrival': arrival,
        'configured_rate_requests_s': rate, 'rate_rng_seed_text': str(rate),
        'seed': seed, 'split': split, 'request_count': requests,
        'AMD_trace_sha256': amd_digest, 'content_sha256': src['content_digest'](rows),
        'requests': rows, 'demand': src['demand'](rows),
        'warmup_requests': int(requests*.10),
        'warmup_rule': 'First ten percent in original offered order; retained in raw offered/attempted counts and excluded only from the declared measured-window SLO denominator.',
        'prompt_generation': 'Unmodified AMD uniform token IDs; no text tokenizer substitution, prefix caching disabled.',
        'model': {'id': 'Qwen/Qwen2.5-32B-Instruct', 'revision': MODEL_REVISION,
                  'vocab_size': VOCAB, 'max_positions': MAX_POSITIONS,
                  'precision': 'BF16 weights and KV', 'tensor_parallel_size': 1},
        'scope': 'Same generator inputs permit workload matching. H200 scheduling, transport, correctness, isolation, actual SM grants, clocks, latency, goodput and energy require fresh measurements. No AMD threshold or capacity is adopted.'}
    trace['sha256'] = digest(trace)
    return trace


def validate(trace):
    assert trace['version'] == 'H200_portable_AMD_requests_v1'
    assert digest(trace) == trace['sha256'], 'sealed portable trace changed'
    regenerated = make_trace(trace['workload'], trace['arrival'], trace['configured_rate_requests_s'],
        trace['request_count'], trace['seed'], trace['split'])
    # Runtime metadata may differ on a reader; all actual inputs and hashes must not.
    for key in ('AMD_reference_commit', 'AMD_generator_path', 'AMD_generator_sha256',
                'requests', 'AMD_trace_sha256', 'content_sha256', 'demand',
                'rate_rng_seed_text', 'model', 'warmup_requests', 'warmup_rule',
                'prompt_generation'):
        assert trace[key] == regenerated[key], 'portable input mismatch: '+key
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', required=True, choices=sorted(generator()['WORKLOADS']))
    parser.add_argument('--arrival', required=True, choices=('poisson', 'bursty'))
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--requests', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--split', required=True, choices=('development', 'judging'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    trace = make_trace(args.workload, args.arrival, args.rate, args.requests, args.seed, args.split)
    validate(trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(trace, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: trace[k] for k in ('sha256', 'AMD_trace_sha256', 'content_sha256',
        'request_count', 'warmup_requests', 'demand')}, indent=2))


if __name__ == '__main__':
    main()
