"""Prospective paired open-loop arrivals and request accounting.

This module supplies no GPU qualification. A fresh policy/reference pilot,
instrumentation assessment and independent acquisitions are still required.
Arrival times never move when an engine step or admission is late.
"""
import hashlib
import json
import math
import random


def make_trace(seed, rate, requests=24, split='judging'):
    assert isinstance(seed, int) and math.isfinite(rate) and rate > 0
    assert isinstance(requests, int) and requests >= 12
    assert split in ('development', 'judging', 'functional', 'warmup')
    rng = random.Random(seed)
    arrivals = [0.]
    for _ in range(requests - 1):
        arrivals.append(arrivals[-1] + rng.expovariate(rate))
    trace = {'version': 'poisson_open_loop_v1', 'seed': seed,
             'split': split, 'configured_rate_requests_s': rate,
             'offered_s': arrivals, 'request_count': requests,
             'prompt_policy': 'same pinned synthetic prompt for all requests; prefix caching disabled',
             'generated_tokens_per_request': 64}
    trace['sha256'] = trace_digest(trace)
    return trace


def trace_digest(trace):
    return hashlib.sha256(json.dumps({k: v for k, v in trace.items() if k != 'sha256'},
                                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def trial_seed(acquisition, trial, split):
    """Disjoint development/judging streams; all policies reuse each stream."""
    assert acquisition in range(3) and trial in range(10)
    assert split in ('development', 'judging', 'warmup')
    return {'development': 202611270000, 'judging': 202611280000,
            'warmup': 202611290000}[split] + acquisition * 100 + trial


def requests_for(trace):
    assert trace_digest(trace) == trace['sha256'], 'arrival trace changed'
    arrivals = trace['offered_s']
    assert len(arrivals) == trace['request_count'] and arrivals[0] == 0
    assert all(math.isfinite(v) and v >= 0 for v in arrivals)
    assert all(a < b for a, b in zip(arrivals, arrivals[1:]))
    return {f'r{i:04d}': {'offered_s': offered, 'admitted_s': None,
                         'output_ids': [], 'token_observed_s': []}
            for i, offered in enumerate(arrivals)}


def record_output(record, output, observed_s, expected):
    assert math.isfinite(observed_s)
    previous = record['output_ids']
    ids = list(output.outputs[0].token_ids)
    assert ids[:len(previous)] == previous and len(ids) == len(previous) + 1
    assert ids == expected[:len(ids)], 'greedy output differs from serial reference'
    assert record['admitted_s'] is not None
    assert observed_s >= record['admitted_s'] >= record['offered_s']
    times = record['token_observed_s']
    assert not times or observed_s >= times[-1]
    record['output_ids'] = ids
    times.append(observed_s)
    if len(ids) == 1:
        record.update(first_token_s=observed_s,
                      ttft_s=observed_s-record['offered_s'],
                      admission_delay_s=record['admitted_s']-record['offered_s'],
                      admitted_to_first_token_s=observed_s-record['admitted_s'])
    if output.finished:
        assert len(ids) == len(expected) == 64
        assert output.outputs[0].finish_reason == 'length'
        record['finished_s'] = observed_s
        record['inter_token_s'] = [b-a for a, b in zip(times, times[1:])]
        assert len(record['inter_token_s']) == 63
    else:
        assert len(ids) < len(expected)


def summarize(requests, end_s, slo):
    """Retain offered denominator and censored requests; no steady-state claim."""
    assert set(slo) == {'ttft_s', 'maximum_inter_token_s'}
    assert all(math.isfinite(v) and v > 0 for v in slo.values())
    assert math.isfinite(end_s) and end_s > 0 and requests
    completed = [r for r in requests.values() if 'finished_s' in r]
    for r in completed:
        assert len(r['output_ids']) == 64 and len(r['inter_token_s']) == 63
        assert r['finished_s'] <= end_s
    compliant = [r for r in completed if r['ttft_s'] <= slo['ttft_s']
                 and max(r['inter_token_s']) <= slo['maximum_inter_token_s']]
    return {'offered_requests': len(requests), 'admitted_requests': sum(
                r['admitted_s'] is not None for r in requests.values()),
            'completed_requests': len(completed),
            'unfinished_request_ids': [rid for rid, r in requests.items() if 'finished_s' not in r],
            'compliant_completed_requests': len(compliant), 'slo': dict(slo),
            'observation_wall_s': end_s,
            'completed_requests_per_s_including_drain': len(completed)/end_s,
            'compliant_requests_per_s_including_drain': len(compliant)/end_s,
            'generated_tokens_completed_requests': len(completed)*64,
            'all_requests_completed': len(completed) == len(requests),
            'steady_state_demonstrated': False,
            'scope': 'instrumented finite offered trace including drain; retain per-request TTFT/63ITLs, admission delay and unfinished requests; not automatically clean or steady-state serving'}
