"""Chunked-prefill campaign: deterministic open-loop request traces (experiments 2, 3, 5).

A trace is fully determined by (workload, arrival process, mean rate, request count, seed): request ids, prompt and
output lengths and prompt contents come from the seed alone, so every arrival process and rate replays the identical
request set; timestamps come from (seed, arrival, rate). trace_sha256 covers ids, lengths, contents and timestamps.
Development seeds (tuning) and held-out seeds (judging) are disjoint (DEV_SEEDS, HELDOUT_SEEDS).

Workloads (lengths uniform over the closed ranges; P:D = mean prompt / mean output tokens):
  dheavy    prompt 128-512,   output 256-768   P:D ~ 0.6   (decode-heavy: under one prompt token per decode tick)
  balanced  prompt 512-2048,  output 128-384   P:D ~ 5
  pheavy    prompt 2048-6144, output 32-128    P:D ~ 51    (prefill-heavy)
  lo512 / lo2048 / lo4096   prompt 128-512, output 384-640 / 1536-2560 / 3072-5120   P:D ~ 0.62 / 0.16 / 0.08 (long output)
  lctx      prompt 8192-16384, output 512-1536 P:D ~ 12   (long resident context)
  lprompt   prompt 16384-30720, output 32-256  P:D ~ 163  (long uncached prompt)
  prefix    shared-structure workload of experiment 3: every request = its own prefix (prefix_len) + fixed suffix,
            fixed output
Arrivals: 'poisson' (exponential gaps) or 'bursty' (gamma gaps, coefficient of variation 4, same mean).
"""
import hashlib
import json
import random

WORKLOADS = {'dheavy': ((128, 512), (256, 768)), 'balanced': ((512, 2048), (128, 384)),
             'pheavy': ((2048, 6144), (32, 128)),
             # long-context campaign (docs/amd/lc_campaign_design_20261124.md section 7); added 2026-11-24, earlier
             # workloads and their traces are unchanged
             'lo512': ((128, 512), (384, 640)), 'lo2048': ((128, 512), (1536, 2560)), 'lo4096': ((128, 512), (3072, 5120)),
             'lctx': ((8192, 16384), (512, 1536)), 'lprompt': ((16384, 30720), (32, 256))}
DEV_SEEDS = (101, 102)
HELDOUT_SEEDS = (201, 202, 203)
BURST_CV = 4.0


def lengths(workload, n, seed):
    (p0, p1), (o0, o1) = WORKLOADS[workload]
    r = random.Random(f'{seed}:{workload}:lengths')
    return [(r.randint(p0, p1), r.randint(o0, o1)) for _ in range(n)]


def arrivals(arrival, rate, n, seed):
    r = random.Random(f'{seed}:{arrival}:{rate}:arrivals')
    t, out = 0.0, []
    for _ in range(n):
        if arrival == 'poisson':
            gap = r.expovariate(rate)
        elif arrival == 'bursty':
            k = 1 / BURST_CV ** 2
            gap = r.gammavariate(k, 1 / (k * rate))
        else:
            raise ValueError(arrival)
        t += gap
        out.append(t)
    return [x - out[0] for x in out]


def make(workload, arrival, rate, n, seed, vocab, prefix_len=0, suffix=256, output=128):
    """Returns list of {id, arrival_s, prompt_ids, max_tokens, prefix_len} and the trace digest."""
    rc = random.Random(f'{seed}:{workload}:{prefix_len}:contents')
    if workload == 'prefix':
        lens = [(prefix_len + suffix, output)] * n
    else:
        lens = lengths(workload, n, seed)
    times = arrivals(arrival, rate, n, seed)
    reqs = []
    for i, ((p, o), t) in enumerate(zip(lens, times)):
        ids = [rc.randrange(vocab) for _ in range(p)]
        reqs.append({'id': f'{workload}_s{seed}_r{i}', 'arrival_s': t, 'prompt_ids': ids, 'max_tokens': o,
                     'prefix_len': prefix_len if workload == 'prefix' else 0})
    h = hashlib.sha256(json.dumps([[q['id'], round(q['arrival_s'], 9), q['max_tokens'], q['prompt_ids']] for q in reqs]).encode())
    return reqs, h.hexdigest()


def content_digest(reqs):
    """Digest of the request set without timestamps (identical across arrival processes and rates)."""
    return hashlib.sha256(json.dumps([[q['id'], q['max_tokens'], q['prompt_ids']] for q in reqs]).encode()).hexdigest()


def demand(reqs):
    p = sum(len(q['prompt_ids']) for q in reqs)
    o = sum(q['max_tokens'] for q in reqs)
    return {'requests': len(reqs), 'prompt_tokens': p, 'output_tokens': o, 'p_to_d': p / o,
            'span_s': reqs[-1]['arrival_s'] if reqs else 0}
