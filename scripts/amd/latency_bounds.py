#!/usr/bin/env python3
"""E/F bounded latency decomposition and J9 c/r/f sensitivity for the focused MI300X cases.

Inputs (all from this allocation): T = graph-mode decode ms/step (M1 summary); F = useful FLOPs/step
(analytical: 2*B*linear parameter elements + 4*B*Hq*d*L_hist per layer); D = bytes/step (counter
FETCH_SIZE+WRITE_SIZE where a qualified capture exists, otherwise logical U_tick, labelled);
P = measured register-resident BF16 MFMA rate; beta = measured large-footprint streaming read rate.
Framework (common contract): T = C + M - X + O, C = F/P, M = D/beta, 0 <= X <= min(C,M), O >= 0;
O_feasible = [max(0, T-C-M), T-max(C,M)]. An empty interval rejects the chosen rates.
P and beta are independent microbenchmark controls (no circular D/T substitution); both are optimistic
service rates, so C and M are lower bounds on the time those resources need.
Sensitivity: C_new = C/c, M_new = M*(1-f)/r, with O and X either held (same-O/same-X, conditional)
or bounded by the free-overlap envelope [max(C_new,M_new)+O, C_new+M_new+O].
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_counters import safetensor_bytes


def flops_per_step(cfg_path, step=5):
    cfg = json.loads(Path(cfg_path).read_text())
    snap = Path(cfg['model_path'])
    mc = json.loads((snap / 'config.json').read_text())
    sizes = safetensor_bytes(snap)
    b, c = cfg['batch_sizes'][0], cfg['context_lengths'][0]
    emb = sum(v for k, v in sizes.items() if 'embed_tokens' in k)
    head = sum(v for k, v in sizes.items() if k.startswith('lm_head'))
    norms = sum(v for k, v in sizes.items() if 'norm' in k or k.endswith('bias'))
    tied = mc.get('tie_word_embeddings', False) or head == 0
    linear_bytes = sum(sizes.values()) - emb - head - norms + (emb if tied else head)
    hq, d = mc['num_attention_heads'], mc.get('head_dim') or mc['hidden_size'] // mc['num_attention_heads']
    layers, hkv = mc['num_hidden_layers'], mc.get('num_key_value_heads', hq)
    hist = c + step + 1
    f_linear = 2 * b * linear_bytes / 2
    f_attn = 4 * b * hq * d * hist * layers
    weights = sum(sizes.values()) - emb - head + (emb if tied else head) + b * mc['hidden_size'] * 2
    kv = layers * 2 * b * hist * hkv * d * 2
    return {'F_flops': f_linear + f_attn, 'F_linear': f_linear, 'F_attention': f_attn,
            'U_tick_bytes': weights + kv, 'batch': b, 'context': c}


def main(args):
    m1 = json.loads(args.m1.read_text())['conditions']
    svc = {(r['condition'], r['cus']): r for r in json.loads(args.service.read_text())['conditions']}
    counters = json.loads(args.counters.read_text())['cases'] if args.counters.exists() else {}
    P = svc[('mfma_iters20000_reps1', 304)]['mean'] * 1e12
    beta = svc[('read_bytes4294967296_passes1_reps5', 304)]['mean'] * 1e9
    out = {'classification': 'analytical bounds from measured inputs', 'scope': __doc__.strip(),
           'P_bf16_flops_per_s': P, 'beta_read_bytes_per_s': beta, 'cases': {}}
    grid = {'c': [1, .75, .5, .25], 'r': [.7, 1, 1.2, 1.4], 'f': [0, .05, .1, .2]}
    for name, cond in m1.items():
        if '_graph_' not in name or not cond.get('mean_ms_per_step') or name.startswith('smoke'):
            continue
        cfg = args.configs / f'{name}.json'
        if not cfg.exists():
            continue
        w = flops_per_step(cfg)
        eager = name.replace('_graph_', '_eager_')
        D, d_source = w['U_tick_bytes'], 'logical U_tick (analytical)'
        steps = counters.get(eager, {}).get('steps', [])
        if steps and all('read_bytes' in s for s in steps):
            D = sum(s['read_bytes'] + s.get('write_bytes', 0) for s in steps) / len(steps)
            d_source = f'counters FETCH_SIZE+WRITE_SIZE ({eager}, ticks 4-6, eager program)'
        T = cond['mean_ms_per_step'] / 1e3
        C, M = w['F_flops'] / P, D / beta
        lo, hi = max(0.0, T - C - M), T - max(C, M)
        row = {'T_s': T, 'T_ci95_ms': cond.get('ci95_ms_per_step'), 'C_s': C, 'M_s': M, 'D_bytes': D, 'D_source': d_source,
               'F_flops': w['F_flops'], 'O_feasible_s': [lo, hi] if lo <= hi else None,
               'feasible': lo <= hi, 'M_over_T': M / T, 'C_over_T': C / T,
               'bandwidth_time_fraction_upper': M / T}
        if lo <= hi:
            O_mid = lo  # most overlap-free reading: X = C + M + O - T at O = lo
            X = C + M + O_mid - T
            sens = []
            for c in grid['c']:
                for r in grid['r']:
                    for f in grid['f']:
                        Cn, Mn = C / c, M * (1 - f) / r
                        same = Cn + Mn - min(X, Cn, Mn) + O_mid
                        sens.append({'c': c, 'r': r, 'f': f, 'T_same_O_X_s': same,
                                     'T_envelope_s': [max(Cn, Mn) + lo, Cn + Mn + hi],
                                     'speedup_same_O_X': T / same, 'memory_service_improves': f > 1 - r})
            row['sensitivity'] = sens
        out['cases'][name] = row
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out['cases'].items():
        print(f"{k:28s} T={v['T_s']*1e3:.3f}ms C={v['C_s']*1e3:.3f} M={v['M_s']*1e3:.3f} M/T={v['M_over_T']:.2f} "
              f"O=[{', '.join(f'{x*1e3:.3f}' for x in v['O_feasible_s']) if v['feasible'] else 'EMPTY'}] D:{v['D_source'][:8]}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m1', type=Path, required=True)
    p.add_argument('--service', type=Path, required=True)
    p.add_argument('--counters', type=Path, required=True)
    p.add_argument('--configs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
