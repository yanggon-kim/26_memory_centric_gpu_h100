#!/usr/bin/env python3
"""Figures for docs/amd/lc_campaign_results_20261124.md from the published lc analysis tables.

  lc_figures.py --lc1 results_published/amd_20261124_mi300x_lc_lc1/lc_tables.json [--lc2 ...] [--lc3 lc3/lc_tables.json lc3fx/lc_tables.json]
                --out docs/amd/figs/lc
Needs matplotlib (a plotting venv; not the vLLM container). Static PNGs, light surface, fixed categorical order.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SER = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'lines.linewidth': 2, 'lines.markersize': 5, 'legend.frameon': False})


def line(ax, xs, ys, i, label):
    ax.plot(xs, ys, color=SER[i], marker='o', label=label, markeredgecolor='#fcfcfb', markeredgewidth=1.5)
    ax.annotate(label, (xs[-1], ys[-1]), xytext=(4, 0), textcoords='offset points', color=INK2, va='center', fontsize=8)


def fig_l1(t, out):
    rows = {}
    for r in t['l1']['ratios']:
        if r['metric'] == 'ttft_s':
            rows.setdefault(r['arm'], []).append((r['L'], r['ratio_median']))
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, arm in enumerate(('mix256', 'mix1024', 'mix4096')):
        v = sorted(rows[arm])
        line(ax, [x / 1000 for x, _ in v], [y for _, y in v], i, f'B = {arm[3:]}')
    ax.axhline(1, color=INK2, lw=1, ls='--')
    ax.set_xlabel('prompt length (K tokens)')
    ax.set_ylabel('TTFT chunked / whole')
    ax.set_title('L1: chunked vs whole prefill, Qwen2.5-32B, MI300X', fontsize=9, loc='left', color=INK)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 36)
    fig.tight_layout()
    fig.savefig(out / 'l1_chunked_over_whole.png', dpi=150)


def fig_l2(t, out):
    cells = [r for r in t['l2']['profile'] if r['role'] == 'ref' and r['C'] == 0 and r['N'] and r['P'] == 4096]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for i, N in enumerate(sorted({r['N'] for r in cells})):
        v = sorted((r['S'], r['decode_attn'], r['decode_attn'] / r['busy_ms']) for r in cells if r['N'] == N)
        if len(v) < 2:
            continue
        line(axs[0], [s / 1000 for s, _, _ in v], [a for _, a, _ in v], i, f'N = {N}')
        line(axs[1], [s / 1000 for s, _, _ in v], [100 * f for _, _, f in v], i, f'N = {N}')
    axs[0].set_ylabel('decode attention per step (ms)')
    axs[1].set_ylabel('decode attention share of step busy time (%)')
    for ax in axs:
        ax.set_xlabel('resident context per request S (K tokens)')
        ax.set_xlim(0, 36)
        ax.legend(loc='upper left')
    axs[0].set_title('L2: decode-only steps (CUDA graphs)', fontsize=9, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(out / 'l2_decode_attention.png', dpi=150)


def fig_pd(points, out):
    """points: (label, effective P:D, pure-decode exec share)"""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    groups = {}
    for g, x, y in points:
        groups.setdefault(g, []).append((x, y))
    for i, (g, v) in enumerate(groups.items()):
        v.sort()
        ax.plot([x for x, _ in v], [100 * y for _, y in v], color=SER[i], marker='o', ls='none', label=g,
                markeredgecolor='#fcfcfb', markeredgewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('effective prompt : decode tokens')
    ax.set_ylabel('pure-decode share of execution time (%)')
    ax.set_title('L3: pure-decode time share vs P:D', fontsize=9, loc='left', color=INK)
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(out / 'l3_decode_share_vs_pd.png', dpi=150)


POL_ORDER = ['whole', 'sep2048', 'mix256', 'mix512', 'mix1024', 'mix2048', 'mix4096', 'mix8192',
             'colo4_whole', 'colo4_mix2048', 'd1p3', 'd2p2', 'd3p1']
# separate policies in neutral ink (whole dashed so a tie with sep2048 stays visible); mixed budgets in fixed hue order
POL_STYLE = {'whole': (INK, '--'), 'sep2048': (INK2, '-'), 'mix256': (SER[0], '-'), 'mix512': (SER[2], '-'),
             'mix1024': (SER[3], '-'), 'mix2048': (SER[1], '-'), 'mix4096': (SER[6], '-'), 'mix8192': (SER[4], '-'),
             # amendment 5: colocated four-GPU deployments keep their single-GPU policy colour; disaggregated ratios
             'colo4_whole': (INK, '--'), 'colo4_mix2048': (SER[1], '-'), 'd1p3': (SER[0], '-'), 'd2p2': (SER[2], '-'),
             'd3p1': (SER[6], '-')}


def fig_goodput(series, out, name, rows=(('goodput_rps', 'SLO goodput (rps)'),)):
    """series: {workload: {policy: {key: [(rate, value)]}}}; one row of panels per key"""
    wls = list(series)
    fig, axs = plt.subplots(len(rows), len(wls), figsize=(3.2 * len(wls), 2.8 * len(rows)), squeeze=False)
    for i, (key, lab) in enumerate(rows):
        for ax, w in zip(axs[i], wls):
            for p in sorted(series[w], key=lambda p: POL_ORDER.index(p) if p in POL_ORDER else 99):
                by = {}  # mean over seeds at each offered rate
                for x, y in series[w][p].get(key, []):
                    by.setdefault(x, []).append(y or 0)
                v = sorted((x, sum(ys) / len(ys)) for x, ys in by.items())
                if not v:
                    continue
                c, ls = POL_STYLE.get(p, (SER[7], '-'))
                ax.plot([x for x, _ in v], [y for _, y in v], color=c, ls=ls, marker='o', label=p, markeredgecolor='#fcfcfb',
                        markeredgewidth=1.2, lw=1.6 if ls == '-' else 2.2)
            ax.set_title(f'{w}' + ('' if i == 0 else ' (max ITL <= 1 s)'), fontsize=9, loc='left', color=INK)
            if i == len(rows) - 1:
                ax.set_xlabel('offered load (rps)')
        axs[i][0].set_ylabel(lab)
    hl = {}  # union over panels: a series present in one workload only still gets a legend entry
    for ax in axs.flat:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(lb, h)
    order = sorted(hl, key=lambda p: POL_ORDER.index(p) if p in POL_ORDER else 99)
    axs[0][-1].legend([hl[k] for k in order], order, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f'{name}.png', dpi=150)


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    t1 = json.loads(a.lc1.read_text())
    fig_l1(t1, a.out)
    fig_l2(t1, a.out)
    pts = [('two-request tail (lc1)', (128 + c['I']) / (c['O'] + (256 if c['I'] else 0)), c['decode_frac_exec'])
           for c in t1['l3a']['cases'] if c['policy'] == 'mix2048']
    extra = [json.loads(p.read_text()) for p in [a.lc2] + (a.lc3 or []) if p]
    # an lc3fx re-run (<step>_fx, amendment 4) replaces the failed original step
    redone = {r['step'][:-3] for t in extra for r in ((t.get('serving') or {}).get('rows', []) if isinstance(t.get('serving'), dict) else [])
              if r.get('step', '').endswith('_fx')}
    good = {}
    for t in [t1] + extra:
        for w, ladder in (t.get('pilot', {}).get('ladder') or {}).items():
            pts += [(f'serving, stock B 2048', r['effective_p_to_d'], r['step_fraction_time'].get('decode', 0))
                    for r in ladder if r.get('effective_p_to_d')]
        sv = t.get('serving') or {}
        for r in (sv.get('rows', []) if isinstance(sv, dict) else sv):
            if r.get('step') in redone:
                continue
            if r.get('model', 'q32') == 'q32' and r.get('effective_p_to_d') and r.get('policy') in ('mix2048', 'whole'):
                pts.append((f"serving, {r['policy']}", r['effective_p_to_d'], r['step_fraction_time'].get('decode', 0)))
            g = good.setdefault(r['experiment'], {}).setdefault(r['workload'], {}).setdefault(r['policy'], {})
            for k in ('goodput_rps', 'goodput_stall1000_rps'):
                if k in r:
                    g.setdefault(k, []).append((r['rate'], r[k]))
    fig_pd(pts, a.out)
    for exp, s in good.items():
        fig_goodput(s, a.out, f'goodput_{exp}', (('goodput_rps', 'SLO goodput (rps)'),
                                                  ('goodput_stall1000_rps', 'goodput, stall cap (rps)')))
    if a.ld:  # amendment 5/6: ld tables (crashed steps already replaced by their _fx re-runs)
        s = {}
        for r in json.loads(a.ld.read_text())['rows']:
            g = s.setdefault(r['workload'], {}).setdefault(r['policy'], {})
            for k in ('goodput_rps', 'goodput_stall1000_rps'):
                g.setdefault(k, []).append((r['rate'], r[k]))
        fig_goodput(s, a.out, 'goodput_ld', (('goodput_rps', 'SLO goodput, 4 GPUs (rps)'),
                                             ('goodput_stall1000_rps', 'goodput, stall cap (rps)')))
    print('figures ->', a.out)


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--lc1', type=Path, required=True)
    q.add_argument('--lc2', type=Path)
    q.add_argument('--lc3', type=Path, nargs='+', help='lc3 tables, then the lc3fx re-run tables')
    q.add_argument('--ld', type=Path, help='ld_tables.json (lc_analyze.py ld)')
    q.add_argument('--out', type=Path, required=True)
    main(q.parse_args())
