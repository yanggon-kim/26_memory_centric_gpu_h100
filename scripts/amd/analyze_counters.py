#!/usr/bin/env python3
"""M2: physical L2->memory bytes per decode step vs logical demand, from lane_profile.sh outputs.

Physical: FETCH_SIZE / WRITE_SIZE (KB, summed over dispatches whose start lies in a
decode_step marker) measure traffic leaving the per-XCD L2s toward the memory side.
On MI300X that traffic can still hit the 256 MB Infinity Cache (MALL), which has no
counter here, so these are upper bounds on HBM bytes, not HBM bytes.
Logical (analytical): U_tick = weights read per step + historical KV; computed from the
safetensors headers and model config. Tied embeddings are read in full by the LM head.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import struct

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_trace import classify


def safetensor_bytes(snapshot):
    sizes = {}
    files = sorted(Path(snapshot).glob('*.safetensors'))
    if any(not f.name.startswith('consolidated') for f in files):
        files = [f for f in files if not f.name.startswith('consolidated')]  # Mistral ships both layouts
    for f in files:
        with open(f, 'rb') as h:
            n = struct.unpack('<Q', h.read(8))[0]
            header = json.loads(h.read(n))
        for k, v in header.items():
            if k != '__metadata__':
                sizes[k] = v['data_offsets'][1] - v['data_offsets'][0]
    return sizes


def logical(config_path, step):
    cfg = json.loads(Path(config_path).read_text())
    snap = Path(cfg['model_path'])
    mc = json.loads((snap / 'config.json').read_text())
    sizes = safetensor_bytes(snap)
    if not sizes:
        return None
    total = sum(sizes.values())
    emb = sum(v for k, v in sizes.items() if 'embed_tokens' in k)
    head = sum(v for k, v in sizes.items() if k.startswith('lm_head'))
    b, c = cfg['batch_sizes'][0], cfg['context_lengths'][0]
    hidden = mc['hidden_size']
    layers = mc['num_hidden_layers']
    hkv = mc.get('num_key_value_heads', mc['num_attention_heads'])
    d = mc.get('head_dim') or hidden // mc['num_attention_heads']
    tied = mc.get('tie_word_embeddings', False) or head == 0
    # Decoder + final norm + LM head (tied: the embedding matrix itself) + B embedding rows.
    weights = (total - emb - head) + (emb if tied else head) + b * hidden * 2
    hist_len = c + step + 1
    kv_hist = layers * 2 * b * hist_len * hkv * d * 2
    kv_write = layers * 2 * b * hkv * d * 2
    return {'weights_read_bytes': weights, 'kv_historical_read_bytes': kv_hist, 'kv_new_write_bytes': kv_write,
            'U_tick_bytes': weights + kv_hist, 'checkpoint_bytes': total, 'tied_embeddings': tied,
            'historical_tokens': hist_len, 'layers': layers, 'kv_heads': hkv, 'head_dim': d, 'batch': b, 'context': c,
            'scope': 'analytical logical demand; activations/scratch/metadata omitted'}


def markers(trace_dir):
    f = next(Path(trace_dir).rglob('*marker_api_trace.csv'))
    rows = [r for r in csv.DictReader(open(f)) if re.fullmatch(r'decode_step_\d+', r['Function'])]
    return {int(r['Function'].rsplit('_', 1)[1]): (int(r['Start_Timestamp']), int(r['End_Timestamp'])) for r in rows}


def per_step(group_dir, steps):
    f = next(Path(group_dir).rglob('*counter_collection.csv'), None)
    if f is None:
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        s = int(r['Start_Timestamp'])
        for k, (a, b) in steps.items():
            if a <= s <= b:
                cell = out.setdefault(k, {}).setdefault(r['Counter_Name'], {'total': 0.0, 'by_class': {}})
                v = float(r['Counter_Value'])
                cell['total'] += v
                c = classify(r['Kernel_Name'])
                cell['by_class'][c] = cell['by_class'].get(c, 0.0) + v
                break
    return out


def main(args):
    result = {'classification': 'measured-real-GPU counters (profiled) + analytical logical demand', 'cases': {}}
    for case_dir in sorted(p for p in args.root.glob('*') if p.is_dir()):
        name = case_dir.name
        cfg = args.configs / f'{name}.json'
        groups = {g.name: g for g in case_dir.iterdir() if g.is_dir()}
        row = {'groups_done': sorted(g for g, p in groups.items() if (p / 'done').exists()),
               'groups_failed': sorted(g for g, p in groups.items() if not (p / 'done').exists())}
        merged = {}
        for g, p in groups.items():
            if not (p / 'done').exists() or g == 'trace':
                continue
            try:
                steps = markers(p / 'rocprof')
                counts = per_step(p / 'rocprof', steps)
            except StopIteration:
                continue
            for k, v in (counts or {}).items():
                merged.setdefault(k, {}).update(v)
        sel = {k: v for k, v in merged.items() if k in args.steps}
        table = []
        for k, v in sorted(sel.items()):
            u = logical(cfg, k) if cfg.exists() else None
            entry = {'step': k, 'logical': u}
            if 'FETCH_SIZE' in v:
                entry['read_bytes'] = v['FETCH_SIZE']['total'] * 1024
                entry['read_bytes_by_class'] = {c: b * 1024 for c, b in v['FETCH_SIZE']['by_class'].items()}
                if u:
                    entry['D_read_over_U_tick'] = entry['read_bytes'] / u['U_tick_bytes']
            if 'WRITE_SIZE' in v:
                entry['write_bytes'] = v['WRITE_SIZE']['total'] * 1024
            for counter in ('TCC_HIT', 'TCC_MISS', 'SQ_INSTS_VALU_MFMA_MOPS_BF16', 'SQ_WAVES', 'GRBM_GUI_ACTIVE'):
                if counter in v:
                    entry[counter] = v[counter]['total']
            if 'TCC_HIT' in v and 'TCC_MISS' in v:
                entry['tcc_hit_fraction_of_requests'] = v['TCC_HIT']['total'] / (v['TCC_HIT']['total'] + v['TCC_MISS']['total'])
            table.append(entry)
        row['steps'] = table
        result['cases'][name] = row
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    for name, row in result['cases'].items():
        for e in row['steps']:
            print(name, e['step'], {k: (round(v / 1e9, 4) if isinstance(v, float) and v > 1e6 else v) for k, v in e.items()
                                    if k in ('read_bytes', 'write_bytes', 'D_read_over_U_tick', 'tcc_hit_fraction_of_requests')},
                  'U_GB', round(e['logical']['U_tick_bytes'] / 1e9, 4) if e['logical'] else None)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True, help='results/.../m2_profile')
    p.add_argument('--configs', type=Path, required=True, help='results/.../configs_profile')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[4, 5, 6])
    main(p.parse_args())
