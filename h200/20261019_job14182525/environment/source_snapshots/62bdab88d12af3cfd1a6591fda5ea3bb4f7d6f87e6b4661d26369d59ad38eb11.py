#!/usr/bin/env python3
"""CPU-only logical tensor/chunk cache study. Never a GPU DRAM trace."""
import argparse
from collections import OrderedDict, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

MiB = 1024 ** 2

class ByteLRU:
    def __init__(self, capacity):
        if capacity<0:raise ValueError('Cache capacity must be nonnegative')
        self.capacity, self.used = capacity, 0
        self.items = OrderedDict()

    def access(self, key, size):
        if key in self.items:
            self.items.move_to_end(key)
            return True
        if size > self.capacity:
            return False
        while self.used + size > self.capacity:
            _, removed = self.items.popitem(last=False)
            self.used -= removed
        self.items[key] = size
        self.used += size
        return False

class PolicyCache:
    """Extra pinned storage is exclusive of unchanged baseline L2."""
    def __init__(self, l2, extra, policy, selected):
        if l2<0 or extra<0:raise ValueError('Baseline and added cache capacities must be nonnegative')
        self.policy = policy
        self.selected = selected
        self.pinned = set()
        self.lru = ByteLRU(l2 + extra if policy == 'lru' else l2)

    def access(self, key, size):
        if self.policy != 'lru' and key in self.selected:
            hit = key in self.pinned
            self.pinned.add(key)
            return hit
        return self.lru.access(key, size)

    def read(self, key, allocation_bytes, requested_bytes):
        # A synthetic block becomes valid only after paying for its full fill.
        hit = self.access(key, allocation_bytes)
        return hit, 0 if hit else allocation_bytes, 0 if hit else requested_bytes

def break_even_fraction(remote_to_dram, overhead_to_dram):
    """f(1 - remote/DRAM) > overhead/DRAM, normalized per baseline synthetic read-fill byte."""
    if not 0 <= remote_to_dram < 1 or overhead_to_dram < 0:
        raise ValueError('Require 0 <= remote/DRAM < 1 and overhead >= 0')
    return overhead_to_dram / (1 - remote_to_dram)

def manifest(snapshot):
    snapshot = Path(snapshot)
    config = json.loads((snapshot / 'config.json').read_text())
    tensors, header_hashes = {}, {}
    for shard in sorted(snapshot.glob('*.safetensors')):
        with shard.open('rb') as handle:
            raw_header = handle.read(struct.unpack('<Q', handle.read(8))[0])
            header = json.loads(raw_header)
            header_hashes[shard.name] = hashlib.sha256(raw_header).hexdigest()
        for name, value in header.items():
            if name == '__metadata__':
                continue
            start, end = value['data_offsets']
            tensors[name] = dict(shape=value['shape'], dtype=value['dtype'], size_bytes=end-start,
                                 shard=shard.name, shard_data_offset=start)
    if not tensors:
        raise ValueError(f'No cached safetensors in {snapshot}')
    if any(tensor['dtype'] not in {'BF16','F16'} for tensor in tensors.values()):
        raise ValueError('Dense BF16-size model requires two-byte checkpoint tensors; derive actual BF16 runtime storage for other formats')
    aliases = {}
    if config.get('tie_word_embeddings') and 'lm_head.weight' not in tensors:
        aliases['lm_head.weight'] = 'model.embed_tokens.weight'
    return dict(snapshot=str(snapshot), config=config, tensors=tensors, aliases=aliases,
                config_sha256=hashlib.sha256((snapshot/'config.json').read_bytes()).hexdigest(),
                safetensors_header_sha256=header_hashes,
                unique_weight_bytes=sum(t['size_bytes'] for t in tensors.values()),
                classification='analytical-derived-from-local-model-metadata')

def regions(model, case, steps):
    """Semantic decode order. No activation/RF or kernel rereads are inferred."""
    conf, tensors = model['config'], model['tensors']
    if conf.get('model_type') not in {'qwen2','mistral'}:
        raise ValueError('Unsupported model_type; validate a family-specific logical access adapter')
    if any('sliding' in kind for kind in (conf.get('layer_types') or [])) or (conf.get('sliding_window') is not None and conf.get('use_sliding_window',True)):
        raise ValueError('Active sliding-window attention needs a validated window-aware access adapter')
    if any('experts' in name for name in tensors):
        raise ValueError('Dense logical model does not support MoE routing; implement and validate a separate expert-access trace')
    if not any('model.layers.0.self_attn.q_proj.weight' == name for name in tensors):
        raise ValueError('Unsupported dense tensor naming/layout; validate adapter before analytical study')
    batch, context = case['batch_size'], case['context_length']
    shared = int(context * case.get('shared_prefix_fraction', 0))
    hidden, layers = conf['hidden_size'], conf['num_hidden_layers']
    heads = conf['num_key_value_heads']
    head_dim = conf.get('head_dim', hidden // conf['num_attention_heads'])
    # This study intentionally fixes BF16 KV, regardless of weight storage dtype.
    kv_row = head_dim * 2
    embed = 'model.embed_tokens.weight'
    row_bytes = tensors[embed]['size_bytes'] // conf['vocab_size']
    def weight(name, step, layer, role='weight'):
        real = model['aliases'].get(name, name)
        if real in tensors:
            return dict(step=step, layer=layer, head='NA', tensor=real, role=role,
                        group='batch', offset=0, size=tensors[real]['size_bytes'],
                        allocation=tensors[real]['size_bytes'], operation='read')
        return None
    for step in range(steps):
        for seq in range(batch):
            token = (step * batch + seq) % conf['vocab_size']
            yield dict(step=step, layer='NA', head='NA', tensor=embed, role='embedding',
                       group='batch', offset=token*row_bytes, size=row_bytes,
                       allocation=tensors[embed]['size_bytes'], operation='read')
        for layer in range(layers):
            base = f'model.layers.{layer}.'
            for suffix in ['input_layernorm.weight', 'self_attn.q_proj.weight', 'self_attn.q_proj.bias',
                           'self_attn.k_proj.weight', 'self_attn.k_proj.bias',
                           'self_attn.v_proj.weight', 'self_attn.v_proj.bias']:
                event = weight(base+suffix, step, layer)
                if event:
                    yield event
            # K/V use unique GQA heads. Shared-prefix blocks physically alias across sequences.
            for seq in range(batch):
                for kind in ['K', 'V']:
                    for head in range(heads):
                        tensor = f'kv.layer{layer}.{kind}.head{head}'
                        suffix_tokens = context - shared + step + 1
                        alloc = (context - shared + steps) * kv_row
                        yield dict(step=step, layer=layer, head=head, tensor=tensor, role='kv_append',
                                   group=f'seq{seq}', offset=(suffix_tokens-1)*kv_row,
                                   size=kv_row, allocation=alloc, operation='write')
                        if shared:
                            yield dict(step=step, layer=layer, head=head, tensor=tensor,
                                       role='kv_prefix', group='shared_prefix', offset=0,
                                       size=shared*kv_row, allocation=shared*kv_row, operation='read')
                        yield dict(step=step, layer=layer, head=head, tensor=tensor, role='kv_suffix',
                                   group=f'seq{seq}', offset=0, size=suffix_tokens*kv_row,
                                   allocation=alloc, operation='read')
            for suffix in ['self_attn.o_proj.weight', 'post_attention_layernorm.weight',
                           'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']:
                event = weight(base+suffix, step, layer)
                if event:
                    yield event
        yield weight('model.norm.weight', step, 'NA')
        yield weight('lm_head.weight', step, 'NA', 'output_projection')

def chunks(event, granularity):
    start, end = event['offset'], event['offset'] + event['size']
    for chunk in range(start // granularity, (end + granularity - 1) // granularity):
        lo = chunk * granularity
        key = (event['tensor'], event['group'], chunk)
        size = min(granularity, event['allocation'] - lo)
        touched = min(end, lo+granularity) - max(start, lo)
        yield key, size, touched

def pin_candidates(model, case, steps, granularity, policy):
    # Order is declared policy, not a fitted oracle. Prefix pinning prioritizes KV;
    # weight pinning starts with small norm/bias tensors then contiguous weight chunks.
    if policy == 'pin_prefix':
        seen = set()
        for event in regions(model, case, steps):
            if event['step'] > 0:
                break
            if event['role'] != 'kv_prefix':
                continue
            for key, size, _ in chunks(event, granularity):
                if key not in seen:
                    seen.add(key)
                    yield key, size
    tensors = model['tensors']
    for name in sorted(tensors, key=lambda name: (tensors[name]['size_bytes'], name)):
        total = tensors[name]['size_bytes']
        for index in range((total+granularity-1)//granularity):
            yield (name, 'batch', index), min(granularity, total-index*granularity)

def select_pins(candidates, capacity):
    selected, used = set(), 0
    for key, size in candidates:
        if used + size > capacity:
            break
        selected.add(key)
        used += size
    return selected, used

def analyze(model, case, capacities, granularity, steps, warmup, l2):
    policies = ['lru', 'pin_weights', 'pin_prefix']
    configurations = []
    for policy in policies:
        candidates = list(pin_candidates(model, case, steps, granularity, policy)) if policy != 'lru' else []
        for extra in capacities:
            pins, used = select_pins(candidates, extra)
            configurations.append((policy, extra, used, PolicyCache(l2, extra, policy, pins), defaultdict(int)))
    logical = defaultdict(int)
    trace_sample = []
    for ordinal, event in enumerate(regions(model, case, steps)):
        if event is None:
            continue
        if len(trace_sample) < 40:
            trace_sample.append(dict(event, classification='analytical', address_kind='synthetic-tensor-offset',
                                     address=f"{event['tensor']}:{event['group']}:{event['offset']}",
                                     size_bytes=event['size'],
                                     logical_timestamp=ordinal, gpu_timestamp_ns='NA', sm='NA', cta='NA',
                                     consumer='decode_batch', share_group=event['group']))
        measured = event['step'] >= warmup
        role = event['role']
        if measured:
            logical[event['operation']] += event['size']
        for key, size, touched in chunks(event, granularity):
            if event['operation'] == 'write':
                # Write-through, no write allocation. Updates do not invalidate resident bytes.
                if measured:
                    for _, _, _, _, counter in configurations:
                        counter['write'] += touched
                continue
            for _, _, _, cache, counter in configurations:
                hit, fill_bytes, miss_payload = cache.read(key, size, touched)
                if measured:
                    counter['read'] += touched
                    counter['hit' if hit else 'miss'] += touched
                    counter['fill'] += fill_bytes
                    counter[f'{role}_read'] += touched
                    counter[f'{role}_miss'] += miss_payload
                    counter[f'{role}_fill'] += fill_bytes
    denominator = (steps-warmup) * case['batch_size']
    baseline = next(counter for policy, extra, _, _, counter in configurations if policy == 'lru' and extra == 0)
    rows, reuse = [], []
    for policy, extra, used, _, counter in configurations:
        saved = baseline['fill'] - counter['fill']
        row = dict(case_id=case['case_id'], model=case['model'], classification='analytical',
                   batch_size=case['batch_size'], context_length=case['context_length'],
                   shared_prefix_fraction=case.get('shared_prefix_fraction', 0),
                   policy=policy, chunk_bytes=granularity, cache_granularity='synthetic_blocks_not_gpu_cachelines',
                   l2_bytes=l2, additional_sram_bytes=extra,
                   pinned_bytes=used, decode_steps=steps, warmup_steps=warmup,
                   logical_read_bytes_per_token=logical['read']/denominator,
                   modeled_offchip_read_bytes_per_token=counter['fill']/denominator,
                   read_fill_bytes_per_token=counter['fill']/denominator,
                   read_miss_payload_bytes_per_token=counter['miss']/denominator,
                   modeled_offchip_write_bytes_per_token=counter['write']/denominator,
                   baseline_offchip_read_bytes_per_token=baseline['fill']/denominator,
                   baseline_read_fill_bytes_per_token=baseline['fill']/denominator,
                   baseline_read_miss_payload_bytes_per_token=baseline['miss']/denominator,
                   incremental_read_bytes_saved_per_token=saved/denominator,
                   incremental_read_fill_bytes_saved_per_token=saved/denominator,
                   incremental_miss_payload_bytes_saved_per_token=(baseline['miss']-counter['miss'])/denominator,
                   incremental_read_fraction_saved=saved/baseline['fill'] if baseline['fill'] else 0,
                   read_hit_fraction=counter['hit']/counter['read'], joules_per_token='NA',
                   latency_ms='NA', notes='synthetic full-block fills; fully associative; write-through/no-write-allocate; no GPU cacheline/PPA/timing claim')
        for role in ['weight', 'output_projection', 'embedding', 'kv_prefix', 'kv_suffix']:
            row[f'{role}_offchip_read_bytes_per_token'] = counter[f'{role}_fill']/denominator
            row[f'{role}_read_miss_payload_bytes_per_token'] = counter[f'{role}_miss']/denominator
            if policy == 'lru':
                reuse.append(dict(case_id=case['case_id'], classification='analytical', role=role,
                                  chunk_bytes=granularity, sampled_cache_capacity_bytes=l2+extra,
                                  read_bytes=counter[f'{role}_read'], miss_bytes=counter[f'{role}_miss'],
                                  read_fill_bytes=counter[f'{role}_fill'],
                                  byte_weighted_hit_fraction=1-counter[f'{role}_miss']/counter[f'{role}_read']
                                  if counter[f'{role}_read'] else 'NA',
                                  interpretation='sampled LRU capacity/hit curve; not exact instruction-level reuse distance'))
        rows.append(row)
    return rows, reuse, trace_sample

def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', action='append', help='Restrict to named case(s).')
    parser.add_argument('--chunk-bytes', type=int, help='Override chunk size for sensitivity runs.')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if settings['l2_bytes']<=0 or any(value<0 for value in settings['additional_sram_mib']):
        raise ValueError('Measured baseline L2 bytes must be positive; added diagnostic capacities must be nonnegative')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    models = {name: manifest(path) for name, path in settings['models'].items()}
    (output/'per_tensor_manifest.json').write_text(json.dumps(models, indent=2)+'\n')
    rows, reuse, traces = [], [], []
    for case in settings['cases']:
        if args.case and case['case_id'] not in args.case:
            continue
        print(f"Analyzing {case['case_id']}", flush=True)
        result, curve, sample = analyze(models[case['model']], case,
                                       [int(x*MiB) for x in settings['additional_sram_mib']],
                                       args.chunk_bytes or settings['chunk_bytes'], settings['decode_steps'],
                                       settings['warmup_steps'], settings['l2_bytes'])
        rows.extend(result)
        reuse.extend(curve)
        traces.extend(dict(event, case_id=case['case_id']) for event in sample)
    if not rows:
        raise ValueError('No cases selected')
    write_csv(output/'traffic_bound_curves.csv', rows)
    write_csv(output/'reuse_distance.csv', reuse)
    with (output/'trace_sample.jsonl').open('w') as handle:
        for event in traces:
            handle.write(json.dumps(event)+'\n')
    energy = [dict(classification='illustrative-dimensionless-sensitivity',
                   remote_energy_over_dram_energy=r, extra_overhead_over_baseline_dram_energy=o,
                   required_incremental_read_fraction=break_even_fraction(r, o))
              for r in [0.05, 0.1, 0.25, 0.5] for o in [0, 0.01, 0.05, 0.1]]
    write_csv(output/'remote_energy_break_even.csv', energy)
    summary = dict(classification='analytical', config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   settings=settings, selected_cases=args.case, chunk_override=args.chunk_bytes,
                   rows=len(rows), measured_dram_bytes=False, gpu_execution=False,
                   read_fill_policy='full_allocated_synthetic_block_on_first_read_miss',
                   write_policy='write_through_no_write_allocate',
                   primary_traffic_metric='full_synthetic_block_read_fill_bytes',
                   python_version=sys.version, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'analysis_metadata.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Wrote {len(rows)} traffic rows to {output}')

if __name__ == '__main__':
    main()
