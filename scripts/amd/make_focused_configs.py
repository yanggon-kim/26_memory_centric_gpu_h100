#!/usr/bin/env python3
"""Materialize the plan-2.0 focused AMD cases as explicitly versioned configs (no GPU execution).

Each config holds one (batch, context) case for one engine mode with the common-contract
statistics: 3 warmups, 10 trials per acquisition, 64 generated tokens. Acquisitions are
separate processes launched by the caller. Cases outside the H100 baseline stage
(e.g. 32B C4096, which fits in MI300X's 192 GB) are added here under stage
'amd_focused_v1' rather than by editing the historical stages.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

CASES = [  # model, batch, context, purpose
    ('bridge', 1, 128, 'smoke'),
    ('bridge', 1, 1024, 'small-model low-batch'),
    ('bridge', 8, 4096, 'larger KV and execution demand'),
    ('qwen7b', 1, 4096, 'larger weight stream'),
    ('qwen7b', 8, 4096, 'compute-active counterexample'),
    ('mistral7b', 1, 4096, 'larger weight stream, different GQA layout'),
    ('mistral7b', 8, 4096, 'compute-active counterexample'),
    ('qwen14b', 1, 4096, 'model-scale extension'),
    ('qwen32b', 1, 4096, 'model-scale extension; fits 192 GB'),
]
ENGINES = {'graph': {'enforce_eager': False}, 'eager': {'enforce_eager': True}}


def snapshot(hf_home, model_id, revision):
    path = Path(hf_home) / 'hub' / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
    return path if any(path.glob('*.safetensors')) else None


def main(args):
    package = Path(__file__).resolve().parents[2]
    models = json.loads((package / 'configs/models.json').read_text())['models']
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for name, batch, context, purpose in CASES:
        model = models[name]
        snap = snapshot(args.hf_home, model['model_id'], model['revision'])
        entry = {'model': name, 'batch': batch, 'context': context, 'purpose': purpose}
        if snap is None:
            entry.update(status='unavailable', reason='pinned snapshot not present locally (download failed or gated)')
            index.append(entry)
            continue
        base = args.output / f'{name}_base.json'
        if not base.exists():
            subprocess.run([sys.executable, str(package / 'scripts/materialize_config.py'), '--model', name,
                            '--stage', 'smoke', '--snapshot', str(snap), '--output', str(base)], check=True)
        config = json.loads(base.read_text())
        for engine, overrides in ENGINES.items():
            case = dict(config, batch_sizes=[batch], context_lengths=[context], warmup_trials=3,
                        trials=3 if purpose == 'smoke' else 10,
                        generated_tokens=64, stage='amd_focused_v1')
            case['engine'] = dict(config['engine'], **overrides, max_num_seqs=max(batch, 1),
                                  max_num_batched_tokens=max(8192, batch * context))
            case['purpose'] = purpose
            path = args.output / f'{name}_{engine}_b{batch}_c{context}.json'
            path.write_text(json.dumps(case, indent=2) + '\n')
            index.append(dict(entry, engine=engine, status='materialized', config=str(path)))
    (args.output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(index, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hf-home', default='/work1/proj_blaisetine/yanggon/hf')
    main(p.parse_args())
