#!/usr/bin/env python3
"""Strictly compare completed vLLM token IDs with a matching HF bridge case."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vllm-run', type=Path, required=True)
    p.add_argument('--hf-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    vllm = json.loads(a.vllm_run.read_text())
    hf = json.loads(a.hf_run.read_text())
    assert vllm['status'] == 'complete' and hf['status'] == 'complete'
    assert vllm['config']['model_revision'] == hf['config']['model_revision']
    assert vllm['config']['precision'] == hf['config']['precision'] == 'BF16'
    reference = {(c['batch_size'], c['context_length'], c['generated_tokens']): c
                 for c in hf['cases'] if c['outputs_identical'] and c['compiled_correctness']['passed']}
    rows = []
    for case in vllm['cases']:
        key = (case['batch_size'], case['context_length'], case['generated_tokens'])
        if key not in reference:
            continue
        control = reference[key]
        if case['prompt_sha256'] != control['prompt_sha256']:
            rows.append({'case': key, 'status': 'failed', 'reason': 'prompt SHA256 mismatch',
                         'vllm_prompt_sha256': case['prompt_sha256'],
                         'hf_prompt_sha256': control['prompt_sha256']})
            continue
        expected = control['trials'][0]['output_ids']
        assert len(case['trials']) >= 3 and len(control['trials']) >= 3
        trial_rows = []
        for trial_index, trial in enumerate(case['trials']):
            actual = trial['output_ids']
            assert len(actual) == len(expected) and all(len(a) == len(e) for a, e in zip(actual, expected))
            differences = [{'request': i, 'token_index': j, 'expected': expected[i][j],
                            'actual': actual[i][j]}
                           for i in range(len(expected)) for j in range(len(expected[i]))
                           if actual[i][j] != expected[i][j]]
            trial_rows.append({'trial': trial_index, 'identical': not differences,
                               'difference_count': len(differences),
                               'first_differences': differences[:20]})
        rows.append({'case': key, 'status': 'passed' if all(x['identical'] for x in trial_rows) else 'failed',
                     'prompt_sha256': case['prompt_sha256'], 'trials': trial_rows})
    assert rows, 'no matching completed case found; do not claim cross-engine correctness'
    result = {'classification': 'cross-engine-token-correctness',
              'vllm_run': str(a.vllm_run), 'hf_run': str(a.hf_run),
              'status': 'passed' if all(r['status'] == 'passed' for r in rows) else 'failed',
              'cases': rows,
              'note': 'Exact generated token ID equality for identical prompts, BF16 model revision, greedy EOS-ignored output. Differences are preserved and never tolerated away.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'], len(rows), 'matching cases')
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
