#!/usr/bin/env python3
"""Index attempted H100 cases without discarding failed or partial evidence."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cases = []
    for folder in sorted(Path('results').glob('bridge_*')):
        run = folder / 'run.json'
        if not run.is_file():
            continue
        data = json.loads(run.read_text())
        failure = folder / 'failure.json'
        config = data.get('config', {})
        case_rows = data.get('cases', [])
        shapes = [{'batch_size': c['batch_size'], 'context_length': c['context_length'],
                   'outputs_identical': c.get('outputs_identical'),
                   'measured_trials': len(c.get('trials', []))} for c in case_rows]
        cases.append({'id': folder.name, 'kind': 'software_decode', 'status': data['status'],
                      'classification': data.get('classification'), 'engine': config.get('engine'),
                      'compile': config.get('compile'), 'mode': data.get('mode'),
                      'requested_batch_sizes': config.get('batch_sizes'),
                      'requested_context_lengths': config.get('context_lengths'),
                      'recorded_cases': shapes, 'source': str(run), 'source_sha256': sha(run),
                      'failure_source': str(failure) if failure.is_file() else None,
                      'failure_sha256': sha(failure) if failure.is_file() else None})
    for folder in sorted(Path('results').glob('bridge_eager_kernel_profile*')):
        if (folder / 'run.json').is_file():
            continue
        inventory = folder / 'kernel_inventory.json'
        failure = folder / 'failure.json'
        source = inventory if inventory.is_file() else failure
        if not source.is_file():
            continue
        cases.append({'id': folder.name, 'kind': 'instrumented_kernel_inventory',
                      'status': 'complete' if inventory.is_file() else 'failed',
                      'source': str(source), 'source_sha256': sha(source),
                      'failure_source': str(failure) if failure.is_file() else None,
                      'failure_sha256': sha(failure) if failure.is_file() else None})
    for folder in sorted(Path('results').glob('vllm_*')):
        run = folder / 'run.json'
        if not run.is_file():
            continue
        data = json.loads(run.read_text())
        failure = folder / 'failure.json'
        shapes = [{'batch_size': c['batch_size'], 'context_length': c['context_length'],
                   'output_digests_identical': c.get('output_digests_identical'),
                   'measured_trials': len(c.get('trials', []))} for c in data.get('cases', [])]
        cases.append({'id': folder.name, 'kind': 'optimized_software_decode',
                      'status': data['status'], 'classification': data.get('classification'),
                      'selected_cases': data.get('selected_cases'), 'recorded_cases': shapes,
                      'source': str(run), 'source_sha256': sha(run),
                      'failure_source': str(failure) if failure.is_file() else None,
                      'failure_sha256': sha(failure) if failure.is_file() else None})
    roofline = Path('results/roofline_ceilings_cu128.json')
    if roofline.is_file():
        cases.append({'id': 'roofline_ceilings_cu128', 'kind': 'measured_control_ceilings',
                      'status': 'complete', 'source': str(roofline), 'source_sha256': sha(roofline)})
    profiler = Path('results/profiler_pilot/capabilities.json')
    if profiler.is_file():
        cases.append({'id': 'profiler_pilot', 'kind': 'profiler_capability',
                      'status': 'partial', 'source': str(profiler), 'source_sha256': sha(profiler)})
    attempts = Path('results/runtime_build/attempt_manifest.json')
    if attempts.is_file():
        for attempt in json.loads(attempts.read_text())['attempts']:
            log = Path(attempt['log'])
            cases.append({'id': 'vllm_build_attempt_' + str(attempt['id']),
                          'kind': 'runtime_build', 'status': attempt['status'],
                          'reason': attempt['reason'], 'source': str(log),
                          'source_sha256': sha(log)})
    for folder in sorted(Path('traces').iterdir()):
        if not folder.is_dir() or not any((folder / name).is_file() for name in
                                            ('launch_manifest.json', 'export_manifest.json',
                                             'failure.json', 'capture.log')):
            continue
        export = folder / 'export_manifest.json'
        failure = folder / 'failure.json'
        launch = folder / 'launch_manifest.json'
        capture = folder / 'capture.log'
        source = export if export.is_file() else failure if failure.is_file() else launch if launch.is_file() else capture
        trace_status = ('capture_and_postprocess_verified' if export.is_file() else
                        'partial_failed' if failure.is_file() else
                        'in_progress' if launch.is_file() else
                        'diagnostic_capture_passed_unverified' if 'dynamic micro PASS' in capture.read_text(errors='replace') else
                        'diagnostic_unverified')
        recovery = Path('results') / (folder.name + '_trace_result.json')
        cases.append({'id': folder.name, 'kind': 'native_trace',
                      'status': trace_status,
                      'recovered_after_failure': export.is_file() and failure.is_file(),
                      'recovery_source': str(recovery) if recovery.is_file() else None,
                      'recovery_sha256': sha(recovery) if recovery.is_file() else None,
                      'source': str(source), 'source_sha256': sha(source),
                      'failure_source': str(failure) if failure.is_file() else None,
                      'failure_sha256': sha(failure) if failure.is_file() else None})
    out = {'schema_version': 1, 'target_uuid': 'GPU-18558721-7ba2-20a7-f883-b2c4e2593aff',
           'cases': cases, 'note': 'Recorded attempts only; failed and partial artifacts remain in place.'}
    Path('results/case_manifest.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f'Recorded {len(cases)} attempted cases')


if __name__ == '__main__':
    main()
