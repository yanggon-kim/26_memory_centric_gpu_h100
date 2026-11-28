#!/usr/bin/env python3
"""Saved-data failure audit; no GPU execution or retrospective qualification."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

PACKAGE=Path(__file__).resolve().parents[1]


def main(output):
    rows=[]
    for name,pattern in [('20261019_job14177052','J8*.json'),('20261019_job14182525','J6_eager_b8_c4096_*acq*.json')]:
        root=PACKAGE/'h200'/name
        for p in sorted((root/'profiles').glob(pattern)):
            d=json.loads(p.read_text())
            if d.get('status')!='rejected':continue
            run=root/d.get('run_source','missing')
            if not run.is_file():continue
            native=json.loads(run.read_text());trials=[t for c in native['cases'] for t in c['trials']]
            rows.append({'path':str(p.relative_to(PACKAGE)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                'run_sha256':hashlib.sha256(run.read_bytes()).hexdigest(),'exact_ids':d.get('exact_reference_ids'),
                'clock_gate':d.get('clock_qualification'),
                'trial_before_sm_mhz':[t['hardware_before']['sm_clock_mhz'] for t in trials],
                'trial_after_sm_mhz':[t['hardware_after']['sm_clock_mhz'] for t in trials]})
    root=PACKAGE/'h200/20261019_job14180351'
    energy=json.loads((root/'results/energy_qualification_compact_v1.json').read_text())
    for case in energy['cases']:
        if 'qualified_energy' not in case or case['qualified_energy']:continue
        p=root/'runs'/case['id']/'run.json';d=json.loads(p.read_text());trials=d['cases'][0]['trials']
        rows.append({'path':str(p.relative_to(PACKAGE)),'run_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'exact_ids':case.get('exact_64_token_baseline_prefix'),'clock_gate':case.get('clock_qualification'),
            'trial_before_sm_mhz':[t['hardware_before']['sm_clock_mhz'] for t in trials],
            'trial_after_sm_mhz':[t['hardware_after']['sm_clock_mhz'] for t in trials]})
    diagnostics=[]
    for version in ('v1','v2'):
        p=PACKAGE/f'h200/20261016_job14100402/runs/J1_conditioning_pilot_{version}/run.json'
        d=json.loads(p.read_text());trials=d['cases'][0]['trials']
        diagnostics.append({'source':str(p.relative_to(PACKAGE)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'conditioning_ms':[t['post_prefill_conditioning']['gpu_event_ms'] for t in trials],
            'before_sm_mhz':[t['hardware_before']['sm_clock_mhz'] for t in trials],
            'after_sm_mhz':[t['hardware_after']['sm_clock_mhz'] for t in trials]})
    fit=PACKAGE/'h200/20261017_job14133085/results/J4_completed_v1/shoreline_native_heldout_fit.json'
    model=json.loads(fit.read_text())
    report={'classification':'CPU analysis of saved measurements; no new measurements',
        'clock_rejections':rows,'old_conditioning_pilots':diagnostics,
        'native_fit':{'source':str(fit.relative_to(PACKAGE)),'sha256':hashlib.sha256(fit.read_bytes()).hexdigest(),
            'status':model['status'],'held_out':model['held_out'],
            'decision':'common overlap coefficient fails held-out10% gate; no demonstrated implementation bug. Keep rejection, do not refit to exposed holdout and call it validation. New model requires independent validation.'},
        'C4096':{'decision':'new BF16 static/dynamic SDPA/eager cache-route and common-prefix score diagnostics; preserve exact-ID gate; no C4096 timing expansion'},
        'graph_restriction':'unsupported graph/context route; no safe implementation established, no identical retry',
        'trace':'wait/arrival/TMA semantics and H200 calibration unresolved; no new trace capture',
        'archive':'private destination inaccessible under current credentials; no new upload retry without access change',
        'new_clock_protocol':{'name':'memory_settle_v3','duration_s':1.0,'private_buffers_bytes':2*256*1024**2,
            'scope':'cache-disturbing conditioned decode, not a repair of unconditioned latency or TTFT. No clock/power settings changed. Full original <=5% gate, exact outputs and isolation retained; one pilot per route before3x10 expansion; no retry.'}}
    output.mkdir(parents=True,exist_ok=True)
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Failed/rejected H200 experiment audit','',
        f'Audited {len(rows)} rejected J6/J8/J10 captures. Originals remain rejected.',
        '', 'The repeated lower clock at the start and1980MHz at the end is consistent with a post-prefill DVFS transition, rather than a counter-reader crash. It is not evidence of a driver fault. The two earlier conditioning pilots lasted only about25–60ms and failed.',
        '', 'A new bounded memory-workload precondition lasts one second after the verified prefill barrier, before timed decode. It changes cache history and adds untimed first-token latency. Only this separately labelled program can expand if its pilots pass the unchanged clock/output/isolation gates.',
        '', 'New GPU sequence: C4096 score/cache correctness diagnosis; B8 eager/graph clock pilots; bounded fixed-partition competitor, event-control and long-energy pilots; only passing families expand to3 acquisitions/10 trials; CPU analysis and publication.',
        '', 'J4 is a failed predictive model, not a confirmed software defect. Keep the exposed holdout rejected. Graph restriction, trace semantics and private archive access lack a safe repair in this change and receive no repeat captures.',
        '', 'The earlier J6 teardown and J7 range bugs are already repaired and are not repeated.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'clock_rejected_cases_audited':len(rows),'old_conditioning_pilots':len(diagnostics),'new_gpu_measurements':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
