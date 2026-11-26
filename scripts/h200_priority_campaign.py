#!/usr/bin/env python3
"""November 25 two-H200 campaign: new queue, independent UUIDs and strict gates.

Coordinator never initializes CUDA. GPU workers use the reviewed single-device
supervisor and immutable numerical snapshot. No Slurm submission occurs here.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback

PACKAGE = Path(__file__).resolve().parents[1]
FRACTIONS = (1.0, .85, .75, .60, .45, .30)
ROWS = ('P0', 'P1a', 'P1b', 'P2a', 'P2b', 'P3a', 'P3b', 'P4a', 'P4b', 'P4c', 'P5a', 'P5b', 'P5c', 'P5d')


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(Path(path).read_bytes()).hexdigest()


def limitation(root, case, reason, evidence, tried, resume):
    row = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'case': case,
           'limitation': reason, 'evidence': evidence, 'attempted_remedy': tried,
           'consequence': 'No claim beyond the recorded qualified scope.', 'resume_condition': resume}
    with (PACKAGE / '.git/h200_limitations.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (root / 'results/limitations.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        safe = lambda s: str(s).replace('|', '/').replace('\n', ' ')
        with (PACKAGE / 'docs/h200_measurement_limitations_20261125.md').open('a') as stream:
            stream.write('| ' + ' | '.join(map(safe, [row['utc'] + ' ' + str(root.relative_to(PACKAGE)) + ' ' + case,
                reason + '; evidence: ' + evidence, row['consequence'], tried,
                'open; resume: ' + resume])) + ' |\n')
        with (root / 'analysis/report.md').open('a') as stream:
            stream.write('\n- Limitation ' + case + ': ' + reason + '. Evidence: ' + evidence +
                         '. Tried: ' + tried + '. Resume: ' + resume + '.\n')


def publish(root, message):
    subprocess.run([sys.executable, PACKAGE / 'scripts/checkpoint_h200.py', '--root', root,
                    '--message', message, '--push'], cwd=PACKAGE, check=True, timeout=300)


def config(root, engine, context=4096, trials=10, batch=1):
    d = load(PACKAGE / 'h200/20261125_priority_preparation/qwen32b_smoke.json')
    d.update(batch_sizes=[batch], context_lengths=[context], warmup_trials=3, trials=trials,
             compile=False, maximum_cache_length=max(8192, context+64))
    d['engine'].update(max_model_len=max(8192, context+64), max_num_seqs=max(8, batch),
                       max_num_batched_tokens=max(8192, batch*context), enforce_eager=engine == 'eager',
                       gpu_memory_utilization=.85)
    if engine == 'eager':
        d['engine']['compilation_config'] = {'mode': 0}
    name = f'qwen32b_{engine}_b{batch}_c{context}_t{trials}.json'
    path = root / 'configs' / name
    if path.exists():
        assert load(path) == d, 'immutable config differs'
    else:
        write(path, d)
    return path


def valid_tokens(data, expected=None):
    if data.get('status') != 'complete' or len(data.get('cases', [])) != 1:
        return False
    case = data['cases'][0]
    outputs = case.get('warmup_output_ids', []) + [t['output_ids'] for t in case['trials']]
    expected = outputs[0] if expected is None and outputs else expected
    return (len(case.get('warmup_output_ids', [])) >= 3 and len(case['trials']) >= 3 and
            case.get('output_digests_identical') is True and outputs and
            all(x == expected for x in outputs) and
            all(t['prefill_barrier']['passed'] and t['decode_iterations'] == case['generated_tokens'] - 1
                for t in case['trials']))


class Worker:
    def __init__(self, root, session):
        self.root = root
        os.environ['H200_SESSION_CONFIG'] = str(session)
        import h200_session as hs
        self.hs = hs
        self.settings = load(session)
        self.script = Path(self.settings['frozen_source_root']) / 'scripts'
        self.py = self.settings['python']
        self.cuda = Path(self.settings['cuda_home'])
        self.uuid = self.settings['gpu_uuid']
        self.expected = None
        self.records = []
        self.rebind_step = False

    def execute(self, label, command, gpu=True, timeout=1200):
        return self.hs.run(self.root, label, command, timeout, gpu=gpu, env=self.hs.environment())

    def case(self, engine, fraction, acq, pilot=False, stock=False, suffix='', context=4096, batch=1):
        label = f'{engine}_b{batch}_c{context}_{"stock" if stock else "green"}_f{round(fraction*100)}_a{acq}' + ('_proof' if pilot else '') + suffix + ('_rebind_v1' if self.rebind_step else '')
        output = self.root / 'runs' / label
        c = config(self.root, engine, context, 3 if pilot else 10, batch)
        command = [self.py, self.script / 'shoreline_restricted_decode.py', '--config', c,
                   '--gpu-uuid', self.uuid, '--output', output, '--fraction', str(fraction), '--mode', 'timing']
        if not stock:
            command += ['--force-green']
            if self.rebind_step:
                command += ['--rebind-step']
        report = self.root / 'profiles' / label
        if pilot:
            command = [self.cuda / 'bin/nsys', 'profile', '--sample=none', '--cpuctxsw=none',
                       '--trace=cuda,nvtx', '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi',
                       '--capture-range-end=stop', '--output=' + str(report)] + command + ['--external-profiler']
        process = self.execute(label, command)
        row = {'id': label, 'engine': engine, 'fraction': fraction, 'acquisition': acq, 'pilot': pilot,
               'stock': stock, 'suffix': suffix, 'rebind_step':self.rebind_step,'gpu_uuid': self.uuid, 'batch': batch, 'context': context,
               'status': 'failed', 'process_status': process['status'], 'qualified': False,
               'source_identity': self.settings['frozen_source_identity'], 'run': str(output / 'run.json')}
        allocation = output / 'green_allocation.json'
        if allocation.exists():
            row['allocation'] = load(allocation)['allocation']
            row['actual_sm_count'] = row['allocation']['actual_sm_count']
        native = output / 'run.json'
        if native.exists():
            d = load(native)
            row['run_sha256'] = sha(native)
            row['exact_outputs'] = bool(valid_tokens(d, self.expected))
            if process['status'] == 'complete' and row['exact_outputs']:
                row['status'] = 'complete'
                trials = d['cases'][0]['trials']
                row['step_ms'] = [t['wall_ms']/t['decode_iterations'] for t in trials]
                row['gpu_event_step_ms'] = [t['cuda_event_ms']/t['decode_iterations'] for t in trials]
                row['tokens_s'] = [t['tokens_per_s'] for t in trials]
                if not pilot:
                    import shoreline_campaign as campaign
                    row['clock'] = campaign.clock_qualification(d, process['monitor']['source'])
                    row['qualified'] = row['clock']['passed'] and len(trials) == 10
        if pilot and report.with_suffix('.nsys-rep').exists() and allocation.exists():
            db = report.with_suffix('.sqlite')
            export = self.execute(label + '_export', [self.cuda / 'bin/nsys', 'export', '--type=sqlite',
                '--output=' + str(db), report.with_suffix('.nsys-rep')], gpu=False, timeout=180)
            if export['status'] == 'complete':
                try:
                    from shoreline_timeline import read
                    row['timeline'] = read(db, row['allocation']['green_context_id'] or 0)
                    row['placement_passed'] = (row['timeline']['restriction_passed'] and
                        (engine != 'graph' or all(w['graph_kernel_count'] > 0 for w in row['timeline']['windows'])))
                    row['qualified'] = row['status'] == 'complete' and row['placement_passed']
                except Exception as error:
                    row['proof_error'] = str(error)
        self.records.append(row)
        write(self.root / 'results' / (label + '.json'), row)
        if row['status'] != 'complete' or (pilot and not row['qualified']):
            limitation(self.root, label, 'Producer, exact output, or measured decoder placement gate failed',
                       str((self.root/'logs'/(label+'.log')).relative_to(PACKAGE)),
                       'Versioned green-context path and separate NSYS measured-decode proof',
                       'Inspect preserved failure; bounded repair before reusing this arm')
        elif not pilot and not row['qualified']:
            limitation(self.root, label, 'Clock envelope failed; all trials retained',
                       str((self.root/'results'/(label+'.json')).relative_to(PACKAGE)),
                       'Passive clock sampling and per-trial endpoints; no clock settings changed',
                       'Independent source/clock-qualified acquisition; no favorable trial selection')
        return row

    def reference(self, source, label):
        path = self.root / 'results' / (label + '_hf.json')
        result = self.execute(label + '_hf', [self.py, self.script / 'hf_full_reference.py',
                    '--source-run', source, '--gpu-uuid', self.uuid, '--output', path], timeout=1800)
        passed = result['status'] == 'complete' and path.exists() and load(path)['status'] == 'passed'
        if not passed:
            limitation(self.root, label, 'Strict independent HF SDPA exact-token gate failed',
                str(path.relative_to(PACKAGE)), 'Independent full-length dynamic-cache reference; unchanged exact gate',
                'Diagnose first differing token without weakening correctness')
        return passed

    def curve(self, engine, batch=1, context=4096):
        stock_proof=self.case(engine,1.,0,pilot=True,stock=True,batch=batch,context=context)
        if not stock_proof['qualified']:
            analyze_curve(self.root,self.records,engine,batch,context)
            return False
        proofs = []
        for fraction in FRACTIONS:
            proof = self.case(engine, fraction, 0, pilot=True, batch=batch, context=context)
            proofs.append(proof)
            if not proof['qualified']:
                analyze_curve(self.root,self.records,engine,batch,context)
                if engine=='graph' and not self.rebind_step and proof.get('allocation'):
                    limitation(self.root,proof['id'],'Optimized placement pilot failed; bounded explicit per-tick rebinding repair follows',
                               str((self.root/'results'/(proof['id']+'.json')).relative_to(PACKAGE)),
                               'Initial context/stream binding and NSYS graph-node placement check',
                               'Require a fresh proof for every count using the versioned rebind_v1 route')
                    self.rebind_step=True
                    return self.curve(engine,batch,context)
                return False
        # Broad curve is completed before refinement or additional workloads.
        for acq in range(3):
            self.case(engine, 1., acq, stock=True, batch=batch, context=context)
            self.case(engine, 1., acq, batch=batch, context=context)
            for index, fraction in enumerate(FRACTIONS[1:]):
                self.case(engine, fraction, acq, batch=batch, context=context)
                if index == 1:
                    self.case(engine, 1., acq, suffix='_interleaved', batch=batch, context=context)
            self.case(engine, 1., acq, suffix='_end', batch=batch, context=context)
            analyze_curve(self.root, self.records, engine, batch, context)
            publish(self.root, f'H200 {engine} B{batch}/C{context} broad curve acquisition {acq}: retain all gates')
        result = analyze_curve(self.root, self.records, engine, batch, context)
        return result['complete_qualified_curve']

    def traffic(self, engine='graph', context=4096, batch=1):
        """Three independent app-range producers; physical bytes never use clean profiler time."""
        import csv
        import io
        from shoreline_evidence import qualify_replays
        reference=self.case(engine,1.,0,stock=True,suffix='_traffic_reference',context=context,batch=batch)
        if reference['status']!='complete' or not self.reference(reference['run'],f'{engine}_b{batch}_c{context}_traffic'):
            return {'status':'blocked','reason':'exact traffic reference unavailable'}
        expected=load(reference['run'])['cases'][0]
        results=[]
        for acq in range(3):
            label=f'P2a_{engine}_b{batch}_c{context}_a{acq}'
            c=load(config(self.root,engine,context,3,batch))
            c['shoreline_protocol']={'counter_range_protocol':'ticks_only_v5'}
            cfg=self.root/'configs'/(label+'.json');write(cfg,c)
            output=self.root/'runs'/label
            report=self.root/'profiles'/(label+'.ncu-rep')
            command=[self.cuda/'bin/ncu','--replay-mode','app-range','--clock-control','none',
                '--cache-control','none','--nvtx','--nvtx-include','shoreline_ticks_4_6/',
                '--metrics','dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum',
                '--export',report,self.py,self.script/'shoreline_replay.py','--config',cfg,
                '--gpu-uuid',self.uuid,'--mode','timing','--output',output,'--external-profiler']
            producer=self.execute(label,command,timeout=1800)
            replay=qualify_replays(output,expected,self.root)
            values={}
            if report.exists():
                imp=self.execute(label+'_import',[self.cuda/'bin/ncu','--import',report,'--page','raw',
                          '--print-units','base','--csv'],gpu=False,timeout=180)
                text=(self.root/'logs'/(label+'_import.log')).read_text()
                lines=text.splitlines()
                header=next((i for i,l in enumerate(lines) if l.startswith('"ID"')),None)
                if header is not None:
                    for row in csv.DictReader(io.StringIO('\n'.join(lines[header:]))):
                        metric=row.get('Metric Name')
                        if metric in ('dram__bytes_read.sum','dram__bytes_write.sum','gpu__time_duration.sum'):
                            try:values.setdefault(row['ID'],{})[metric]=float(row['Metric Value'].replace(',',''))
                            except (ValueError,KeyError):pass
            valid=(producer['status']=='complete' and replay['passed'] and len(values)==1 and
                all(v.get('dram__bytes_read.sum',0)>0 and v.get('dram__bytes_write.sum',-1)>=0 for v in values.values()))
            row={'status':'complete' if valid else 'failed','acquisition':acq,'gpu_uuid':self.uuid,
                'classification':'measured-real-GPU-instrumented','ranges':values,'replays':replay,
                'ticks':[4,5,6],'token_denominator':3*batch,'ordinary_latency':None,
                'clock_join':'unqualified until pooled replay clock/source/isolation join',
                'scope':'physical full-range DRAM reads/writes; aggregate counters do not prove address-level exactly-once access'}
            write(self.root/'results'/(label+'.json'),row);results.append(row)
            if not valid:
                limitation(self.root,label,'Physical-counter replay/output/range gate failed',
                    str((self.root/'results'/(label+'.json')).relative_to(PACKAGE)),
                    'Application-range replay with native per-pass publications and unchanged exact outputs',
                    'Inspect NCU permissions/ranges and preserved publications; no analytical-byte substitution')
            publish(self.root,f'H200 P2a {label}: physical-counter disposition')
            if not valid:break
        return {'status':'captured' if len(results)==3 and all(x['status']=='complete' for x in results) else 'partial',
                'scope':'instrumented physical traffic; timing joins remain separate'}

    def primary(self):
        inv = self.execute('inventory', [self.py, self.script/'inventory.py', '--gpu-uuid', self.uuid,
                    '--check-cuda', '--output', self.root/'environment/inventory.json'], timeout=180)
        if inv['status'] != 'complete':
            return {'status': 'blocked', 'reason': 'inventory failed'}
        for engine in ('graph', 'eager'):
            smoke = self.case(engine, 1., 0, stock=True, context=128, suffix='_smoke')
            if smoke['status'] != 'complete' or not self.reference(smoke['run'], engine + '_smoke'):
                return {'status': 'blocked', 'reason': engine + ' C128 exact smoke/reference gate'}
        target = self.case('graph', 1., 0, stock=True, suffix='_target_control')
        if target['status'] != 'complete' or not self.reference(target['run'], 'graph_target'):
            return {'status': 'blocked', 'reason': 'C4096 exact HF/graph gate'}
        self.expected = load(target['run'])['cases'][0]['trials'][0]['output_ids']
        graph = self.curve('graph')
        if not graph:
            limitation(self.root, 'P1a_graph', 'Complete qualified optimized restricted curve unavailable',
                'results/curve_graph.json and individual proof/acquisition records',
                'One initial versioned optimized-path proof cycle; preserve all diagnostics for bounded repair',
                'Repair actual decoder context/graph placement, then re-pilot; eager fallback remains separate')
            eager = self.case('eager', 1., 0, stock=True, suffix='_target_control')
            if eager['status'] == 'complete':
                self.curve('eager')
        publish(self.root, 'H200 P1a: publish measured curve and explicit blocked gates')
        if graph:
            for batch in (8,32):
                if time.time()+5400 >= dt.datetime.fromisoformat(self.settings['launch_cutoff']).timestamp():
                    break
                self.expected=None
                control=self.case('graph',1.,0,stock=True,suffix='_width_control',batch=batch)
                if control['status']=='complete' and self.reference(control['run'],f'graph_b{batch}_c4096'):
                    self.expected=load(control['run'])['cases'][0]['trials'][0]['output_ids']
                    self.curve('graph',batch=batch)
            self.expected=load(target['run'])['cases'][0]['trials'][0]['output_ids']
        physical=self.traffic()
        write(self.root/'results/P2a_disposition.json',physical)
        # Full-GPU long-context fallbacks remain eligible if restriction is blocked.
        for context in (16384,32768):
            if time.time()+2400 >= dt.datetime.fromisoformat(self.settings['launch_cutoff']).timestamp():
                break
            expected=self.expected;self.expected=None
            control=self.case('graph',1.,0,stock=True,suffix='_long_control',context=context)
            if control['status']!='complete' or not self.reference(control['run'],f'long_c{context}'):
                self.expected=expected
                continue
            self.expected=load(control['run'])['cases'][0]['trials'][0]['output_ids']
            if graph and time.time()+5400 < dt.datetime.fromisoformat(self.settings['launch_cutoff']).timestamp():
                self.curve('graph',context=context)
            else:
                for acq in range(3):
                    self.case('graph',1.,acq,stock=True,suffix='_long',context=context)
            self.expected=expected
            publish(self.root,f'H200 long-context C{context}: separate full-GPU acquisitions')
        return {'status': 'partial', 'optimized_curve_qualified': graph,
                'next': 'P2a full-GPU physical traffic and P3a serving; inspect existing adapter compatibility'}

    def secondary(self):
        self.execute('inventory', [self.py, self.script/'inventory.py', '--gpu-uuid', self.uuid,
                    '--check-cuda', '--output', self.root/'environment/inventory.json'], timeout=180)
        label='P0_native_ncu_stream512'
        command=[self.cuda/'bin/ncu','--replay-mode','kernel','--clock-control','none',
                 '--cache-control','none','--profile-from-start','off',
                 '--metrics','dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum',
                 '--export',self.root/'profiles'/label,self.py,self.script/'shoreline_native.py',
                 '--gpu-uuid',self.uuid,'--output',self.root/'runs'/label,'--acquisition','0',
                 '--pilot','--profile-case','stream_512mib']
        counter=self.execute(label,command,timeout=600)
        if counter['status']!='complete':
            limitation(self.root,label,'Known-payload NCU permission/control pilot failed',
                       str((self.root/'logs'/(label+'.log')).relative_to(PACKAGE)),
                       'Native stream control with physical DRAM metrics, no clock/cache modification',
                       'Inspect profiler permissions and raw NCU log; independent unprofiled controls remain eligible')
        # Independent controls on the spare physical board, never pooled with the decoder GPU.
        for acq in range(3):
            for fraction in FRACTIONS:
                label = f'native_f{round(fraction*100)}_a{acq}'
                command = [self.py, self.script/'shoreline_native.py', '--gpu-uuid', self.uuid,
                           '--output', self.root/'runs'/label, '--fraction', str(fraction),
                           '--acquisition', str(acq), '--pilot']
                result = self.execute(label, command, timeout=900)
                if result['status'] != 'complete':
                    limitation(self.root, label, 'Independent native service control failed',
                        str((self.root/'logs'/(label+'.log')).relative_to(PACKAGE)),
                        'Existing known-value streaming/gather and arithmetic native adapter',
                        'Inspect native exception; do not infer decoder bandwidth from useful payload')
                    return {'status': 'partial'}
            publish(self.root, f'H200 spare GPU native service acquisition {acq}; decoder transfer not inferred')
        return {'status': 'native_controls_captured', 'qualification': 'clock/source/physical-counter joins remain separate'}


def analyze_curve(root, records, engine, batch=1, context=4096):
    from shoreline_statistics import hierarchical_interval, paired_ratio
    clean = [r for r in records if r['engine'] == engine and not r['pilot'] and not r['stock'] and
             r['batch'] == batch and r['context'] == context]
    points = []
    proofs = {r['fraction']: r for r in records if r['engine'] == engine and r['pilot'] and not r['stock'] and r.get('qualified') and r['batch']==batch and r['context']==context}
    stock_proofs=[r for r in records if r['engine']==engine and r['pilot'] and r['stock'] and r.get('qualified') and r['batch']==batch and r['context']==context]
    refs = sorted([r for r in records if r['engine'] == engine and not r['pilot'] and r['stock']
                   and r['batch'] == batch and r['context'] == context and not r.get('suffix')], key=lambda r:r['acquisition'])
    green_refs = sorted([r for r in clean if r['fraction'] == 1. and not r.get('suffix')], key=lambda r:r['acquisition'])
    for fraction in FRACTIONS:
        rows = sorted([r for r in clean if r['fraction'] == fraction and
                       not r.get('suffix')], key=lambda r:r['acquisition'])
        point = {'fraction': fraction, 'qualified': False, 'sources': [r['id'] for r in rows],
                 'actual_sm_count': rows[0].get('actual_sm_count') if rows else None,
                 'raw_acquisition_medians_ms': [statistics.median(r['step_ms']) if r.get('step_ms') else None for r in rows],
                 'raw_scope': 'retained observations, not qualified comparisons until every gate passes'}
        combined = rows + refs + green_refs + [r for r in clean if r.get('suffix') in ('_end', '_interleaved')]
        clocks = [r.get('clock', {}).get('clocks', {}) for r in combined]
        if (len(rows) == len(refs) == 3 and all(r.get('qualified') for r in combined) and
            fraction in proofs and 1. in proofs and stock_proofs and
            len({r['source_identity'] for r in combined})==1 and
            all(r['source_identity']==rows[0]['source_identity'] for r in (proofs[fraction],proofs[1.],stock_proofs[-1])) and
            len({r['actual_sm_count'] for r in rows}) == 1 and
            proofs[fraction]['actual_sm_count']==rows[0]['actual_sm_count'] and
            green_refs and proofs[1.]['actual_sm_count']==green_refs[0]['actual_sm_count'] and
            stock_proofs[-1]['actual_sm_count']==refs[0]['actual_sm_count'] and
            [r['acquisition'] for r in rows]==[r['acquisition'] for r in refs]==[0,1,2]):
            pooled = {m: max(c[m]['maximum'] for c in clocks)/min(c[m]['minimum'] for c in clocks)
                      for m in ('sm_clock_mhz', 'memory_clock_mhz')}
            point['pooled_clock_ratio'] = pooled
            if all(v <= 1.05 for v in pooled.values()) and len(green_refs) == 3:
                point.update(qualified=True, latency_ms=hierarchical_interval([r['step_ms'] for r in rows]),
                    ratio=paired_ratio([r['step_ms'] for r in rows], [r['step_ms'] for r in refs]),
                    ratio_to_full_green=paired_ratio([r['step_ms'] for r in rows], [r['step_ms'] for r in green_refs]),
                    tokens_s=hierarchical_interval([r['tokens_s'] for r in rows]))
        points.append(point)
    thresholds = {}
    for margin in (.05, .10, .25):
        candidates = [p for p in points if p['qualified'] and p['ratio']['ci95'][1] <= 1+margin]
        thresholds[str(margin)] = {'smallest_tested_demonstrated_sm_count': min((p['actual_sm_count'] for p in candidates), default=None),
                                  'scope': 'tested counts only; unresolved/censored between points, no AMD thresholds imported'}
    result = {'engine': engine, 'batch':batch,'context':context,'classification': 'measured-real-GPU', 'points': points, 'thresholds': thresholds,
              'complete_qualified_curve': all(p['qualified'] for p in points),
              'scope': 'pure decode after verified prefill barrier, wall ms/tick; primary ratios use unrestricted stock; full-green ratios also retained; physical area/HBM capacity unchanged'}
    write(root/'results'/f'curve_{engine}_b{batch}_c{context}.json', result)
    if batch==1 and context==4096:
        write(root/'results'/f'curve_{engine}.json', result)
    lines = [f'\n## {engine} B{batch}/C{context} curve\n', '| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |',
             '|---|---:|---|---:|---|']
    for p in points:
        lines.append(f"| {p['fraction']} | {p['actual_sm_count']} | {p['qualified']} | {p.get('latency_ms', {}).get('value', 'NA')} | {p.get('ratio', {}).get('ci95', 'NA')} |")
    with (root/'analysis/report.md').open('a') as stream:
        stream.write('\n'.join(lines)+'\n')
    return result


def coordinator(args):
    assert os.environ.get('SLURM_JOB_ID', '').isdigit(), 'requires allocated sbatch/srun'
    root = PACKAGE/'h200'/f'{dt.datetime.now():%Y%m%d}_job{os.environ["SLURM_JOB_ID"]}_priority'
    root.mkdir(exist_ok=False)
    for folder in ('environment', 'results', 'logs', 'configs', 'runs', 'profiles', 'analysis'):
        (root/folder).mkdir()
    (root/'analysis/report.md').write_text('# Two-H200 priority campaign\n\nNo measurements yet. Primary: Qwen2.5-32B BF16/TP1 B1/C4096 SM curve. Secondary UUID: independent native controls.\n')
    state = subprocess.check_output(['scontrol', 'show', 'job', '-o', os.environ['SLURM_JOB_ID']], text=True)
    (root/'environment/slurm.txt').write_text(state)
    fields = dict(item.split('=',1) for item in state.split() if '=' in item)
    assert fields['JobState'] == 'RUNNING'
    end = dt.datetime.fromisoformat(fields['EndTime']).astimezone()
    query = "import json,torch; assert torch.cuda.device_count()==2; print(json.dumps([{'uuid':'GPU-'+str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-'),'name':torch.cuda.get_device_properties(i).name,'sm_count':torch.cuda.get_device_properties(i).multi_processor_count} for i in range(2)]))"
    devices = json.loads(subprocess.check_output([sys.executable, '-c', query], text=True))
    assert all('H200' in d['name'] for d in devices)
    write(root/'environment/allocated_devices.json', {'devices':devices, 'slurm_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'hostname':socket.gethostname()})
    for name, command in [('topology', ['nvidia-smi', 'topo', '-m']), ('runtime', [sys.executable,'-m','pip','freeze'])]:
        (root/'environment'/(name+'.txt')).write_text(subprocess.check_output(command, text=True, stderr=subprocess.STDOUT))
    base = {'schema_version':1, 'slurm_job_id':os.environ['SLURM_JOB_ID'], 'expected_hostname':socket.gethostname(),
            'python':sys.executable, 'cuda_home':args.cuda, 'allocation_end':end.isoformat(),
            'launch_cutoff':(end-dt.timedelta(minutes=30)).isoformat(), 'closeout_buffer_s':1800,
            'monitor_period_s':.05, 'idle_required_s':10, 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'seed':20261013, 'acquisitions':3, 'warmups':3, 'trials':10}
    jobs=[]
    for i, device in enumerate(devices):
        subroot=root/f'gpu{i}'
        for folder in ('environment','results','logs','configs','runs','profiles','analysis'):
            (subroot/folder).mkdir(parents=True,exist_ok=True)
        (subroot/'analysis/report.md').write_text(f'# GPU {i}: {device["uuid"]}\n')
        session=dict(base, gpu_uuid=device['uuid'], workspace=str(Path('/tmp')/f'h200_priority_{os.getuid()}_{os.environ["SLURM_JOB_ID"]}_{i}'))
        Path(session['workspace']).mkdir(exist_ok=False)
        write(subroot/'session.json',session)
        subprocess.run([sys.executable,PACKAGE/'scripts/freeze_shoreline.py','--session',subroot/'session.json','--activate'],check=True)
        jobs.append((i,subroot))
    write(root/'results/case_manifest.json',{'schema_version':1,'rows':[{'id':r,'status':'pending' if r!='P5b' else 'deferred','reason':'requires four allocated GPUs' if r=='P5b' else None} for r in ROWS]})
    write(root/'STATUS.json',{'status':'inventoried','devices':devices,'allocation_end':end.isoformat()})
    publish(root,'H200 priority: publish two-device allocation and frozen source manifests')
    processes=[]
    for i,subroot in jobs:
        stream=(root/'logs'/f'worker{i}.log').open('w')
        command=[sys.executable,Path(__file__).resolve(),'--worker',str(i),'--root',str(subroot),'--session',str(subroot/'session.json')]
        processes.append((subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT),stream))
    while any(p.poll() is None for p,s in processes):
        time.sleep(15)
    for p,s in processes:s.close()
    manifest=load(root/'results/case_manifest.json')
    for row in manifest['rows']:
        if row['id']=='P0':
            row.update(status='partial',reason='Inventory, correctness and profiler/placement records require component-level inspection')
        elif row['id']=='P1a':
            curves=[load(p) for p in (root/'gpu0/results').glob('curve_*.json')
                    if load(p).get('batch')==1 and load(p).get('context')==4096]
            row.update(status='complete' if any(c['complete_qualified_curve'] for c in curves) else 'blocked',
                       reason='Engine labels and all gates remain explicit in gpu0/results/curve_*.json')
        elif row['id']=='P1b':
            row.update(status='partial' if list((root/'gpu0/results').glob('curve_*_b*_c*.json')) else 'pending',
                       reason='Only source/output/clock/placement-qualified shapes count; unmeasured refinements remain pending')
        elif row['id']=='P2a':
            row.update(status='partial' if list((root/'gpu0/results').glob('P2a*.json')) else 'pending',
                       reason='Physical traffic captures retain replay/range gates; counter-clock joins and kernel retuning are separate')
    write(root/'results/case_manifest.json',manifest)
    write(root/'STATUS.json',{'status':'workers_finished_scientific_gates_separate','worker_exit_codes':[p.returncode for p,s in processes],
          'allocation_end':end.isoformat(), 'rows':manifest['rows'],
          'next_command':f'squeue --start -j {os.environ["SLURM_JOB_ID"]}'})
    with (root/'analysis/report.md').open('a') as stream:
        stream.write('\n## Queue disposition\n\n'+'\n'.join('- '+r['id']+': '+r['status']+'; '+str(r.get('reason') or 'not acquired in this allocation') for r in manifest['rows'])+'\n')
    publish(root,'H200 priority: publish worker outcomes and handoff')
    print('CAMPAIGN_ROOT',root,flush=True)



def bootstrap(args):
    """Heavy installation/download runs only inside the authorized allocation."""
    root=PACKAGE/'h200'/f'{dt.datetime.now():%Y%m%d}_job{os.environ["SLURM_JOB_ID"]}_bootstrap'
    for folder in ('environment','results','logs','analysis'):
        (root/folder).mkdir(parents=True,exist_ok=True)
    (root/'analysis/report.md').write_text('# Allocated staging and launch\n\nNo model measurement inferred from installation.\n')
    state=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)
    (root/'environment/slurm.txt').write_text(state)
    (root/'environment/host.txt').write_text(socket.gethostname()+'\n')
    write(root/'STATUS.json',{'status':'allocated_staging','job_id':os.environ['SLURM_JOB_ID']})
    publish(root,'H200 priority: allocated runtime/model staging started')
    runtime=Path('/storage/scratch1/6/jkim4112/envs/h200_priority_20261125/bin/python')
    stages=[('runtime_install',[runtime,'-m','pip','install','--no-cache-dir','-r',PACKAGE/'configs/runtime.lock.txt'],5400),
            ('runtime_check',[runtime,'-c',"import torch,vllm,transformers; assert torch.__version__.split('+')[0]=='2.11.0'; assert vllm.__version__=='0.22.1'; assert transformers.__version__=='4.57.6'; assert torch.cuda.device_count()==2; print(torch.__version__,vllm.__version__,transformers.__version__,torch.version.cuda); print([(str(torch.cuda.get_device_properties(i).uuid),torch.cuda.get_device_properties(i).name,torch.cuda.get_device_properties(i).multi_processor_count,torch.cuda.mem_get_info(i)) for i in range(2)])"],180),
            ('model_download',[runtime,PACKAGE/'scripts/materialize_config.py','--model','qwen32b','--stage','smoke','--revision','5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd','--download','--output',PACKAGE/'h200/20261125_priority_preparation/qwen32b_smoke.json'],5400),
            ('cpu_contracts',[runtime,PACKAGE/'scripts/test_cpu_contracts.py'],300),
            ('priority_contracts',[runtime,PACKAGE/'scripts/test_h200_priority.py'],300)]
    try:
        for label,command,limit in stages:
            print('STAGING',label,flush=True)
            started=time.time()
            with (root/'logs'/(label+'.log')).open('x') as stream:
                result=subprocess.run(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,timeout=limit)
            write(root/'results'/(label+'.json'),{'status':'complete' if result.returncode==0 else 'failed','exit_code':result.returncode,'elapsed_s':time.time()-started,'command':list(map(str,command))})
            if result.returncode:
                raise RuntimeError(label+' failed with exit '+str(result.returncode))
        write(root/'STATUS.json',{'status':'staging_complete','next':'coordinator'})
        publish(root,'H200 priority: pinned runtime and immutable32B staging complete')
        subprocess.run([runtime,'-u',PACKAGE/'scripts/h200_priority_campaign.py','--cuda',args.cuda],check=True)
    except BaseException as error:
        write(root/'STATUS.json',{'status':'blocked','error':str(error),'traceback':traceback.format_exc()})
        limitation(root,'bootstrap','Allocated staging/launch failed: '+str(error),str(root.relative_to(PACKAGE)),
                   'Preserved each stage log and pinned requirements; no system changes','Inspect failed stage and use existing allocation if time remains')
        raise
    finally:
        publish(root,'H200 priority: durable staging/launch checkpoint')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',action='store_true');p.add_argument('--cuda');p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--root',type=Path);p.add_argument('--session',type=Path)
    a=p.parse_args()
    if a.bootstrap:
        bootstrap(a)
    elif a.worker is None:
        coordinator(a)
    else:
        worker=Worker(a.root,a.session)
        try:
            outcome=worker.primary() if a.worker==0 else worker.secondary()
            write(a.root/'results/worker_outcome.json',outcome)
        except BaseException as error:
            write(a.root/'results/worker_outcome.json',{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
            limitation(a.root,'worker','Worker stopped with preserved exception','results/worker_outcome.json',
                       'Allocation-bound supervised launches and retained logs','Inspect exception and resume at a fresh identified case')
            raise
        finally:
            publish(a.root,'H200 priority worker checkpoint')


if __name__=='__main__':
    main()
