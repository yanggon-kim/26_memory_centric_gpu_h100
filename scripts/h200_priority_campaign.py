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


def estimate_curve_seconds(stock_pilot_s, green_pilot_s, idle_s=10):
    """Conservative scheduling estimate, never an inference-latency result."""
    assert set(green_pilot_s)==set(FRACTIONS)
    assert stock_pilot_s>0 and all(v>0 for v in green_pilot_s.values())
    # Three stock runs; nine full-green runs including drift references;
    # three runs at each of five restricted levels. Pilots execute3+3 trials,
    # ordinary runs3+10. Scaling startup/profiler cost too is conservative.
    pilot_sum=3*stock_pilot_s+9*green_pilot_s[1.]+3*sum(green_pilot_s[f] for f in FRACTIONS[1:])
    return pilot_sum*(13/6)*1.25+27*idle_s+300


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


def prior_kv_software_validation(path, model_config):
    """Reuse only this adapter's prior functional scope, never a new pair's result."""
    d=load(path)
    assert d['status']=='complete' and d['real_KV_handoff_verified'] is True
    assert d['model_revision']==model_config['model_revision']
    assert d['model_config_sha256']==model_config['config_sha256']
    assert d['receiver_prefill_executed'] is False and len(d['trials'])==6
    assert all(t['correctness']=='passed' for t in d['trials'])
    assert d['versions']=={'torch':'2.11.0+cu130','transformers':'4.57.6','cuda':'13.0'}
    assert d['warmups']==3 and sum(t['warmup'] for t in d['trials'])==3
    assert len(d['cache_layouts'])==128
    assert {(x['layer'],x['role']) for x in d['cache_layouts']}=={(i,r) for i in range(64) for r in ('key','value')}
    assert all(x['dtype']=='torch.bfloat16' for x in d['cache_layouts'])
    assert set(d['source_sha256'])=={'h200_real_kv_smoke.py','h200_p2p_smoke.py','h200_session.py'}
    assert all(sha(PACKAGE/'scripts'/name)==digest for name,digest in d['source_sha256'].items())
    return {'status':'prior_software_functional_validation_only','prior_result':str(path),
            'prior_result_sha256':sha(path),'prior_physical_pair':d['devices'],
            'model_revision':d['model_revision'],'new_pair_handoff_validated':False,
            'scope':'unchanged all-layer HF adapter validated on the original pair; fresh pair handoff deferred until after core SM sweep'}


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
        self.last_curve_stop = None

    def execute(self, label, command, gpu=True, timeout=1200):
        return self.hs.run(self.root, label, command, timeout, gpu=gpu, env=self.hs.environment())

    def case(self, engine, fraction, acq, pilot=False, stock=False, suffix='', context=4096, batch=1):
        label = f'{engine}_b{batch}_c{context}_{"stock" if stock else "green"}_f{round(fraction*100)}_a{acq}' + ('_proof' if pilot else '') + suffix + ('_rebind_v1' if self.rebind_step else '')
        output = self.root / 'runs' / label
        c = config(self.root, engine, context, 3 if pilot else 10, batch)
        command = [self.py, self.script / 'shoreline_restricted_decode.py', '--config', c,
                   '--gpu-uuid', self.uuid, '--output', output, '--fraction', str(fraction), '--mode', 'timing',
                   '--rounding','nearest']
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
        row['producer_elapsed_s']=process.get('elapsed_s')
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

    def curve(self, engine, batch=1, context=4096, stock_proof=None):
        self.last_curve_stop=None
        if stock_proof is None:
            stock_proof=self.case(engine,1.,0,pilot=True,stock=True,batch=batch,context=context)
        else:
            assert stock_proof['pilot'] and stock_proof['stock']
            assert (stock_proof['engine'],stock_proof['batch'],stock_proof['context'])==(engine,batch,context)
            assert stock_proof['source_identity']==self.settings['frozen_source_identity']
            assert stock_proof['gpu_uuid']==self.uuid and Path(stock_proof['run']).is_relative_to(self.root)
        if not stock_proof['qualified']:
            self.last_curve_stop='proof'
            analyze_curve(self.root,self.records,engine,batch,context)
            self.unrestricted_after_block(engine,batch,context)
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
                self.unrestricted_after_block(engine,batch,context)
                self.last_curve_stop='proof'
                return False
        estimate=estimate_curve_seconds(stock_proof['producer_elapsed_s'],
                  {p['fraction']:p['producer_elapsed_s'] for p in proofs},self.settings.get('idle_required_s',10))
        allocation=self.hs.verify_allocation()
        remaining=allocation['effective_launch_cutoff_unix_s']-time.time()
        budget_path=self.root/'results'/f'budget_{engine}_b{batch}_c{context}.json'
        write(budget_path,{'classification':'analytical scheduling estimate from instrumented process runtimes',
              'estimated_s':estimate,'available_before_closeout_s':remaining,'admitted':estimate<=remaining,
              'method':'13/6 trial scaling,25percent margin,27idle guards,300s publication/analysis allowance',
              'pilot_sources':[stock_proof['id']]+[p['id'] for p in proofs]})
        if estimate>remaining:
            self.last_curve_stop='budget'
            limitation(self.root,f'{engine}_b{batch}_c{context}','Complete3x10 curve deferred by measured pilot budget',
                       str(budget_path.relative_to(PACKAGE)),
                       'Proved all allocations; conservatively scaled total process time including prefill',
                       'Resume with sufficient remaining allocation time; no cross-allocation timing pooling')
            analyze_curve(self.root,self.records,engine,batch,context)
            publish(self.root,'H200 curve deferred: complete-comparison budget does not fit')
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
        if not result['complete_qualified_curve']:
            self.last_curve_stop='quality'
            limitation(self.root,f'{engine}_b{batch}_c{context}_quality',
                       'All requested curve acquisitions attempted; strict scientific gates remain unqualified despite passing placement pilots',
                       f'results/curve_{engine}_b{batch}_c{context}.json',
                       'Retained every acquisition, interleaved reference and clock/source gate; no favorable resampling',
                       'Inspect the specific quality failures; advance eligible traffic/scheduler work rather than treating clock rejection as restriction failure')
        return result['complete_qualified_curve']

    def unrestricted_after_block(self,engine,batch,context):
        """Retain a 3x10 optimized stock reference even if restriction is blocked."""
        from shoreline_statistics import hierarchical_interval
        rows=[self.case(engine,1.,acq,stock=True,suffix='_unrestricted_fallback',
                        batch=batch,context=context) for acq in range(3)]
        passed=all(r.get('qualified') for r in rows)
        if passed:
            clocks=[r['clock']['clocks'] for r in rows]
            passed=all(max(c[m]['maximum'] for c in clocks)/min(c[m]['minimum'] for c in clocks)<=1.05
                       for m in ('sm_clock_mhz','memory_clock_mhz'))
        result={'engine':engine,'batch':batch,'context':context,'qualified_timing':passed,
                'sources':[r['id'] for r in rows],
                'raw_acquisition_medians_ms':[statistics.median(r['step_ms']) if r.get('step_ms') else None for r in rows],
                'latency_ms':hierarchical_interval([r['step_ms'] for r in rows]) if passed else None,
                'scope':'unrestricted stock software timing; restriction proof remains a separate failed gate; never used as eager normalization'}
        write(self.root/'results'/f'unrestricted_{engine}_b{batch}_c{context}.json',result)
        with (self.root/'analysis/report.md').open('a') as stream:
            stream.write(f'\nUnrestricted {engine} B{batch}/C{context} after blocked restriction: timing qualified={passed}; see results/unrestricted_{engine}_b{batch}_c{context}.json.\n')
        publish(self.root,f'H200 {engine} B{batch}/C{context}: independent full reference after blocked restriction')

    def traffic(self, engine='graph', context=4096, batch=1, acquisitions=None, reference_case=None):
        """Three independent app-range producers; physical bytes never use clean profiler time."""
        from shoreline_evidence import qualify_replays
        if reference_case is None:
            reference=self.case(engine,1.,0,stock=True,suffix='_traffic_reference',context=context,batch=batch)
            if reference['status']!='complete' or not self.reference(reference['run'],f'{engine}_b{batch}_c{context}_traffic'):
                return {'status':'blocked','reason':'exact traffic reference unavailable'}
            expected=load(reference['run'])['cases'][0]
        else:
            # The resume collector validates the immutable source/HF artifacts,
            # physical UUID, exact prompt and every warmup/trial before reuse.
            expected=reference_case
            assert expected['batch_size']==batch and expected['context_length']==context
            assert expected['generated_tokens']==64
        acquisitions=tuple(range(3)) if acquisitions is None else tuple(acquisitions)
        assert acquisitions and len(set(acquisitions))==len(acquisitions) and set(acquisitions)<={0,1,2}
        results=[]
        for acq in acquisitions:
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
            from h200_ncu_read import parse_csv,resolve_report
            actual_report=None
            reader_error=None
            try:actual_report=resolve_report(report)
            except ValueError as error:reader_error=str(error)
            if actual_report is not None:
                imp=self.execute(label+'_import',[self.cuda/'bin/ncu','--import',actual_report,'--page','raw',
                          '--print-units','base','--csv'],gpu=False,timeout=180)
                text=(self.root/'logs'/(label+'_import.log')).read_text()
                if imp['status']=='complete':
                    try:values=parse_csv(text)
                    except ValueError as error:reader_error=str(error)
                else:reader_error='Profiler report import process failed'
            valid=(producer['status']=='complete' and replay['passed'] and len(values)==1 and
                all(v.get('dram__bytes_read.sum',0)>0 and v.get('dram__bytes_write.sum',-1)>=0 for v in values.values()))
            row={'status':'complete' if valid else 'failed','acquisition':acq,'gpu_uuid':self.uuid,
                'classification':'measured-real-GPU-instrumented','ranges':values,'replays':replay,
                'counter_artifact':str(actual_report) if actual_report else None,'counter_reader_error':reader_error,
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

    def primary(self, core_first=False):
        inv = self.execute('inventory', [self.py, self.script/'inventory.py', '--gpu-uuid', self.uuid,
                    '--check-cuda', '--output', self.root/'environment/inventory.json'], timeout=180)
        if inv['status'] != 'complete':
            return {'status': 'blocked', 'reason': 'inventory failed'}
        if not core_first:
            for engine in ('graph', 'eager'):
                smoke = self.case(engine, 1., 0, stock=True, context=128, suffix='_smoke')
                if smoke['status'] != 'complete' or not self.reference(smoke['run'], engine + '_smoke'):
                    return {'status': 'blocked', 'reason': engine + ' C128 exact smoke/reference gate'}
        # The resume uses the target's bounded3+3 NSYS smoke and exact HF gate
        # before ordinary timing. Reuse its same-allocation stock proof once.
        target = self.case('graph',1.,0,stock=True,pilot=True) if core_first else self.case('graph', 1., 0, stock=True, suffix='_target_control')
        if core_first and not target['qualified']:
            return {'status':'blocked','reason':'target smoke exact graph/placement gate'}
        if target['status'] != 'complete' or not self.reference(target['run'], 'graph_target'):
            return {'status': 'blocked', 'reason': 'C4096 exact HF/graph gate'}
        self.expected = load(target['run'])['cases'][0]['trials'][0]['output_ids']
        graph = self.curve('graph',stock_proof=target if core_first else None)
        if not graph and self.last_curve_stop not in ('budget','quality'):
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
        if time.time()+1800 < dt.datetime.fromisoformat(self.settings['launch_cutoff']).timestamp():
            write(self.root/'results/native_service_disposition.json',self.native_controls())
        # Full-GPU long-context fallbacks remain eligible if restriction is blocked.
        # Qwen2.5's declared32768 positions include the generated-token budget.
        for context in (16384,32704):
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

    def native_controls(self):
        # Same-board service controls; no cross-device timing or service pooling.
        for acq in range(3):
            for fraction in FRACTIONS:
                label = f'native_f{round(fraction*100)}_a{acq}'
                command = [self.py, self.script/'shoreline_native.py', '--gpu-uuid', self.uuid,
                           '--output', self.root/'runs'/label, '--fraction', str(fraction),
                           '--acquisition', str(acq), '--pilot','--rounding','nearest']
                result = self.execute(label, command, timeout=900)
                if result['status'] != 'complete':
                    limitation(self.root, label, 'Independent native service control failed',
                        str((self.root/'logs'/(label+'.log')).relative_to(PACKAGE)),
                        'Existing known-value streaming/gather and arithmetic native adapter',
                        'Inspect native exception; do not infer decoder bandwidth from useful payload')
                    return {'status': 'partial'}
            publish(self.root, f'H200 {self.uuid} native service acquisition {acq}; decoder transfer not inferred')
        return {'status':'native_controls_captured','qualification':'clock/source/physical-counter joins remain separate'}

    def secondary(self, core_first=False):
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
        if core_first:
            physical=self.traffic()
            write(self.root/'results/P2a_disposition.json',physical)
            publish(self.root,'H200 resumed physical traffic comparison before additional native controls')
            native=self.native_controls()
        else:
            native=self.native_controls()
            physical=self.traffic()
        write(self.root/'results/P2a_disposition.json',physical)
        return {'status': 'captures_saved_scientific_gates_separate', 'native':native,'physical':physical,
                'qualification': 'device-specific results; clock/source/physical-counter joins remain separate'}


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
    stock={'qualified':False,'scope':'unrestricted stock reference; its normalized self-ratio is exactly1'}
    if (len(refs)==3 and stock_proofs and all(r.get('qualified') for r in refs) and
        len({r['source_identity'] for r in refs})==1 and len({r['actual_sm_count'] for r in refs})==1 and
        stock_proofs[-1]['source_identity']==refs[0]['source_identity'] and
        stock_proofs[-1]['actual_sm_count']==refs[0]['actual_sm_count']):
        pooled={m:max(r['clock']['clocks'][m]['maximum'] for r in refs)/min(r['clock']['clocks'][m]['minimum'] for r in refs)
                for m in ('sm_clock_mhz','memory_clock_mhz')}
        if all(v<=1.05 for v in pooled.values()):
            stock.update(qualified=True,actual_sm_count=refs[0]['actual_sm_count'],pooled_clock_ratio=pooled,
                         latency_ms=hierarchical_interval([r['step_ms'] for r in refs]),sources=[r['id'] for r in refs])
    for fraction in FRACTIONS:
        rows = sorted([r for r in clean if r['fraction'] == fraction and
                       not r.get('suffix')], key=lambda r:r['acquisition'])
        point = {'fraction': fraction, 'qualified': False, 'sources': [r['id'] for r in rows],
                 'actual_sm_count': rows[0].get('actual_sm_count') if rows else None,
                 'raw_acquisition_medians_ms': [statistics.median(r['step_ms']) if r.get('step_ms') else None for r in rows],
                 'raw_scope': 'retained observations, not qualified comparisons until every gate passes'}
        combined = rows + refs + green_refs + [r for r in clean if r.get('suffix') in ('_end', '_interleaved')]
        clocks = [r.get('clock', {}).get('clocks', {}) for r in combined]
        if (stock['qualified'] and len(rows) == len(refs) == 3 and all(r.get('qualified') for r in combined) and
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
        counts=[p['actual_sm_count'] for p in candidates]+([stock['actual_sm_count']] if stock['qualified'] else [])
        thresholds[str(margin)] = {'smallest_tested_demonstrated_sm_count':min(counts,default=None),
                                  'restricted_path_smallest_tested_sm_count':min((p['actual_sm_count'] for p in candidates),default=None),
                                  'scope': 'tested counts only, includes qualified full stock itself; no interpolation or AMD thresholds'}
    result = {'engine': engine, 'batch':batch,'context':context,'classification': 'measured-real-GPU', 'points': points, 'thresholds': thresholds,
              'unrestricted_stock':stock,'complete_qualified_curve': stock['qualified'] and all(p['qualified'] for p in points),
              'scope': 'pure decode after verified prefill barrier, wall ms/tick; primary ratios use unrestricted stock; full-green ratios also retained; physical area/HBM capacity unchanged'}
    write(root/'results'/f'curve_{engine}_b{batch}_c{context}.json', result)
    if batch==1 and context==4096:
        write(root/'results'/f'curve_{engine}.json', result)
    lines = [f'\n## {engine} B{batch}/C{context} curve\n', '| Requested fraction | Actual SMs | Qualified | ms/tick | Ratio 95% CI |',
             '|---|---:|---|---:|---|']
    lines.append(f"| stock | {stock.get('actual_sm_count','NA')} | {stock['qualified']} | {stock.get('latency_ms',{}).get('value','NA')} | 1 (normalization) |")
    for p in points:
        lines.append(f"| {p['fraction']} | {p['actual_sm_count']} | {p['qualified']} | {p.get('latency_ms', {}).get('value', 'NA')} | {p.get('ratio', {}).get('ci95', 'NA')} |")
    with (root/'analysis/report.md').open('a') as stream:
        stream.write('\n'.join(lines)+'\n')
    return result



def continuation_task(task, root):
    """Validate a reviewed single-device task before freezing or launching it."""
    import re
    assert re.fullmatch(r'[A-Za-z0-9_]+',task['id']), 'invalid task identity'
    assert task['gpu'] in (0,1), 'only allocated device slots0/1'
    name=task['script']
    assert Path(name).name==name and name.endswith('.py'), 'script basename required'
    assert (PACKAGE/'scripts'/name).is_file(), 'versioned repository script required'
    assert isinstance(task['arguments'],list) and all(isinstance(x,str) for x in task['arguments'])
    assert not any(x=='--gpu-uuid' or x.startswith('--gpu-uuid=') for x in task['arguments']), 'UUID is fixed by the allocation'
    assert 1 <= task.get('timeout_s',1800) <= 7200
    subroot=root/f'gpu{task["gpu"]}'
    assert not (subroot/'results/processes'/(task['id']+'.json')).exists(), 'attempt already exists'
    return subroot


def continue_allocation(root, end):
    """Keep the same authorized job available for result-driven follow-up work.

    Tasks are explicit versioned script/argument records, never shell strings.
    GPU producers retain the same allocation/UUID/isolation/deadline supervisor.
    """
    queue=root/'control';queue.mkdir(exist_ok=True)
    write(root/'STATUS.json',{'status':'awaiting_reviewed_continuation','allocation_end':end.isoformat(),
                             'control_directory':str(queue),'closeout_buffer_s':1800})
    publish(root,'H200 initial queue checkpoint: retain allocation for reviewed follow-up cases')
    cutoff=end.timestamp()-1800
    print('CONTINUATION_QUEUE',queue,flush=True)
    while time.time()<cutoff and not (queue/'finalize').exists():
        for source in sorted(queue.glob('*.task.json')):
            result_path=source.with_name(source.name.replace('.task.json','.result.json'))
            if result_path.exists():continue
            try:
                task=load(source);subroot=continuation_task(task,root)
                session=load(subroot/'session.json')
                session['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
                session_path=subroot/('session_'+task['id']+'.json')
                write(session_path,session)
                subprocess.run([sys.executable,PACKAGE/'scripts/freeze_shoreline.py','--session',session_path,'--activate'],check=True)
                session=load(session_path)
                command=[sys.executable,PACKAGE/'scripts/h200_session.py','--root',subroot,
                         '--label',task['id'],'--gpu','--timeout',str(task.get('timeout_s',1800)),'--',
                         session['python'],Path(session['frozen_source_root'])/'scripts'/task['script'],
                         *task['arguments'],'--gpu-uuid',session['gpu_uuid']]
                env=os.environ.copy();env['H200_SESSION_CONFIG']=str(session_path)
                write(result_path,{'status':'running','task_sha256':sha(source),'session':str(session_path)})
                with (root/'logs'/(task['id']+'_supervisor.log')).open('x') as stream:
                    result=subprocess.run(list(map(str,command)),env=env,stdout=stream,stderr=subprocess.STDOUT)
                write(result_path,{'status':'producer_finished' if result.returncode==0 else 'failed',
                      'exit_code':result.returncode,'task_sha256':sha(source),'scientific_gate':'separate; never inferred from exit0'})
                if result.returncode:
                    limitation(root,task['id'],'Continuation producer failed with exit '+str(result.returncode),
                               str((root/'logs'/(task['id']+'_supervisor.log')).relative_to(PACKAGE)),
                               'Versioned source and allocation-bound supervisor; all original output retained',
                               'Inspect exception/output/clock/isolation gates before a new bounded attempt')
                publish(root,'H200 continuation '+task['id']+': producer disposition')
            except Exception as error:
                write(result_path,{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
                limitation(root,source.name,'Continuation task failed: '+str(error),str(result_path.relative_to(PACKAGE)),
                           'Validated task identity, frozen source and allocation-bound producer',
                           'Inspect the failed task; record a new attempt identity')
        time.sleep(15)


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
    write(root/'environment/allocated_devices.json', {'devices':devices,'slurm_job_id':os.environ['SLURM_JOB_ID'], 'slurm_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'hostname':socket.gethostname()})
    for name, command in [('topology', ['nvidia-smi', 'topo', '-m']), ('runtime', [sys.executable,'-m','pip','freeze'])]:
        (root/'environment'/(name+'.txt')).write_text(subprocess.check_output(command, text=True, stderr=subprocess.STDOUT))
    # Bounded paired smoke precedes isolated measurements; no transfer overlaps a curve.
    p2p_command=[Path(args.cuda)/'bin/nsys','profile','--sample=none','--cpuctxsw=none',
                 '--trace=cuda,nvtx','--capture-range=cudaProfilerApi','--capture-range-end=stop',
                 '--output='+str(root/'profiles/P0_p2p'),sys.executable,PACKAGE/'scripts/h200_p2p_smoke.py',
                 '--devices',root/'environment/allocated_devices.json','--output',root/'results/P0_p2p.json','--profile']
    try:
        with (root/'logs/P0_p2p.log').open('x') as stream:
            subprocess.run(list(map(str,p2p_command)),stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
    except (subprocess.SubprocessError,OSError) as error:
        limitation(root,'P0_p2p','Paired synthetic peer-copy pilot failed: '+str(error),
                   str((root/'logs/P0_p2p.log').relative_to(PACKAGE)),
                   'Only the two allocated UUIDs; explicit peer-access and known-value copy checks',
                   'Inspect topology/profiler/runtime diagnostics; single-device cases remain eligible')
    real_kv_command=[Path(args.cuda)/'bin/nsys','profile','--sample=none','--cpuctxsw=none',
                 '--trace=cuda,nvtx','--capture-range=cudaProfilerApi','--capture-range-end=stop',
                 '--output='+str(root/'profiles/P0_real_hf_kv'),sys.executable,PACKAGE/'scripts/h200_real_kv_smoke.py',
                 '--devices',root/'environment/allocated_devices.json',
                 '--config',PACKAGE/'h200/20261125_priority_preparation/qwen32b_smoke.json',
                 '--output',root/'results/P0_real_hf_kv.json','--profile']
    if args.core_first:
        reused=prior_kv_software_validation(args.prior_kv_pilot,load(PACKAGE/'h200/20261125_priority_preparation/qwen32b_smoke.json'))
        write(root/'results/P0_prior_KV_software_validation.json',reused)
        limitation(root,'P0_new_pair_KV','Fresh paired HF handoff deferred until after first complete SM sweep; prior adapter validation applies only to its recorded physical pair',
                   'results/P0_prior_KV_software_validation.json',
                   'Validated unchanged adapter/model source hashes; fresh allocated-pair peer-copy smoke remains separate',
                   'At a safe completed-curve boundary, run the existing real-KV pilot on the new allocated pair; never relabel prior UUIDs')
    else:
        try:
            with (root/'logs/P0_real_hf_kv.log').open('x') as stream:
                subprocess.run(list(map(str,real_kv_command)),stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=600)
        except (subprocess.SubprocessError,OSError) as error:
            limitation(root,'P0_real_hf_kv','Real whole-model HF KV functional handoff pilot failed: '+str(error),
                       str((root/'logs/P0_real_hf_kv.log').relative_to(PACKAGE)),
                       'All-layer BF16 transfer and receiver decode compared with source and independent generate; exact gate unchanged',
                       'Inspect cache layout/position/output diagnostics; serving connector remains unvalidated')
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
        if args.core_first:command+=['--core-first']
        processes.append((subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT),stream))
    while any(p.poll() is None for p,s in processes):
        time.sleep(15)
    for p,s in processes:s.close()
    continue_allocation(root,end)
    manifest=load(root/'results/case_manifest.json')
    for row in manifest['rows']:
        if row.get('reviewed_evidence'):
            continue
        if row['id']=='P0':
            row.update(status='partial',reason='Inventory, correctness and profiler/placement records require component-level inspection')
        elif row['id']=='P1a':
            curves=[load(p) for p in (root/'gpu0/results').glob('curve_*.json')
                    if load(p).get('batch')==1 and load(p).get('context')==4096]
            graph_ok=any(c['engine']=='graph' and c['complete_qualified_curve'] for c in curves)
            eager_ok=any(c['engine']=='eager' and c['complete_qualified_curve'] for c in curves)
            row.update(status='complete' if graph_ok else 'partial' if eager_ok else 'blocked',
                       optimized_curve_qualified=graph_ok,eager_fallback_qualified=eager_ok,
                       reason='Eager fallback does not complete a blocked optimized curve; engine-specific evidence remains explicit')
        elif row['id']=='P1b':
            expanded=[load(p) for p in (root/'gpu0/results').glob('graph_b*_c*.json') if load(p).get('batch',1)>1 or load(p).get('context',0)>4096]
            row.update(status='partial' if expanded else 'pending',
                       reason='Only source/output/clock/placement-qualified shapes count; unmeasured refinements remain pending')
        elif row['id']=='P2a':
            captures=[p for i in (0,1) for p in (root/f'gpu{i}/results').glob('P2a*.json')]
            row.update(status='partial' if captures else 'pending',
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
    root=PACKAGE/'h200'/(f'{dt.datetime.now():%Y%m%d}_job{os.environ["SLURM_JOB_ID"]}_bootstrap'+
                        (f'_repair{args.bootstrap_attempt}' if args.bootstrap_attempt else ''))
    root.mkdir(exist_ok=False)
    for folder in ('environment','results','logs','analysis'):
        (root/folder).mkdir(parents=True,exist_ok=True)
    (root/'analysis/report.md').write_text('# Allocated staging and launch\n\nNo model measurement inferred from installation.\n')
    state=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)
    (root/'environment/slurm.txt').write_text(state)
    (root/'environment/host.txt').write_text(socket.gethostname()+'\n')
    if os.environ.get('H200_VISIBILITY_MANIFEST'):
        visibility=load(os.environ['H200_VISIBILITY_MANIFEST'])
        assert visibility['slurm_job_id']==os.environ['SLURM_JOB_ID']
        assert visibility['selected_cuda_visible_devices']==os.environ['CUDA_VISIBLE_DEVICES']
        write(root/'environment/allocated_visibility_repair.json',visibility)
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
        flags=['--core-first','--prior-kv-pilot',str(args.prior_kv_pilot)] if args.core_first else []
        subprocess.run([runtime,'-u',PACKAGE/'scripts/h200_priority_campaign.py','--cuda',args.cuda]+flags,check=True)
    except Exception as error:
        write(root/'STATUS.json',{'status':'blocked','error':str(error),'traceback':traceback.format_exc()})
        limitation(root,'bootstrap','Allocated staging/launch failed: '+str(error),str(root.relative_to(PACKAGE)),
                   'Preserved each stage log and pinned requirements; no system changes','Inspect failed stage and use existing allocation if time remains')
        publish(root,'H200 staging failure: preserve logs and hold existing allocation for bounded repair')
        control=root/'control';control.mkdir(exist_ok=True)
        fields=dict(x.split('=',1) for x in state.split() if '=' in x)
        cutoff=dt.datetime.fromisoformat(fields['EndTime']).astimezone().timestamp()-1800
        print('STAGING_RECOVERY_HOLD',control,flush=True)
        while time.time()<cutoff and not (control/'finalize').exists():
            trigger=control/'retry_staging'
            if trigger.exists() and args.bootstrap_attempt==0:
                trigger.rename(root/'results/retry_staging.request')
                subprocess.run([sys.executable,'-u',PACKAGE/'scripts/h200_priority_campaign.py',
                    '--bootstrap','--bootstrap-attempt','1','--cuda',args.cuda]+
                    (['--core-first','--prior-kv-pilot',str(args.prior_kv_pilot)] if args.core_first else []),check=True)
                break
            time.sleep(15)
    finally:
        publish(root,'H200 priority: durable staging/launch checkpoint')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',action='store_true');p.add_argument('--bootstrap-attempt',type=int,choices=(0,1),default=0)
    p.add_argument('--cuda');p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--root',type=Path);p.add_argument('--session',type=Path)
    p.add_argument('--core-first',action='store_true',help='target3+3 graph smoke/HF before SM sweep; defer fresh paired KV handoff')
    p.add_argument('--prior-kv-pilot',type=Path)
    a=p.parse_args()
    if a.core_first and a.worker is None and a.prior_kv_pilot is None:p.error('--core-first coordinator/bootstrap requires --prior-kv-pilot')
    if a.bootstrap:
        bootstrap(a)
    elif a.worker is None:
        coordinator(a)
    else:
        worker=Worker(a.root,a.session)
        try:
            outcome=worker.primary(core_first=a.core_first) if a.worker==0 else worker.secondary(core_first=a.core_first)
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
