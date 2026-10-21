#!/usr/bin/env python3
"""Independent references, full-scope decoder counters and frozen held-out validation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback


def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(session):
    root=session.resolve().parent;package=Path(__file__).resolve().parents[1]
    os.environ['H200_SESSION_CONFIG']=str(session.resolve())
    import h200_session as env
    import shoreline_campaign as campaign
    # Fit/feature mathematics must use the same sealed tree as the GPU workers.
    sys.path.insert(0,str(Path(env.SESSION['frozen_source_root'])/'scripts'))
    from shoreline_jobs import records,value
    from decoder_calibration_model import CONDITIONS,design_identity,fit,validate,predict,parameter_uncertainty,resampled_center,full_decode_publication
    from shoreline_statistics import hierarchical_interval,latency_bounds
    from shoreline_jobs import functional_smoke_reference
    env.verify_allocation()
    for key in ('PACKAGE','PYTHON','UUID','dump','run','python_command'):setattr(campaign,key,getattr(env,key))
    state={'status':'running','protocol':'decoder_calibration_v4','design_identity':design_identity(),'steps':[]}
    ledger=root/'results/campaign_queue.json';result_path=root/'results/decoder_calibration_v4.json'
    result={'status':'collecting','conditions':[],'independent_memory':{},'prediction':None,
        'design_identity':design_identity(),'issued_compute_work':None,
        'scope':'new memory-conditioned graph BF16/TP1 decoder family; useful-work service-reference model and assumption bounds, not physical busy time'}
    def save():env.dump(ledger,state);env.dump(result_path,result)
    def checkpoint(message):save();campaign.checkpoint(root,message)
    save()
    def execute(label,command,gpu=True,timeout=900):
        step={'id':label,'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        state['steps'].append(step);save()
        outcome=env.run(root,label,command,timeout,gpu=gpu)
        step.update(status='command_complete' if outcome['status']=='complete' else outcome['status'],
            exit_code=outcome.get('exit_code'),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if outcome['status']=='pending':raise RuntimeError('launch cutoff; preserve calibration without expansion')
        save();return outcome
    def worker(label,mode,headers,config=None,b=None,c=None,pattern='stream',profile=False):
        base=root/'runs'/label
        command=env.python_command('decoder_calibration_worker.py','--gpu-uuid',env.UUID,'--output',base,
            '--mode',mode,'--headers',headers,'--pattern',pattern)
        if config:command+=['--config',str(config)]
        if b is not None:command+=['--batch',str(b),'--context',str(c)]
        if profile:
            command+=['--external-profiler']
            include='calibration_decode_selected/' if mode=='decode' else 'calibration_memory_selected/'
            report=root/'profiles'/(label+'.ncu-rep')
            command=[str(env.CUDA/'bin/ncu'),'--replay-mode','app-range','--clock-control','none',
                '--cache-control','none','--nvtx','--nvtx-include',include,'--metrics',
                'dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum','--export',str(report)]+command
        outcome=execute(label,command,timeout=1200)
        natives=list(base.glob('replay_*/run.json'))
        row={'id':label,'qualified':False,'producer':str((root/'results/processes'/(label+'.json')).relative_to(package)),
            'source_freeze_identity':outcome.get('source_freeze_identity')}
        if outcome['status']!='complete':return row
        if not profile:
            if len(natives)!=1 or load(natives[0])['status']!='complete':return row
            data=load(natives[0]);q=campaign.clock_qualification(data,outcome['monitor']['source'])
            row.update(qualified=q['passed'],clock=q,native=str(natives[0].relative_to(package)),native_sha256=sha(natives[0]))
            return row
        export=execute(label+'_export',[str(env.CUDA/'bin/ncu'),'--import',str(report),
            '--page','raw','--print-units','base','--csv'],gpu=False,timeout=120)
        metrics=records(root/'logs'/(label+'_export.log')) if export['status']=='complete' else []
        selected={r['Metric Name']:value(r['Metric Value']) for r in metrics}
        row['ranges']=len({r['ID'] for r in metrics})
        row['physical_bytes']=sum(selected.get(k) or 0 for k in ('dram__bytes_read.sum','dram__bytes_write.sum'))
        row['read_bytes']=selected.get('dram__bytes_read.sum');row['write_bytes']=selected.get('dram__bytes_write.sum')
        paths=list(base.glob('replay_*/'+('profile_interval_output.json' if mode=='decode' else 'counter_publication.json')))
        pubs=[load(p) for p in paths]
        q=campaign.clock_qualification({'cases':[{'trials':pubs}]},outcome['monitor']['source']) if pubs else {'passed':False}
        row.update(publications=[{'path':str(p.relative_to(package)),'sha256':sha(p)} for p in paths],clock=q)
        if mode=='decode':
            reference=load(functional_smoke_reference(root,'bridge','graph',b,c))['cases'][0]
            correct=bool(pubs) and all(full_decode_publication(p,reference,b) for p in pubs)
        else:correct=bool(pubs) and all(p.get('correctness')=='passed' and p['case']==pattern for p in pubs)
        row.update(exact_publications=correct,qualified=row['ranges']==1 and correct and q['passed'] and
            isinstance(row['read_bytes'],(float,int)) and row['read_bytes']>0 and isinstance(row['write_bytes'],(float,int)))
        return row
    def pooled(rows):
        if not rows or not all(r['qualified'] for r in rows):return False
        identities={r['source_freeze_identity'] for r in rows}
        if len(identities)!=1 or None in identities:return False
        for metric in ('sm_clock_mhz','memory_clock_mhz'):
            clocks=[r['clock']['clocks'][metric] for r in rows]
            if max(c['maximum'] for c in clocks)/min(c['minimum'] for c in clocks)>1.05:return False
        return True
    try:
        prerequisite=env.SESSION.get('required_graph_pilot_job_id')
        if prerequisite:
            candidates=list((package/'h200').glob('*job'+str(prerequisite)))
            if len(candidates)!=1:raise RuntimeError('required recovery allocation evidence is unavailable')
            proof=candidates[0]/'results/D3_graph_clock.json'
            producer=candidates[0]/'results/processes/D3_graph_clock.json'
            if not proof.exists() or not producer.exists():raise RuntimeError('graph clock pilot was not completed; no identical clock-rejection retry')
            p=load(proof);native=load(producer)
            result['readiness_prerequisite']={'path':str(proof.relative_to(package)),'sha256':sha(proof),
                'scope':'adapter readiness only; no timing/sample pooling across allocations'}
            if not (p['status']=='complete' and p.get('conditioned') and p.get('exact_ids') and
                    native['status']=='complete' and not native.get('monitor',{}).get('failure')):
                raise RuntimeError('previous conditioned graph pilot rejected; calibration blocked rather than retrying the same failed protocol')
        campaign.materialize(root,'bridge')
        headers=root/'environment/model_headers/bridge_v2.json'
        # First physical native reference is the bounded reader/producer pilot.
        for pattern in ('stream','paged128'):
            rows=[]
            for acq in range(3):
                timed=worker(f'C4_memory_{pattern}_acq{acq:02d}','memory',headers,pattern=pattern)
                if not timed['qualified']:
                    result['independent_memory'][pattern]=rows+[{'ordinary':timed}]
                    checkpoint(f'Memory-reference pilot {pattern} failed; no counter or decoder expansion')
                    raise RuntimeError('independent memory ordinary gate failed')
                counter=worker(f'C4_memory_{pattern}_counter_acq{acq:02d}','memory',headers,pattern=pattern,profile=True)
                rows.append({'ordinary':timed,'counter':counter});result['independent_memory'][pattern]=rows
                checkpoint(f'Calibration memory reference {pattern} acquisition{acq}: ordinary={timed["qualified"]} counter={counter["qualified"]}')
                if not timed['qualified'] or not counter['qualified']:raise RuntimeError('independent memory-service pilot/gate failed; no decoder expansion')
        all_rows=[v for rows in result['independent_memory'].values() for r in rows for v in r.values()]
        if not pooled(all_rows):raise RuntimeError('independent memory references fail pooled source/clock gate')

        def collect(identity,b,c,split):
            row={'id':identity,'batch':b,'context':c,'split':split,'qualified':False,'acquisitions':[]}
            result['conditions'].append(row);save()
            for engine in ('hf_eager','graph'):
                q=campaign.acquisition(root,'bridge',engine,b,c,0,smoke=True)
                if q['status']!='passed':raise RuntimeError('fresh exact functional reference producer failed')
            if functional_smoke_reference(root,'bridge','graph',b,c) is None:raise RuntimeError('exact HF/graph reference unavailable')
            cfg=campaign.make_config(root,'bridge','graph',b,c,10)
            profile=load(cfg);profile['trials']=3;profile['calibration_whole_decode_range']=True
            profile['shoreline_protocol'].update(trials_per_acquisition=3,counter_range_protocol='full_decode_v4',
                scope='instrumented full63-step range; only first measured trial selected; no ordinary timing join from profiler time')
            p=root/'configs'/('C4_'+identity+'_counter.json');env.dump(p,profile)
            for acq in range(3):
                compute=worker(f'C4_compute_{identity}_acq{acq:02d}','compute',headers,b=b,c=c)
                a={'compute':compute};row['acquisitions'].append(a)
                if not compute['qualified']:
                    checkpoint(f'Compute-reference gate rejected {identity}; decoder expansion stopped')
                    raise RuntimeError('compute-reference pilot/gate failed')
                timed=worker(f'C4_decode_{identity}_acq{acq:02d}','decode',headers,config=cfg,b=b,c=c)
                a['ordinary']=timed
                if timed['qualified']:
                    native=load(package/timed['native']);reference=load(functional_smoke_reference(root,'bridge','graph',b,c))['cases'][0]
                    timed['exact_reference_ids']=native['cases'][0]['output_digests_identical'] and all(
                        t['output_ids']==reference['trials'][0]['output_ids'] for t in native['cases'][0]['trials'])
                    timed['qualified']=timed['exact_reference_ids']
                if not timed['qualified']:
                    checkpoint(f'Ordinary decoder clock/producer gate rejected {identity}; counter expansion stopped')
                    raise RuntimeError('ordinary decoder pilot/gate failed')
                counter=worker(f'C4_decode_{identity}_counter_acq{acq:02d}','decode',headers,config=p,b=b,c=c,profile=True)
                a['counter']=counter
                checkpoint(f'Decoder calibration {identity} acquisition{acq}: qualified={all(x["qualified"] for x in a.values())}')
                if not all(x['qualified'] for x in a.values()):raise RuntimeError('bounded decoder/reference acquisition failed; no favorable retries')
            qrows=all_rows+[v for a in row['acquisitions'] for v in a.values()]
            row['qualified']=pooled(qrows)
            C={tile:[] for tile in (512,1024)};M={p:[] for p in ('stream','paged128')};T=[]
            for index,a in enumerate(row['acquisitions']):
                controls=load(package/a['compute']['native'])['cases']
                assert len(controls)==14 and all(c['correctness']=='passed' and len(c['trials'])==10 for c in controls)
                d=load(package/a['ordinary']['native'])['cases'][0]
                assert len(d['trials'])==10 and all(t['decode_iterations']==63 for t in d['trials'])
                T.append([t['wall_ms'] for t in d['trials']])
                for tile in C:
                    selected=[c for c in controls if c['metadata']['tile']==tile]
                    C[tile].append([sum(c['metadata']['full_scope_useful_flops']/c['metadata']['useful_flops_per_launch']*
                        c['trials'][j]['gpu_event_ms'] for c in selected) for j in range(10)])
                for pattern in M:
                    ref=result['independent_memory'][pattern][index]
                    native=load(package/ref['ordinary']['native'])['cases'][0]
                    D=a['counter']['physical_bytes'];service_D=ref['counter']['physical_bytes']
                    M[pattern].append([D/service_D*t['gpu_event_ms'] for t in native['trials']])
            row.update(C_acquisitions=C[1024],M_acquisitions=M['stream'],T_acquisitions=T,
                C_ms=statistics.median(statistics.median(a) for a in C[1024]),
                M_ms=statistics.median(statistics.median(a) for a in M['stream']),
                T_ms=statistics.median(statistics.median(a) for a in T))
            row['statistical_intervals']={name:hierarchical_interval(values) for name,values in
                [('C_reference',C[1024]),('M_reference',M['stream']),('ordinary_T',T)]}
            row['reference_assumption_scenarios']={'C_tile512_ms':statistics.median(statistics.median(a) for a in C[512]),
                'M_paged128_ms':statistics.median(statistics.median(a) for a in M['paged128'])}
            row['coupled_bounds']=latency_bounds(row['T_ms'],row['C_ms'],row['M_ms'])
            row['qualified']=row['qualified'] and row['coupled_bounds']['feasible']
            checkpoint(f'Qualified service-reference features and bounds: {identity} qualified={row["qualified"]}')
            return row

        training=[collect(*condition) for condition in CONDITIONS if condition[3]=='train']
        # No held-out decoder timing has been launched at this point.
        frozen=fit(training);frozen['parameter_uncertainty']=parameter_uncertainty(training)
        frozen_path=root/'results/decoder_model_frozen_v4.json'
        env.dump(frozen_path,frozen);freeze=sha(frozen_path)
        result['frozen_model']={'path':str(frozen_path.relative_to(package)),'sha256':freeze}
        checkpoint('Freeze decoder coefficients and training gates before unseen held-out decoder acquisition')
        if not frozen['training_gate']:
            raise RuntimeError('training calibration exceeds predeclared10% error; held-out conditions remain unexposed')
        held=[collect(*condition) for condition in CONDITIONS if condition[3]=='held_out']
        assert sha(frozen_path)==freeze,'frozen model was modified after held-out exposure'
        result['validation']=validate(load(frozen_path),held,freeze)
        import numpy as np
        rng=np.random.default_rng(20261013)
        for r in held:
            predictions=[predict(p,{**r,'C_ms':resampled_center(r['C_acquisitions'],rng),
                'M_ms':resampled_center(r['M_acquisitions'],rng)}) for p in frozen['parameter_uncertainty']['draws']]
            r['parameter_reference_prediction_interval']=np.quantile(predictions,[.025,.975]).tolist()
            r['interval_scope']='conditional parameter/reference uncertainty; excludes model misspecification and is not an accuracy acceptance gate'
        global_rows=all_rows+[v for c in result['conditions'] for a in c['acquisitions'] for v in a.values()]
        if not pooled(global_rows):
            result['validation'].update(status='rejected',reason='global decoder/reference source/clock envelope failed')
        result['reference_sensitivity']=[{'id':r['id'],'main_ms':predict(frozen,r),
            'alternative_reference_ms':predict(frozen,{**r,'C_ms':r['reference_assumption_scenarios']['C_tile512_ms'],
                'M_ms':r['reference_assumption_scenarios']['M_paged128_ms']})} for r in held]
        result['classification_status']='service-equivalent bounds only; physical overlap is not identified'
        result['status']=result['validation']['status']
    except Exception as error:
        result.update(status='unavailable_or_rejected',error=str(error),traceback=traceback.format_exc())
    finally:
        lines=['# Decoder calibration and held-out validation','',result['scope'],
            '', 'Status: '+result['status'], '', 'No exact physical compute/memory/overlap partition is claimed.',
            '', 'Useful-work reference, reference-shape/access-pattern assumptions and uncertainty are retained in `results/decoder_calibration_v4.json`.']
        if result.get('validation'):lines+=['',json.dumps(result['validation'],indent=2)]
        lines+=['','| Condition | Split | Qualified | Decode ms/step | C reference ms/step | M reference ms/step |',
            '|---|---|---|---:|---:|---:|']
        for row in result['conditions']:
            values=[str(row[k]/63) if k in row else 'NA' for k in ('T_ms','C_ms','M_ms')]
            lines.append('|'+ '|'.join([row['id'],row['split'],str(row['qualified'])]+values)+'|')
        lines+=['','C/M are reference-service equivalents, not measured physical busy durations. Coupled residual/overlap endpoints, uncertainty and reference sensitivity are in the JSON; do not stack arbitrary endpoint extrema.']
        if result.get('error'):lines+=['',result['error']]
        (root/'analysis/decoder_calibration_v4.md').write_text('\n'.join(lines)+'\n')
        state['status']='commands_finished_scientific_gates_separate'
        campaign.update_job(root,'J9','partial','Decoder calibration disposition: '+result['status']+
            '; service-reference validation is separate from physical overlap and architecture feasibility')
        env.dump(root/'results/decoder_calibration_disposition.json',{'scientific_status':result['status'],
            'design_identity':design_identity(),'validation':result.get('validation'),'error':result.get('error')})
        checkpoint('Publish decoder calibration, frozen model/held-out gates, rejected cases and reference bounds')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True)
    main(p.parse_args().session)
