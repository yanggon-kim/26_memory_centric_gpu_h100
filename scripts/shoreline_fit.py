#!/usr/bin/env python3
"""Common-family held-out native fit; independent reference rates remain scoped."""
import argparse
import json
from pathlib import Path
import statistics
import hashlib


def load(p):return json.loads(Path(p).read_text())
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')


def fit(root):
    import numpy as np
    from shoreline_statistics import hierarchical_interval,latency_bounds
    runs=[]
    for acq in range(3):
        p=root/'runs'/f'J4_service_extension_acq{acq:02d}/run.json'
        if p.exists() and load(p)['status']=='complete':runs.append((p,load(p)))
    output={'status':'unavailable','prediction':None,'missing_reason':'three complete independent extension acquisitions required',
            'classification':'empirical native service-equivalent fit; not physical wall-time decomposition or decoder calibration'}
    if len(runs)!=3:
        save(root/'results/shoreline_native_heldout_fit.json',output);return output
    assert len({json.dumps(d['source_sha256'],sort_keys=True) for p,d in runs})==1,'native numerical source changed across acquisitions'
    proof=root/'results/J4_same_pass_overlap.json'
    if not proof.exists() or load(proof).get('status')!='complete':
        output['missing_reason']='six-condition actual same-pass overlap proof required'
        save(root/'results/shoreline_native_heldout_fit.json',output);return output
    from shoreline_campaign import clock_qualification
    conditions=[]
    for ratio in (1,2,4,8,16,32):
        modes={};clocks=[];all_clock_checks=True
        for mode in ('empty','compute','memory','concurrent'):
            acquisitions=[]
            for p,d in runs:
                case=next(c for c in d['cases'] if c['id']==f'overlap_ratio{ratio}_{mode}')
                assert case['correctness']=='passed' and len(case['trials'])==10
                acquisitions.append([t['gpu_event_ms'] for t in case['trials']])
                process=load(root/'results/processes'/(p.parent.name+'.json'))
                assert process['status']=='complete' and not process['monitor']['failure'],'native isolation rejected'
                q=clock_qualification({'cases':[case]},process['monitor']['source'])
                all_clock_checks=all_clock_checks and q['passed']
                for metric,extrema in q.get('clocks',{}).items():
                    clocks.append((metric,extrema['minimum'],extrema['maximum']))
            modes[mode]={'raw_acquisitions':acquisitions,'summary':hierarchical_interval(acquisitions)}
        pooled={metric:max(x[2] for x in clocks if x[0]==metric)/min(x[1] for x in clocks if x[0]==metric)
                if any(x[0]==metric for x in clocks) else None
                for metric in ('sm_clock_mhz','memory_clock_mhz')}
        valid=all_clock_checks and all(v is not None and v<=1.05 for v in pooled.values())
        conditions.append({'ratio':ratio,'modes':modes,'clock_qualified_endpoints':valid,
                           'pooled_clock_ratios_including_interior_samples':pooled})
    eligible=[c for c in conditions if c['clock_qualified_endpoints']]
    output.update(conditions=conditions)
    if len(eligible)<5:
        output['missing_reason']='fewer than five clock-qualified distinct native conditions'
        save(root/'results/shoreline_native_heldout_fit.json',output);return output
    rng=np.random.default_rng(20261013);order=rng.permutation(len(eligible))
    ntrain=int(.7*len(eligible));train=[eligible[i] for i in order[:ntrain]];held=[eligible[i] for i in order[ntrain:]]
    def values(c):
        return tuple(c['modes'][m]['summary']['value'] for m in ('compute','memory','concurrent'))
    C,M,T=map(np.asarray,zip(*(values(c) for c in train)))
    best=None
    for eta in np.linspace(0,1,1001):
        residual=max(0,float(np.mean(T-C-M+eta*np.minimum(C,M))))
        prediction=C+M-eta*np.minimum(C,M)+residual
        error=float(np.mean((prediction-T)**2))
        if best is None or error<best[0]:best=(error,float(eta),residual)
    _,eta,residual=best
    predictions=[]
    for c in held:
        compute,memory,t=values(c);pred=compute+memory-eta*min(compute,memory)+residual
        predictions.append({'ratio':c['ratio'],'measured_ms':t,'prediction_ms':pred,
                            'relative_error':abs(pred-t)/t,'coupled_feasibility':latency_bounds(t,compute,memory)})
    threshold=load(root/'session.json').get('native_prediction_max_relative_error',.10)
    valid=all(x['relative_error']<=threshold for x in predictions)
    output.update(status='validated_for_native_family' if valid else 'rejected',
                  missing_reason=None if valid else 'held-out native error exceeds predeclared threshold',
                  parameters={'eta_common':eta,'O_common_ms':residual},
                  train_condition_ids=[c['ratio'] for c in train],held_out_condition_ids=[c['ratio'] for c in held],
                  held_out=predictions,seed=20261013,split='70/30 condition-level, rounded to whole conditions',
                  heldout_max_relative_error_threshold=threshold,
                  sources=[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,d in runs],
                  decoder_transfer='not validated; no native-fit coefficient is automatically applied to decoder')
    save(root/'results/shoreline_native_heldout_fit.json',output)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    print(json.dumps({k:v for k,v in fit(p.parse_args().root.resolve()).items() if k in ('status','missing_reason','parameters','held_out')}))
