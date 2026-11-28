"""CPU-only candidate startup-budget repair, not a GPU controller.

Use only source/model/settings-matched, archive-verified lifecycle observations.
Clock-rejected/partial captures supply runtime bounds only, never performance
samples. Current queued connector experiment and original cohort are unchanged.
"""
import math


def role_timeout(elapsed_s,completed_lifecycle_s,status):
    assert status in ('complete','timeout')
    assert type(elapsed_s) in (int,float) and math.isfinite(elapsed_s) and elapsed_s>0
    assert 0<len(completed_lifecycle_s)<=13
    assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in completed_lifecycle_s)
    assert sum(completed_lifecycle_s)<=elapsed_s+.001
    missing=13-len(completed_lifecycle_s)
    projected=elapsed_s+missing*max(completed_lifecycle_s)
    timeout=max(300,math.ceil(1.5*projected+60))
    assert timeout<=660, 'complete role bound exceeds unchanged660second watchdog'
    return {'timeout_s':timeout,'observed_process_elapsed_s':elapsed_s,
        'observed_completed_requests':len(completed_lifecycle_s),'missing_to_fixed13requests':missing,
        'elapsed_is_lower_bound':status=='timeout','projected_full_process_s':projected,
        'formula':'max(300,ceil(1.5*(observed elapsed + missing-to13 * slowest completed lifecycle)+60))',
        'performance_samples_used':0}


def propose(pilot,observations):
    assert pilot['acquisitions']==3 and pilot['warmups_per_condition']==3 and pilot['timed_per_condition']==10
    expected={('stock_before','stock'),('native','producer'),('native','consumer'),('stock_after','stock')}
    phases={k:max(300,v) for k,v in pilot['phase_timeout_s'].items()}
    assert set(phases)=={'stock_before','native','stock_after'}
    revised=[]
    for observation in observations:
        assert (observation['condition'],observation['role']) in expected
        assert observation['source_role_sha256']==pilot['role_source_sha256']
        assert observation['context']==pilot['context'] and observation['engine_settings']==pilot['engine_settings']
        row=role_timeout(observation['elapsed_s'],observation['completed_lifecycle_s'],observation['process_status'])
        phases[observation['condition']]=max(phases[observation['condition']],row['timeout_s'])
        revised.append(dict(condition=observation['condition'],role=observation['role'],**row))
    assert observations and all(v<=660 for v in phases.values())
    preserved={k:v for k,v in pilot.items() if k not in ('roles','phase_timeout_s','required_s','classification','margin_scope')}
    return {**preserved,'protocol':'runtime_startup_envelope_v2_candidate','classification':'CPU runtime admission proposal only',
        'phase_timeout_s':phases,'required_s':3*sum(phases.values())+360,
        'acquisitions':3,'warmups_per_condition':3,'timed_per_condition':10,
        'all_bounds_within_existing660second_watchdog':True,
        'original_pilot_phase_timeout_s':dict(pilot['phase_timeout_s']),
        'original_pilot_role_estimates':pilot.get('roles',[]),
        'margin_scope':'At least300s per role,50percent runtime envelope plus60s, then360s cohort overhead; unchanged660s role watchdog and actual allocation closeout reserve still apply.',
        'observations':revised,'performance_samples_used':0,'GPU_execution':False,
        'scope':'Operational timeout repair only; role/collector/output/source/clock/isolation/drift gates and trial counts unchanged. A future controller must verify observation provenance, freeze/promote this version and recheck actual complete-cohort time. No current GPU launch or favorable-clock resampling.'}


if __name__=='__main__':
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raw=args.inputs.read_bytes();inputs=json.loads(raw)
    result=propose(inputs['pilot'],inputs['observations'])
    result['inputs_sha256']=hashlib.sha256(raw).hexdigest()
    result['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('phase_timeout_s','required_s','performance_samples_used','GPU_execution')}))
