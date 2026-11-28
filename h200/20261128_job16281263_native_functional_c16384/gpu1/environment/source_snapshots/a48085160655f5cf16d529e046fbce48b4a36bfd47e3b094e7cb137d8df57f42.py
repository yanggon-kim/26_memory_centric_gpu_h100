#!/usr/bin/env python3
"""CPU ownership and historical-functional-evidence gates for the next allocation."""
import copy
import json
from pathlib import Path
import tempfile
import types
from unittest.mock import patch
import h200_allocated_visibility as visibility
import h200_priority_campaign as campaign


def rejected(call):
    try:
        call()
    except (AssertionError, KeyError):
        return
    raise AssertionError('invalid allocation/evidence accepted')


def main():
    state='JobId=123 JobState=RUNNING NumNodes=1 NodeList=node-1 UserId=test(42) AllocTRES=cpu=16,gres/gpu:h200=2'
    inventory='GPU-a, NVIDIA H200\nGPU-b, NVIDIA H200\n'
    check=lambda s=state,i=inventory:visibility.validate(s,i,'123','node-1.example',42)
    assert check()==['GPU-a','GPU-b']
    for old,new in [('RUNNING','PREEMPTED'),('node-1','node-2'),('test(42)','test(43)'),('h200=2','h200=8'),('JobId=123','JobId=456')]:
        rejected(lambda old=old,new=new:check(state.replace(old,new)))
    for bad in [inventory*4,inventory.replace('GPU-b','GPU-a'),inventory.replace('H200','H100'),inventory.splitlines()[0]]:
        rejected(lambda bad=bad:check(i=bad))
    root=campaign.PACKAGE
    prior=json.loads((root/'h200/20261126_job16163464_priority/results/P0_real_hf_kv.json').read_text())
    cfg=json.loads((root/'h200/20261125_priority_preparation/qwen32b_smoke.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'prior.json'
        path.write_text(json.dumps(prior))
        result=campaign.prior_kv_software_validation(path,cfg)
        assert result['new_pair_handoff_validated'] is False
        assert result['status']=='prior_software_functional_validation_only'
        for mutate in [lambda d:d.update(real_KV_handoff_verified=False),
                       lambda d:d['versions'].update(transformers='wrong'),
                       lambda d:d['source_sha256'].update({'h200_session.py':'wrong'}),
                       lambda d:d['trials'][0].update(correctness='failed'),
                       lambda d:d['cache_layouts'].pop(),
                       lambda d:d.update(receiver_prefill_executed=True)]:
            bad=copy.deepcopy(prior);mutate(bad);path.write_text(json.dumps(bad))
            rejected(lambda:campaign.prior_kv_software_validation(path,cfg))
    # A quality-rejected complete graph acquisition set must not trigger an
    # expensive eager restriction fallback. Failed graph proof may trigger it.
    for stop,reference_passed in [('quality',True),('proof',True),('quality',False)]:
        with tempfile.TemporaryDirectory() as tmp, patch.object(campaign,'limitation'), patch.object(campaign,'publish'):
            root=Path(tmp);(root/'results').mkdir()
            run=root/'run.json';run.write_text(json.dumps({'cases':[{'trials':[{'output_ids':[[1,2]]}]}]}))
            calls=[];curves=[]
            def case(engine,*args,**kwargs):
                if engine=='eager':assert worker.capture_bind is False
                calls.append((engine,kwargs))
                return {'status':'complete','qualified':True,'run':str(run)}
            worker=types.SimpleNamespace(root=root,py='CPU-fake',script=root,uuid='GPU-fake',expected=None,
                capture_bind=True,
                settings={'launch_cutoff':'2000-01-01T00:00:00+00:00'},last_curve_stop=None,
                execute=lambda *a,**kw:{'status':'complete'},case=case,
                reference=lambda *a,**kw:reference_passed,traffic=lambda:{'status':'CPU-fake'})
            def curve(engine,**kwargs):
                curves.append((engine,kwargs));worker.last_curve_stop=stop;return False
            worker.curve=curve
            outcome=campaign.Worker.primary(worker,core_first=True)
            assert calls[0][0]=='graph' and calls[0][1]['pilot'] is True
            assert all(call[1].get('context',4096)==4096 for call in calls)
            if not reference_passed:
                assert outcome['status']=='blocked' and not curves
            else:
                assert curves[0][1]['stock_proof']['run']==str(run)
                assert any(engine=='eager' for engine,kw in calls)==(stop=='proof')
    print('H200 resume ownership, prior-evidence and core-first fallback contracts passed')


if __name__=='__main__':
    main()
