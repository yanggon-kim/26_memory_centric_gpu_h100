#!/usr/bin/env python3
"""CPU contracts for resource rounding, token gates and incomplete-curve rejection."""
import copy
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import h200_priority_campaign as campaign
import shoreline_green


def main():
    ids=[[3,4,5]]
    data={'status':'complete','cases':[{'generated_tokens':3,'warmup_output_ids':[ids,ids,ids],
          'output_digests_identical':True,'trials':[{'output_ids':ids,'prefill_barrier':{'passed':True},
          'decode_iterations':2} for _ in range(3)]}]}
    assert campaign.valid_tokens(data,ids)
    bad=copy.deepcopy(data);bad['cases'][0]['warmup_output_ids'][0]=[[3,4,6]]
    assert not campaign.valid_tokens(bad,ids)
    bad=copy.deepcopy(data);bad['cases'][0]['trials'][0]['decode_iterations']=3
    assert not campaign.valid_tokens(bad,ids)
    bad=copy.deepcopy(data);bad['status']='running'
    assert not campaign.valid_tokens(bad,ids)
    with tempfile.TemporaryDirectory() as name:
        root=Path(name);(root/'results').mkdir();(root/'analysis').mkdir()
        result=campaign.analyze_curve(root,[],'graph')
        assert not result['complete_qualified_curve']
        assert all(x['smallest_tested_demonstrated_sm_count'] is None for x in result['thresholds'].values())
    # Synthetic positive control plus independent negative provenance/placement gates.
    records=[]
    def record(fraction,acq,pilot=False,stock=False):
        count={1.:132,.85:120,.75:104,.60:80,.45:64,.30:40}[fraction]
        return {'id':f'{fraction}_{acq}_{pilot}_{stock}','engine':'graph','batch':1,'context':4096,
                'fraction':fraction,'acquisition':acq,'pilot':pilot,'stock':stock,'suffix':'',
                'qualified':True,'actual_sm_count':count,'source_identity':'frozenA',
                'step_ms':[10.]*10,'tokens_s':[100.]*10,
                'clock':{'clocks':{m:{'minimum':1000.,'maximum':1000.} for m in ('sm_clock_mhz','memory_clock_mhz')}}}
    records.append(record(1.,0,pilot=True,stock=True))
    records.extend(record(f,0,pilot=True) for f in campaign.FRACTIONS)
    for acq in range(3):
        records.append(record(1.,acq,stock=True))
        records.extend(record(f,acq) for f in campaign.FRACTIONS)
    def cheap_interval(*args,**kwargs):return {'value':1.,'ci95':[1.,1.]}
    with tempfile.TemporaryDirectory() as name, \
         patch('shoreline_statistics.hierarchical_interval',side_effect=cheap_interval), \
         patch('shoreline_statistics.paired_ratio',side_effect=cheap_interval):
        root=Path(name);(root/'results').mkdir();(root/'analysis').mkdir()
        assert campaign.analyze_curve(root,records,'graph')['complete_qualified_curve']
        bad=copy.deepcopy(records);bad[2]['actual_sm_count']=128
        assert not campaign.analyze_curve(root,bad,'graph')['complete_qualified_curve']
        bad=copy.deepcopy(records);bad[-1]['source_identity']='different_source'
        assert not campaign.analyze_curve(root,bad,'graph')['complete_qualified_curve']
        bad=copy.deepcopy(records);bad[0]['qualified']=False
        assert not campaign.analyze_curve(root,bad,'graph')['complete_qualified_curve']
    requested=[]
    resource=types.SimpleNamespace(sm_count=132,coscheduled_alignment=8,min_partition_size=16)
    class Device:
        def __init__(self,index):self.resources=types.SimpleNamespace(sm=resource)
        def set_current(self):pass
    class Green:
        def set_context(self):pass
        def Stream(self):return object()
    def create(n,index):requested.append(n);return Green()
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(GreenContext=types.SimpleNamespace(create=create),
                                        current_stream=lambda:object()))
    with patch.dict(sys.modules,{'cuda':types.ModuleType('cuda'),'cuda.core':types.SimpleNamespace(Device=Device)}), \
         patch.object(shoreline_green,'retain_for_process'), \
         patch.object(shoreline_green,'stream_proof',return_value={'green_context_handle':1,'actual_sm_count':132}):
        shoreline_green.create(torch,1.,make_current=True,force_green=True)
        assert requested==[132], 'full allocation must not round above device SM count'
        shoreline_green.create(torch,.75,make_current=True,force_green=True)
        assert requested==[132,104]
        shoreline_green.create(torch,1.)
        assert requested==[132,104], 'stock path must remain separately available'
    with tempfile.TemporaryDirectory() as name:
        root=Path(name)
        task={'id':'P3a_pilot_v1','gpu':0,'script':'shoreline_native.py','arguments':[],'timeout_s':900}
        assert campaign.continuation_task(task,root)==root/'gpu0'
        for changed in ({'gpu':2},{'script':'../outside.py'},{'arguments':['--gpu-uuid=GPU-unallocated']},{'id':'../escape'}):
            invalid=dict(task,**changed)
            try:campaign.continuation_task(invalid,root)
            except (AssertionError,ValueError):pass
            else:raise AssertionError('invalid continuation accepted')
        prior=root/'gpu0/results/processes/P3a_pilot_v1.json';prior.parent.mkdir(parents=True);prior.write_text('{}')
        try:campaign.continuation_task(task,root)
        except AssertionError:pass
        else:raise AssertionError('existing attempt accepted')
    import os
    import gpu_target
    import shoreline_restricted_decode as restricted
    class ScopedGreen:
        depth=1
        def set_context(self):self.depth+=1
        def pop_context(self):self.depth-=1
    green=ScopedGreen();stream=object();steps=[]
    def original_step():steps.append(green.depth)
    engine=types.SimpleNamespace(step=original_step)
    baseline=types.ModuleType('controlled_vllm_experiments')
    def original_trial(engine):
        engine.step();engine.step()
        return {}
    baseline.run_trial=original_trial
    baseline.main=lambda a:baseline.run_trial(engine)
    torch=types.SimpleNamespace(cuda=types.SimpleNamespace(current_stream=lambda:stream,
                           set_stream=lambda s:None,stream=lambda s:contextlib.nullcontext(s)))
    proof={'green_context_handle':7,'actual_sm_count':104}
    with tempfile.TemporaryDirectory() as name, \
         patch.dict(os.environ), \
         patch.dict(sys.modules,{'torch':torch,'controlled_vllm_experiments':baseline}), \
         patch.object(gpu_target,'verify_cuda'), \
         patch.object(shoreline_green,'create',return_value=(green,stream,proof)), \
         patch.object(shoreline_green,'stream_proof',return_value=proof):
        args=types.SimpleNamespace(gpu_uuid='GPU-synthetic',fraction=.75,force_green=True,rebind_step=True,
                                   output=Path(name),external_profiler=False)
        restricted.main(args)
        assert steps==[3,3] and green.depth==1, 'repair must balance both per-trial and per-step contexts'
        assert engine.step is original_step, 'scheduler hook must be restored'
    print('H200 priority CPU contracts passed')


if __name__=='__main__':main()
