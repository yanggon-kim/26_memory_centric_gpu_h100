#!/usr/bin/env python3
"""CPU contracts for resource rounding, token gates and incomplete-curve rejection."""
import copy
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
    print('H200 priority CPU contracts passed')


if __name__=='__main__':main()
