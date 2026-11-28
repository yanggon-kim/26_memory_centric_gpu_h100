#!/usr/bin/env python3
"""CPU contracts for context restoration and sparse actual-kernel proof windows."""
import types
import unittest
from unittest.mock import patch
from h200_resident_sweep import bound_trial,conditions


class Binding(unittest.TestCase):
    def run_bound(self,fail=False):
        primary=object();stream=object();events=[];stack=[]
        cuda=types.SimpleNamespace(current_stream=lambda:cuda.current,
             set_stream=lambda s:setattr(cuda,'current',s),synchronize=lambda:events.append('sync'),
             profiler=types.SimpleNamespace(start=lambda:events.append('start'),stop=lambda:events.append('stop')),
             nvtx=types.SimpleNamespace(range_push=lambda n:events.append(n),range_pop=lambda:events.append('pop')),
             current=primary)
        green=types.SimpleNamespace(set_context=lambda:stack.append(1),pop_context=lambda:stack.pop())
        def step():
            self.assertIs(cuda.current,stream);self.assertEqual(len(stack),2)
            events.append('kernel')
            if fail:raise RuntimeError('retained failure')
        engine=types.SimpleNamespace(step=step)
        def original(engine,*args,**kwargs):
            for i in range(64):engine.step()
            return {'output_ids':[[1]*64]}
        proof={'actual_sm_count':40,'green_context_id':7}
        with patch('shoreline_green.stream_proof',return_value=proof):
            if fail:
                with self.assertRaisesRegex(RuntimeError,'retained failure'):
                    bound_trial(original,engine,[],types.SimpleNamespace(max_tokens=64),'x',
                                types.SimpleNamespace(cuda=cuda),None,green,stream,proof,True,'selected')
            else:
                bound_trial(original,engine,[],types.SimpleNamespace(max_tokens=64),'x',
                            types.SimpleNamespace(cuda=cuda),None,green,stream,proof,True,'selected')
        self.assertIs(engine.step,step);self.assertIs(cuda.current,primary);self.assertEqual(stack,[])
        return events

    def test_exact_three_step_capture_and_restoration(self):
        events=self.run_bound()
        self.assertEqual(events.count('kernel'),64)
        captured=events[events.index('start')+1:events.index('stop')]
        self.assertEqual(captured.count('kernel'),3)
        self.assertEqual(events.count('start'),1);self.assertEqual(events.count('stop'),1)

    def test_failure_restores_context_and_original_step(self):self.run_bound(True)

    def test_full_curve_and_drift_reference_order(self):
        names=[x[0] for x in conditions()]
        self.assertEqual(len(names),9);self.assertEqual(len(set(names)),9)
        self.assertEqual(names,['stock','green132','green112','green96','green132_interleaved',
                                'green80','green56','green40','green132_end'])
        self.assertEqual(len(conditions(proof=True)),7)
        self.assertEqual([x[0] for x in conditions(pilot=True)],['stock','green132','green40'])


if __name__=='__main__':unittest.main()
