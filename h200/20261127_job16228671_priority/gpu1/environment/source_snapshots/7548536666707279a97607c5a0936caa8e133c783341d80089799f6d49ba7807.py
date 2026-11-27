#!/usr/bin/env python3
"""CPU contracts for observer cleanup and the core-before-context boundary."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import h200_eager_first as campaign
from h200_scheduler_pilot import run_observed


class Contracts(unittest.TestCase):
    def test_observer_preserves_result_and_closes_outside_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs=types.SimpleNamespace(identity={'scope':'CPU fake'},rows=[
                {'prompt_input_queries':2,'post_prompt_input_queries':0,'composition':'prefill_only','any_preempted_request':False},
                *[{'prompt_input_queries':0,'post_prompt_input_queries':1,'composition':'decode_only','any_preempted_request':False} for _ in range(3)]],closed=False)
            obs.close=lambda:setattr(obs,'closed',True)
            result={'exact_original':'retained'}
            got=run_observed(lambda *a,**kw:result,lambda e:obs,object(),[[1,2]],types.SimpleNamespace(max_tokens=4),'trial',None,Path(tmp))
            self.assertIs(got,result);self.assertTrue(obs.closed)
            self.assertTrue((Path(tmp)/'scheduler_trial.json').is_file())

    def test_failed_model_retains_observer_rows_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs=types.SimpleNamespace(identity={},rows=[],closed=False);obs.close=lambda:setattr(obs,'closed',True)
            def fail(*a,**kw):raise ValueError('model failed')
            with self.assertRaisesRegex(ValueError,'model failed'):
                run_observed(fail,lambda e:obs,None,[[1]],types.SimpleNamespace(max_tokens=4),'failed',None,Path(tmp))
            self.assertTrue(obs.closed);self.assertTrue((Path(tmp)/'scheduler_failed.json').is_file())

    def test_core_route_and_expansion_gate_follow_published_disposition(self):
        for stop,allowed in [('proof',True),('quality',True),('budget',False)]:
            with self.subTest(stop=stop),tempfile.TemporaryDirectory() as tmp, \
                 patch.object(campaign,'limitation'),patch.object(campaign,'scheduler_pilot') as scheduler:
                root=Path(tmp);(root/'results').mkdir();run=root/'run.json'
                campaign.write(run,{'cases':[{'trials':[{'output_ids':[[1,2]]}]}]})
                worker=types.SimpleNamespace(root=root,py='CPU-fake',script=root,uuid='GPU-owned',
                       rebind_step=False,capture_bind=True,last_curve_stop=None,
                       settings={'launch_cutoff':'2000-01-01T00:00:00+00:00'},
                       execute=lambda *a,**kw:{'status':'complete'},reference=lambda *a:True)
                def case(engine,*a,**kw):
                    self.assertEqual(engine,'eager');self.assertTrue(worker.rebind_step);self.assertFalse(worker.capture_bind)
                    return {'qualified':True,'run':str(run)}
                def curve(*a,**kw):worker.last_curve_stop=stop;return False
                worker.case=case;worker.curve=curve
                def publish(*a):self.assertFalse((root/'results/eager_core_disposition.json').exists())
                with patch.object(campaign,'publish',side_effect=publish):campaign.primary(worker)
                self.assertEqual(campaign.load(root/'results/eager_core_disposition.json')['allow_context_expansion'],allowed)
                scheduler.assert_not_called()


if __name__=='__main__':unittest.main()
