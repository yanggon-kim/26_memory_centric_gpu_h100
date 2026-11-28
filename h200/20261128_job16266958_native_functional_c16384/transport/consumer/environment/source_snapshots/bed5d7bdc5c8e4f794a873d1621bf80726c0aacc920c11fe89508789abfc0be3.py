import unittest
from pathlib import Path
import tempfile
import time
import types
from unittest.mock import patch
from h200_postcore_pilots import validate_idle,pilots


class Idle(unittest.TestCase):
    def test_short_budget_does_not_start_a_model_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            launched=[]
            worker=types.SimpleNamespace(root=Path(directory),py='CPU-fake',script=Path('/CPU-fake'),uuid='GPU-CPU-fake',
                hs=types.SimpleNamespace(verify_allocation=lambda:{'effective_launch_cutoff_unix_s':time.time()+30}),
                execute=lambda label,*args,**kwargs:launched.append(label) or {'status':'complete'})
            with patch('h200_resident_campaign.build_environment'):
                result=pilots(worker)
            self.assertEqual(launched,['inventory'])
            self.assertEqual(result['results'][0]['status'],'deferred_budget')

    def test_completed_core_and_idle_gpu_are_required(self):
        core={'complete_acquisition_set':True};outcome={'status':'resident_curve_disposition'}
        validate_idle(core,{'status':'idle'},outcome,[])
        for c,current,o,pids in (({}, {'status':'idle'},outcome,[]),
            (core,{'status':'running'},outcome,[]),(core,{'status':'idle'},outcome,[123]),
            (core,{'status':'idle'},{'status':'failed'},[])):
            with self.assertRaises(AssertionError):validate_idle(c,current,o,pids)

    def test_active_conditioned_continuation_blocks_idle_transition(self):
        with self.assertRaises(AssertionError):
            validate_idle({'complete_acquisition_set':True},{'status':'idle'},
                {'status':'resident_curve_disposition','conditioned_followup':{'status':'admitted'}},[])


if __name__=='__main__':unittest.main()
