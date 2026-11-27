import unittest
import tempfile
import time
import types
from pathlib import Path
from unittest.mock import patch
from h200_spare_serving_followup import validate_idle,capture


class SpareAdmission(unittest.TestCase):
    def test_short_budget_launches_no_gpu_producer(self):
        with tempfile.TemporaryDirectory() as folder:
            worker=types.SimpleNamespace(root=Path(folder),hs=types.SimpleNamespace(
                verify_allocation=lambda:{'effective_launch_cutoff_unix_s':time.time()+10000}))
            with patch('h200_spare_serving_followup.limitation') as recorded:
                result=capture(worker,time.time()+30)
            recorded.assert_called_once()
            self.assertEqual(result['status'],'deferred_budget')
            self.assertFalse(result['budget']['admitted'])

    def test_complete_source_review_and_idle_owned_gpu_are_required(self):
        outcome={'status':'prefill_acquisition_disposition','complete_acquisition_set':True}
        review={'complete_acquisition_set':True,'errors':[],'rows':[{}]*15,'session_sha256':'actual'}
        validate_idle(outcome,{'status':'idle'},[],review,'actual')
        for o,c,p,r,s in (({**outcome,'complete_acquisition_set':False},{'status':'idle'},[],review,'actual'),
                         (outcome,{'status':'running'},[],review,'actual'),
                         (outcome,{'status':'idle'},[123],review,'actual'),
                         (outcome,{'status':'idle'},[],{**review,'errors':['bad source']},'actual'),
                         (outcome,{'status':'idle'},[],review,'changed')):
            with self.assertRaises(AssertionError):validate_idle(o,c,p,r,s)

    def test_previous_clock_rejection_does_not_qualify_or_block_independent_pilot(self):
        outcome={'status':'prefill_acquisition_disposition','complete_acquisition_set':True}
        review={'complete_acquisition_set':True,'errors':[],'rows':[{}]*15,'session_sha256':'actual',
                'instrumented_service_qualified':False,'ordinary_TTFT_qualified':False}
        validate_idle(outcome,{'status':'idle'},[],review,'actual')


if __name__ == '__main__':unittest.main()
