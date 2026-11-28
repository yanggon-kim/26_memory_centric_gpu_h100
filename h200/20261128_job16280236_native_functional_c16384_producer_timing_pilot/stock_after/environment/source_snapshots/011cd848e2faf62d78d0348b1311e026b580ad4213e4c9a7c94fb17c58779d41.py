import copy
import unittest
from unittest.mock import patch
from h200_reference_determinism_followup import validate_idle
from h200_deterministic_hf_reference import tag


class ReferenceRepairGuards(unittest.TestCase):
    def test_active_producer_cannot_be_replaced_by_a_diagnostic(self):
        old={'gpu_uuid':'GPU-a'}
        outcome={'status':'blocked_resident_pilot'}
        current={'status':'idle'}
        reference={'status':'failed','gpu_uuid':'GPU-a','batch':8,'context':4096,
            'generated_tokens':64,'prompt_ids':[[1]],'model_id':'model','revision':'revision','precision':'BF16'}
        source={'cases':[{'prompt_ids':[[1]]}],
            'config':{'model_id':'model','model_revision':'revision','precision':'BF16'}}
        with patch('h200_reference_determinism_followup.valid_tokens',return_value=True):
            validate_idle(old,outcome,current,[],reference,source)
            for changed_current,pids in [({'status':'running'},[]),(current,[123])]:
                with self.assertRaises(AssertionError):
                    validate_idle(old,outcome,changed_current,pids,reference,source)
            with self.assertRaises(AssertionError):
                validate_idle(old,{'status':'running'},current,[],reference,source)
            corrupt=copy.deepcopy(reference);corrupt['prompt_ids']=[[2]]
            with self.assertRaises(AssertionError):
                validate_idle(old,outcome,current,[],corrupt,source)

    def test_partial_and_failed_reference_records_keep_algorithm_and_scope(self):
        settings={'torch_deterministic_algorithms':True,'warn_only':False}
        for status in ('running','failed','passed'):
            record=tag({'status':status,'source_sha256':'original'},settings)
            self.assertEqual(record['status'],status)
            self.assertEqual(record['source_sha256'],'original')
            self.assertEqual(record['deterministic_reference_diagnostic'],settings)
            self.assertFalse(record['promotes_original_default_reference'])


if __name__=='__main__': unittest.main()
