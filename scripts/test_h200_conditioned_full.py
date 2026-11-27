import copy
from pathlib import Path
import tempfile
import types
import unittest
from h200_priority_campaign import case_conditioning_verified
from h200_eager_first import shape_summary


class ConditionedFull(unittest.TestCase):
    def test_every_warmup_and_timed_conditioning_receipt_is_required(self):
        receipt={'protocol':'memory_settle_v3','copies':64,'buffer_bytes':256*1024**2,
                 'wall_ms':1001.,'gpu_event_ms':1000.}
        data={'config':{'decode_conditioning':{'protocol':'memory_settle_v3','seconds':1.,'buffer_bytes':256*1024**2,'scope':'conditioned_decode_only'}},
              'cases':[{'trials':[{'post_prefill_conditioning':copy.deepcopy(receipt)} for _ in range(10)],
                        'warmup_conditioning_receipts':[copy.deepcopy(receipt) for _ in range(3)]}]}
        self.assertTrue(case_conditioning_verified(data,True,10))
        data['cases'][0]['warmup_conditioning_receipts'][-1]=None
        self.assertFalse(case_conditioning_verified(data,True,10))
        data['cases'][0]['warmup_conditioning_receipts'][-1]=copy.deepcopy(receipt)
        data['cases'][0]['trials'][-1]['post_prefill_conditioning']=None
        self.assertFalse(case_conditioning_verified(data,True,10))

    def test_different_execution_protocols_cannot_form_one_qualified_shape(self):
        clocks={'sm_clock_mhz':{'maximum':1980,'minimum':1980},'memory_clock_mhz':{'maximum':3201,'minimum':3201}}
        rows=[{'acquisition':a,'qualified':True,'step_ms':[20.]*10,'source_identity':'CPU-fake',
               'clock':{'clocks':clocks},'id':f'CPU-fake-{a}','repair_route':'original_memory_settle_v3',
               'conditioning':{'protocol':'memory_settle_v3'}} for a in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            worker=types.SimpleNamespace(root=Path(directory))
            self.assertTrue(shape_summary(worker,rows,'graph',4096)['qualified_timing'])
            rows[-1]['repair_route']='original';rows[-1]['conditioning']=None
            self.assertFalse(shape_summary(worker,rows,'graph',4096)['qualified_timing'])


if __name__=='__main__':unittest.main()
