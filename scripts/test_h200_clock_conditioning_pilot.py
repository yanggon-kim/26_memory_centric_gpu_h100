import copy
import unittest
from h200_clock_conditioning_pilot import conditioning_gate


class Gate(unittest.TestCase):
    def test_conditioned_execution_requires_all_trial_receipts(self):
        declaration={'protocol':'memory_settle_v3','seconds':1.0,'buffer_bytes':256*1024**2,'scope':'conditioned_decode_only'}
        receipt={'protocol':'memory_settle_v3','copies':64,'buffer_bytes':256*1024**2,'wall_ms':1001.,'gpu_event_ms':999.}
        data={'config':{'decode_conditioning':declaration},'cases':[{'trials':[{'post_prefill_conditioning':copy.deepcopy(receipt)} for _ in range(3)]}]}
        self.assertTrue(conditioning_gate(data,True))
        del data['cases'][0]['trials'][1]['post_prefill_conditioning']
        self.assertFalse(conditioning_gate(data,True))

    def test_unconditioned_arm_rejects_hidden_conditioning(self):
        data={'config':{},'cases':[{'trials':[{} for _ in range(3)]}]}
        self.assertTrue(conditioning_gate(data,False))
        data['cases'][0]['trials'][0]['post_prefill_conditioning']={'protocol':'memory_settle_v3'}
        self.assertFalse(conditioning_gate(data,False))


if __name__=='__main__':unittest.main()
