import copy
import unittest
from h200_review_prefill import CONDITIONS,pooled_gate


class Qualification(unittest.TestCase):
    def rows(self):
        clock={'passed':True,'clocks':{'sm_clock_mhz':{'minimum':1980,'maximum':1980},
            'memory_clock_mhz':{'minimum':3201,'maximum':3201}}}
        return [{'gpu_uuid':'GPU-CPU-fake','source_identity':'CPU-fake','source_verified':True,
            'acquisition':a,'condition':name,'service_wall_ms':[1.]*10,'service_gpu_event_ms':[1.]*10,
            'submitted_first_token_wall_ms':[1.]*10,'service_clock':copy.deepcopy(clock)}
            for a in range(3) for name in CONDITIONS]

    def test_complete_single_device_source_is_required(self):
        rows=self.rows();self.assertTrue(pooled_gate(rows,'service_clock')[0])
        rows[-1]['gpu_uuid']='GPU-other';self.assertFalse(pooled_gate(rows,'service_clock')[0])
        rows=self.rows();rows[-1]['source_identity']='other';self.assertFalse(pooled_gate(rows,'service_clock')[0])
        self.assertFalse(pooled_gate(self.rows()[:-1],'service_clock')[0])

    def test_individually_stable_but_different_clocks_fail_pooled_gate(self):
        rows=self.rows();rows[-1]['service_clock']['clocks']['sm_clock_mhz']={'minimum':1800,'maximum':1800}
        passed,ratios=pooled_gate(rows,'service_clock')
        self.assertFalse(passed);self.assertGreater(ratios['sm_clock_mhz'],1.05)

    def test_missing_trial_or_invalid_latency_fails(self):
        rows=self.rows();rows[-1]['submitted_first_token_wall_ms'].pop()
        self.assertFalse(pooled_gate(rows,'service_clock')[0])
        rows=self.rows();rows[-1]['service_gpu_event_ms'][-1]=float('inf')
        self.assertFalse(pooled_gate(rows,'service_clock')[0])


if __name__=='__main__':unittest.main()
