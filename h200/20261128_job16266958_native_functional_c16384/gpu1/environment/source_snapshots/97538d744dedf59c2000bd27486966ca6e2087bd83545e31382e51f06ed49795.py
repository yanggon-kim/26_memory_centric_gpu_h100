import copy
import unittest
from h200_review_producer_quant import analyze,KEYS


class QuantitativeProducer(unittest.TestCase):
    def fixture(self):
        rows=[]
        for a in range(3):
            for condition,role in KEYS:
                value=150 if role=='producer' else 100
                rows.append({'acquisition':a,'condition':condition,'role':role,'context':16384,'gpu_uuid':'GPU-b' if role=='consumer' else 'GPU-a',
                    'source_identity':'source','reference_run_sha256':'reference','functional_CPU_review_sha256':'certificate','source_verified':True,
                    'metrics':{'submit_to_final_step_return_host_ms':[value]*10},
                    'clock':{'passed':True,'clocks':{m:{'minimum':v,'maximum':v} for m,v in [('sm_clock_mhz',1980),('memory_clock_mhz',3201)]}}})
        return rows

    def test_primary_ratio_uses_three_fresh_before_controls(self):
        result=analyze(self.fixture(),[],draws=128)
        self.assertTrue(result['performance_qualified']);self.assertAlmostEqual(result['qualified_producer_ratio']['value'],1.5)
        self.assertEqual(result['pilot_samples_in_estimator'],0);self.assertIsNone(result['primary_J_per_token'])

    def test_clock_drift_missing_case_or_new_UUID_censors_all_ratios(self):
        variants=[]
        rows=self.fixture();rows[0]['clock']['passed']=False;variants.append(rows)
        rows=self.fixture();next(r for r in rows if r['condition']=='stock_after')['metrics']['submit_to_final_step_return_host_ms']=[110]*10;variants.append(rows)
        rows=self.fixture();rows.pop();variants.append(rows)
        rows=self.fixture();rows[0]['gpu_uuid']='GPU-other';variants.append(rows)
        rows=self.fixture();rows[0]['metrics']['submit_to_final_step_return_host_ms'].pop();variants.append(rows)
        for rows in variants:
            result=analyze(rows,[],draws=128)
            self.assertFalse(result['performance_qualified']);self.assertIsNone(result['qualified_producer_ratio'])

    def test_error_or_mixed_source_cannot_hide_behind_complete_counts(self):
        rows=self.fixture();rows[-1]['source_identity']='different'
        self.assertFalse(analyze(rows,[],128)['performance_qualified'])
        self.assertFalse(analyze(self.fixture(),[{'error':'actual reference failure'}],128)['performance_qualified'])


if __name__=='__main__':unittest.main()
