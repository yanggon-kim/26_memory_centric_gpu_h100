#!/usr/bin/env python3
import copy
import unittest
from h200_finish_eager_curve import allowance,captured_acquisition,FRACTIONS


class Contracts(unittest.TestCase):
    def test_only_analytical_allowance_changes(self):
        result=allowance([1200],1,2600)
        self.assertTrue(result['admitted']);self.assertEqual(result['remaining_bundle_estimate_s'],2520)
        self.assertFalse(allowance([1200],1,2500)['admitted'])
        self.assertEqual(allowance([1200,1250],2,1400)['remaining_bundle_estimate_s'],1310)

    def test_complete_point_set_keeps_both_drift_references(self):
        def row(stock,fraction,suffix=''):
            return {'engine':'eager','pilot':False,'acquisition':0,'status':'complete','step_ms':[1]*10,
                    'stock':stock,'fraction':fraction,'suffix':suffix}
        rows=[row(True,1.)]+[row(False,f) for f in FRACTIONS]+[row(False,1.,s) for s in ('_interleaved','_end')]
        self.assertTrue(captured_acquisition(rows,0));self.assertFalse(captured_acquisition(rows,1))
        self.assertFalse(captured_acquisition(rows[:-1],0))
        bad=copy.deepcopy(rows);bad[-1]['step_ms']=[1]*9;self.assertFalse(captured_acquisition(bad,0))
        bad=copy.deepcopy(rows);bad[-1]['status']='failed';self.assertFalse(captured_acquisition(bad,0))


if __name__=='__main__':unittest.main()
