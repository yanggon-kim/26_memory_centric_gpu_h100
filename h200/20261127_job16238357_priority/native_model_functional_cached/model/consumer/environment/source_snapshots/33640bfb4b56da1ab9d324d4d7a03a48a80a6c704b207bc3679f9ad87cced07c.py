#!/usr/bin/env python3
"""CPU scheduling contracts: cold startup, measured warm cost, hard cutoffs."""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import h200_priority_campaign as campaign


class Contracts(unittest.TestCase):
    def test_stages_require_whole_bundles_and_keep_margin(self):
        first=campaign.staged_curve_budget(1800,[],700)
        self.assertEqual(first['estimated_bundle_s'],600)
        self.assertTrue(first['admitted'])
        rest=campaign.staged_curve_budget(1800,[100],449)
        self.assertEqual(rest['estimated_bundle_s'],450)
        self.assertFalse(rest['admitted'])
        self.assertEqual(campaign.staged_curve_budget(1800,[100,200],350)['estimated_bundle_s'],350)
        with self.assertRaises(AssertionError):campaign.staged_curve_budget(1800,[0],1000)

    def exercise(self,available,observed,expected_clean):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'results').mkdir();(root/'analysis').mkdir()
            worker=types.SimpleNamespace(root=root,records=[],settings={'idle_required_s':10},rebind_step=False,
                hs=types.SimpleNamespace(verify_allocation=lambda:{'effective_launch_cutoff_unix_s':1000+available}))
            def case(engine,fraction,acq,pilot=False,stock=False,suffix='',**kwargs):
                row={'id':f'{engine}_{fraction}_{acq}_{pilot}_{stock}_{suffix}','pilot':pilot,'stock':stock,
                     'qualified':True,'fraction':fraction,'producer_elapsed_s':100 if stock and pilot else 10}
                worker.records.append(row);return row
            worker.case=case
            worker.unrestricted_after_block=lambda *args:(_ for _ in ()).throw(AssertionError('not a proof failure'))
            def analysis(*args):
                return {'complete_qualified_curve':sum(not r['pilot'] for r in worker.records)==27}
            times=[]
            for n in range(3):times.extend([n*observed,(n+1)*observed])
            with patch.object(campaign,'PACKAGE',root),patch.object(campaign.time,'time',return_value=1000), \
                 patch.object(campaign.time,'monotonic',side_effect=times),patch.object(campaign,'analyze_curve',side_effect=analysis), \
                 patch.object(campaign,'publish'),patch.object(campaign,'limitation'):
                complete=campaign.Worker.curve(worker,'graph')
            self.assertEqual(sum(not r['pilot'] for r in worker.records),expected_clean)
            budget=campaign.load(root/'results/budget_graph_b1_c4096.json')
            self.assertGreater(budget['initial_full_curve_estimate_s'],available)
            return complete,worker.last_curve_stop,budget

    def test_first_broad_acquisition_refines_cold_pilot_cost(self):
        complete,stop,budget=self.exercise(1000,20,27)
        self.assertTrue(complete);self.assertIsNone(stop)
        self.assertEqual(len(budget['stages']),3)
        self.assertEqual(budget['observed_acquisition_wall_s'],[20,20,20])
        self.assertTrue(all(s['admitted'] for s in budget['stages']))

    def test_insufficient_first_bundle_starts_no_timing_acquisition(self):
        complete,stop,budget=self.exercise(100,20,0)
        self.assertFalse(complete);self.assertEqual(stop,'budget')
        self.assertFalse(budget['stages'][0]['admitted'])

    def test_measured_cost_can_stop_remaining_bundle_without_discarding_first(self):
        complete,stop,budget=self.exercise(1000,500,9)
        self.assertFalse(complete);self.assertEqual(stop,'budget')
        self.assertEqual(budget['observed_acquisition_wall_s'],[500])
        self.assertFalse(budget['stages'][1]['admitted'])


if __name__=='__main__':unittest.main()
