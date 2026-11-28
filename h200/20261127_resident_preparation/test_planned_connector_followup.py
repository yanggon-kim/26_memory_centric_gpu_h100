"""CPU continuation gates: no GPU source/clock/trial requirement is relaxed."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from h200_planned_connector_followup import completed_comparison_gate,run_if_planned


class Continuation(unittest.TestCase):
    def fixture(self):
        rows=[{'acquisition':a,'condition':c,'role':r,'metrics':{'submit_to_final_step_return_host_ms':[1.]*10}}
            for a in range(3) for c,roles in (('stock_before',('stock',)),('native',('producer','consumer')),('stock_after',('stock',))) for r in roles]
        outcome={'status':'captured_rejected_fixed3x10'}
        review={'complete':True,'source_gate':True,'same_physical_pair_gate':True,'errors':[],
            'pilot_samples_in_estimator':0,'clock_gate':False,'performance_qualified':False,'rows':rows}
        captures={'mode':'comparison','planned_acquisitions':3,'warmups':3,'timed':10,'captures':rows}
        archive={'status':'archive_transfer_verified','every_archive_member_verified':True,
            'archive_sha256':'same','destination_sha256':'same','regular_files':10}
        return outcome,review,captures,archive

    def test_complete_clock_rejected_comparison_remains_rejected_and_eligible(self):
        args=self.fixture();self.assertTrue(completed_comparison_gate(*args))
        self.assertFalse(args[1]['performance_qualified']);self.assertFalse(args[1]['clock_gate'])

    def test_incomplete_source_wrong_pair_or_archive_failure_cannot_advance(self):
        for kind in ('partial','source','pair','archive','missing_acquisition','short_timing','pilot_samples'):
            args=list(copy.deepcopy(self.fixture()))
            if kind=='partial':args[0]['status']='blocked_or_partial_fixed3x10'
            if kind=='source':args[1]['source_gate']=False
            if kind=='pair':args[1]['same_physical_pair_gate']=False
            if kind=='archive':args[3]['every_archive_member_verified']=False
            if kind=='missing_acquisition':args[2]['captures'].pop()
            if kind=='short_timing':args[1]['rows'][0]['metrics']['submit_to_final_step_return_host_ms'].pop()
            if kind=='pilot_samples':args[1]['pilot_samples_in_estimator']=1
            with self.subTest(kind=kind),self.assertRaises(AssertionError):completed_comparison_gate(*args)

    def test_absent_exact_job_plan_never_launches_or_queries_scheduler(self):
        with tempfile.TemporaryDirectory() as folder,patch('h200_planned_connector_followup.PACKAGE',Path(folder)),patch.dict('os.environ',{'SLURM_JOB_ID':'123'}),patch('h200_planned_connector_followup.subprocess.run') as run,patch('h200_planned_connector_followup.subprocess.check_output') as query:
            self.assertEqual(run_if_planned('/unused','/unused')['status'],'no_plan')
            run.assert_not_called();query.assert_not_called()

    def test_incomplete_source_plan_is_recorded_before_any_scheduler_or_GPU_action(self):
        with tempfile.TemporaryDirectory() as folder:
            package=Path(folder);core=package/'h200/core';quant=package/'h200/quant'
            (quant/'results').mkdir(parents=True);(quant/'analysis').mkdir()
            plan=package/'h200/20261127_resident_preparation/results/connector_followup_123.json';plan.parent.mkdir(parents=True)
            plan.write_text(json.dumps({'enabled':True,'job_id':'123','GPUs':2,'context':16384,
                'entry':'h200_connector_capture_campaign.py','source_sha256':{}}))
            with patch('h200_planned_connector_followup.PACKAGE',package),patch.dict('os.environ',{'SLURM_JOB_ID':'123'}),patch('h200_planned_connector_followup.subprocess.run') as run,patch('h200_planned_connector_followup.subprocess.check_output') as query,patch('h200_planned_connector_followup.publish'),patch('h200_planned_connector_followup.limitation') as notice:
                result=run_if_planned(core,quant)
                self.assertEqual(result['status'],'connector_followup_admission_or_execution_failed')
                run.assert_not_called();query.assert_not_called();notice.assert_called_once()
                self.assertTrue((quant/'results/planned_connector_followup.json').exists())


if __name__=='__main__':unittest.main()
