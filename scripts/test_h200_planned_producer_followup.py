import datetime as dt
import os
import socket
import time
import unittest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import h200_planned_producer_followup as continuation
from h200_priority_campaign import sha
from h200_planned_producer_followup import admission,ENTRY,REQUIRED_SECONDS


class ContinuationAdmission(unittest.TestCase):
    def test_real_short_budget_path_preserves_core_without_starting_a_child(self):
        with tempfile.TemporaryDirectory() as folder:
            package=Path(folder);core=package/'h200/20261128_job123_native_functional_c16384'
            (core/'results').mkdir(parents=True);(core/'analysis').mkdir()
            cert=core/'results/cert.json';cert.write_text(json.dumps({'status':'native_model_functional_review_passed','job_id':'123'}))
            original=json.dumps({'status':'functional_prerequisites_passed','model_review':str(cert),'model_review_sha256':sha(cert)})
            (core/'results/worker_outcome.json').write_text(original)
            scripts=package/'scripts';scripts.mkdir();(scripts/ENTRY).write_text('unused CPU test entry\n')
            helper=Path(continuation.__file__);(scripts/helper.name).write_bytes(helper.read_bytes())
            now=time.time();end=dt.datetime.fromtimestamp(now+4500,dt.timezone.utc).isoformat()
            cutoff=dt.datetime.fromtimestamp(now+1800,dt.timezone.utc).isoformat()
            for i,uuid in enumerate(('GPU-a','GPU-b')):
                base=core/f'gpu{i}';base.mkdir()
                (base/'session.json').write_text(json.dumps({'python':sys.executable,'slurm_job_id':'123',
                    'expected_hostname':socket.gethostname(),'gpu_uuid':uuid,'frozen_source_identity':'one',
                    'frozen_source_root':str(package),'launch_cutoff':cutoff}))
            plan=package/'h200/20261127_resident_preparation/results/same_allocation_followup_123.json';plan.parent.mkdir(parents=True)
            plan.write_text(json.dumps({'job_id':'123','context':16384,'enabled':True,'entry':ENTRY,
                'entry_sha256':sha(scripts/ENTRY),'GPUs':2}))
            raw='JobId=123 JobState=RUNNING UserId=u('+str(os.getuid())+') NodeList='+socket.gethostname().split('.')[0]+' AllocTRES=cpu=16,gres/gpu:h200=2 EndTime='+end
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch.object(continuation,'PACKAGE',package), \
                 patch.object(continuation.subprocess,'check_output',return_value=raw), \
                 patch.object(continuation.subprocess,'run') as run,patch.object(continuation,'publish'), \
                 patch.object(continuation,'limitation') as limitation:
                result=continuation.run_if_planned(16384)
            self.assertEqual(result['status'],'deferred_insufficient_time');run.assert_not_called()
            self.assertEqual((core/'results/worker_outcome.json').read_text(),original)
            self.assertFalse(result['GPU_followup_launched']);limitation.assert_called_once()

    def fixture(self):
        now=time.time();end=dt.datetime.fromtimestamp(now+4500,dt.timezone.utc).isoformat()
        plan={'job_id':'123','context':16384,'enabled':True,'entry':ENTRY,'GPUs':2}
        fields={'JobId':'123','JobState':'RUNNING','UserId':'user('+str(os.getuid())+')',
            'NodeList':socket.gethostname().split('.')[0],'AllocTRES':'cpu=16,gres/gpu:h200=2','EndTime':end}
        sessions=[{'slurm_job_id':'123','expected_hostname':socket.gethostname(),'gpu_uuid':uuid,
            'frozen_source_identity':'one','launch_cutoff':end} for uuid in ('GPU-a','GPU-b')]
        return [plan,'123',16384,fields,sessions,{'status':'functional_prerequisites_passed'},
            {'status':'native_model_functional_review_passed','job_id':'123'},now]

    def test_exact_owned_successful_pair_is_required(self):
        args=self.fixture();self.assertTrue(admission(*args)['eligible'])
        args[5]['status']='blocked_or_partial_functional'
        with self.assertRaises(AssertionError):admission(*args)
        args=self.fixture();args[0]['job_id']='122'
        with self.assertRaises(AssertionError):admission(*args)
        args=self.fixture();args[3]['JobState']='PENDING'
        with self.assertRaises(AssertionError):admission(*args)
        args=self.fixture();args[4][1]['gpu_uuid']='GPU-a'
        with self.assertRaises(AssertionError):admission(*args)

    def test_short_budget_is_a_deferral_without_loosening_the_case(self):
        args=self.fixture();args[3]['EndTime']=dt.datetime.fromtimestamp(args[-1]+REQUIRED_SECONDS+899,dt.timezone.utc).isoformat()
        result=admission(*args)
        self.assertFalse(result['eligible']);self.assertEqual(result['required_s'],1920)
        self.assertIn('preserve completed functional',result['reason'])


if __name__=='__main__':unittest.main()
