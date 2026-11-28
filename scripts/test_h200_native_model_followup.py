import datetime as dt
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import h200_native_model_followup as followup


class ModelFollowupAdmission(unittest.TestCase):
    def test_failed_component_or_wrong_job_cannot_launch_model(self):
        with tempfile.TemporaryDirectory() as folder:
            package=Path(folder);source=package/'h200/source';paths=[]
            for i in range(2):
                base=source/f'gpu{i}';(base/'results').mkdir(parents=True)
                session={'gpu_uuid':f'GPU-{i}','python':'/p','cuda_home':'/cuda',
                    'slurm_job_id':'123','expected_hostname':socket.gethostname(),
                    'launch_cutoff':(dt.datetime.now().astimezone()+dt.timedelta(hours=2)).isoformat()}
                (base/'session.json').write_text(json.dumps(session));paths.append(base/'session.json')
                (base/'results/current_process.json').write_text('{"status":"idle"}')
                (base/'results/worker_outcome.json').write_text(json.dumps({'status':
                    'resident_curve_disposition' if i==0 else 'prefill_acquisition_disposition',
                    'complete_acquisition_set':True}))
            end=(dt.datetime.now().astimezone()+dt.timedelta(hours=2)).replace(tzinfo=None).isoformat()
            state=f'JobId=123 JobState=RUNNING UserId=owner({os.getuid()}) NodeList={socket.gethostname().split(".")[0]} AllocTRES=cpu=16,gres/gpu:h200=2 EndTime={end}'
            args=SimpleNamespace(root=package/'h200/model',component=package/'h200/component',source_session=paths)
            with patch.object(followup,'PACKAGE',package),patch.dict(os.environ,{'SLURM_JOB_ID':'123'}), \
                 patch('h200_native_model_followup.subprocess.check_output',return_value=state), \
                 patch('h200_native_model_followup.subprocess.Popen') as spawn, \
                 patch('h200_review_native_transport.review',return_value={'status':'failed_CPU_review'}) as review:
                with self.assertRaises(AssertionError):followup.main(args)
                self.assertFalse(args.root.exists());spawn.assert_not_called();review.assert_called_once()
            with patch.object(followup,'PACKAGE',package),patch.dict(os.environ,{'SLURM_JOB_ID':'124'}), \
                 patch('h200_native_model_followup.subprocess.check_output',return_value=state), \
                 patch('h200_native_model_followup.subprocess.Popen') as spawn:
                with self.assertRaises(AssertionError):followup.main(args)
                self.assertFalse(args.root.exists());spawn.assert_not_called()


if __name__=='__main__':unittest.main()
