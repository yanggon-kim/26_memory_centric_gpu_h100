import datetime as dt
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from h200_priority_campaign import sha
import h200_producer_quant_campaign as campaign


class FixedCohortAdmission(unittest.TestCase):
    def test_short_actual_allocation_never_starts_a_role_or_creates_a_case(self):
        with tempfile.TemporaryDirectory() as folder:
            package=Path(folder);core=package/'h200/core';root=package/'h200/new_comparison'
            (core/'model/results').mkdir(parents=True);(core/'results').mkdir()
            now=time.time();end=dt.datetime.fromtimestamp(now+1000,dt.timezone.utc).isoformat()
            for i,uuid in enumerate(('GPU-a','GPU-b')):
                base=core/f'gpu{i}';(base/'results').mkdir(parents=True)
                (base/'session.json').write_text(json.dumps({'slurm_job_id':'123','expected_hostname':socket.gethostname(),
                    'gpu_uuid':uuid,'python':'pinned','cuda_home':'cuda','frozen_source_identity':'source','launch_cutoff':end}))
                (base/'results/current_process.json').write_text(json.dumps({'status':'idle'}))
            source=core/'reference.json';source.write_text(json.dumps({'engine_settings':{}}))
            pair={'job_id':'123','hostname':socket.gethostname(),'protocol':'native_model_KV_long_context_functional_v2',
                'context':16384,'generated_tokens':64,'reference_run':str(source),
                'roles':{'producer':{'gpu_uuid':'GPU-a'},'consumer':{'gpu_uuid':'GPU-b'}}}
            (core/'model/pair.json').write_text(json.dumps(pair))
            cert=core/'model/results/native_model_CPU_review.json';cert.write_text(json.dumps({'status':'native_model_functional_review_passed','job_id':'123'}))
            (core/'results/worker_outcome.json').write_text(json.dumps({'status':'functional_prerequisites_passed','model_review_sha256':sha(cert)}))
            raw='JobId=123 JobState=RUNNING UserId=u('+str(os.getuid())+') NodeList='+socket.gethostname().split('.')[0]+' AllocTRES=cpu=16,gres/gpu:h200=2 EndTime='+end
            estimate={'all_bounds_within_existing660second_watchdog':True,'context':16384,'engine_settings':{},'required_s':3600}
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch.object(campaign,'PACKAGE',package), \
                 patch.object(campaign,'budget',return_value=estimate),patch.object(campaign.subprocess,'check_output',return_value=raw), \
                 patch.object(campaign,'run_one') as one,patch.object(campaign,'run_pair') as pair_run:
                with self.assertRaisesRegex(AssertionError,'fixed3x10cohort'):
                    campaign.main(SimpleNamespace(functional_root=core,root=root,pilot_root=package/'h200/old_pilot'))
            one.assert_not_called();pair_run.assert_not_called();self.assertFalse(root.exists())


if __name__=='__main__':unittest.main()
