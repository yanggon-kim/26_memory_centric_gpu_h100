"""CPU boundary/idle guards only; no export, allocation or GPU execution."""
import datetime as dt
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from export_functional_boundary_caches import admission,idle_devices,archived_certificate


class Cache(unittest.TestCase):
    def args(self):
        now=100000.;cutoff=now+1000
        sessions=[dict(gpu_uuid='GPU-'+str(i),python='/python',cuda_home='/cuda',slurm_job_id='123',
            expected_hostname='node.domain',frozen_source_identity='same',
            launch_cutoff=dt.datetime.fromtimestamp(cutoff,dt.timezone.utc).isoformat()) for i in (0,1)]
        end=dt.datetime.fromtimestamp(cutoff+900,dt.timezone.utc).isoformat()
        raw=f'JobId=123 JobState=RUNNING UserId=owner({os.getuid()}) NodeList=node AllocTRES=gres/gpu:h200=2 EndTime={end}'
        return [sessions,[{'status':'idle'}]*2,'123','node.domain',os.getuid(),raw,
            {'status':'functional_prerequisites_passed'},
            {'status':'native_model_functional_review_passed','job_id':'123','errors':[]},
            [],[{'status':'idle'}]*6,now]

    def test_closed_functional_boundary_and_complete_CPU_budget(self):
        self.assertEqual(admission(*self.args())['CPU_subprocess_bound_s'],120)
        args=self.args();args[-1]+=850
        with self.assertRaisesRegex(AssertionError,'margin required'):admission(*args)

    def test_active_or_future_followup_root_cannot_be_raced(self):
        for condition in ('sibling','active','base_busy'):
            args=self.args()
            if condition=='sibling':args[8]=['future-primary-root']
            if condition=='active':args[9][0]={'status':'running'}
            if condition=='base_busy':args[1][0]={'status':'running'}
            with self.subTest(condition=condition),self.assertRaises(AssertionError):admission(*args)

    def test_wrong_job_failed_functional_or_mixed_sources_rejected(self):
        for condition in ('job','failure','source'):
            args=self.args()
            if condition=='job':args[2]='999'
            if condition=='failure':args[6]={'status':'partial'}
            if condition=='source':args[0][1]['frozen_source_identity']='other'
            with self.subTest(condition=condition),self.assertRaises(AssertionError):admission(*args)

    def test_compute_or_graphics_work_denies_export(self):
        sessions=[dict(gpu_uuid='GPU-owned')]
        for compute,graphics in (([],[]),([123],[]),([],[456])):
            nv=SimpleNamespace(nvmlDeviceGetHandleByUUID=lambda uuid:uuid,
                nvmlDeviceGetComputeRunningProcesses=lambda handle:compute,
                nvmlDeviceGetGraphicsRunningProcesses=lambda handle:graphics)
            if compute or graphics:
                with self.assertRaises(AssertionError):idle_devices(nv,sessions)
            else:idle_devices(nv,sessions)

    def test_verified_archive_must_bind_the_actual_functional_certificate(self):
        for condition in ('valid','incomplete','mismatch','missing_certificate'):
            archive=dict(status='archive_transfer_verified',every_archive_member_verified=True,
                archive_sha256='archive-pin',destination_sha256='archive-pin',source_root='/repo/h200/core')
            manifest=dict(files=[dict(path='model/results/native_model_CPU_review.json',sha256='certificate-pin')])
            if condition=='incomplete':archive['every_archive_member_verified']=False
            if condition=='mismatch':archive['destination_sha256']='other'
            if condition=='missing_certificate':manifest['files']=[]
            if condition=='valid':archived_certificate(archive,manifest,Path('/repo/h200/core'),'certificate-pin')
            else:
                with self.subTest(condition=condition),self.assertRaises(AssertionError):archived_certificate(archive,manifest,Path('/repo/h200/core'),'certificate-pin')


if __name__=='__main__':unittest.main()
