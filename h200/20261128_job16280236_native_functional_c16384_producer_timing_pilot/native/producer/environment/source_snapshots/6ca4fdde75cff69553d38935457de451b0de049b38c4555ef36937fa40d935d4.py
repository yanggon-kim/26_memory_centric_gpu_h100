import os
import unittest
from h200_producer_timing_followup import admission


class ProducerAdmission(unittest.TestCase):
    def fixture(self):
        sessions=[{'slurm_job_id':'123','expected_hostname':'node','gpu_uuid':uuid,'python':'python',
            'cuda_home':'cuda','frozen_source_identity':'same'} for uuid in ('GPU-a','GPU-b')]
        pair={'job_id':'123','hostname':'node','protocol':'native_model_KV_long_context_functional_v2',
            'context':16384,'generated_tokens':64,'roles':{'producer':{'gpu_uuid':'GPU-a'},'consumer':{'gpu_uuid':'GPU-b'}}}
        outcome={'status':'functional_prerequisites_passed'}
        cert={'status':'native_model_functional_review_passed','job_id':'123'}
        slurm='JobId=123 JobState=RUNNING UserId=user('+str(os.getuid())+') NodeList=node AllocTRES=cpu=16,gres/gpu:h200=2'
        return [sessions,[{'status':'idle'},{'status':'idle'}],'123','node',os.getuid(),slurm,outcome,pair,cert]

    def test_only_closed_owned_same_source_long_functional_pair_is_admitted(self):
        args=self.fixture();admission(*args)
        args[6]['status']='running_native_model_functional'
        with self.assertRaises(AssertionError):admission(*args)
        args=self.fixture();args[7]['context']=128
        with self.assertRaises(AssertionError):admission(*args)
        args=self.fixture();args[0][1]['frozen_source_identity']='different'
        with self.assertRaises(AssertionError):admission(*args)
        args=self.fixture();args[5]=args[5].replace('RUNNING','PREEMPTED')
        with self.assertRaises(AssertionError):admission(*args)
        args=self.fixture();args[1][1]['status']='running'
        with self.assertRaises(AssertionError):admission(*args)


if __name__=='__main__':unittest.main()
