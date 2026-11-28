import copy
import unittest
import types
from h200_native_transport_probe import validate_pair,all_native_sends_completed


class PairScope(unittest.TestCase):
    def test_empty_queue_does_not_prove_last_native_send_completed(self):
        ids={f'request#layer{i}' for i in range(64)}
        completed=ids-{'request#layer63'}
        engine=types.SimpleNamespace(send_queue=[],send_request_id_to_tensor_ids={'request':completed})
        self.assertFalse(all_native_sends_completed(engine,'request',ids))
        completed.add('request#layer63')
        self.assertTrue(all_native_sends_completed(engine,'request',ids))

    def test_exact_owned_pair_job_host_and_role_are_required(self):
        pair={'job_id':'123','hostname':'owned-node','host':'127.0.0.1','synthetic_payload':True,
              'request_prefix':'job123_probe','roles':{'producer':{'gpu_uuid':'GPU-a','port':21001},
                                                    'consumer':{'gpu_uuid':'GPU-b','port':21002}}}
        session={'slurm_job_id':'123','expected_hostname':'owned-node','gpu_uuid':'GPU-a'}
        validate_pair(pair,session,'producer','GPU-a','123','owned-node')
        for key,value in (('host','192.0.2.1'),('job_id','other'),('synthetic_payload',False)):
            invalid=copy.deepcopy(pair);invalid[key]=value
            with self.assertRaises(AssertionError):validate_pair(invalid,session,'producer','GPU-a','123','owned-node')
        with self.assertRaises(AssertionError):validate_pair(pair,session,'consumer','GPU-a','123','owned-node')
        duplicate=copy.deepcopy(pair);duplicate['roles']['consumer']['gpu_uuid']='GPU-a'
        with self.assertRaises(AssertionError):validate_pair(duplicate,session,'producer','GPU-a','123','owned-node')


if __name__=='__main__':unittest.main()
