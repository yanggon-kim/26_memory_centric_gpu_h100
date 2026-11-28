import copy
import unittest
from h200_producer_timing_role import descriptor_scope,checked_outputs


class ProducerTimingContracts(unittest.TestCase):
    def fixture(self):
        pair={'protocol':'matched_native_producer_host_timing_v1','synthetic_payload':False,
            'job_id':'123','hostname':'node','host':'127.0.0.1','request_prefix':'timing123',
            'roles':{'producer':{'gpu_uuid':'GPU-a','port':12001},'consumer':{'gpu_uuid':'GPU-b','port':12002}},
            'context':16384,'generated_tokens':64,'acquisition':0,'mode':'pilot','warmup_trials':3,'timed_trials':3}
        session={'slurm_job_id':'123','expected_hostname':'node','gpu_uuid':'GPU-a'}
        return pair,session

    def test_stock_and_producer_share_only_the_owned_producer_device(self):
        pair,session=self.fixture()
        for role in ('stock','producer'):descriptor_scope(pair,session,role,'GPU-a','123','node')
        with self.assertRaises(AssertionError):descriptor_scope(pair,session,'stock','GPU-b','123','node')
        with self.assertRaises(AssertionError):descriptor_scope(pair,session,'producer','GPU-a','124','node')
        pair['roles']['consumer']['gpu_uuid']='GPU-a'
        with self.assertRaises(AssertionError):descriptor_scope(pair,session,'stock','GPU-a','123','node')

    def test_pilot_cannot_be_relabeled_as_ten_trial_comparison(self):
        pair,session=self.fixture();pair['mode']='comparison'
        with self.assertRaises(AssertionError):descriptor_scope(pair,session,'producer','GPU-a','123','node')
        pair['timed_trials']=10;descriptor_scope(pair,session,'producer','GPU-a','123','node')
        pair['context']=128
        with self.assertRaises(AssertionError):descriptor_scope(pair,session,'producer','GPU-a','123','node')

    def test_outputs_keep_p0_and_consumer_suffix_distinct(self):
        expected=list(range(64))
        checked_outputs([0],expected,'stock');checked_outputs([0],expected,'producer')
        checked_outputs([1,2,3],expected,'consumer')
        with self.assertRaises(AssertionError):checked_outputs([0,1],expected,'producer')
        with self.assertRaises(AssertionError):checked_outputs([0,1],expected,'consumer')
        with self.assertRaises(AssertionError):checked_outputs([1,9],expected,'consumer')


if __name__=='__main__':unittest.main()
