import copy
import types
import unittest
from unittest.mock import patch
from h200_native_model_pilot import validate_scope, validate_queries, NativeLedger, observe_output


class NativeModelContracts(unittest.TestCase):
    def test_first_bad_token_or_extra_progress_is_retained_before_rejection(self):
        expected = list(range(64)); outputs = []
        with self.assertRaises(AssertionError): observe_output(outputs,[999],expected,'consumer',123)
        self.assertEqual(outputs,[{'output_ids':[999],'host_observed_unix_ns':123}])
        outputs = []
        with self.assertRaises(AssertionError): observe_output(outputs,[1,2],expected,'consumer',124)
        self.assertEqual(outputs[0]['output_ids'],[1,2])
        outputs = []
        observe_output(outputs,[1],expected,'consumer',125)
        observe_output(outputs,[1,2],expected,'consumer',126)
        self.assertEqual(len(outputs),2)

    def test_receiver_executes_first_token_query_and62more_without_reprefill(self):
        rows = [{'any_preempted_request':False, 'requests':[{
            'request_id':'rid','computed_tokens_before_step':i,
            'original_prompt_tokens':129,'scheduled_input_queries':1,
            'prompt_input_queries':int(i==128),'post_prompt_input_queries':int(i>128)}]}
            for i in range(128,191)]
        validate_queries(rows,'consumer','rid')
        for change in ('reprefill','missing','preempted','wrong_id'):
            bad = copy.deepcopy(rows)
            if change == 'reprefill': bad[0]['requests'][0]['computed_tokens_before_step'] = 0
            elif change == 'missing': bad.pop()
            elif change == 'preempted': bad[0]['any_preempted_request'] = True
            else: bad[0]['requests'][0]['request_id'] = 'other'
            with self.assertRaises(AssertionError): validate_queries(bad,'consumer','rid')

    def test_model_and_synthetic_pair_scopes_cannot_be_confused(self):
        pair = {'protocol':'native_model_KV_functional_v1','synthetic_payload':False,
            'job_id':'123','hostname':'owned','host':'127.0.0.1','request_prefix':'test',
            'context':128,'generated_tokens':64,'roles':{
                'producer':{'gpu_uuid':'GPU-a','port':20001},
                'consumer':{'gpu_uuid':'GPU-b','port':20002}}}
        session = {'slurm_job_id':'123','expected_hostname':'owned','gpu_uuid':'GPU-a'}
        validate_scope(pair,session,'producer','GPU-a','123','owned')
        for key,value in (('synthetic_payload',True),('context',4096),('generated_tokens',65),('job_id','124')):
            bad = copy.deepcopy(pair); bad[key] = value
            with self.assertRaises(AssertionError): validate_scope(bad,session,'producer','GPU-a','123','owned')

    def test_metadata_control_bypasses_intrusive_payload_injection_hook(self):
        calls = []
        native = types.SimpleNamespace(send_tensor=lambda *a: True,recv_tensor=lambda *a: object(),
            get_finished=lambda *a:(None,None),recv_request_id_to_tensor_ids={})
        original_load = lambda *a,**kw: calls.append('native_load')
        connector = types.SimpleNamespace(p2p_nccl_engine=native,start_load_kv=original_load,is_producer=False)
        hook = NativeLedger(connector)
        try:
            hook.active = {'payload_proof':False,'received':[]}
            # No forward-context metadata or tensor copies are needed in this
            # arm; a hash/order-changing hook would fail this check.
            connector.start_load_kv(object())
            self.assertEqual(calls,['native_load'])
        finally: hook.close()
        self.assertIs(connector.start_load_kv,original_load)


if __name__ == '__main__': unittest.main()
