import copy
import types
import unittest
from unittest.mock import patch
from h200_native_model_pilot import validate_scope, validate_queries, NativeLedger, observe_output
from h200_native_pd_contract import phase_contract
from h200_review_native_model import host_pool_accounting


class NativeModelContracts(unittest.TestCase):
    def test_long_context_scope_and_query_counts_are_separate_from_C128(self):
        pair={'protocol':'native_model_KV_long_context_functional_v2','synthetic_payload':False,
            'job_id':'123','hostname':'owned','host':'127.0.0.1','request_prefix':'long',
            'context':16384,'generated_tokens':64,'roles':{
                'producer':{'gpu_uuid':'GPU-a','port':20001},'consumer':{'gpu_uuid':'GPU-b','port':20002}}}
        session={'slurm_job_id':'123','expected_hostname':'owned','gpu_uuid':'GPU-a'}
        validate_scope(pair,session,'producer','GPU-a','123','owned')
        for protocol,context in (('native_model_KV_functional_v1',16384),
                                 ('native_model_KV_long_context_functional_v2',128),
                                 ('native_model_KV_long_context_functional_v2',32768)):
            wrong=dict(pair,protocol=protocol,context=context)
            with self.assertRaises(AssertionError):validate_scope(wrong,session,'producer','GPU-a','123','owned')
        self.assertEqual(phase_contract(list(range(16384)),1)['analytical_requested_KV_tensor_bytes'],4*1024**3)
        rows=[{'any_preempted_request':False,'requests':[{'request_id':'rid','computed_tokens_before_step':i,
            'original_prompt_tokens':16385,'scheduled_input_queries':1,
            'prompt_input_queries':int(i==16384),'post_prompt_input_queries':int(i>16384)}]} for i in range(16384,16447)]
        validate_queries(rows,'consumer','rid',context=16384)
        rows[0]['requests'][0]['computed_tokens_before_step']=0
        with self.assertRaises(AssertionError):validate_queries(rows,'consumer','rid',context=16384)

    def test_delayed_native_pool_free_stays_with_its_original_request(self):
        tensor=types.SimpleNamespace(numel=lambda:64,element_size=lambda:2)
        blocks={}
        def store(value):blocks[100]=types.SimpleNamespace(size=128);return 100
        def load(*args):return tensor
        def free(addr):blocks.pop(addr)
        pool=types.SimpleNamespace(store_tensor=store,load_tensor=load,free=free,allocated_blocks=blocks)
        native=types.SimpleNamespace(pool=pool,send_tensor=lambda *a:True,recv_tensor=lambda *a:None,
            get_finished=lambda *a:(None,None),recv_request_id_to_tensor_ids={})
        connector=types.SimpleNamespace(p2p_nccl_engine=native,start_load_kv=lambda *a:None,is_producer=False)
        hook=NativeLedger(connector)
        first={'request_id':'first','host_pool_events':[]};following={'request_id':'next','host_pool_events':[]}
        try:
            hook.active=first;addr=pool.store_tensor(tensor);pool.load_tensor(addr,None,(64,),None)
            self.assertEqual(hook.host_pool_snapshot(),[{'address':100,'request_id':'first','requested_tensor_bytes':128}])
            hook.active=following;pool.free(addr)
            self.assertEqual(following['host_pool_events'],[])
            self.assertEqual(len(first['host_pool_events']),3)
            self.assertEqual(hook.host_pool_snapshot(),[])
            result=host_pool_accounting({'host_pool_events':[]},first,128)
            self.assertEqual(result['native_host_spill_tensor_bytes'],128)
            self.assertEqual(result['retained_before_owned_cleanup'],[])
            wrong=copy.deepcopy(first);wrong['host_pool_events'].pop(1)
            with self.assertRaises(AssertionError):host_pool_accounting({'host_pool_events':[]},wrong,128)
        finally:hook.close()
        self.assertIs(pool.store_tensor,store);self.assertIs(pool.load_tensor,load);self.assertIs(pool.free,free)

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
