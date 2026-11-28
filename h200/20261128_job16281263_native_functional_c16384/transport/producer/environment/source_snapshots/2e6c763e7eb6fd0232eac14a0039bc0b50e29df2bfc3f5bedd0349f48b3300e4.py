import copy
import unittest
from h200_review_native_model import check_pair_trials
from h200_native_pd_contract import validate_layers


class ModelPairReader(unittest.TestCase):
    def fixture(self):
        pair = {'roles':{'producer':{'port':20001},'consumer':{'port':20002}}}
        prompt = list(range(128)); expected = list(range(64)); source = []; target = []
        for index in range(12):
            payload = index>=6; rid = f'r{index}'
            common = {'status':'functional_phase_passed','mode':'payload_proof' if payload else 'metadata_control',
                'payload_proof':payload,'trial':index%6,'warmup':index%6<3,'request_id':rid,'internal_request_id':rid}
            sent = []
            for layer in range(64):
                row = {'tensor_id':f'{rid}#model.layers.{layer}.self_attn.attn','layer_index':layer,
                    'shape':[2,8,16,8,128],'dtype':'torch.bfloat16','contiguous':True,
                    'nbytes':524288,'completed':True,'remote_address':'127.0.0.1:20002'}
                if payload: row['sha256'] = 'a'*64
                sent.append(row)
            schedule = lambda computed,prompt_len,n,prefill: {'any_preempted_request':False,'requests':[{
                'request_id':rid,'computed_tokens_before_step':computed,'original_prompt_tokens':prompt_len,
                'scheduled_input_queries':n,'prompt_input_queries':prefill,'post_prompt_input_queries':n-prefill}]}
            a = dict(common,sent=sent,received=[],injected=[],outputs=[{'output_ids':expected[:1]}],
                scheduler_rows=[schedule(0,128,128,128)])
            recvs = copy.deepcopy(sent)
            for row in recvs: row['remote_address'] = '127.0.0.1:20001'
            injected = copy.deepcopy(recvs) if payload else []
            for row in injected: row['injected'] = True
            b = dict(common,sent=[],received=recvs,injected=injected,
                outputs=[{'output_ids':expected[1:i+2]} for i in range(63)],
                scheduler_rows=[schedule(128+i,129,1,int(i==0)) for i in range(63)],
                native_receive_completions=[r['tensor_id'] for r in recvs])
            if payload: b['payload_gate'] = validate_layers(sent,injected,rid,33554432)
            source.append(a); target.append(b)
        return source,target,expected,prompt,pair

    def test_complete_functional_fixture_does_not_qualify_deployment(self):
        args = self.fixture(); rows = check_pair_trials(*args)
        self.assertEqual(len(rows),12)
        self.assertTrue(all(r['canonical_outputs']==64 and r['consumer_original_prompt_queries']==0 for r in rows))

    def test_bad_injection_control_hashes_phase_count_and_peer_are_rejected(self):
        for failure in ('injection','control_hash','outputs','reprefill','peer','received_hash'):
            a,b,expected,prompt,pair = self.fixture()
            if failure == 'injection': b[-1]['injected'].pop()
            elif failure == 'control_hash': a[0]['sent'][0]['sha256'] = 'a'*64
            elif failure == 'outputs': b[0]['outputs'][-1]['output_ids'].append(0)
            elif failure == 'reprefill': b[0]['scheduler_rows'][0]['requests'][0]['scheduled_input_queries'] = 129
            elif failure == 'peer': a[0]['sent'][0]['remote_address'] = '127.0.0.1:20001'
            else: b[-1]['received'][0]['sha256'] = 'b'*64
            with self.assertRaises(AssertionError): check_pair_trials(a,b,expected,prompt,pair)


if __name__ == '__main__': unittest.main()
