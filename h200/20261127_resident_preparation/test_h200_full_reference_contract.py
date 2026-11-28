"""CPU false-positive rejection for full-output reference contracts."""
import copy
import unittest
from h200_full_reference_contract import check_tokens


class Tokens(unittest.TestCase):
    def fixture(self):
        model=dict(id='anchor',revision='pinned',precision='BF16 weights and KV',
            tensor_parallel_size=1,max_positions=32768,vocab_size=100)
        contract=dict(model=model,seed=20261013,prompt_ids=[1,2,3],max_tokens=2,
            greedy=True,ignore_eos=True,repetition_penalty=1.,prefix_cache=False)
        ids=[[7,8]]
        source=dict(status='complete',config=dict(precision='BF16',seed=20261013,model_id='anchor',model_revision='pinned'),
            engine_settings=dict(dtype='bfloat16',tensor_parallel_size=1,seed=20261013,enable_prefix_caching=False),
            hardware=dict(uuid='GPU-fixture'),cases=[dict(batch_size=1,context_length=3,generated_tokens=2,
                prompt_ids=[[1,2,3]],prompt_sha256='fixture',warmup_output_ids=[ids]*3,output_digests_identical=True,
                trials=[dict(output_ids=ids,prefill_barrier=dict(passed=True),decode_iterations=1) for _ in range(3)])])
        rows=[dict(output_ids=ids,cache_length=4) for _ in range(3)]
        hf=dict(status='passed',dynamic_cache_gate_passed=True,source_sha256='pinned-source',
            model_id='anchor',revision='pinned',precision='BF16',batch=1,context=3,generated_tokens=2,
            prompt_ids=[[1,2,3]],prompt_sha256='fixture',gpu_uuid='GPU-fixture',
            warmups=copy.deepcopy(rows),trials=copy.deepcopy(rows),
            static_cache_gate=dict(status='passed',passed=True,warmups=copy.deepcopy(rows),trials=copy.deepcopy(rows)))
        return source,hf,contract

    def test_complete_default_and_static_outputs_agree(self):
        self.assertEqual(check_tokens(*self.fixture(),'pinned-source'),[7,8])

    def test_flags_cannot_hide_wrong_final_token_in_any_HF_phase(self):
        for arm in ('dynamic','static'):
            for phase in ('warmups','trials'):
                source,hf,contract=self.fixture()
                target=hf if arm=='dynamic' else hf['static_cache_gate']
                target[phase][-1]['output_ids']=[[7,9]]
                with self.subTest(arm=arm,phase=phase),self.assertRaises(AssertionError):check_tokens(source,hf,contract,'pinned-source')

    def test_short_output_or_missing_HF_trial_is_rejected(self):
        for change in ('short','missing'):
            source,hf,contract=self.fixture()
            if change=='short':hf['trials'][-1]['output_ids']=[[7]]
            else:hf['static_cache_gate']['warmups'].pop()
            with self.subTest(change=change),self.assertRaises(AssertionError):check_tokens(source,hf,contract,'pinned-source')

    def test_cache_progress_prompt_GPU_and_conditioning_are_binding(self):
        for change in ('cache','prompt','GPU','conditioning','diagnostic'):
            source,hf,contract=self.fixture()
            if change=='cache':hf['static_cache_gate']['trials'][-1]['cache_length']=3
            if change=='prompt':hf['prompt_ids']=[[3,2,1]]
            if change=='GPU':hf['gpu_uuid']='GPU-other'
            if change=='conditioning':source['config']['decode_conditioning']={'protocol':'different-history'}
            if change=='diagnostic':hf['deterministic_reference_diagnostic']=True
            with self.subTest(change=change),self.assertRaises(AssertionError):check_tokens(source,hf,contract,'pinned-source')

    def test_numeric_equality_cannot_accept_float_tokens_in_either_engine(self):
        for route in ('source_warmup','source_trial','HF_dynamic','HF_static'):
            source,hf,contract=self.fixture()
            if route=='source_warmup':source['cases'][0]['warmup_output_ids'][0]=[[7.0,8]]
            if route=='source_trial':source['cases'][0]['trials'][0]['output_ids']=[[7,8.0]]
            if route=='HF_dynamic':hf['trials'][0]['output_ids']=[[7.0,8]]
            if route=='HF_static':hf['static_cache_gate']['warmups'][0]['output_ids']=[[7,8.0]]
            with self.subTest(route=route),self.assertRaisesRegex(AssertionError,'integer token IDs'):
                check_tokens(source,hf,contract,'pinned-source')

    def test_prompt_ID_schema_is_required_for_contract_and_both_engines(self):
        for route in ('contract','source','HF'):
            source,hf,contract=self.fixture()
            if route=='contract':contract['prompt_ids']=[1.0,2,3]
            if route=='source':source['cases'][0]['prompt_ids']=[[1,2.0,3]]
            if route=='HF':hf['prompt_ids']=[[1,2,3.0]]
            with self.subTest(route=route),self.assertRaisesRegex(AssertionError,'integer token IDs'):
                check_tokens(source,hf,contract,'pinned-source')


if __name__=='__main__':unittest.main()
