import contextlib
import types
import unittest
from unittest.mock import patch
from hf_full_reference import static_cache_gate
from h200_priority_campaign import strict_reference_record


class Static(unittest.TestCase):
    def check(self,ids,cache_length):
        model=types.SimpleNamespace(config=object())
        torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        transformers=types.SimpleNamespace(StaticCache=lambda **kwargs:object())
        case={'context_length':3,'generated_tokens':2,'trials':[{'output_ids':[[7,8]]} for _ in range(3)]}
        with patch('software_decode_experiments.make_decode',return_value=object()) as decode,patch(
            'software_decode_experiments.controlled_trial',return_value={'output_ids':ids,'output_sha256':'CPU-fake','cache_length':cache_length}) as trial:
            result=static_cache_gate(model,torch,transformers,None,case,[[7,8]],lambda value:None)
        self.assertEqual(trial.call_count,6);self.assertIs(decode.call_args.args[-1],False)
        self.assertEqual(len(result['warmups']),3);self.assertEqual(len(result['trials']),3)
        return result

    def test_original_static_path_and_cache_progress_must_match(self):
        self.assertTrue(self.check([[7,8]],4)['passed'])
        self.assertFalse(self.check([[7,8]],3)['passed'])

    def test_dynamic_agreement_never_promotes_failed_static_ids(self):
        result=self.check([[7,9]],4)
        self.assertFalse(result['passed']);self.assertEqual(result['status'],'failed')
        self.assertEqual(result['first_differences'][0]['token_index'],1)

    def test_reference_requires_both_modes_and_exact_source_checksum(self):
        data={'status':'passed','dynamic_cache_gate_passed':True,'static_cache_gate':{'status':'passed','passed':True},'source_sha256':'CPU-source'}
        self.assertTrue(strict_reference_record(data,'CPU-source'))
        self.assertFalse(strict_reference_record(data,'different-source'))
        data.pop('static_cache_gate')
        self.assertFalse(strict_reference_record(data,'CPU-source'))


if __name__=='__main__':unittest.main()
