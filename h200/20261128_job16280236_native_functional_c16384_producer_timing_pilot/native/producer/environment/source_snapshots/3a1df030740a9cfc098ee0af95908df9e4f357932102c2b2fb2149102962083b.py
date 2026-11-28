import copy
import hashlib
import re
import unittest
from h200_native_pd_contract import request_id, phase_contract, validate_layers


class NativeContract(unittest.TestCase):
    def test_native_address_patterns_and_globally_distinct_ids(self):
        ids = [request_id(f'job123_a0_trial{i}', '127.0.0.1', 21001, '127.0.0.1', 21002)
               for i in range(10)]
        self.assertEqual(len(set(ids)), 10)
        for rid in ids:
            self.assertEqual(re.search(r'___decode_addr_(.*):(\d+)', rid).groups(), ('127.0.0.1', '21002'))
            self.assertEqual(re.search(r'___prefill_addr_(.*):(\d+)___', rid).groups(), ('127.0.0.1', '21001'))
        with self.assertRaises(AssertionError):
            request_id('unsafe___prefill_addr_x', '127.0.0.1', 21001, '127.0.0.1', 21002)

    def test_first_token_boundary_and_payload_padding_are_separate(self):
        plan = phase_contract([3]*128, 7)
        self.assertEqual((plan['producer_generated_tokens'], plan['consumer_generated_tokens'], plan['canonical_generated_tokens']), (1, 63, 64))
        self.assertEqual(plan['consumer_prompt_ids'], [3]*128+[7])
        self.assertEqual((plan['consumer_prefill_classified_queries_expected'], plan['consumer_post_prompt_queries_expected'], plan['canonical_decode_queries_expected']), (1, 62, 63))
        self.assertEqual((plan['producer_blocks'], plan['consumer_blocks']), (8, 9))
        self.assertEqual(plan['analytical_requested_KV_tensor_bytes'], 32*1024**2)
        padded = phase_contract([3]*129, 7)
        self.assertEqual(padded['padding_tokens'], 15)
        self.assertGreater(padded['analytical_requested_KV_tensor_bytes'], padded['logical_meaningful_KV_bytes'])
        self.assertIsNone(plan['physical_HBM_or_NVLink_bytes'])

    def test_complete_pair_ledger_rejects_missing_corrupt_or_uninjected_layers(self):
        rid = request_id('job123_pilot0', '127.0.0.1', 21001, '127.0.0.1', 21002)
        rows = [{'layer_index': i, 'tensor_id': rid+f'#layer{i}', 'shape': [2,8,16,8,128],
                 'dtype': 'torch.bfloat16', 'nbytes': 524288,
                 'sha256': hashlib.sha256(str(i).encode()).hexdigest(),
                 'contiguous': True, 'completed': True, 'injected': True} for i in range(64)]
        self.assertEqual(validate_layers(rows, copy.deepcopy(rows), rid, 32*1024**2)['status'], 'paired_layer_ledger_passed')
        with self.assertRaises(AssertionError):validate_layers(rows, rows[:-1], rid, 32*1024**2)
        for key,value in (('completed', False), ('injected', False), ('dtype','torch.float16'), ('sha256','0'*64), ('shape',[262144])):
            other=copy.deepcopy(rows);other[9][key]=value
            with self.assertRaises(AssertionError):validate_layers(rows, other, rid, 32*1024**2)


if __name__ == '__main__':unittest.main()
