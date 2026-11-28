import types
import unittest
from h200_serving_policy_pilot import update_request


def output(ids,finished=False):
    return types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=ids,finish_reason='length')],finished=finished)


class Requests(unittest.TestCase):
    def record(self):return {'offered_s':1.,'admitted_s':3.,'output_ids':[],'token_observed_s':[]}

    def test_ttft_starts_at_offered_arrival_and_first_token_is_separate(self):
        record=self.record()
        update_request(record,output([7]),4.,[7,8])
        self.assertEqual(record['ttft_s'],3.)
        self.assertEqual(record['admission_to_first_token_s'],1.)
        update_request(record,output([7,8],True),4.2,[7,8])
        self.assertEqual(record['finished_s'],4.2)

    def test_changed_greedy_prefix_preserved_then_rejected(self):
        record=self.record()
        with self.assertRaisesRegex(AssertionError,'differs'):update_request(record,output([9]),4.,[7,8])
        self.assertEqual(record['output_ids'],[9])

    def test_multi_token_jump_and_premature_finish_are_rejected(self):
        with self.assertRaises(AssertionError):update_request(self.record(),output([7,8]),4.,[7,8])
        with self.assertRaises(AssertionError):update_request(self.record(),output([7],True),4.,[7,8])


if __name__=='__main__':unittest.main()
