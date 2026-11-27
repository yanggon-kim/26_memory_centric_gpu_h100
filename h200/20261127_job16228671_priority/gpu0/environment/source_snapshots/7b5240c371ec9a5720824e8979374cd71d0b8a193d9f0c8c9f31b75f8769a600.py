import types
import unittest
from h200_prefill_resident import drain_prefill


def output(n=1,finished=False):
    return types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[7]*n)],finished=finished)


class Barrier(unittest.TestCase):
    def test_chunked_prefill_reaches_same_one_token_barrier(self):
        replies=iter([[],[],[],[output()]])
        result,calls=drain_prefill(lambda:next(replies))
        self.assertEqual(calls,4);self.assertEqual(len(result[0].outputs[0].token_ids),1)

    def test_decode_tokens_and_premature_finish_rejected(self):
        with self.assertRaises(AssertionError):drain_prefill(lambda:[output(2)])
        with self.assertRaises(AssertionError):drain_prefill(lambda:[output(finished=True)])

    def test_missing_barrier_has_finite_attempt_bound(self):
        with self.assertRaisesRegex(RuntimeError,'bounded'):drain_prefill(lambda:[],limit=3)


if __name__=='__main__':unittest.main()
