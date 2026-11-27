import types
import unittest
from h200_prefill_resident import drain_prefill,submission_clock


def output(n=1,finished=False):
    return types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[7]*n)],finished=finished)


class Barrier(unittest.TestCase):
    def test_submission_clock_preserves_native_return_and_restores_on_failure(self):
        def add(value):return value
        engine=types.SimpleNamespace(add_request=add)
        with submission_clock(engine) as record:
            self.assertEqual(engine.add_request('id'),'id')
            self.assertGreaterEqual(record['end_perf'],record['start_perf'])
            with self.assertRaises(AssertionError):engine.add_request('duplicate')
        self.assertIs(engine.add_request,add)
        def fail():raise RuntimeError('native submit failed')
        engine.add_request=fail
        with self.assertRaisesRegex(RuntimeError,'native submit'):
            with submission_clock(engine):engine.add_request()
        self.assertIs(engine.add_request,fail)

    def test_clock_endpoint_precedes_submission(self):
        events=[]
        engine=types.SimpleNamespace(add_request=lambda:events.append('submit'))
        def sample():events.append('hardware');return {'sm_clock_mhz':1980}
        with submission_clock(engine,sample) as record:
            engine.add_request()
        self.assertEqual(events,['hardware','submit'])
        self.assertEqual(record['hardware_before']['sm_clock_mhz'],1980)

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
