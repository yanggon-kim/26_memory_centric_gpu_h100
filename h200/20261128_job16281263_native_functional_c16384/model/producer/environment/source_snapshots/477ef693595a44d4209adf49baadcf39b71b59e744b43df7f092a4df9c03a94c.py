from types import SimpleNamespace
import unittest
from h200_serving_trace import make_trace, requests_for, trial_seed, record_output, summarize


class TraceContracts(unittest.TestCase):
    def test_late_admission_keeps_offered_ttft_and_censored_denominator(self):
        records = requests_for(make_trace(42, 1., requests=12))
        r = records['r0000']
        r['admitted_s'] = 2.
        expected = list(range(64))
        for i in range(64):
            output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=expected[:i+1],
                         finish_reason='length' if i == 63 else None)], finished=i == 63)
            record_output(r, output, 3.+i*.02, expected)
        assert r['ttft_s'] == 3. and r['admission_delay_s'] == 2.
        result = summarize(records, 5., {'ttft_s': 2., 'maximum_inter_token_s': .1})
        assert result['completed_requests'] == 1 and result['offered_requests'] == 12
        assert result['compliant_completed_requests'] == 0
        assert len(result['unfinished_request_ids']) == 11
        assert not result['all_requests_completed'] and not result['steady_state_demonstrated']


    def test_matched_traces_are_sealed_and_development_streams_disjoint(self):
        streams = [{trial_seed(a, t, split) for a in range(3) for t in range(10)}
                   for split in ('development', 'judging', 'warmup')]
        assert len(set.union(*streams)) == 90
        trace = make_trace(trial_seed(0, 0, 'judging'), .5)
        assert trace == make_trace(trial_seed(0, 0, 'judging'), .5)
        trace['offered_s'][1] += .01
        with unittest.TestCase().assertRaisesRegex(AssertionError, 'trace changed'):
            requests_for(trace)


    def test_output_mismatch_and_early_length_finish_fail(self):
        r = requests_for(make_trace(42, 1., 12))['r0000'];r['admitted_s'] = 0.
        output = SimpleNamespace(outputs=[SimpleNamespace(token_ids=[9], finish_reason='length')], finished=True)
        with unittest.TestCase().assertRaisesRegex(AssertionError, 'greedy output'):
            record_output(r, output, 1., list(range(64)))
        output.outputs[0].token_ids = [0]
        with unittest.TestCase().assertRaises(AssertionError):
            record_output(r, output, 1., list(range(64)))

if __name__ == "__main__":
    unittest.main()
