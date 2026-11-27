#!/usr/bin/env python3
"""CPU contract checks; these are not measurements or scheduler GPU validation."""
from types import SimpleNamespace as NS
import unittest
from h200_scheduler_observer import composition, Observer


def fixture():
    output = NS(scheduled_new_reqs=[NS(req_id='new', num_computed_tokens=0,
                                      prompt_token_ids=[11]*8)],
                scheduled_cached_reqs=NS(req_ids=['old'], num_computed_tokens=[100]),
                num_scheduled_tokens={'new': 4, 'old': 1}, total_num_scheduled_tokens=5,
                scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={})
    # Live state was advanced by schedule(): reading it would misclassify old
    # chunk boundaries. Composition must instead use output's retained offsets.
    requests = {'new': NS(num_prompt_tokens=8, num_computed_tokens=4, num_preemptions=0),
                'old': NS(num_prompt_tokens=100, num_computed_tokens=101, num_preemptions=0)}
    return output, requests


class Scheduler:
    def __init__(self):
        self.output, self.requests = fixture()
        self.running, self.waiting = [self.requests['old']], [self.requests['new']]

    def schedule(self):
        return self.output


class Contract(unittest.TestCase):
    def test_composition_and_retained_offsets(self):
        output, requests = fixture()
        self.assertEqual(composition(output, requests)['composition'], 'mixed')
        output.scheduled_cached_reqs.num_computed_tokens = [99]
        row = composition(output, requests)
        self.assertEqual(row['composition'], 'prefill_only')
        self.assertEqual(row['prompt_input_queries'], 5)
        self.assertEqual(row['post_prompt_input_queries'], 0)

    def test_preemption_and_missing_metadata(self):
        output, requests = fixture()
        requests['old'].num_preemptions = 1
        self.assertTrue(composition(output, requests)['any_preempted_request'])
        output.scheduled_cached_reqs.req_ids = []
        with self.assertRaises(AssertionError):
            composition(output, requests)

    def test_speculative_decode_rejected(self):
        output, requests = fixture()
        output.scheduled_spec_decode_tokens = {'old': [7]}
        with self.assertRaises(AssertionError):
            composition(output, requests)

    def test_hook_preserves_output_and_restores_method(self):
        InprocClient = type('InprocClient', (), {})
        client = InprocClient()
        scheduler = Scheduler()
        client.engine_core = NS(scheduler=scheduler)
        original = scheduler.schedule
        observer = Observer(NS(engine_core=client))
        self.assertIs(scheduler.schedule(), scheduler.output)
        self.assertEqual(observer.rows[0]['composition'], 'mixed')
        self.assertFalse(observer.identity['clean_timing_eligible'])
        observer.close()
        self.assertEqual(scheduler.schedule, original)
        observer.close()


if __name__ == '__main__':
    unittest.main()
