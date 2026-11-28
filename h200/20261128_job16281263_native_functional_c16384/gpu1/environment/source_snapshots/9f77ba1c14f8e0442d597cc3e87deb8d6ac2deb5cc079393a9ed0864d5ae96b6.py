"""CPU contracts against independent, pinned AMD input digests; no GPU result."""
import copy
import unittest
from unittest.mock import patch

import h200_portable_serving_trace as portable


class PortableInputs(unittest.TestCase):
    def test_independent_AMD_digest_fixtures_and_bursty_ties(self):
        # Obtained from git show 8c4cc5b9:scripts/amd/cp/cp_trace.py,
        # executed directly without importing this H200 adapter.
        fixtures = [
            ('lclo', 'poisson', 1.0, 101, 'development',
             'c4e2e139fd3c3ae39678b958361df96bebfd71ca8b32cb0f24de2d5c631f9dab',
             '383d0e067b39f2de5cc3ad6f4d923650dff5bf2c6f97712966eedd3befb33bc1'),
            ('mixdoc', 'bursty', 1.0, 101, 'development',
             '70a92a725b4272741b96f8f75b7d395645e46b6414ca1fd56f4255cada12325b',
             '06f7013d2c59acc518beaae94fbdbcab26cee4bad68c0396844368b5775b58a2'),
            ('lctx', 'poisson', .1, 201, 'judging',
             '6b791ec240a8d9aaf6324429b8e0935e70a640b51d391bbe51ec53c6edb4db6b',
             'ea0d3b3342718fc30bd790fa5c5c152c22556215b98c1d626e6ed466264e7f6a')]
        for workload, arrival, rate, seed, split, trace_hash, content_hash in fixtures:
            with self.subTest(workload=workload):
                trace = portable.make_trace(workload, arrival, rate, 30, seed, split)
                self.assertEqual(trace['AMD_trace_sha256'], trace_hash)
                self.assertEqual(trace['content_sha256'], content_hash)
                self.assertTrue(portable.validate(trace))
                if arrival == 'bursty':
                    times = [r['arrival_s'] for r in trace['requests']]
                    self.assertEqual(sum(a == b for a, b in zip(times, times[1:])), 1)

    def test_contents_do_not_change_with_rate_or_arrival(self):
        traces = [portable.make_trace('mixdoc', arrival, rate, 30, 102, 'development')
                  for arrival, rate in [('poisson', 1), ('poisson', 2.), ('bursty', 1.)]]
        self.assertEqual(len({t['content_sha256'] for t in traces}), 1)
        self.assertEqual(len({t['AMD_trace_sha256'] for t in traces}), 3)
        for trace in traces:
            self.assertEqual(trace['warmup_requests'], 3)
            self.assertIs(type(trace['configured_rate_requests_s']), float)
            self.assertTrue(all(len(r['prompt_ids'])+r['max_tokens'] <= 32768
                                for r in trace['requests']))

    def test_resealing_a_changed_prompt_does_not_bypass_generator_check(self):
        trace = portable.make_trace('mixdoc', 'poisson', 1., 30, 101, 'development')
        corrupt = copy.deepcopy(trace)
        corrupt['requests'][-1]['prompt_ids'][-1] ^= 1
        with self.assertRaisesRegex(AssertionError, 'sealed portable trace changed'):
            portable.validate(corrupt)
        corrupt['sha256'] = portable.digest(corrupt)
        with self.assertRaisesRegex(AssertionError, 'portable input mismatch'):
            portable.validate(corrupt)

    def test_split_leakage_and_insufficient_warmups_are_rejected(self):
        for seed, split, count in [(201, 'development', 30), (101, 'judging', 30),
                                   (101, 'development', 29), (True, 'development', 30)]:
            with self.subTest(seed=seed, split=split, count=count):
                with self.assertRaises(AssertionError):
                    portable.make_trace('lclo', 'poisson', 1., count, seed, split)

    def test_modified_vendored_source_is_rejected_before_execution(self):
        with patch.object(portable.Path, 'read_bytes', return_value=b'raise RuntimeError()'):
            with self.assertRaisesRegex(AssertionError, 'generator source changed'):
                portable.generator()


if __name__ == '__main__':
    unittest.main()
