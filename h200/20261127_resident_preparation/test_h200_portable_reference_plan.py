"""CPU content-reuse tests; no model, GPU, output certificate or timing."""
import json
from pathlib import Path
import tempfile
import unittest

from h200_portable_reference_plan import plan
from h200_portable_serving_trace import make_trace


class References(unittest.TestCase):
    def save(self, folder, name, trace):
        path=Path(folder)/name
        path.write_text(json.dumps(trace))
        return path

    def test_arrival_variants_share_only_reference_work_not_offered_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[self.save(folder,str(i)+'.json',make_trace('lclo',arrival,rate,30,101,'development'))
                for i,(arrival,rate) in enumerate([('poisson',1.),('bursty',1.),('poisson',2.)])]
            result=plan(paths)
            self.assertEqual(result['offered_request_occurrences'],90)
            self.assertEqual(result['unique_input_contracts'],30)
            self.assertEqual(result['repeated_input_contract_occurrences'],60)
            self.assertEqual(sum(o['trace_warmup_request'] for row in result['items'] for o in row['occurrences']),9)
            self.assertTrue(all(len(row['occurrences'])==3 for row in result['items']))
            self.assertTrue(all(row['qualified_reference'] is None for row in result['items']))
            self.assertFalse(result['performance_qualified'])

    def test_independent_seed_and_full_output_contracts_remain_distinct(self):
        with tempfile.TemporaryDirectory() as folder:
            traces=[make_trace('lclo','poisson',1.,30,seed,'development') for seed in (101,102)]
            result=plan([self.save(folder,str(i)+'.json',trace) for i,trace in enumerate(traces)])
            self.assertEqual(result['unique_input_contracts'],60)
            self.assertEqual(result['repeated_input_contract_occurrences'],0)
            contracts={(tuple(r['contract']['prompt_ids']),r['contract']['max_tokens']) for r in result['items']}
            self.assertEqual(len(contracts),60)
            self.assertTrue(all(r['output_tokens']==r['contract']['max_tokens'] for r in result['items']))

    def test_changed_sealed_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            trace=make_trace('lclo','poisson',1.,30,101,'development')
            trace['requests'][0]['max_tokens']-=1
            with self.assertRaises(AssertionError):plan([self.save(folder,'bad.json',trace)])

    def test_duplicate_trace_cannot_inflate_reference_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.save(folder,'trace.json',make_trace('lclo','poisson',1.,30,101,'development'))
            with self.assertRaises(AssertionError):plan([path,path])


if __name__=='__main__':unittest.main()
