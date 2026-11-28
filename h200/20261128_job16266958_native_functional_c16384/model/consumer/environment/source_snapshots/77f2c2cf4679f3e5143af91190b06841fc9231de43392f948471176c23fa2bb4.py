"""Synthetic CPU observations test accounting boundaries, never GPU evidence."""
import copy
import unittest
import h200_portable_serving_trace as inputs
import h200_portable_serving_score as scoring


class Accounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace=inputs.make_trace('mixdoc','poisson',1.,30,101,'development')
        cls.expected={q['id']:[7]*q['max_tokens'] for q in cls.trace['requests']}
        cls.records=[{'id':q['id'],'status':'complete','submitted_s':q['arrival_s']+.1,
            'output_ids':cls.expected[q['id']][:],
            'token_observed_s':[q['arrival_s']+.5+j*.01 for j in range(q['max_tokens'])],
            'finish_reason':'length'} for q in cls.trace['requests']]
        cls.end=max(r['token_observed_s'][-1] for r in cls.records)+10

    def test_missing_failed_and_wrong_outputs_keep_denominator(self):
        records=copy.deepcopy(self.records)
        records[-2]['output_ids'][-1]=8
        records[-3].update(status='timeout',output_ids=[],token_observed_s=[])
        missing=records.pop()['id']
        result=scoring.score(self.trace,records,self.expected,self.end)
        measured=result['measured']
        self.assertEqual(result['all_offered'],30)
        self.assertEqual(measured['offered'],27)
        self.assertEqual(measured['completed_by_count_and_length'],25)
        self.assertEqual(measured['completed_with_wrong_outputs'],1)
        self.assertEqual(measured['exact_reference_completed'],24)
        self.assertEqual(measured['objectives']['primary']['compliant'],24)
        self.assertEqual(measured['objectives']['primary']['attainment'],24/27)
        self.assertEqual(result['missing_record_ids'],[missing])
        self.assertEqual(len(result['all_failed_or_incomplete_ids']),3)
        self.assertIsNone(result['decoder_step_latency_ms'])
        self.assertFalse(result['quantitative_performance_qualified'])
        self.assertIsNone(result['warmup']['objectives']['primary']['compliant_requests_per_s_including_drain'])

    def test_a_single_stall_hidden_by_p95_changes_stall_objective(self):
        records=copy.deepcopy(self.records)
        times=records[-1]['token_observed_s']
        for index in range(50,len(times)): times[index]+=1.5
        result=scoring.score(self.trace,records,self.expected,self.end+2)
        row=result['per_request'][-1]
        self.assertLess(row['request_itl_p95_ms'],50)
        self.assertTrue(row['primary'])
        self.assertTrue(row['itl50'])
        self.assertFalse(row['stall1000'])
        self.assertTrue(row['stall2000'])

    def test_mixdoc_class_limits_and_class_counts(self):
        records=copy.deepcopy(self.records)
        for record in records:
            record['token_observed_s']=[t+3 for t in record['token_observed_s']]
        result=scoring.score(self.trace,records,self.expected,self.end+4)
        classes=result['measured_by_class']
        self.assertEqual(sum(c['offered'] for c in classes.values()),27)
        self.assertGreater(classes['document']['offered'],0)
        self.assertEqual(classes['chat']['objectives']['primary']['compliant'],0)
        self.assertEqual(classes['document']['objectives']['primary']['compliant'],classes['document']['offered'])

    def test_missing_references_and_timestamp_corruption_cannot_qualify(self):
        expected=dict(self.expected); expected.pop(self.records[-1]['id'])
        result=scoring.score(self.trace,self.records,expected,self.end)
        self.assertEqual(result['measured']['completed_without_exact_reference'],1)
        self.assertEqual(result['measured']['objectives']['primary']['compliant'],26)
        records=copy.deepcopy(self.records)
        records[-1]['token_observed_s'][1]=records[-1]['token_observed_s'][0]-.01
        with self.assertRaisesRegex(AssertionError,'timestamps moved backward'):
            scoring.score(self.trace,records,self.expected,self.end)


class Capacity(unittest.TestCase):
    @staticmethod
    def row(rate,seed,attainment=1.,qualified=True,crash=False):
        return {'rate':rate,'seed':seed,'scientific_qualified':qualified,'engine_abort':crash,
            'incomplete':0,'offered':27,'exact_reference_completed':27,
            'objectives':{'primary':{'attainment':attainment}}}

    def test_lower_rate_failure_prevents_a_higher_rate_recovery(self):
        rows=[self.row(r,s,.8 if (r,s)==(2.,202) else 1.) for r in (1.,2.,3.) for s in (201,202)]
        out=scoring.sustainable(rows,[1.,2.,3.],[201,202])
        self.assertEqual(out['sustainable_rate_requests_s'],1.)
        self.assertEqual(out['first_unpassed_rate']['rate'],2.)
        self.assertFalse(out['right_censored'])

    def test_missing_seed_crash_or_unqualified_boundary_is_censored(self):
        for kind in ('missing','crash','unqualified'):
            with self.subTest(kind=kind):
                rows=[self.row(r,s) for r in (1.,2.) for s in (201,202)]
                if kind=='missing': rows.pop()
                elif kind=='crash': rows[-1]['engine_abort']=True
                else: rows[-1]['scientific_qualified']=False
                out=scoring.sustainable(rows,[1.,2.],[201,202])
                self.assertEqual(out['sustainable_rate_requests_s'],1.)
                self.assertTrue(out['right_censored'])
                self.assertFalse(out['top_of_ladder_passed'])

    def test_nan_and_duplicate_cells_cannot_pass_the_ladder(self):
        with self.assertRaises(AssertionError):
            scoring.sustainable([self.row(1.,201,float('nan'))],[1.],[201])
        with self.assertRaises(AssertionError):
            scoring.sustainable([self.row(1.,201)]*2,[1.],[201])
        out=scoring.sustainable([self.row(1.,201)],[1.],[201])
        self.assertTrue(out['top_of_ladder_passed'])
        self.assertTrue(out['right_censored'])


if __name__=='__main__': unittest.main()
