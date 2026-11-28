import unittest
from h200_producer_startup_budget_v2 import role_timeout,propose


class Budget(unittest.TestCase):
    def fixture(self):
        pilot=dict(acquisitions=3,warmups_per_condition=3,timed_per_condition=10,
            phase_timeout_s={'stock_before':251,'native':392,'stock_after':165},
            role_source_sha256='pinned',context=16384,engine_settings={'dtype':'bfloat16'})
        row=dict(condition='stock_after',role='stock',source_role_sha256='pinned',context=16384,
            engine_settings={'dtype':'bfloat16'},elapsed_s=165.455,completed_lifecycle_s=[2.2]*6,process_status='timeout')
        return pilot,row

    def test_partial_elapsed_is_lower_bound_and_missing_requests_are_budgeted(self):
        pilot,row=self.fixture();result=propose(pilot,[row])
        self.assertGreater(result['phase_timeout_s']['stock_after'],300)
        self.assertTrue(result['observations'][0]['elapsed_is_lower_bound'])
        self.assertEqual(result['observations'][0]['missing_to_fixed13requests'],7)
        self.assertEqual((result['acquisitions'],result['warmups_per_condition'],result['timed_per_condition']),(3,3,10))
        self.assertEqual(result['performance_samples_used'],0)

    def test_startup_bound_cannot_shorten_original_or_exceed_watchdog(self):
        self.assertGreaterEqual(role_timeout(90,[2.]*6,'complete')['timeout_s'],300)
        with self.assertRaisesRegex(AssertionError,'watchdog'):role_timeout(600,[2.]*6,'timeout')

    def test_source_model_settings_mismatch_cannot_supply_runtime_bound(self):
        for key,value in [('source_role_sha256','other'),('context',32704),('engine_settings',{'dtype':'float8'})]:
            pilot,row=self.fixture();row[key]=value
            with self.subTest(field=key),self.assertRaises(AssertionError):propose(pilot,[row])

    def test_invalid_or_empty_runtime_observations_are_rejected(self):
        for elapsed,times,status in [(float('nan'),[1.],'complete'),(1.,[],'timeout'),(1.,[2.],'complete'),(10.,[1.],'failed')]:
            with self.subTest(values=(elapsed,times,status)),self.assertRaises(AssertionError):role_timeout(elapsed,times,status)

    def test_original_controller_identity_and_watchdog_contract_is_preserved(self):
        pilot,row=self.fixture();result=propose(pilot,[row])
        self.assertEqual(result['context'],pilot['context'])
        self.assertEqual(result['engine_settings'],pilot['engine_settings'])
        self.assertEqual(result['role_source_sha256'],pilot['role_source_sha256'])
        self.assertTrue(result['all_bounds_within_existing660second_watchdog'])
        self.assertEqual(result['required_s'],3*sum(result['phase_timeout_s'].values())+360)


if __name__=='__main__':unittest.main()
