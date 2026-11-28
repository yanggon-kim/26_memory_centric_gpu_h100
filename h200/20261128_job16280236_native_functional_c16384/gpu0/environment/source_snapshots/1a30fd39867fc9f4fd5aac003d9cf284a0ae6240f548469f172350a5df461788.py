import unittest
from h200_producer_quant_budget import role_bound


class RuntimeBudget(unittest.TestCase):
    def test_model_initialization_is_not_scaled_as_thirteen_requests(self):
        result=role_bound(180,[5]*6)
        self.assertEqual(result['observed_initialization_warmups_and_other_s'],165)
        self.assertEqual(result['timeout_s'],299)
        self.assertTrue(result['within_existing_role_watchdog'])

    def test_cold_first_warmup_is_not_extrapolated_to_each_timed_request(self):
        result=role_bound(200,[50,5,5,1,1,1])
        self.assertEqual(result['timeout_s'],289)
        self.assertEqual(result['observed_initialization_warmups_and_other_s'],197)

    def test_inconsistent_or_nonfinite_pilot_cannot_admit_a_cohort(self):
        for elapsed,requests in [(float('nan'),[1]*6),(5,[1]*6),(10,[1]*5),(10,[1,1,1,1,1,float('inf')])]:
            with self.assertRaises(AssertionError):role_bound(elapsed,requests)

    def test_existing_role_watchdog_is_not_silently_extended(self):
        self.assertFalse(role_bound(650,[1]*6)['within_existing_role_watchdog'])


if __name__=='__main__':unittest.main()
