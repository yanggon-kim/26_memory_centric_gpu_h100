import copy
import unittest
from h200_review_functional_pilots import prefix_coverage


class Coverage(unittest.TestCase):
    def receipt(self):
        return {'input_prompt_tokens':1024,'first_selected_tokens':1,'actual_prefill_steps':4,
            'whole':False,'token_budget':256,'scheduler_rows':[
                {'composition':'prefill_only','any_preempted_request':False,'post_prompt_input_queries':0,
                 'requests':[{'computed_tokens_before_step':offset,'original_prompt_tokens':1024,'prompt_input_queries':256}]}
                for offset in (0,256,512,768)]}

    def test_full_prefix_coverage_and_whole_semantics(self):
        data=self.receipt();self.assertTrue(prefix_coverage(data,1024))
        data['whole']=True
        with self.assertRaises(AssertionError):prefix_coverage(data,1024)

    def test_duplicate_prefix_and_decode_in_prefill_are_rejected(self):
        data=self.receipt();data['scheduler_rows'][1]['requests'][0]['computed_tokens_before_step']=0
        with self.assertRaises(AssertionError):prefix_coverage(data,1024)
        data=self.receipt();data['scheduler_rows'][-1]['post_prompt_input_queries']=1
        with self.assertRaises(AssertionError):prefix_coverage(data,1024)


if __name__=='__main__':unittest.main()
