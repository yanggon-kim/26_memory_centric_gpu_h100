import copy
import unittest
from h200_review_serving_pilot import check_requests,check_schedule
from h200_serving_policy import PINNED_SCHEDULER_SHA256


def fixture():
    stamps=[2.+i*.01 for i in range(64)]
    request={'offered_s':0.,'admitted_s':1.,'internal_request_id':'internal0',
             'output_ids':list(range(64)),'token_observed_s':stamps,'first_token_s':stamps[0],
             'finished_s':stamps[-1],'ttft_s':2.,
             'inter_token_s':[b-a for a,b in zip(stamps,stamps[1:])]}
    rows=[]
    for index in range(64):
        prompt,decode=(4,0) if index==0 else (0,1)
        rows.append({'composition':'prefill_only' if index==0 else 'decode_only',
            'any_preempted_request':False,'prompt_input_queries':prompt,'post_prompt_input_queries':decode,
            'requests':[{'request_id':'internal0','original_prompt_tokens':4,'num_preemptions':0,
                'computed_tokens_before_step':0 if index==0 else 3+index,
                'prompt_input_queries':prompt,'post_prompt_input_queries':decode,
                'scheduled_input_queries':prompt+decode}]})
    return {'classification':'measured-real-GPU-instrumented','requests':{'external0':request},
        'policy_identity':{'policy':'whole_separate','scheduler_source_sha256':PINNED_SCHEDULER_SHA256,
                           'adapter_source_sha256':'a'*64},
        'observer_identity':{'source_sha256':PINNED_SCHEDULER_SHA256,'observer_source_sha256':'b'*64},
        'scheduler_rows':rows,'policy_rows':[{'composition':r['composition']} for r in rows],
        'steps':[{'composition':r['composition'],'wall_ms':1.,'gpu_event_ms':.5} for r in rows]}


class ServingAccounting(unittest.TestCase):
    def test_offered_ttft_and_all_outputs_are_required(self):
        data=fixture();check_requests(data['requests'],list(range(64)),[0.])
        late=copy.deepcopy(data);late['requests']['external0']['ttft_s']=1.
        with self.assertRaises(AssertionError):check_requests(late['requests'],list(range(64)),[0.])
        partial=copy.deepcopy(data);partial['requests']['external0']['output_ids'].pop()
        with self.assertRaises(AssertionError):check_requests(partial['requests'],list(range(64)),[0.])

    def test_schedule_rejects_unaccounted_queries_and_unknown_requests(self):
        data=fixture();check_schedule(data,4,'whole_separate')
        for key,value in (('post_prompt_input_queries',0),):
            bad=copy.deepcopy(data);bad['scheduler_rows'][1][key]=value
            with self.assertRaises(AssertionError):check_schedule(bad,4,'whole_separate')
        bad=copy.deepcopy(data);bad['scheduler_rows'][1]['requests'][0]['request_id']='unknown'
        with self.assertRaises(AssertionError):check_schedule(bad,4,'whole_separate')

    def test_source_identity_and_finite_steps_cannot_be_relaxed(self):
        data=fixture();check_schedule(data,4,'whole_separate',{'h200_serving_policy.py':'a'*64,'h200_scheduler_observer.py':'b'*64})
        with self.assertRaises(AssertionError):check_schedule(data,4,'whole_separate',{'h200_serving_policy.py':'c'*64,'h200_scheduler_observer.py':'b'*64})
        data['steps'][0]['wall_ms']=float('inf')
        with self.assertRaises(AssertionError):check_schedule(data,4,'whole_separate')


if __name__=='__main__':unittest.main()
