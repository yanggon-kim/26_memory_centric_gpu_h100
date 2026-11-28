import copy
import unittest
from h200_review_producer_timing import check_trials,check_pair,raw_energy
from h200_producer_timing_role import check_step_accounting,ACCOUNTING_SOURCES


class ProducerReader(unittest.TestCase):
    def test_counter_reset_and_unavailability_never_yield_primary_energy(self):
        run,_,_=self.fixture('producer');row=run['trials'][0]
        self.assertEqual(raw_energy(row)['raw_counter_delta_mj'],10)
        self.assertIsNone(raw_energy(row)['primary_J_per_token'])
        row['raw_energy_counter']['after']['energy_mj']=1
        self.assertEqual(raw_energy(row)['status'],'counter_reset')
        row['raw_energy_counter']['after'].update(status='unavailable',energy_mj=None,error='unsupported')
        self.assertEqual(raw_energy(row)['status'],'counter_unavailable')
        self.assertIsNone(raw_energy(row)['primary_J_per_token'])

    def fixture(self,role):
        expected=list(range(64));pair={'warmup_trials':3,'timed_trials':3,'mode':'pilot','acquisition':0,
            'protocol':'matched_native_producer_host_timing_v1','context':16384}
        rows=[]
        for trial in range(6):
            rid='r'+str(trial);steps=[];outputs=[]
            for i in range(63 if role=='consumer' else 1):
                start=100+10*i;end=start+5
                steps.append({'start_unix_ns':start,'end_unix_ns':end,'elapsed_ns':5})
                outputs.append({'observed_unix_ns':end,'output_ids':expected[1:i+2] if role=='consumer' else [0],
                    'num_cached_tokens':16384 if role=='consumer' else 0,
                    'computed_tokens_before_step':0 if i==0 else 16384+i,
                    'computed_tokens_after_step':16384+i+1 if role=='consumer' else 16384})
            queries=[{'request_id':rid,'computed_tokens_before_step':16384+i,'scheduled_input_queries':1,
                'original_prompt_tokens':16385,'prompt_input_queries':int(i==0),'post_prompt_input_queries':int(i>0)} for i in range(63)] if role=='consumer' else [
                {'request_id':rid,'computed_tokens_before_step':0,'scheduled_input_queries':16384,'original_prompt_tokens':16384}]
            rows.append({'trial':trial,'warmup':trial<3,'observer_present':trial<3,'request_id':rid,'status':'exact_output_passed',
                'host_step_intervals':steps,'outputs':outputs,'submit_unix_ns':99,'submit_to_final_step_return_ns':steps[-1]['end_unix_ns']-99,
                'final_step_return_unix_ns':steps[-1]['end_unix_ns'],'first_token_file_read_unix_ns':98,
                'producer_first_step_return_unix_ns':105,'all_native_sends_observed_unix_ns':108,
                'consumer_completion_unix_ns':725,'native_sent_tensor_count':64,
                'warmup_scheduler_rows':[{'any_preempted_request':False,'requests':queries}] if trial<3 else None,
                'completed_unix_ns':900,
                'engine_request_preemptions':0,
                'raw_energy_counter':{'before':{'query_start_unix_ns':90,'query_end_unix_ns':91,'status':'available','energy_mj':100,'error':None},
                    'after':{'query_start_unix_ns':800,'query_end_unix_ns':801,'status':'available','energy_mj':110,'error':None},
                    'counter_unit':'mJ','primary_J_per_token':None}})
        for row in rows:row['query_accounting']=check_step_accounting(row,role,16384)
        return {'status':'complete','role':role,'mode':'pilot','protocol':pair['protocol'],'acquisition':0,'trials':rows,
            'request_accounting_source_sha256':ACCOUNTING_SOURCES},pair,expected

    def test_actual_timed_prefix_recompute_or_preemption_is_rejected(self):
        run,pair,expected=self.fixture('consumer');run['trials'][3]['outputs'][0]['num_cached_tokens']=0
        with self.assertRaises(AssertionError):check_trials(run,pair,expected)
        run,pair,expected=self.fixture('consumer');run['trials'][4]['engine_request_preemptions']=1
        with self.assertRaises(AssertionError):check_trials(run,pair,expected)

    def test_raw_decoder_ticks_exclude_the_first_handoff_query(self):
        run,pair,expected=self.fixture('consumer');metrics=check_trials(run,pair,expected)
        self.assertEqual(len(metrics['consumer_first_handoff_query_host_ms']),3)
        self.assertEqual(len(metrics['consumer_following62query_host_ms']),3*62)
        run['trials'][3]['outputs'][7]['output_ids']=[9]
        with self.assertRaises(AssertionError):check_trials(run,pair,expected)

    def test_missing_send_or_instrumented_timed_row_cannot_pass(self):
        run,pair,expected=self.fixture('producer');check_trials(run,pair,expected)
        broken=copy.deepcopy(run);broken['trials'][3]['native_sent_tensor_count']=63
        with self.assertRaises(AssertionError):check_trials(broken,pair,expected)
        broken=copy.deepcopy(run);broken['trials'][4]['observer_present']=True
        with self.assertRaises(AssertionError):check_trials(broken,pair,expected)

    def test_actual_p0_must_precede_receiver_handoff_and_ack_match(self):
        producer,_,_=self.fixture('producer');consumer,_,_=self.fixture('consumer')
        for row in consumer['trials']:
            row['first_token_file_read_unix_ns']=106
        check_pair(producer,consumer)
        consumer['trials'][2]['first_token_file_read_unix_ns']=104
        with self.assertRaises(AssertionError):check_pair(producer,consumer)


if __name__=='__main__':unittest.main()
