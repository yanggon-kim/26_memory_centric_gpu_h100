"""CPU selection/provenance association contracts, never GPU attribution."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.append(str(Path(__file__).resolve().parents[2]/'scripts'))
from h200_review_connector_capture import checked_gate,method_associations,paired_barrier_consistency,PROTOCOL


class Review(unittest.TestCase):
    def fixture(self):
        trials=[{'request_id':mode+'_'+str(i),'mode':mode,'trial':i,'warmup':i<3,
            'payload_proof':mode=='payload_proof','status':'functional_phase_passed',
            'submit_unix_ns':100 if i==3 else 300 if i==4 else 1,'completed_unix_ns':200 if i==3 else 301}
            for mode in ('metadata_control','payload_proof') for i in range(6)]
        gate={'protocol':PROTOCOL,'actor_pid':100,'capture_count':1,'active':False,
            'events':[{'mode':'metadata_control','trial':3,'request_id':'metadata_control_3',
                'status':'range_closed_pending_GPU_coverage_review','errors':[],
                'profiler_start_returned':True,'profiler_stop_attempted':True,'profiler_stop_returned':True,
                'nvtx_push_returned':True,'nvtx_pop_returned':True,'request_functional_phase_passed':True,
                'host_begin_unix_ns':50,'host_end_unix_ns':250}]}
        gate['peer_barriers']=[{'stage':stage,'request_id':'metadata_control_3','status':'peer_ready_verified',
            'actor_pid':100,'role':'producer','gpu_uuid':'GPU-a','peer_actor_pid':200,'peer_role':'consumer','peer_gpu_uuid':'GPU-b',
            'self_ready_unix_ns':ready,'peer_ready_unix_ns':ready+5,'peer_received_unix_ns':ready+10}
            for stage,ready in (('started',60),('stopped',260))]
        return gate,{'trials':trials,'role':'producer','gpu_uuid':'GPU-a'}

    def test_selected_gate_requires_all_original_requests_and_balanced_success(self):
        gate,run=self.fixture();self.assertEqual(checked_gate(gate,run)['request_id'],'metadata_control_3')
        for mutation in ('wrong_request','missing_stop','warmup','incomplete_sequence','invalid_pid'):
            g,r=copy.deepcopy(gate),copy.deepcopy(run)
            if mutation=='wrong_request':g['events'][0]['request_id']='other'
            if mutation=='missing_stop':g['events'][0]['profiler_stop_returned']=False
            if mutation=='warmup':r['trials'][3]['warmup']=True
            if mutation=='incomplete_sequence':r['trials'].pop()
            if mutation=='invalid_pid':g['actor_pid']=True
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):checked_gate(g,r)

    def test_missing_or_unverified_peer_stage_cannot_establish_capture_alignment(self):
        for kind in ('missing_start','missing_stop','failed'):
            gate,run=self.fixture()
            if kind=='missing_start':gate['peer_barriers'].pop(0)
            if kind=='missing_stop':gate['peer_barriers'].pop()
            if kind=='failed':gate['peer_barriers'][0]['status']='waiting_for_peer'
            with self.subTest(kind=kind),self.assertRaises(AssertionError):checked_gate(gate,run)

    def test_request_submission_cannot_precede_start_or_next_request_precede_stop(self):
        for kind in ('early_selected','early_next'):
            gate,run=self.fixture()
            if kind=='early_selected':run['trials'][3]['submit_unix_ns']=65
            else:run['trials'][4]['submit_unix_ns']=265
            with self.subTest(kind=kind),self.assertRaises(AssertionError):checked_gate(gate,run)

    def test_peer_pid_uuid_timestamp_and_request_are_checked_across_actors(self):
        gate,run=self.fixture()
        a={'role':'producer','gpu_uuid':'GPU-a','actor_pid':100,'selected_request_id':'metadata_control_3',
            'selected_submit_unix_ns':100,'next_submit_unix_ns':300,'peer_barriers':gate['peer_barriers']}
        b=copy.deepcopy(a);b.update(role='consumer',gpu_uuid='GPU-b',actor_pid=200)
        for event in b['peer_barriers']:
            event.update(peer_role='producer',peer_actor_pid=100,peer_gpu_uuid='GPU-a')
            event['self_ready_unix_ns'],event['peer_ready_unix_ns']=event['peer_ready_unix_ns'],event['self_ready_unix_ns']
        self.assertTrue(paired_barrier_consistency([a,b]))
        for fault in ('pid','gpu','timestamp','request'):
            wrong=copy.deepcopy(b)
            if fault=='pid':wrong['peer_barriers'][0]['peer_actor_pid']=999
            if fault=='gpu':wrong['peer_barriers'][0]['peer_gpu_uuid']='GPU-other'
            if fault=='timestamp':wrong['peer_barriers'][0]['peer_ready_unix_ns']+=1
            if fault=='request':wrong['selected_request_id']='other-request'
            with self.subTest(fault=fault),self.assertRaises(AssertionError):paired_barrier_consistency([a,wrong])

    def test_overlapping_kernel_unions_and_unassigned_request_ids_are_distinct(self):
        events=[{'index':0,'owner':'Native','method':'send','status':'returned','request_id':'request'},
            {'index':1,'owner':'Pool','method':'load','status':'returned','request_id':None}]
        timeline={'selected_kernels':[{'start':a,'end':b,'innermost_connector_NVTX':label} for a,b,label in
            ((0,10,'h200_connector/Native.send/0'),(5,15,'h200_connector/Native.send/0'),(20,30,'h200_connector/Pool.load/1'),(40,50,None))]}
        rows,errors=method_associations(timeline,{'events':events});self.assertEqual(errors,[])
        self.assertEqual(rows['Native.send']['kernel_duration_sum_ns'],20)
        self.assertEqual(rows['Native.send']['kernel_active_union_ns'],15)
        self.assertEqual(rows['Pool.load']['unassigned_request_event_indices'],[1])

    def test_unknown_sidecar_marker_is_not_attributed(self):
        row={'selected_kernels':[{'start':0,'end':10,'innermost_connector_NVTX':'h200_connector/Native.send/99'}]}
        summary,errors=method_associations(row,{'events':[]})
        self.assertEqual(summary,{})
        self.assertEqual(errors[0]['reason'],'missing_sidecar_event')


if __name__=='__main__':unittest.main()
