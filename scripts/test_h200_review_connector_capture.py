"""CPU selection/provenance association contracts, never GPU attribution."""
import copy
from pathlib import Path
import sys
import unittest

from h200_review_connector_capture import checked_gate,method_associations,PROTOCOL


class Review(unittest.TestCase):
    def fixture(self):
        trials=[{'request_id':mode+'_'+str(i),'mode':mode,'trial':i,'warmup':i<3,
            'payload_proof':mode=='payload_proof','status':'functional_phase_passed'}
            for mode in ('metadata_control','payload_proof') for i in range(6)]
        gate={'protocol':PROTOCOL,'actor_pid':100,'capture_count':1,'active':False,
            'events':[{'mode':'metadata_control','trial':3,'request_id':'metadata_control_3',
                'status':'range_closed_pending_GPU_coverage_review','errors':[],
                'profiler_start_returned':True,'profiler_stop_attempted':True,'profiler_stop_returned':True,
                'nvtx_push_returned':True,'nvtx_pop_returned':True,'request_functional_phase_passed':True,
                'host_begin_unix_ns':1,'host_end_unix_ns':2}]}
        return gate,{'trials':trials}

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
