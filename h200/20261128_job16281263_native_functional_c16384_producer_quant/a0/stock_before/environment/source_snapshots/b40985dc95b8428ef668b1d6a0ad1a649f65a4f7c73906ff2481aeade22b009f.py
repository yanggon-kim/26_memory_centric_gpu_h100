"""CPU failure/selection contracts only, never a GPU profiler test."""
import unittest
from h200_connector_capture_gate import CaptureGate


class Gate(unittest.TestCase):
    def row(self,trial=3,payload=False):
        return {'request_id':'request-'+str(trial),'mode':'payload_proof' if payload else 'metadata_control',
            'trial':trial,'warmup':trial<3,'payload_proof':payload,'status':'running'}

    def make(self):
        calls=[]
        gate=CaptureGate(lambda:calls.append('start'),lambda:calls.append('stop'),
            lambda name:calls.append('push'),lambda:calls.append('pop'))
        return gate,calls

    def test_only_first_diagnostic_metadata_request_is_selected(self):
        gate,calls=self.make()
        for payload in (False,True):
            for trial in range(6):
                row=self.row(trial,payload);gate.observe_active(row)
                row['status']='functional_phase_passed';gate.observe_active(None)
        self.assertEqual(calls,['start','push','pop','stop'])
        result=gate.summary();self.assertEqual(result['capture_count'],1)
        self.assertEqual(result['events'][0]['status'],'range_closed_pending_GPU_coverage_review')
        self.assertFalse(result['GPU_kernel_coverage_verified'])

    def test_interruption_closes_range_and_retains_partial_status(self):
        gate,calls=self.make();gate.observe_active(self.row());gate.close();gate.close()
        self.assertEqual(calls,['start','push','pop','stop'])
        self.assertEqual(gate.events[0]['status'],'partial_or_failed_range')
        self.assertFalse(gate.events[0]['request_functional_phase_passed'])

    def test_push_failure_stops_profiler_without_unbalanced_pop(self):
        gate,calls=self.make()
        def failed(name):raise RuntimeError('NVTX push failed')
        gate.push=failed
        with self.assertRaisesRegex(RuntimeError,'NVTX push failed'):gate.observe_active(self.row())
        self.assertEqual(calls,['start','stop']);self.assertIsNone(gate.active)
        self.assertEqual(gate.events[0]['status'],'partial_or_failed_range')

    def test_pop_failure_still_attempts_stop_and_preserves_error(self):
        gate,calls=self.make()
        def failed():raise RuntimeError('NVTX pop failed')
        gate.pop=failed;gate.observe_active(self.row())
        with self.assertRaisesRegex(RuntimeError,'NVTX pop failed'):gate.close()
        self.assertEqual(calls,['start','push','stop']);self.assertIsNone(gate.active)
        self.assertTrue(gate.events[0]['profiler_stop_returned'])
        self.assertEqual(gate.events[0]['errors'][0]['operation'],'pop')

    def test_duplicate_selected_request_never_starts_second_capture(self):
        gate,calls=self.make();row=self.row();gate.observe_active(row)
        row['status']='functional_phase_passed';gate.observe_active(None)
        with self.assertRaisesRegex(AssertionError,'only be captured once'):gate.observe_active(self.row())
        self.assertEqual(calls,['start','push','pop','stop'])


if __name__=='__main__':unittest.main()
