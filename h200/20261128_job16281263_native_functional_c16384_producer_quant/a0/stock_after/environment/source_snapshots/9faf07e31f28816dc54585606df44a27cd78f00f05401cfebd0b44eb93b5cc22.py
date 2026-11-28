"""CPU-prepared profiler gate; not integrated with a GPU entrypoint yet.

Callbacks belong to a future explicitly instrumented role. A balanced host
range does not prove CUDA launch coverage, layer attribution or clean timing.
"""
import time


class CaptureGate:
    def __init__(self,start,stop,push,pop):
        self.start=start;self.stop=stop;self.push=push;self.pop=pop
        self.events=[];self.active=None;self.row=None

    def observe_active(self,row):
        if row is None:
            if self.active is not None:self._finish(False)
            return
        assert self.active is None,'overlapping request while profiler range active'
        if row['mode']!='metadata_control' or row['trial']!=3:
            return
        assert row['warmup'] is False and row['payload_proof'] is False
        assert not self.events,'selected request may only be captured once'
        event={'request_id':row['request_id'],'mode':row['mode'],'trial':row['trial'],
            'host_begin_unix_ns':time.time_ns(),'host_end_unix_ns':None,
            'status':'starting','profiler_start_returned':False,
            'profiler_stop_attempted':False,'profiler_stop_returned':False,
            'nvtx_push_returned':False,'nvtx_pop_returned':False,
            'request_functional_phase_passed':False,'errors':[]}
        self.events.append(event);self.active=event;self.row=row
        try:
            self.start();event['profiler_start_returned']=True
            self.push('h200_connector_metadata_request/'+row['request_id'])
            event['nvtx_push_returned']=True;event['status']='capturing'
        except BaseException as error:
            event['errors'].append({'operation':'start_or_push','type':type(error).__name__,'error':str(error)})
            self._finish(True)
            raise

    def _finish(self,partial):
        event=self.active
        assert event is not None
        error=None
        try:
            if event['nvtx_push_returned']:
                try:self.pop();event['nvtx_pop_returned']=True
                except BaseException as caught:
                    error=caught;event['errors'].append({'operation':'pop','type':type(caught).__name__,'error':str(caught)})
            if event['profiler_start_returned']:
                event['profiler_stop_attempted']=True
                try:self.stop();event['profiler_stop_returned']=True
                except BaseException as caught:
                    error=error or caught;event['errors'].append({'operation':'stop','type':type(caught).__name__,'error':str(caught)})
            event['request_functional_phase_passed']=self.row.get('status')=='functional_phase_passed'
            event['status']='range_closed_pending_GPU_coverage_review' if not partial and not event['errors'] and event['request_functional_phase_passed'] else 'partial_or_failed_range'
        finally:
            event['host_end_unix_ns']=time.time_ns()
            self.active=None;self.row=None
        if error is not None:raise error

    def close(self):
        if self.active is not None:self._finish(True)

    def summary(self):
        return {'classification':'host profiler gate accounting only; no GPU execution or CUDA coverage demonstrated by this module',
            'selected_mode':'metadata_control','selected_trial':3,
            'capture_count':len(self.events),'active':self.active is not None,
            'events':self.events,'GPU_kernel_coverage_verified':False,
            'GPU_layer_compute_attributed':False,'clean_timing_qualified':False,
            'scope':'Future role must retain original twelve-request functional proof, separate intrusive payload arm, CUDA/NVTX trace, source/runtime/exact-output and actual launch coverage checks. Profiler callback success alone is insufficient.'}
