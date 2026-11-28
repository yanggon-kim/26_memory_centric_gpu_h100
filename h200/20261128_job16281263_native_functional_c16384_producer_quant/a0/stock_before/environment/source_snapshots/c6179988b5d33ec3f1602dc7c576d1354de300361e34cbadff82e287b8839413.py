"""Prospective bounded connector capture; not yet promoted to GPU runtime.

Retains the original twelve-request functional protocol. Only metadata trial3
opens cudaProfilerApi/NVTX capture. Its serialized handoff and host observers
are instrumented controls, never clean producer timing or deployed goodput.
"""
import argparse
import json
import os
from pathlib import Path
import sys

PROTOCOL='host_API_NVTX_metadata_barrier_v3'


def main(args):
    pair=json.loads(args.pair_config.read_text())
    assert pair.get('connector_attribution_protocol')==PROTOCOL
    assert pair['protocol']=='native_model_KV_long_context_functional_v2'
    args.output.mkdir(parents=True,exist_ok=False)
    import h200_native_model_pilot as functional
    from h200_connector_spans import ConnectorSpans
    from h200_connector_capture_gate import CaptureGate
    from h200_connector_capture_barrier import PeerBarrier
    from h200_native_transport_probe import wait_json
    from software_decode_experiments import dump
    original=functional.NativeLedger
    failures=[];closed=[]

    class ObservedLedger(original):
        def __init__(self,connector):
            self.capture_gate=None;self._active=None
            super().__init__(connector)
            import torch
            self.capture_gate=CaptureGate(torch.cuda.profiler.start,torch.cuda.profiler.stop,
                torch.cuda.nvtx.range_push,torch.cuda.nvtx.range_pop)
            self.peer_barrier=PeerBarrier(pair,args.role,os.getpid(),dump,wait_json)
            try:
                self.connector_spans=ConnectorSpans(connector,torch.cuda.nvtx.range_push,torch.cuda.nvtx.range_pop)
            except BaseException:
                super().close()
                raise

        @property
        def active(self):return self._active

        @active.setter
        def active(self,row):
            previous=self._active
            self._active=row
            if self.capture_gate is not None:
                self.capture_gate.observe_active(row)
                if row is not None and row['mode']=='metadata_control' and row['trial']==3:
                    self.peer_barrier.meet('started',row['request_id'])
                elif row is None and previous is not None and previous['mode']=='metadata_control' and previous['trial']==3:
                    self.peer_barrier.meet('stopped',previous['request_id'])

        def close(self):
            # Defer diagnostic failures until functional.main has executed its
            # owned-engine cleanup; a raised close must not strand that cleanup.
            def failed(operation,error):
                failures.append({'operation':operation,'type':type(error).__name__,'error':str(error)})
            try:self.capture_gate.close()
            except BaseException as error:failed('profiler_gate_close',error)
            gate=self.capture_gate.summary();gate['protocol']=PROTOCOL;gate['actor_pid']=os.getpid()
            gate['peer_barriers']=self.peer_barrier.history
            closed.append(gate)
            spans=self.connector_spans
            document={'protocol':PROTOCOL,'actor_pid':os.getpid(),'classification':spans.classification,
                'events':spans.events,'clean_timing_qualified':False,'GPU_compute_attributed':False,
                'scope':'Host API/NVTX spans, not GPU duration. Nested/concurrent intervals are not additive. Payload-proof arms include hashing overhead and are excluded from selected metadata capture. Serialized handoff is not deployment latency.'}
            restored=False
            try:
                document['summary']=spans.summary()
                spans.close();restored=True
            except BaseException as error:failed('connector_spans_close',error)
            if restored:
                try:super().close()
                except BaseException as error:failed('functional_hooks_restore',error)
            for name,data in (('profiler_capture_gate.json',gate),('connector_host_spans.json',document)):
                data['diagnostic_close_errors']=list(failures)
                try:dump(args.output/name,data)
                except BaseException as error:
                    failed('write_'+name,error)
                    sys.stderr.write('Failed to persist connector diagnostic '+name+' ('+type(error).__name__+')\n')

    functional.NativeLedger=ObservedLedger
    try:
        result=functional.main(args)
    finally:
        functional.NativeLedger=original
    if failures:
        dump(args.output/'connector_capture_failures.json',{'status':'diagnostic_failure','errors':failures})
        raise RuntimeError('Connector profiler diagnostics failed after owned functional cleanup')
    assert len(closed)==1 and closed[0]['capture_count']==1 and not closed[0]['active']
    assert closed[0]['events'][0]['status']=='range_closed_pending_GPU_coverage_review'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-config',type=Path,required=True)
    parser.add_argument('--role',choices=('producer','consumer'),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
