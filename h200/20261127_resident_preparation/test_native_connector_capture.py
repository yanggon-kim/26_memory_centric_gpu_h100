"""CPU lifecycle integration with fake engine/CUDA callbacks; no GPU evidence."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from h200_native_connector_capture import main,PROTOCOL


class Spans:
    classification='CPU fixture'
    def __init__(self,*args):self.events=[]
    def summary(self):return {'CPU_fixture':True}
    def close(self):pass


class Lifecycle(unittest.TestCase):
    def setup_modules(self,interrupt=False,pop_failure=False):
        calls=[];functional=ModuleType('h200_native_model_pilot')
        class Original:
            def __init__(self,connector):self.active=None
            def close(self):calls.append('original_hooks_restored')
        functional.NativeLedger=Original
        def delegated(args):
            hook=functional.NativeLedger(object())
            try:
                for payload in (False,True):
                    for trial in range(6):
                        row={'request_id':str(payload)+'_'+str(trial),
                            'mode':'payload_proof' if payload else 'metadata_control',
                            'payload_proof':payload,'trial':trial,'warmup':trial<3,'status':'running'}
                        hook.active=row;calls.append(('request',payload,trial))
                        if interrupt and not payload and trial==3:raise RuntimeError('toy request interrupted')
                        row['status']='functional_phase_passed';hook.active=None
            finally:
                try:hook.close()
                finally:calls.append('owned_engine_cleanup')
            return 'functional-return'
        functional.main=delegated
        torch=ModuleType('torch')
        def pop():
            calls.append('pop')
            if pop_failure:raise RuntimeError('toy NVTX pop failed')
        torch.cuda=SimpleNamespace(profiler=SimpleNamespace(start=lambda:calls.append('start'),stop=lambda:calls.append('stop')),
            nvtx=SimpleNamespace(range_push=lambda name:calls.append('push'),range_pop=pop))
        spans=ModuleType('h200_connector_spans');spans.ConnectorSpans=Spans
        return {'h200_native_model_pilot':functional,'torch':torch,'h200_connector_spans':spans},calls,Original

    def args(self,root,protocol=PROTOCOL):
        pair=root/'pair.json';pair.write_text(json.dumps({'protocol':'native_model_KV_long_context_functional_v2','connector_attribution_protocol':protocol}))
        return SimpleNamespace(pair_config=pair,output=root/'new-output',role='producer',gpu_uuid='CPU-fixture')

    def test_original_twelve_requests_and_exactly_one_capture_are_retained(self):
        modules,calls,original=self.setup_modules()
        with tempfile.TemporaryDirectory() as folder,patch.dict('sys.modules',modules):
            args=self.args(Path(folder));self.assertEqual(main(args),'functional-return')
            saved=json.loads((args.output/'profiler_capture_gate.json').read_text())
            self.assertEqual(saved['capture_count'],1);self.assertFalse(saved['GPU_kernel_coverage_verified'])
            self.assertEqual(saved['actor_pid'],os.getpid())
            self.assertEqual(len([r for r in calls if isinstance(r,tuple)]),12)
            self.assertEqual([r for r in calls if isinstance(r,str)],['start','push','pop','stop','original_hooks_restored','owned_engine_cleanup'])
            self.assertIs(modules['h200_native_model_pilot'].NativeLedger,original)

    def test_interrupted_selected_request_keeps_partial_capture_and_cleanup(self):
        modules,calls,original=self.setup_modules(interrupt=True)
        with tempfile.TemporaryDirectory() as folder,patch.dict('sys.modules',modules):
            args=self.args(Path(folder))
            with self.assertRaisesRegex(RuntimeError,'toy request interrupted'):main(args)
            saved=json.loads((args.output/'profiler_capture_gate.json').read_text())
            self.assertEqual(saved['events'][0]['status'],'partial_or_failed_range')
            self.assertEqual(calls[-2:],['original_hooks_restored','owned_engine_cleanup'])
            self.assertIs(modules['h200_native_model_pilot'].NativeLedger,original)

    def test_profiler_failure_is_raised_after_engine_cleanup(self):
        modules,calls,original=self.setup_modules(pop_failure=True)
        with tempfile.TemporaryDirectory() as folder,patch.dict('sys.modules',modules):
            args=self.args(Path(folder))
            with self.assertRaisesRegex(RuntimeError,'NVTX pop failed'):main(args)
            self.assertIn('owned_engine_cleanup',calls)
            saved=json.loads((args.output/'profiler_capture_gate.json').read_text())
            self.assertEqual(saved['events'][0]['errors'][0]['operation'],'pop')
            self.assertTrue(saved['events'][0]['profiler_stop_returned'])
            self.assertIs(modules['h200_native_model_pilot'].NativeLedger,original)

    def test_unreviewed_protocol_does_not_create_output_or_import_CUDA(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.args(Path(folder),'host_API_NVTX_functional_v1')
            with self.assertRaises(AssertionError):main(args)
            self.assertFalse(args.output.exists())

    def test_incomplete_native_spans_are_preserved_without_stranding_cleanup(self):
        modules,calls,original=self.setup_modules(interrupt=True)
        class Incomplete(Spans):
            def close(self):raise AssertionError('native spans still active')
        modules['h200_connector_spans'].ConnectorSpans=Incomplete
        with tempfile.TemporaryDirectory() as folder,patch.dict('sys.modules',modules):
            args=self.args(Path(folder))
            with self.assertRaisesRegex(RuntimeError,'toy request interrupted'):main(args)
            self.assertIn('owned_engine_cleanup',calls)
            self.assertNotIn('original_hooks_restored',calls)
            saved=json.loads((args.output/'connector_host_spans.json').read_text())
            self.assertEqual(saved['diagnostic_close_errors'][0]['operation'],'connector_spans_close')
            self.assertIs(modules['h200_native_model_pilot'].NativeLedger,original)


if __name__=='__main__':unittest.main()
