#!/usr/bin/env python3
"""CPU fakes verify alias restoration, stream routing and pinned API rejection."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import h200_vllm_capture_binding as binding
import shoreline_green


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'audit.json'
        self.stream=types.SimpleNamespace(cuda_stream=77)
        self.cuda=types.SimpleNamespace(value=types.SimpleNamespace(cuda_stream=0))
        self.cuda.set_stream=lambda value:setattr(self.cuda,'value',value)
        self.cuda.current_stream=lambda:self.cuda.value
        self.torch=types.SimpleNamespace(cuda=self.cuda,device=lambda value:types.SimpleNamespace(type=value.split(':')[0],index=int(value.split(':')[1]) if ':' in value else None))
        class Green:
            depth=1
            def set_context(self):self.depth+=1
            def pop_context(self):self.depth-=1
        self.green=Green();self.groups=[]
        @contextmanager
        def group(context):
            self.assertIs(context.stream,self.stream);self.groups.append(context);yield context
        @contextmanager
        def original(device):
            raise AssertionError('ordinary capture factory should be replaced within the adapter')
            yield
        self.original=original
        self.ps=types.SimpleNamespace(graph_capture=original,GraphCaptureContext=lambda stream:types.SimpleNamespace(stream=stream),
                                      get_tp_group=lambda:types.SimpleNamespace(graph_capture=group),
                                      get_pp_group=lambda:types.SimpleNamespace(graph_capture=group))
        self.export=types.SimpleNamespace(graph_capture=original);self.runner=types.SimpleNamespace(graph_capture=original)
        self.targets=[('parallel',self.ps),('export',self.export),('runner',self.runner)]
        self.proof={'green_context_handle':7,'green_context_id':9,'actual_sm_count':80}

    def tearDown(self):self.tmp.cleanup()

    def bound(self):
        return binding.bind_api(self.torch,self.green,self.stream,self.ps,self.targets,self.path)

    def test_explicit_stream_preserves_both_group_contexts_and_restores_aliases(self):
        with patch.object(shoreline_green,'stream_proof',return_value=self.proof),self.bound() as data:
            with self.runner.graph_capture(device='cuda:0') as context:
                self.assertIs(context.stream,self.stream);self.assertEqual(self.green.depth,2)
            self.assertEqual(self.green.depth,1)
        self.assertEqual(len(self.groups),2);self.assertEqual(len(data['capture_calls']),1)
        self.assertEqual(data['status'],'complete')
        self.assertTrue(all(m.graph_capture is self.original for _,m in self.targets))

    def test_model_exception_restores_aliases_and_context_depth(self):
        with patch.object(shoreline_green,'stream_proof',return_value=self.proof),self.assertRaisesRegex(ValueError,'model failed'):
            with self.bound() as data:
                with self.runner.graph_capture(device='cuda:0'):raise ValueError('model failed')
        self.assertEqual(self.green.depth,1);self.assertEqual(data['status'],'failed')
        self.assertTrue(all(m.graph_capture is self.original for _,m in self.targets))

    def test_unreviewed_alias_rejected_without_partial_patch(self):
        self.runner.graph_capture=lambda device:None
        with patch.object(shoreline_green,'stream_proof',return_value=self.proof),self.assertRaises(AssertionError):
            with self.bound():pass
        self.assertIs(self.ps.graph_capture,self.original);self.assertIs(self.export.graph_capture,self.original)
        self.assertFalse(self.path.exists())

    def test_unsupported_runtime_rejected_before_installation(self):
        with patch.object(binding.importlib,'import_module',return_value=types.SimpleNamespace(__version__='different')),self.assertRaises(AssertionError):
            with binding.install(self.torch,self.green,self.stream,self.path):pass
        self.assertFalse(self.path.exists())

    def test_other_device_index_rejected_before_context_push(self):
        with patch.object(shoreline_green,'stream_proof',return_value=self.proof),self.assertRaises(AssertionError):
            with self.bound():
                with self.runner.graph_capture(device='cuda:1'):pass
        self.assertEqual(self.green.depth,1)
        self.assertTrue(all(m.graph_capture is self.original for _,m in self.targets))

    def test_public_adapter_seeds_current_stream_tracking(self):
        modules={'vllm':types.SimpleNamespace(__version__='0.22.1'),'vllm.distributed.parallel_state':self.ps,
                 'vllm.distributed':self.export,'vllm.v1.worker.gpu_model_runner':self.runner,
                 'vllm.utils.torch_utils':types.SimpleNamespace(current_stream=self.cuda.current_stream)}
        with patch.object(binding.importlib,'import_module',side_effect=modules.__getitem__), \
             patch.object(shoreline_green,'stream_proof',return_value=self.proof), \
             binding.install(self.torch,self.green,self.stream,self.path) as data:
            self.assertIs(self.cuda.current_stream(),self.stream)
            with self.runner.graph_capture(device='cuda:0'):pass
        self.assertEqual(data['vllm_version'],'0.22.1')
        self.assertIs(self.runner.graph_capture,self.original)


if __name__=='__main__':unittest.main()
