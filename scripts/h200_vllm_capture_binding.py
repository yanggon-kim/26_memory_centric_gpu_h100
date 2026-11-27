"""Bounded vLLM0.22.1 repair: seed stream tracking and bind graph capture.

This is an in-process experiment adapter. Installed package files are untouched;
real decoder NSYS mapping and exact outputs remain mandatory proof gates.
"""
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
from pathlib import Path


def _write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


@contextmanager
def bind_api(torch,green,stream,parallel_state,targets,audit_path):
    from shoreline_green import stream_proof
    assert green is not None
    proof=stream_proof(stream)
    assert proof['green_context_handle'] and proof['actual_sm_count']>0
    original=parallel_state.graph_capture
    assert targets and all(module.graph_capture is original for _,module in targets), 'unexpected graph-capture alias/API'
    data={'status':'running','route':'capture_bind_v2','stream_proof':proof,
          'bound_aliases':[name for name,_ in targets],'capture_calls':[],
          'original_graph_capture_sha256':hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
          'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'capture-stream audit only; actual decoder kernels still require independent NSYS greenContextId validation'}
    @contextmanager
    def capture(device):
        selected=torch.device(device)
        assert selected.type=='cuda' and selected.index in (None,0), 'only explicitly visible owned GPU supported'
        green.set_context()
        try:
            torch.cuda.set_stream(stream)
            context=parallel_state.GraphCaptureContext(stream)
            # Preserve both group communicator contexts, supplying the explicit
            # green stream instead of constructing an ordinary pooled stream.
            with parallel_state.get_tp_group().graph_capture(context), parallel_state.get_pp_group().graph_capture(context):
                assert torch.cuda.current_stream().cuda_stream==stream.cuda_stream
                data['capture_calls'].append({'stream_handle':stream.cuda_stream,'actual_sm_count':proof['actual_sm_count'],
                                              'green_context_id':proof['green_context_id']})
                yield context
        finally:
            green.pop_context()
    torch.cuda.set_stream(stream)
    _write(audit_path,data)
    for _,module in targets:module.graph_capture=capture
    complete=False
    try:
        yield data
        complete=True
    finally:
        for _,module in targets:module.graph_capture=original
        data.update(status='complete' if complete else 'failed',aliases_restored=True)
        _write(audit_path,data)


@contextmanager
def install(torch,green,stream,audit_path):
    vllm=importlib.import_module('vllm')
    assert vllm.__version__=='0.22.1', 'repair requires reviewed pinned vLLM API'
    ps=importlib.import_module('vllm.distributed.parallel_state')
    distributed=importlib.import_module('vllm.distributed')
    runner=importlib.import_module('vllm.v1.worker.gpu_model_runner')
    utils=importlib.import_module('vllm.utils.torch_utils')
    # vLLM installs a set_stream tracker at import. Seed it after importing the
    # module, before engine construction can create its default dedicated stream.
    torch.cuda.set_stream(stream)
    assert utils.current_stream().cuda_stream==stream.cuda_stream
    targets=[('vllm.distributed.parallel_state.graph_capture',ps),
             ('vllm.distributed.graph_capture',distributed),
             ('vllm.v1.worker.gpu_model_runner.graph_capture',runner)]
    with bind_api(torch,green,stream,ps,targets,audit_path) as data:
        data['vllm_version']=vllm.__version__
        yield data
