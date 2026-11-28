#!/usr/bin/env python3
"""Dynamic-input NVBit H100 micro capture before model-scale tracing."""
import argparse
from ctypes import CDLL, c_char_p
import hashlib
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        import pynvml as nvml
        from gpu_target import select, verify_cuda
        nvml.nvmlInit()
        select(nvml, args.gpu_uuid)
        import torch
        props = verify_cuda(torch, args.gpu_uuid)
        tracer_path = os.environ['CUDA_INJECTION64_PATH']
        assert Path(tracer_path).is_file()
        tracer = CDLL(tracer_path)
        tracer.enable_nvbit_instrumentation.restype = None
        tracer.disable_nvbit_instrumentation.restype = None
        tracer.set_nvbit_instrumentation_tag.argtypes = [c_char_p]
        tracer.set_nvbit_instrumentation_tag.restype = None
        torch.manual_seed(20261013)
        a = torch.randn(1024, device='cuda')
        b = torch.randn(1024, device='cuda')
        torch.cuda.synchronize()
        host_expected = a.cpu() + b.cpu()
        tracer.set_nvbit_instrumentation_tag(b'dynamic_add_sm90')
        tracer.enable_nvbit_instrumentation()
        output = a + b
        torch.cuda.synchronize()
        tracer.disable_nvbit_instrumentation()
        assert torch.equal(output.cpu(), host_expected)
        result = {'status': 'complete', 'classification': 'measured-real-GPU-instrumented',
                  'uuid': args.gpu_uuid, 'name': props.name,
                  'capability': list(torch.cuda.get_device_capability()),
                  'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                  'tracer_path': tracer_path, 'elements': 1024,
                  'input_sha256': [hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest() for t in (a,b)],
                  'output_sha256': hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest(),
                  'scope': 'one dynamic-input vector addition after initialization; NVBit instructions require separate nonempty validation'}
        (args.output/'micro_metadata.json').write_text(json.dumps(result, indent=2)+'\n')
        nvml.nvmlShutdown()
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps({
            'status': 'partial_failed', 'type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc()
        }, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
