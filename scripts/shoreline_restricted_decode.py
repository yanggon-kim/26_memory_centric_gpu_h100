#!/usr/bin/env python3
"""Green-stream decoder pilot; NSYS kernel evidence must prove actual restriction."""
import argparse
import json
import os
from pathlib import Path
import traceback


def main(a):
    # UUID visibility is explicit before CUDA initializes; the baseline repeats NVML/CUDA identity checks.
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      VLLM_ENABLE_V1_MULTIPROCESSING='0')
    import torch
    from gpu_target import verify_cuda
    from shoreline_green import create,stream_proof
    from software_decode_experiments import dump
    verify_cuda(torch,a.gpu_uuid)
    green,stream,proof=create(torch,a.fraction,make_current=True)
    assert a.fraction==1 or proof['green_context_handle']
    dump(a.output/'green_allocation.json',{'status':'allocated','allocation':proof,
          'all_kernel_restriction_proof':'pending Nsight Systems greenContextId validation during selected decode ticks'})
    import controlled_vllm_experiments as baseline
    original=baseline.run_trial
    def trial(*args,**kwargs):
        before=stream_proof(torch.cuda.current_stream())
        result=original(*args,**kwargs)
        after=stream_proof(torch.cuda.current_stream())
        assert before['actual_sm_count']==after['actual_sm_count']==proof['actual_sm_count']
        result['green_stream_before']=before;result['green_stream_after']=after
        return result
    baseline.run_trial=trial
    with torch.cuda.stream(stream):
        if a.external_profiler:
            import shoreline_decode
            shoreline_decode.main(a)
        else:baseline.main(a)
    dump(a.output/'green_allocation.json',{'status':'complete','allocation':proof,
          'all_kernel_restriction_proof':'requires matching NSYS kernel greenContextId; stream endpoints alone are insufficient'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--fraction',type=float,required=True)
    p.add_argument('--mode',default='timing');p.add_argument('--external-profiler',action='store_true')
    p.add_argument('--collect-worker-metadata',action='store_true');p.add_argument('--batch',type=int)
    p.add_argument('--context',type=int);p.add_argument('--generated-tokens',type=int,default=64)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except Exception as error:
        from software_decode_experiments import dump
        dump(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
