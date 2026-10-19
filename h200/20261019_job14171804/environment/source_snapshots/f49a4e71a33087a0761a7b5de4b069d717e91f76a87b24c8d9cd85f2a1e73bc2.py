#!/usr/bin/env python3
"""Bounded raw sensor response diagnostic; never rescales or calibrates a sensor."""
import argparse
from pathlib import Path
import time


def main(a):
    import pynvml as nv
    nv.nvmlInit()
    from gpu_target import select,verify_cuda,describe
    handle,_=select(nv,a.gpu_uuid)
    import torch
    from software_decode_experiments import Sensor,dump
    verify_cuda(torch,a.gpu_uuid)
    output={'status':'running','classification':'measured-real-GPU-functional-diagnostic',
            'external_absolute_calibration':False,'rescaling':False,'windows':[]}
    x=torch.ones((2048,2048),device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x)
    for _ in range(3):torch.mm(x,x,out=y)
    torch.cuda.synchronize()
    for treatment,duration in [('idle',2),('owned_bf16_gemm',6),('idle_after',2)]:
        sensor=Sensor(nv,handle,.02);sensor.start();before=sensor.read();start=time.perf_counter();launches=0
        while time.perf_counter()-start<duration:
            if treatment=='owned_bf16_gemm':
                for _ in range(16):torch.mm(x,x,out=y);launches+=1
                torch.cuda.synchronize()
            else:time.sleep(.02)
        end=time.perf_counter();after=sensor.read()
        telemetry=sensor.finish(start,end,before,after,1,5)
        telemetry={k:v for k,v in telemetry.items() if 'per_token' not in k}
        telemetry['scope_note']='native/idle diagnostic has no generated tokens; only raw joules and sensor cadence apply'
        output['windows'].append({'treatment':treatment,'launches':launches,'telemetry':telemetry,
                                  'hardware_after':describe(nv,handle,torch)})
        dump(a.output,output)
    assert torch.equal(y,torch.full_like(y,2048)),'diagnostic numerical reference mismatch'
    output.update(status='complete',interpretation='response/cadence check only; disagreement leaves primary J/token NA and does not authorize an energy-saving claim')
    dump(a.output,output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
