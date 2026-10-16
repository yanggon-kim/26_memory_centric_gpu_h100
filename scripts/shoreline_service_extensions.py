#!/usr/bin/env python3
"""Read-only service and six-condition standalone/concurrent calibration family."""
import argparse
import hashlib
import json
from pathlib import Path
import time


import triton
import triton.language as tl


@triton.jit
def read_only(X,Y,N:tl.constexpr,PATTERN:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    source=i if PATTERN==0 else ((i.to(tl.int64)*17)%N).to(tl.int32)
    value=tl.load(X+source,i<N,other=0,cache_modifier='.cg')
    total=tl.sum(value,axis=0)
    tl.store(Y+tl.program_id(0),total)


def main(a):
    import pynvml as nvml
    nvml.nvmlInit()
    from gpu_target import select,verify_cuda,describe
    handle,_=select(nvml,a.gpu_uuid)
    import torch
    import triton
    import triton.language as tl
    from shoreline_kernels import transfer,arithmetic
    from software_decode_experiments import dump

    verify_cuda(torch,a.gpu_uuid)
    from shoreline_green import create
    green,stream,proof=create(torch,a.fraction)
    data={'status':'running','classification':'measured-real-GPU','acquisition_id':a.acquisition,
          'allocation':proof,'cases':[],'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (Path(__file__),Path(__file__).with_name('shoreline_kernels.py'),Path(__file__).with_name('shoreline_green.py'))},
          'physical_bytes':None,'physical_bytes_reason':'separate actual HBM counters required'}
    def measured(name,launch,verify,meta):
        if a.profile_case and name!=a.profile_case:return
        row={'id':name,'metadata':meta,'trials':[]};data['cases'].append(row)
        with torch.cuda.stream(stream):
            for _ in range(3):launch()
            stream.synchronize()
            for trial in range(10):
                before=describe(nvml,handle,torch)
                if a.profile_case and trial==0:torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push(name)
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();start=time.time_ns();launch();end.record();stream.synchronize();finish=time.time_ns()
                torch.cuda.nvtx.range_pop()
                if a.profile_case and trial==0:torch.cuda.profiler.stop()
                row['trials'].append({'gpu_event_ms':begin.elapsed_time(end),'wall_start_unix_ns':start,
                    'wall_end_unix_ns':finish,'hardware_before':before,'hardware_after':describe(nvml,handle,torch)})
                dump(a.output/'run.json',data)
            verify();row['correctness']='passed';dump(a.output/'run.json',data)

    if a.timeline:torch.cuda.profiler.start()
    with torch.cuda.stream(stream):
        for mib in ([1,512] if a.pilot else [1,48,60,96,512]):
            n=mib*1024**2//4
            x=torch.arange(n,device='cuda',dtype=torch.int32).remainder_(65521).float()
            out=torch.empty(triton.cdiv(n,4096),device='cuda')
            for pattern,name in [(0,'read_only'),(1,'read_stride17')]:
                identity=f'{name}_{mib}mib'
                def launch():return read_only[(triton.cdiv(n,4096),)](x,out,n,pattern,4096)
                def verify():
                    i=torch.arange(n,device='cuda',dtype=torch.int64)
                    source=i if pattern==0 else i*17%n
                    expected=x[source].double().reshape(-1,4096).sum(dim=1).float()
                    assert torch.allclose(out,expected,rtol=1e-6,atol=0),'read-only checksum mismatch'
                measured(identity,launch,verify,{'input_footprint_bytes':n*4,'checksum_output_bytes':out.numel()*4,
                    'useful_read_bytes':n*4,'reference':'full FP64 checksum rounded to FP32, rtol 1e-6',
                    'access':'contiguous scalar .cg loads' if pattern==0 else 'permuted stride17 .cg loads'})
        if not a.pilot and a.fraction==1:
            n=96*1024**2//4;blocks=132;cn=blocks*256
            mx=torch.ones(n,device='cuda');my=torch.empty_like(mx)
            cx=torch.ones(cn,device='cuda');cy=torch.empty_like(cx)
            mids=torch.empty(triton.cdiv(n,4096),device='cuda',dtype=torch.int32)
            cids=torch.empty(blocks,device='cuda',dtype=torch.int32)
            ms,cs=torch.cuda.Stream(),torch.cuda.Stream()
            gate,md,cd=torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Event()
            for ratio in (1,2,4,8,16,32):
                for mode in ('empty','memory','compute','concurrent'):
                    def launch(ratio=ratio,mode=mode):
                        gate.record(stream);ms.wait_event(gate);cs.wait_event(gate)
                        if mode in ('memory','concurrent'):
                            with torch.cuda.stream(ms):
                                for _ in range(4):transfer[(triton.cdiv(n,4096),)](mx,my,mids,n,0,4096)
                        if mode in ('compute','concurrent'):
                            with torch.cuda.stream(cs):
                                for _ in range(ratio):arithmetic[(blocks,)](cx,cy,cids,cn,1024,False,256)
                        md.record(ms);cd.record(cs);stream.wait_event(md);stream.wait_event(cd)
                    def verify(mode=mode):
                        if mode in ('memory','concurrent'):assert torch.equal(mx,my)
                        if mode in ('compute','concurrent'):
                            import numpy as np
                            expected=np.float32(1)
                            for _ in range(1024):expected=np.float32(float(expected)*float(np.float32(1.0001))+float(np.float32(0.0001)))
                            assert np.allclose(cy.cpu().numpy(),expected,rtol=2e-6,atol=0),'arithmetic reference mismatch'
                    measured(f'overlap_ratio{ratio}_{mode}',launch,verify,{'family':'six-condition common native overlap family',
                        'ratio':ratio,'mode':mode,'memory_launches':4,'compute_launches':ratio,
                        'private_allocation':'identical in empty/standalone/concurrent treatments',
                        'proof':'actual same-pass NSYS overlap required; stream submission alone is not overlap proof'})
    if a.timeline:torch.cuda.profiler.stop()
    data['status']='complete';dump(a.output/'run.json',data)
    print(json.dumps({'status':'complete','cases':len(data['cases']),'allocation':proof}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acquisition',type=int,required=True);p.add_argument('--fraction',type=float,default=1)
    p.add_argument('--pilot',action='store_true');p.add_argument('--profile-case')
    p.add_argument('--timeline',action='store_true',help='separate instrumented same-pass overlap proof')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except BaseException as e:
        from software_decode_experiments import dump
        dump(a.output/'failure.json',{'type':type(e).__name__,'message':str(e)})
        raise
