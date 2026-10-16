#!/usr/bin/env python3
"""Independent native service controls with explicit useful-work accounting."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback


def main(a):
    import pynvml as nvml
    nvml.nvmlInit()
    from gpu_target import select,verify_cuda,describe
    handle,_=select(nvml,a.gpu_uuid)
    import torch
    import triton
    import numpy as np
    from shoreline_kernels import transfer,dependent,arithmetic
    from shoreline_green import create,stream_proof
    from software_decode_experiments import dump
    verify_cuda(torch,a.gpu_uuid)
    green,stream,proof=create(torch,a.fraction)
    data={'status':'running','classification':'measured-real-GPU','acquisition_id':a.acquisition,
          'hardware':describe(nvml,handle,torch),'allocation':proof,'cases':[],
          'versions':{'torch':torch.__version__,'triton':triton.__version__},
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__),Path(__file__).with_name('shoreline_kernels.py'),Path(__file__).with_name('shoreline_green.py')]},
          'policy':{'warmups':3,'trials':10,'seed':20261013,'memory_load_policy':'transfer .cg; dependent .ca volatile',
                    'physical_HBM_bytes':None,'missing_reason':'requires separate scope-matched hardware-counter acquisition'}}
    dump(a.output/'run.json',data)

    def measure(identity,launch,verify,metadata,compiled=None):
        if a.profile_case and identity != a.profile_case:
            return
        row={'id':identity,'metadata':metadata,'trials':[],'physical_read_bytes':None,'physical_write_bytes':None,
             'scope':'independent native pattern; useful payload is not physical HBM traffic'}
        data['cases'].append(row)
        with torch.cuda.stream(stream):
            for _ in range(3):launch()
            stream.synchronize()
            # Fixed repetitions within each condition; record actual count.
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();launch();end.record();stream.synchronize()
            one_ms=begin.elapsed_time(end)
            repetitions=max(1,min(1000,int(5/max(one_ms,0.001))+1))
            if a.profile_case:repetitions=1
            row['repetitions']=repetitions
            for trial in range(10):
                before=describe(nvml,handle,torch)
                if a.profile_case and trial==0:torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push(identity)
                begin.record();wall0=time.time_ns()
                for _ in range(repetitions):launch()
                end.record();stream.synchronize();wall1=time.time_ns()
                torch.cuda.nvtx.range_pop()
                if a.profile_case and trial==0:torch.cuda.profiler.stop()
                row['trials'].append({'trial':trial,'gpu_event_ms':begin.elapsed_time(end),
                    'wall_start_unix_ns':wall0,'wall_end_unix_ns':wall1,'hardware_before':before,
                    'hardware_after':describe(nvml,handle,torch),'repetitions':repetitions})
                dump(a.output/'run.json',data)
            verify()
            row['correctness']='passed'
            row['stream_proof']=stream_proof(stream)
            if compiled is not None:
                ids=metadata.pop('_smid_tensor',None)
                if ids is not None:
                    row['observed_smids']=sorted(set(ids.cpu().tolist()))
                row['compiled_metadata']={k:str(getattr(compiled.metadata,k,None)) for k in
                                           ('num_warps','num_ctas','shared','n_regs','n_spills')}
                for kind in ('ptx','sass'):
                    if kind in compiled.asm:
                        p=a.output/(identity+'.'+kind);p.write_text(compiled.asm[kind])
                        row[kind]={'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            dump(a.output/'run.json',data)

    with torch.cuda.stream(stream):
        footprints=[96] if a.pilot else [1,48,60,96,512]
        patterns=[('stream',0),('stride17',1),('gather',2),('paged128',3)]
        for mib in footprints:
            n=mib*1024**2//4
            x=(torch.arange(n,device='cuda',dtype=torch.int32)%65521).to(torch.float32)
            y=torch.empty_like(x);smids=torch.empty((triton.cdiv(n,4096),),device='cuda',dtype=torch.int32)
            for name,pattern in patterns[:1] if a.pilot else patterns:
                identity=f'{name}_{mib}mib'
                if a.profile_case and identity!=a.profile_case:continue
                compiled=transfer[(triton.cdiv(n,4096),)](x,y,smids,n,pattern,4096)
                def launch(x=x,y=y,smids=smids,n=n,pattern=pattern):
                    return transfer[(triton.cdiv(n,4096),)](x,y,smids,n,pattern,4096)
                def verify(x=x,y=y,n=n,pattern=pattern):
                    i=torch.arange(n,device='cuda',dtype=torch.int64)
                    index=i if pattern==0 else i*17%n if pattern==1 else (i*1103515245+12345)%n if pattern==2 else ((i//128*17)%(n//128))*128+i%128
                    assert torch.equal(y,x[index]),'transfer reference mismatch'
                metadata={'footprint_bytes':n*4,'pattern':name,'useful_read_bytes_per_launch':n*4,
                          'useful_write_bytes_per_launch':n*4,'additional_smid_writes':smids.numel()*4,
                          'blocks':smids.numel(),'threads_per_block':128,'block_elements':4096}
                measure(identity,launch,verify,metadata,compiled)
                if data['cases'] and data['cases'][-1]['id']==identity:
                    data['cases'][-1]['observed_smids']=sorted(set(smids.cpu().tolist()))
            del x,y,smids

        for mib in ([1] if a.pilot else [0.03125,48,96]):
            n=int(mib*1024**2)//8
            generator=torch.Generator().manual_seed(20261013)
            order=torch.randperm(n,generator=generator)
            links=torch.empty(n,dtype=torch.int64);links[order]=order.roll(-1)
            next_gpu=links.cuda()
            for chains in ([1] if a.pilot else [1,4,32]):
                identity=f'pointer_{mib:g}mib_chains{chains}'
                if a.profile_case and identity!=a.profile_case:continue
                out=torch.empty(chains,device='cuda',dtype=torch.int64)
                cycles=torch.empty_like(out);smids=torch.empty(1,device='cuda',dtype=torch.int32)
                steps=1024
                compiled=dependent[(1,)](next_gpu,out,cycles,smids,n,chains,steps,num_warps=1)
                def launch():return dependent[(1,)](next_gpu,out,cycles,smids,n,chains,steps,num_warps=1)
                expected=np.arange(chains,dtype=np.int64)*15485863%n
                reference=links.numpy()
                for _ in range(steps):expected=reference[expected]
                def verify():assert np.array_equal(out.cpu().numpy(),expected),'pointer chain mismatch'
                metadata={'footprint_bytes':n*8,'dependent_steps':steps,'independent_chains':chains,
                          'useful_read_bytes_per_launch':chains*steps*8,'cache_level':'conditional; footprint is not residency proof'}
                measure(identity,launch,verify,metadata,compiled)
                if data['cases'] and data['cases'][-1]['id']==identity:
                    data['cases'][-1].update(dependent_clock_cycles=cycles.cpu().tolist(),observed_smids=smids.cpu().tolist())
            del links,order,next_gpu

        for integer in ([False] if a.pilot else [False,True]):
            for blocks in ([132] if a.pilot else [32,132,528]):
                n=blocks*256;steps=1024
                identity=f'{"integer" if integer else "fma"}_blocks{blocks}'
                if a.profile_case and identity!=a.profile_case:continue
                x=torch.full((n,),1,device='cuda',dtype=torch.int32 if integer else torch.float32)
                y=torch.empty_like(x);smids=torch.empty(blocks,device='cuda',dtype=torch.int32)
                compiled=arithmetic[(blocks,)](x,y,smids,n,steps,integer,256)
                def launch():return arithmetic[(blocks,)](x,y,smids,n,steps,integer,256)
                expected=np.uint32(1) if integer else np.float32(1)
                for _ in range(steps):
                    expected=np.uint32((int(expected)*1664525+1013904223)&0xffffffff) if integer else np.float32(float(expected)*float(np.float32(1.0001))+float(np.float32(0.0001)))
                def verify():
                    if integer:assert np.all(y.cpu().numpy().view(np.uint32)==expected)
                    else:assert np.allclose(y.cpu().numpy(),expected,rtol=2e-6,atol=0)
                measure(identity,launch,verify,{'elements':n,'dependent_iterations':steps,
                     'useful_operations_per_launch':2*n*steps,'operation':'uint32 LCG' if integer else 'scalar FP32 FMA',
                     'reference_policy':'exact uint32; FP32 FMA reference rounded once per iteration, rtol 2e-6'},compiled)
                if data['cases'] and data['cases'][-1]['id']==identity:data['cases'][-1]['observed_smids']=sorted(set(smids.cpu().tolist()))

        if not a.pilot:
            for n in (1024,4096):
                identity=f'bf16_gemm_{n}'
                if a.profile_case and identity!=a.profile_case:continue
                x=torch.ones((n,n),device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x)
                def launch():return torch.mm(x,x,out=y)
                def verify():assert torch.equal(y,torch.full_like(y,n)),'exact constant GEMM reference mismatch'
                measure(identity,launch,verify,{'matrix_shape':[n,n,n],'dtype':'bfloat16',
                          'useful_flops_per_launch':2*n**3,'input':'ones; library receives runtime tensors',
                          'instruction_route':'library kernel; requires separate counter/SASS attribution'})
    data['status']='complete';data['hardware_after']=describe(nvml,handle,torch)
    dump(a.output/'run.json',data)
    print(json.dumps({'status':'complete','cases':len(data['cases']),'allocation':proof}))
    nvml.nvmlShutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--acquisition',type=int,required=True);p.add_argument('--fraction',type=float,default=1)
    p.add_argument('--pilot',action='store_true');p.add_argument('--profile-case')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:main(a)
    except Exception as error:
        from software_decode_experiments import dump
        if (a.output/'run.json').exists():
            data=json.loads((a.output/'run.json').read_text());data['status']='failed';dump(a.output/'run.json',data)
        dump(a.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
