#!/usr/bin/env python3
"""Independent synthetic service references or full-decode application replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main(a):
    # Every app-range replay gets an immutable native publication directory.
    a.output=a.output/f'replay_{os.getpid()}'
    a.output.mkdir(parents=True,exist_ok=False)
    if a.mode=='decode':
        import controlled_vllm_experiments as baseline
        a.mode='timing';a.collect_worker_metadata=False;a.generated_tokens=64
        a.batch=None;a.context=None
        baseline.main(a);return
    import torch
    import triton
    import pynvml as nv
    from gpu_target import select,verify_cuda,describe
    from software_decode_experiments import dump
    from decoder_calibration_model import roles
    from decode_precondition import create
    nv.nvmlInit();handle,_=select(nv,a.gpu_uuid);verify_cuda(torch,a.gpu_uuid)
    headers=json.loads(a.headers.read_text());config=headers['model_config']
    torch.manual_seed(20261013)
    settle=create(torch,{'protocol':'memory_settle_v3','seconds':1.0,'buffer_bytes':256*1024**2,'scope':'conditioned_decode_only'})
    report={'status':'running','classification':'independent synthetic service reference; not a decoder replay',
        'batch':a.batch,'context':a.context,'mode':a.mode,'cases':[],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'useful BF16 tiled operator references and separate physical memory service; no issued-work, full-model weight-set or physical compute-only equivalence claim'}
    def measure(name,launch,verify,metadata):
        conditioning=settle()
        capture_stream=torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):launch()
        capture_stream.synchronize()
        repetitions=128
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=capture_stream):
            for _ in range(repetitions):launch()
        row={'id':name,'metadata':metadata,'trials':[],'repetitions':repetitions,
            'setup_conditioning':conditioning,'warmups':3}
        report['cases'].append(row)
        for trial in range(10):
            before=describe(nv,handle,torch)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            selected=a.external_profiler and trial==0
            if selected:torch.cuda.profiler.start()
            if selected:torch.cuda.nvtx.range_push('calibration_memory_selected')
            t0=time.time_ns();begin.record()
            if a.external_profiler:launch()
            else:graph.replay()
            end.record();torch.cuda.synchronize();t1=time.time_ns()
            after=describe(nv,handle,torch)
            if selected:torch.cuda.nvtx.range_pop()
            event_ms=begin.elapsed_time(end)/(1 if a.external_profiler else repetitions)
            row['trials'].append({'gpu_event_ms':event_ms,'wall_start_unix_ns':t0,'wall_end_unix_ns':t1,
                'hardware_before':before,'hardware_after':after})
            verify()
            if selected:
                dump(a.output/'counter_publication.json',{'status':'complete','case':name,
                    'correctness':'passed','metadata':metadata,**row['trials'][-1]})
                torch.cuda.profiler.stop()
            dump(a.output/'run.json',report)
        row['correctness']='passed';del graph
        dump(a.output/'run.json',report)
    if a.mode=='compute':
        from shoreline_accounting import useful
        expected=useful(a.headers,a.batch,a.context)['ticks'][0]['useful_linear_flops']*63
        actual=sum(2*m*k*n*mult for role,m,k,n,mult in roles(config,a.batch,a.context) if not role.startswith('attention_'))
        assert actual==expected,'reference work disagrees with immutable checkpoint geometry'
        for role,m,k,n,multiplicity in roles(config,a.batch,a.context):
            for tile in (512,1024):
                # Fit the synthetic reference operands in L2. K and N tiles are
                # explicit reference shapes, not assumed vLLM issued tiles.
                kk=min(k,tile) if role=='attention_av' else k
                nn=min(n,tile)
                x=torch.ones((m,kk),device='cuda',dtype=torch.bfloat16)
                w=torch.ones((kk,nn),device='cuda',dtype=torch.bfloat16)
                y=torch.empty((m,nn),device='cuda',dtype=torch.bfloat16)
                def launch():torch.mm(x,w,out=y)
                def verify():assert torch.equal(y,torch.full_like(y,kk)),'BF16 reference mismatch'
                measure(f'{role}_tile{tile}',launch,verify,{'role':role,'tile':tile,'shape':[m,kk,nn],
                    'useful_flops_per_launch':2*m*kk*nn,'full_scope_useful_flops':2*m*k*n*multiplicity,
                    'reference_route':'torch.mm hot synthetic operands, repeated CUDA graph; actual decoder issued work/route equivalence unproven'})
                del x,w,y
    else:
        from decoder_calibration_kernels import read_reference
        tensors=headers['safetensors_headers']
        footprint=sum(t['bytes'] for t in tensors) # conservative full checkpoint footprint, including embedding
        n=triton.next_power_of_2((footprint+1)//2)
        x=torch.ones(n,device='cuda',dtype=torch.bfloat16);y=torch.empty(triton.cdiv(n,4096),device='cuda')
        def launch():return read_reference[(triton.cdiv(n,4096),)](x,y,n,a.pattern=='paged128',4096)
        def verify():assert torch.equal(y,torch.full_like(y,4096)),'read-reference checksum mismatch'
        measure(a.pattern,launch,verify,{'pattern':a.pattern,'input_bytes':n*2,'output_bytes':y.numel()*4,
            'physical_bytes':None,'reason':'requires separate NCU counter capture; payload is not physical traffic'})
    report['status']='complete';dump(a.output/'run.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['decode','compute','memory'],required=True)
    p.add_argument('--config');p.add_argument('--headers',type=Path)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--external-profiler',action='store_true')
    p.add_argument('--batch',type=int);p.add_argument('--context',type=int)
    p.add_argument('--pattern',choices=['stream','paged128'],default='stream')
    main(p.parse_args())
