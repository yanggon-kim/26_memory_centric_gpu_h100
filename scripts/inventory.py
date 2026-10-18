#!/usr/bin/env python3
"""Read-only host/profiler inventory; optional selected H100 CUDA or MI300X/MI325X ROCm identity check."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def command(args):
    if not shutil.which(args[0]):
        return {'status':'unavailable','reason':'executable not on PATH'}
    try:
        result=subprocess.run(args,text=True,capture_output=True,timeout=20)
        return {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    except Exception as error:
        return {'status':'failed','reason':str(error)}


def main(args):
    data={'classification':'environment-inventory','platform':platform.platform(),'python':platform.python_version(),
          'target_uuid':args.gpu_uuid,'CUDA_HOME':os.environ.get('CUDA_HOME','unset'),
          'commands':{name:command(cmd) for name,cmd in {
           'gpu_list':['nvidia-smi','-L'],'gpu_details':['nvidia-smi','-q'],'nvcc':['nvcc','--version'],
           'ncu':['ncu','--version'],'nsys':['nsys','--version'],'cpu':['lscpu'],'ram':['free','-h'],
           'storage':['df','-h',str(args.output.parent)],'os':['uname','-a'],
           'amd_smi_static':['amd-smi','static'],'amd_smi_metric':['amd-smi','metric'],'amd_smi_process':['amd-smi','process'],
           'amd_smi_version':['amd-smi','version'],'rocminfo':['rocminfo'],'hipcc':['hipcc','--version'],
           'rocprofv3_avail':['rocprofv3-avail','list'],'rocprof_compute':['rocprof-compute','--version']}.items()},'packages':{}}
    data['environment']={k:os.environ.get(k,'unset') for k in ['ROCM_PATH','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES',
                         'SLURM_JOB_ID','SLURM_JOB_PARTITION','SLURM_JOB_NODELIST','LOADEDMODULES']}
    for package in ['torch','vllm','transformers','nvidia-ml-py','amdsmi','triton','huggingface-hub']:
        try:data['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:data['packages'][package]='unavailable'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')
    if args.check_cuda:
        try:
            assert args.gpu_uuid,'--gpu-uuid required for CUDA check'
            from gpu_target import load_nvml; nvml = load_nvml()
            from gpu_target import select,verify_cuda,query_l2
            nvml.nvmlInit()
            handle,index=select(nvml,args.gpu_uuid)
            import torch
            props=verify_cuda(torch,args.gpu_uuid)
            data['selected_cuda']={'name':props.name,'uuid':str(props.uuid),'nvml_index':index,
                                   'capability':list(torch.cuda.get_device_capability()),'total_memory_bytes':props.total_memory,
                                   'cuda_runtime':torch.version.cuda,'hip_runtime':getattr(torch.version,'hip',None),
                                   'gcn_arch':getattr(props,'gcnArchName',None),'sm_count':props.multi_processor_count}
            data['selected_cuda'].update(query_l2(torch))
            nvml.nvmlShutdown()
        except Exception as error:
            data['cuda_check_error']={'type':type(error).__name__,'message':str(error)}
            args.output.write_text(json.dumps(data,indent=2)+'\n')
            raise
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu-uuid')
    p.add_argument('--check-cuda',action='store_true')
    main(p.parse_args())
