#!/usr/bin/env python3
"""Native fixed-split probes validate the pinned FA3 metadata decoder."""
import hashlib
from pathlib import Path
import subprocess
from h200_session import DEFAULT_ROOT,UUID,WORKSPACE,dump
from fa3_metadata import decode,SOURCE_PIN


def main():
    import pynvml as nvml
    from gpu_target import select,verify_cuda
    nvml.nvmlInit();select(nvml,UUID)
    import torch
    verify_cuda(torch,UUID)
    from vllm.vllm_flash_attn.flash_attn_interface import get_scheduler_metadata
    from vllm.vllm_flash_attn import _vllm_fa3_C
    source=WORKSPACE/'flash-attention-source'
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==SOURCE_PIN
    symbols=subprocess.check_output(['nm','-D',_vllm_fa3_C.__file__],text=True)
    aten=[line for line in symbols.splitlines() if 'mha_fwd_get_scheduler_metadata' in line]
    assert any('N3c1010ScalarTypeERKN2at6Tensor' in line for line in aten),'regular ATen ABI not identified'
    probes=[]
    for batch,length in [(1,1025),(8,4097)]:
        for splits in [1,2,0]:
            lengths=torch.full((batch,),length,dtype=torch.int32,device='cuda')
            offsets=torch.arange(batch+1,dtype=torch.int32,device='cuda')
            metadata=get_scheduler_metadata(batch,1,length,12,2,128,lengths,
                       qkv_dtype=torch.bfloat16,cu_seqlens_q=offsets,page_size=16,
                       causal=True,num_splits=splits,pack_gqa=True)
            raw=metadata.cpu().tolist()
            decoded=decode(raw,batch,True)
            assert decoded['packed_query_lengths']==[6]*batch
            if splits:assert all(1<=x<=splits for x in decoded['per_request_dynamic_splits'])
            if splits==1:assert decoded['per_request_dynamic_splits']==[1]*batch
            probes.append({'batch':batch,'kv_length':length,'requested_split_bound':splits,
                           'raw':raw,'decoded':decoded})
    files=['hopper/flash_api.cpp','hopper/flash_prepare_scheduler.cu','hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp']
    proof={'status':'passed','classification':'measured-real-GPU','gpu_uuid':UUID,
           'scope':'native metadata ABI/control probes only; no decode timing/energy or simulator calibration',
           'source_pin':SOURCE_PIN,'aten_abi_symbols':aten,
           'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files},
           'module_path':_vllm_fa3_C.__file__,
           'module_sha256':hashlib.sha256(Path(_vllm_fa3_C.__file__).read_bytes()).hexdigest(),
           'probes':probes}
    dump(DEFAULT_ROOT/'environment/fa3_scheduler_schema.json',proof)
    print('FA3 metadata fixed-split probes passed',len(probes),flush=True)
    nvml.nvmlShutdown()


if __name__=='__main__':main()
