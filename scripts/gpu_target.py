"""Select exactly one declared physical GPU and verify CUDA/NVML identity."""
import fcntl
import os
from pathlib import Path
import tempfile

_LOCKS=[]

def select(nvml,uuid):
    if not uuid.startswith('GPU-'):
        raise ValueError('Physical GPU UUID required; MIG needs a separately validated allocation')
    lock=(Path(tempfile.gettempdir())/('h100_research_'+uuid+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    _LOCKS.append(lock)
    handle=nvml.nvmlDeviceGetHandleByUUID(uuid)
    assert nvml.nvmlDeviceGetUUID(handle)==uuid
    other=[p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid!=os.getpid()]
    assert not other,f'target GPU has other compute processes: {other}'
    def pci(h):
        value=nvml.nvmlDeviceGetPciInfo(h).busId
        return value.decode() if isinstance(value,bytes) else value
    ordered=sorted((pci(nvml.nvmlDeviceGetHandleByIndex(i)),i) for i in range(nvml.nvmlDeviceGetCount()))
    physical=nvml.nvmlDeviceGetIndex(handle)
    ordinal=next(i for i,(_,index) in enumerate(ordered) if index==physical)
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=str(ordinal) # vLLM0.22.1 requires numeric entries.
    return handle,physical


def verify_cuda(torch,uuid,require_h100=True):
    assert torch.cuda.device_count()==1, 'exactly one CUDA-visible GPU required'
    props=torch.cuda.get_device_properties(0)
    actual=str(props.uuid).removeprefix('GPU-')
    assert actual==uuid.removeprefix('GPU-'), 'CUDA/NVML target UUID mismatch; stop rather than use another GPU'
    if require_h100:
        assert 'H100' in props.name and torch.cuda.get_device_capability(0)==(9,0), 'this package requires a verified H100 sm_90 allocation'
    return props


def query_l2(torch):
    """Read capacity in bytes; never substitute a marketing capacity estimate."""
    props=torch.cuda.get_device_properties(0)
    for name in ['L2_cache_size','l2_cache_size']:
        value=getattr(props,name,None)
        if isinstance(value,int) and value>0:
            return {'l2_bytes':value,'l2_source':'queried CUDA device properties '+name,'l2_unavailable_reason':''}
    try:
        import ctypes
        driver=ctypes.CDLL('libcuda.so.1')
        device,value=ctypes.c_int(),ctypes.c_int()
        assert driver.cuInit(0)==0
        assert driver.cuDeviceGet(ctypes.byref(device),0)==0
        # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE=38 in the CUDA driver header, units bytes.
        assert driver.cuDeviceGetAttribute(ctypes.byref(value),38,device)==0 and value.value>0
        return {'l2_bytes':value.value,'l2_source':'queried CUDA driver CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE=38, selected visible device0','l2_unavailable_reason':''}
    except Exception as error:
        return {'l2_bytes':'NA','l2_source':'unavailable','l2_unavailable_reason':type(error).__name__+': '+str(error)}


def describe(nvml,handle,torch):
    props=torch.cuda.get_device_properties(0)
    data={'uuid':nvml.nvmlDeviceGetUUID(handle),'name':props.name,'sm_count':props.multi_processor_count,
          'compute_capability':list(torch.cuda.get_device_capability()),'total_memory_bytes':props.total_memory,
          'driver':nvml.nvmlSystemGetDriverVersion(),**query_l2(torch),'readout_scope':'capture setup; clocks/power limits observed, not modified'}
    for key,call in {'sm_clock_mhz':lambda:nvml.nvmlDeviceGetClockInfo(handle,nvml.NVML_CLOCK_SM),
                     'memory_clock_mhz':lambda:nvml.nvmlDeviceGetClockInfo(handle,nvml.NVML_CLOCK_MEM),
                     'power_limit_w':lambda:nvml.nvmlDeviceGetPowerManagementLimit(handle)/1000}.items():
        try:data[key]=call()
        except nvml.NVMLError as error:data[key]='NA';data[key+'_unavailable_reason']=str(error)
    return data
