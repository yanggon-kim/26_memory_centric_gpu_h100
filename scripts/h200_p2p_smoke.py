#!/usr/bin/env python3
"""Two allocated H200s: synthetic known-value peer-copy pilot, not real KV handoff."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import threading
import time
from types import SimpleNamespace


def main(a):
    allocation=json.loads(a.devices.read_text())
    job=os.environ.get('SLURM_JOB_ID','')
    assert job.isdigit() and allocation['slurm_job_id']==job, 'current allocation required'
    state=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=20)
    assert 'JobState=RUNNING' in state, 'allocation must be running'
    import pynvml as nv
    import torch
    from h200_session import ProcessMonitor, dump
    expected=allocation['devices']
    assert len(expected)==torch.cuda.device_count()==2
    uuids=['GPU-'+str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(2)]
    assert uuids==[d['uuid'] for d in expected]
    assert all('H200' in torch.cuda.get_device_properties(i).name for i in range(2))
    locks=[]
    for uuid in sorted(uuids):
        lock=(Path(tempfile.gettempdir())/('h100_research_'+uuid+'.lock')).open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    nv.nvmlInit()
    handles=[nv.nvmlDeviceGetHandleByUUID(u) for u in uuids]
    guard=object.__new__(ProcessMonitor);guard.process=SimpleNamespace(pid=os.getpid())
    stopped=threading.Event();failures=[]
    monitor_path=a.output.with_suffix('.monitor.jsonl')
    def sample():
        with monitor_path.open('w') as stream:
            while not stopped.is_set():
                for uuid,h in zip(uuids,handles):
                    try:
                        pids=[p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h)]
                        owned={str(p):guard.owned(p) for p in pids}
                        row={'unix_ns':time.time_ns(),'uuid':uuid,'pids':pids,'owned_ancestry':owned,
                             'sm_clock_mhz':nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_SM),
                             'memory_clock_mhz':nv.nvmlDeviceGetClockInfo(h,nv.NVML_CLOCK_MEM)}
                        if any(v is not True for v in owned.values()):failures.append('unverified/foreign GPU process')
                        stream.write(json.dumps(row)+'\n');stream.flush()
                    except Exception as error:failures.append(str(error))
                stopped.wait(.05)
    for h in handles:
        assert all(guard.owned(p.pid) is True for p in nv.nvmlDeviceGetComputeRunningProcesses(h)), 'GPU busy'
    assert torch.cuda.can_device_access_peer(0,1) and torch.cuda.can_device_access_peer(1,0), 'bidirectional peer access unavailable'
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    data={'status':'running','classification':'measured-real-GPU-pilot','devices':uuids,'trials':[],
          'external_instrumentation':a.profile,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'versions':{'torch':torch.__version__,'cuda':torch.version.cuda},
          'payload_bytes':a.mib*1024**2,'warmups':3,'acquisition':a.acquisition,
          'scope':'synthetic BF16 known-value buffer, single directional peer-copy control; not decoder KV, request migration or serving evidence',
          'real_KV_handoff':False,'physical_link_bytes':None,'physical_link_bytes_reason':'requires validated link counters',
          'monitor':str(monitor_path),'peer_access':[True,True]}
    dump(a.output,data)
    try:
        n=a.mib*1024**2//2
        source=(torch.arange(n,device='cuda:0',dtype=torch.int32)%251).to(torch.bfloat16)
        destination=torch.empty(n,device='cuda:1',dtype=torch.bfloat16)
        torch.cuda.synchronize(0);torch.cuda.synchronize(1)
        with torch.cuda.device(1):
            stream=torch.cuda.Stream(device=1)
            with torch.cuda.stream(stream):
                for _ in range(3):destination.copy_(source,non_blocking=True)
                stream.synchronize()
                for trial in range(10):
                    if a.profile and trial==0:torch.cuda.profiler.start()
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.nvtx.range_push('synthetic_p2p_bf16_control')
                    begin.record();start=time.perf_counter()
                    destination.copy_(source,non_blocking=True)
                    end.record();stream.synchronize()
                    data['trials'].append({'trial':trial,'gpu_event_ms':begin.elapsed_time(end),
                                           'wall_ms':(time.perf_counter()-start)*1000})
                    torch.cuda.nvtx.range_pop()
                    if a.profile and trial==0:torch.cuda.profiler.stop()
                    dump(a.output,data)
        actual=destination.cpu()
        expected_cpu=(torch.arange(n,dtype=torch.int32)%251).to(torch.bfloat16)
        assert torch.equal(actual,expected_cpu), 'peer-copy independent known-value reference mismatch'
        assert not failures, failures
        data.update(status='complete',correctness='passed',isolation='sampled ancestry; arbitrarily short interference cannot be excluded')
    except BaseException as error:
        data.update(status='failed',error=str(error))
        raise
    finally:
        stopped.set();thread.join(timeout=5)
        data['monitor_failures']=failures
        dump(a.output,data)
        nv.nvmlShutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mib',type=int,default=128);p.add_argument('--acquisition',type=int,default=0)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    assert not a.output.exists()
    main(a)
