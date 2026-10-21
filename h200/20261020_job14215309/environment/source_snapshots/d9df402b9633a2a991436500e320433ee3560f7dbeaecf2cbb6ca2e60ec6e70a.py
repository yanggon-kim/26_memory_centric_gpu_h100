"""Explicit cache-disturbing post-prefill memory workload, outside decode timing."""
import time


def validate(config):
    assert config['protocol']=='memory_settle_v3'
    assert config['seconds']==1.0 and config['buffer_bytes']==256*1024**2
    assert config['scope']=='conditioned_decode_only'


def create(torch,config):
    validate(config)
    import triton
    from shoreline_kernels import transfer
    n=config['buffer_bytes']//4
    source=torch.ones(n,device='cuda',dtype=torch.float32)
    target=torch.empty_like(source)
    blocks=triton.cdiv(n,4096)
    smids=torch.empty(blocks,device='cuda',dtype=torch.int32)
    def launch():transfer[(blocks,)](source,target,smids,n,0,4096)
    # Allocation/initialization and first-copy dispatch occur before warmups.
    launch();torch.cuda.synchronize()
    def condition():
        start=time.perf_counter();copies=0
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        while time.perf_counter()-start<config['seconds']:
            for _ in range(32):launch()
            torch.cuda.current_stream().synchronize();copies+=32
            assert copies<=32768,'bounded precondition exceeded copy budget'
        end.record();torch.cuda.current_stream().synchronize()
        return {'protocol':config['protocol'],'copies':copies,'buffer_bytes':config['buffer_bytes'],
            'gpu_event_ms':begin.elapsed_time(end),'wall_ms':1000*(time.perf_counter()-start),
            'scope':'one fixed-duration private-buffer read/write workload after the verified prefill barrier; outside decode timing/energy; changes cache history and TTFT; no clock/power settings changed'}
    return condition
