"""Explicit green-stream resources; requested SM counts are never allocation proof."""
import math


def checked(result):
    assert int(result[0]) == 0, str(result[0])
    return result[1] if len(result) == 2 else result[1:]


def stream_proof(stream):
    from cuda.bindings import driver
    handle = driver.CUstream(stream.cuda_stream)
    resource = checked(driver.cuStreamGetDevResource(handle, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM))
    green = checked(driver.cuStreamGetGreenCtx(handle))
    green_id = checked(driver.cuGreenCtxGetId(green)) if int(green) else None
    return {'stream_handle':stream.cuda_stream,'actual_sm_count':resource.sm.smCount,
            'green_context_handle':int(green),'green_context_id':green_id,
            'proof_source':'CUDA driver cuStreamGetDevResource and cuStreamGetGreenCtx'}


def create(torch, fraction):
    assert 0 < fraction <= 1
    if fraction == 1:
        stream = torch.cuda.current_stream()
        return None, stream, stream_proof(stream)
    from cuda.core import Device
    device = Device(0)
    device.set_current()
    resources = device.resources.sm
    alignment = resources.coscheduled_alignment
    minimum = resources.min_partition_size
    requested = max(minimum, math.ceil(resources.sm_count*fraction/alignment)*alignment)
    green = torch.cuda.GreenContext.create(requested, 0)
    stream = green.Stream()
    proof = stream_proof(stream)
    proof.update(requested_fraction=fraction,requested_sms=requested,
                 device_sm_count=resources.sm_count,min_partition_size=minimum,alignment=alignment)
    assert proof['green_context_handle'] and proof['actual_sm_count'] >= requested
    return green, stream, proof
