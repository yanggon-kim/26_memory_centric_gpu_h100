"""Explicit green-stream resources; requested SM counts are never allocation proof."""
import math


_PROCESS_CONTEXTS=[]


def retain_for_process(green,stream):
    """Keep green resources alive through allocator/graph interpreter teardown.

    CUDA forbids destroying a green context before its streams. Framework
    allocator/graph caches outlive this function and their Python destruction
    order is unspecified. These bounded experiment children deliberately retain
    the two handles until process exit; the driver reclaims their resources.
    """
    import ctypes
    retain=ctypes.pythonapi.Py_IncRef
    retain.argtypes=[ctypes.py_object];retain.restype=None
    retain(green);retain(stream)
    _PROCESS_CONTEXTS.append((green,stream))


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


def create(torch, fraction, make_current=False):
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
    if make_current:green.set_context()
    stream = green.Stream()
    retain_for_process(green,stream)
    proof = stream_proof(stream)
    proof.update(requested_fraction=fraction,requested_sms=requested,
                 device_sm_count=resources.sm_count,min_partition_size=minimum,alignment=alignment,
                 converted_context_made_current=make_current,
                 lifetime_policy='process-lifetime retained; framework cached streams/graphs may outlive local scope; driver reclaims on child exit')
    assert proof['green_context_handle'] and proof['actual_sm_count'] >= requested
    return green, stream, proof
