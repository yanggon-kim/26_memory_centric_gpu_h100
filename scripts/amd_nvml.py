"""pynvml-compatible subset backed by AMD SMI, for ROCm allocations.

Only the calls used by this package are provided. Units follow pynvml: energy mJ, power mW,
clocks MHz. Values come from amdsmi as observed; nothing is set or rescaled.
"""
import glob
import os
import sys
from types import SimpleNamespace

try:
    import amdsmi as _a
except ImportError:
    roots = [os.environ['ROCM_PATH']] if os.environ.get('ROCM_PATH') else sorted(glob.glob('/opt/rocm*'))[::-1]
    sys.path.extend(r + '/share/amd_smi' for r in roots)
    import amdsmi as _a

NVML_CLOCK_SM = _a.AmdSmiClkType.GFX
NVML_CLOCK_MEM = _a.AmdSmiClkType.MEM


class NVMLError(Exception):
    pass


def _wrap(call):
    try:
        return call()
    except _a.AmdSmiException as error:
        raise NVMLError(f'{type(error).__name__}: {error}') from error


def _handles():
    return _a.amdsmi_get_processor_handles()


def _enum(handle):
    return _a.amdsmi_get_gpu_enumeration_info(handle)


def nvmlInit():
    _wrap(lambda: _a.amdsmi_init(_a.AmdSmiInitFlags.INIT_AMD_GPUS))


def nvmlShutdown():
    _wrap(_a.amdsmi_shut_down)


def nvmlSystemGetDriverVersion():
    info = _wrap(lambda: _a.amdsmi_get_gpu_driver_info(_handles()[0]))
    return f"{info['driver_name']} {info['driver_version']}"


def nvmlDeviceGetCount():
    return len(_handles())


def nvmlDeviceGetHandleByIndex(index):
    return sorted(_handles(), key=lambda h: _enum(h)['hip_id'])[index]


def nvmlDeviceGetUUID(handle):
    """HIP UUID as torch reports it, with the package's GPU- prefix."""
    return 'GPU-' + _enum(handle)['hip_uuid'].removeprefix('GPU-')


def nvmlDeviceGetHandleByUUID(uuid):
    matches = [h for h in _handles() if nvmlDeviceGetUUID(h) == uuid]
    if len(matches) != 1:
        raise NVMLError(f'{len(matches)} AMD GPUs match {uuid}')
    return matches[0]


def nvmlDeviceGetIndex(handle):
    """HIP ordinal before any visibility masking."""
    return _enum(handle)['hip_id']


def nvmlDeviceGetName(handle):
    return _wrap(lambda: _a.amdsmi_get_gpu_asic_info(handle))['market_name']


def nvmlDeviceGetPciInfo(handle):
    return SimpleNamespace(busId=_wrap(lambda: _a.amdsmi_get_gpu_device_bdf(handle)))


def _processes(handle):
    return [SimpleNamespace(pid=p['pid'], usedGpuMemory=p['mem'])
            for p in _wrap(lambda: _a.amdsmi_get_gpu_process_list(handle))]


nvmlDeviceGetComputeRunningProcesses = _processes
nvmlDeviceGetGraphicsRunningProcesses = _processes


def nvmlDeviceGetClockInfo(handle, clock):
    return _wrap(lambda: _a.amdsmi_get_clock_info(handle, clock))['clk']


def nvmlDeviceGetPowerManagementLimit(handle):
    return _wrap(lambda: _a.amdsmi_get_power_cap_info(handle))['power_cap'] / 1000  # uW -> mW


def nvmlDeviceGetPowerUsage(handle):
    info = _wrap(lambda: _a.amdsmi_get_power_info(handle))
    value = info['current_socket_power']
    if not isinstance(value, (int, float)):
        value = info['average_socket_power']
    if not isinstance(value, (int, float)):
        raise NVMLError(f'no numeric socket power: {info}')
    return value * 1000  # W -> mW


def nvmlDeviceGetTotalEnergyConsumption(handle):
    info = _wrap(lambda: _a.amdsmi_get_energy_count(handle))
    return info['energy_accumulator'] * info['counter_resolution'] / 1000  # uJ -> mJ


def describe_amd(handle):
    """AMD-only identity/cache fields for inventory; failures are kept as reasons."""
    data = {}
    for key, call in {'asic': lambda: _a.amdsmi_get_gpu_asic_info(handle),
                      'board': lambda: _a.amdsmi_get_gpu_board_info(handle),
                      'vram': lambda: _a.amdsmi_get_gpu_vram_info(handle),
                      'cache': lambda: _a.amdsmi_get_gpu_cache_info(handle),
                      'driver': lambda: _a.amdsmi_get_gpu_driver_info(handle),
                      'power_cap': lambda: _a.amdsmi_get_power_cap_info(handle),
                      'power': lambda: _a.amdsmi_get_power_info(handle),
                      'energy': lambda: _a.amdsmi_get_energy_count(handle),
                      'gfx_clock': lambda: _a.amdsmi_get_clock_info(handle, NVML_CLOCK_SM),
                      'mem_clock': lambda: _a.amdsmi_get_clock_info(handle, NVML_CLOCK_MEM),
                      'enumeration': lambda: _enum(handle)}.items():
        try:
            data[key] = call()
        except Exception as error:
            data[key] = {'unavailable_reason': f'{type(error).__name__}: {error}'}
    return data
