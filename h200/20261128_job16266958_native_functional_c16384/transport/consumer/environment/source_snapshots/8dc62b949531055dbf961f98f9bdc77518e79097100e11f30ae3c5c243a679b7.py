#!/usr/bin/env python3
"""Read-only NVML clock-event snapshots of exactly an owned allocated pair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def main(args):
    sessions = [json.loads(path.read_text()) for path in args.sessions]
    jobs = {s['slurm_job_id'] for s in sessions}
    assert len(sessions) == len({s['gpu_uuid'] for s in sessions}) == 2 and len(jobs) == 1
    job = jobs.pop()
    assert os.environ.get('SLURM_JOB_ID') == job
    assert all(s['expected_hostname'] == socket.gethostname() for s in sessions)
    state = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
    assert 'JobState=RUNNING' in state and f'({os.getuid()})' in state and 'gres/gpu:h200=2' in state
    import pynvml as nv
    nv.nvmlInit()
    rows = []
    def read(call):
        try:
            return call()
        except nv.NVMLError as error:
            return {'unavailable': str(error)}
    try:
        handles = [(s['gpu_uuid'], nv.nvmlDeviceGetHandleByUUID(s['gpu_uuid'])) for s in sessions]
        flags = {name: getattr(nv, 'nvmlClocksEventReason'+name) for name in (
            'GpuIdle', 'ApplicationsClocksSetting', 'SwPowerCap', 'HwSlowdown', 'SyncBoost',
            'SwThermalSlowdown', 'HwThermalSlowdown', 'HwPowerBrakeSlowdown', 'DisplayClockSetting')}
        for _ in range(args.samples):
            for uuid, handle in handles:
                mask = read(lambda: nv.nvmlDeviceGetCurrentClocksEventReasons(handle))
                rows.append({'unix_ns': time.time_ns(), 'gpu_uuid': uuid, 'clock_event_mask': mask,
                    'clock_event_names': [name for name, flag in flags.items() if isinstance(mask, int) and mask & flag],
                    'pstate': read(lambda: nv.nvmlDeviceGetPerformanceState(handle)),
                    'sm_clock_mhz': read(lambda: nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)),
                    'memory_clock_mhz': read(lambda: nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM)),
                    'power_limit_w': read(lambda: nv.nvmlDeviceGetPowerManagementLimit(handle)/1000),
                    'power_w': read(lambda: nv.nvmlDeviceGetPowerUsage(handle)/1000),
                    'temperature_c': read(lambda: nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU))})
            time.sleep(.2)
    finally:
        nv.nvmlShutdown()
    result = {'classification': 'read-only instantaneous diagnostic', 'slurm_job_id': job,
              'hostname': socket.gethostname(), 'samples': rows,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sessions': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.sessions},
              'scope': 'No CUDA contexts/kernels or settings changes. Current reported event masks only; not proof of the cause of earlier clock failures or a timing qualification.'}
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(args.output)
    print(json.dumps({'output': str(args.output), 'samples': len(rows),
                      'reported_reasons': sorted({name for row in rows for name in row['clock_event_names']})}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=25)
    args = parser.parse_args()
    assert 1 <= args.samples <= 50
    main(args)
