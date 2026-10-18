#!/usr/bin/env python3
"""Sample one or more AMD GPUs' clocks/power/temperature/processes to JSONL (observation only).

Runs as a separate host process next to a measurement and never changes device settings.
Trials are later joined to samples by their recorded wall_start/end_unix_ns.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import amd_nvml as n


def main(args):
    n.nvmlInit()
    handles = {u: n.nvmlDeviceGetHandleByUUID(u) for u in args.gpu_uuid}
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    deadline = time.time() + args.max_seconds
    with args.output.open('a') as out:
        while not stop and time.time() < deadline:
            for uuid, h in handles.items():
                row = {'t_unix_ns': time.time_ns(), 'uuid': uuid}
                for key, call in {'gfx_mhz': lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                                  'mem_mhz': lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
                                  'power_mw': lambda: n.nvmlDeviceGetPowerUsage(h),
                                  'energy_mj': lambda: n.nvmlDeviceGetTotalEnergyConsumption(h),
                                  'pids': lambda: [p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h)]}.items():
                    try:
                        row[key] = call()
                    except Exception as error:
                        row[key] = None
                        row[key + '_error'] = str(error)[:120]
                try:
                    row['temp_c'] = n._a.amdsmi_get_temp_metric(h, n._a.AmdSmiTemperatureType.HOTSPOT,
                                                                n._a.AmdSmiTemperatureMetric.CURRENT)
                except Exception:
                    row['temp_c'] = None
                out.write(json.dumps(row) + '\n')
            out.flush()
            time.sleep(args.interval)
    n.nvmlShutdown()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-uuid', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interval', type=float, default=0.1)
    p.add_argument('--max-seconds', type=float, default=43200)
    main(p.parse_args())
