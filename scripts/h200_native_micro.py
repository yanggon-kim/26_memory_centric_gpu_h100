#!/usr/bin/env python3
"""Bounded native H200 tracer and parser smoke; no timing calibration claim."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from h200_session import WORKSPACE, DEFAULT_ROOT, CUDA, UUID, environment, run, dump, PACKAGE, SESSION
from export_trace import export, archive_verified


def main():
    accel=Path(os.environ.get('H200_ACCEL_ROOT',str(WORKSPACE/'simulator/accel-sim-framework')))
    tag=SESSION.get('native_micro_attempt','v1')
    output=DEFAULT_ROOT/'traces'/('native_micro_'+tag) if SESSION else WORKSPACE/'native_micro'
    output.mkdir(exist_ok=False)
    (output/'raw/traces').mkdir(parents=True)
    binary=output/'dynamic_micro'
    result=run(DEFAULT_ROOT,'native_trace_compile_'+tag,[CUDA/'bin/nvcc','-arch=sm_90','-O2',
               PACKAGE/'scripts/nvbit_dynamic_micro.cu','-o',binary],120)
    assert result['status']=='complete'
    tracer=accel/'util/tracer_nvbit/tracer_tool/tracer_tool.so'
    env=environment()
    env.update(CUDA_INJECTION64_PATH=str(tracer),NVBIT_INSTRUMENTATION_ENABLED='1',ACTIVE_FROM_START='1',
               ALLOW_REG_VAL_TRACING='1',SKIP_TMA_MEM='0',SPINLOCK_HANDLING_MODE='2',ENABLE_WATCHDOG='1',
               TRACE_FILE_COMPRESS='1',USER_DEFINED_FOLDERS='1',TRACES_FOLDER=str(output/'raw'))
    env.pop('DYNAMIC_KERNEL_RANGE',None)
    dump(output/'launch_manifest.json',{'classification':'measured-real-GPU-instrumented',
         'gpu_uuid':UUID,'CUDA_HOME':str(CUDA),'source':'scripts/nvbit_dynamic_micro.cu',
         'source_sha256':hashlib.sha256((PACKAGE/'scripts/nvbit_dynamic_micro.cu').read_bytes()).hexdigest(),
         'tracer_sha256':hashlib.sha256(tracer.read_bytes()).hexdigest(),
         'tracer_flags':{k:env[k] for k in ('NVBIT_INSTRUMENTATION_ENABLED','ACTIVE_FROM_START',
             'ALLOW_REG_VAL_TRACING','SKIP_TMA_MEM','SPINLOCK_HANDLING_MODE','ENABLE_WATCHDOG')},
         'simulator_config_scope':'stock H100 parser smoke; H200 timing/power calibration not established'})
    try:
        result=run(DEFAULT_ROOT,'native_trace_capture_'+tag,[binary,UUID,'20261013'],120,gpu=True,env=env)
        assert result['status']=='complete'
        summary=export(output,accel,True,120)
        suffix='_'+tag if SESSION else ''
        dump(DEFAULT_ROOT/'environment'/('native_trace_micro'+suffix+'.json'),summary)
        destination=DEFAULT_ROOT/'environment'/('native_trace_micro'+suffix+'.tar')
        archive_verified(output,destination)
        # Independently verify the persistent archive before accepting transfer.
        proof=json.loads(destination.with_suffix('.tar.sha256.json').read_text())
        assert hashlib.sha256(destination.read_bytes()).hexdigest()==proof['sha256']
        proof['persistent_readback_sha256']=proof['sha256']
        dump(destination.with_suffix('.tar.sha256.json'),proof)
    except Exception as error:
        dump(DEFAULT_ROOT/'environment'/('native_trace_failure_'+tag+'.json'),{'type':type(error).__name__,
             'message':str(error),'retained_partial_path':str(output)})
        raise


if __name__=='__main__':
    main()
