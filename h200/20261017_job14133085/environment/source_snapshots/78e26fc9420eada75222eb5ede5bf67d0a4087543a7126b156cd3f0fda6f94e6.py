#!/usr/bin/env python3
"""Bounded trace launch with actual pinned NVBit1.8 hook API and nonempty checks."""
import argparse
import json
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from software_decode_experiments import dump
from export_trace import export,archive_verified


def terminate_wrapper(signum, frame):
    # Let the owned-child finally block run when the outer supervisor stops us.
    raise SystemExit(128+signum)


def stop_owned_child(process):
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()


def main(a):
    signal.signal(signal.SIGTERM,terminate_wrapper)
    assert 1<=a.steps<=4 and 1<=a.batch<=8 and a.context>0 and a.budget_gib>0 and a.budget_files>0 and a.timeout_s>0 and a.replay_timeout_s>0,'invalid trace budget/shape'
    assert not any(k.startswith(('GPGPUSIM','GPGPU_SIM','ACCELSIM')) for k in os.environ),'native capture rejects inherited simulator setup environment'
    assert not os.environ.get('LD_PRELOAD'),'native capture requires no inherited preload'
    assert not any(term in os.environ.get('LD_LIBRARY_PATH','').lower() for term in ['gpgpu-sim','gpgpusim','accel-sim','accel_sim']),'native capture rejects simulator library paths'
    package=Path(__file__).resolve().parents[1]
    pin=json.loads((package/'configs/simulator_pin.json').read_text())
    actual=subprocess.check_output(['git','-C',str(a.accel_root),'rev-parse','HEAD'],text=True).strip()
    assert actual==pin['framework_sha'],'framework SHA differs from reviewed pin'
    backend=subprocess.check_output(['git','-C',str(a.accel_root/'gpu-simulator/gpgpu-sim'),'rev-parse','HEAD'],text=True).strip()
    assert backend==pin['gpgpu_sim_sha'],'backend SHA differs from reviewed pin'
    toolkit=Path(os.environ['CUDA_HOME'])
    assert (toolkit/'bin/cuobjdump').is_file(),'explicit matching cuobjdump missing'
    tool=a.accel_root/'util/tracer_nvbit/tracer_tool/tracer_tool.so'
    assert tool.is_file(),'build tracer first; H100 runtime validation still required'
    a.output.mkdir(parents=True,exist_ok=False)
    raw=a.output/'raw'
    raw.mkdir() # upstream mkdir is nonrecursive; precreate the entire parent.
    (raw/'traces').mkdir()
    reserve=a.budget_gib*(3 if a.archive_path else 2)
    assert shutil.disk_usage(a.output).free>reserve*1024**3,'insufficient raw+grouped(+archive) trace headroom'
    inode_budget={'maximum_raw_files':a.budget_files,'required_free_user_inodes':3*a.budget_files+512}
    if shutil.which('lfs'):
        quota=subprocess.check_output(['lfs','quota','-u',str(os.getuid()),str(a.output)],text=True,timeout=20)
        dump(a.output/'quota_before.json',{'raw_readonly_quota':quota})
        rows=[line.split() for line in quota.splitlines() if line.lstrip().startswith('/')]
        assert len(rows)==1 and len(rows[0])>=9,'cannot establish user trace inode headroom'
        used=int(rows[0][5].rstrip('*'));limit=int(rows[0][7])
        inode_budget.update(user_inodes_used=used,user_inode_limit=limit,free_user_inodes=limit-used)
        assert limit>0 and limit-used>=inode_budget['required_free_user_inodes'],'insufficient user inode headroom for raw/grouped trace plus reserve'
    else:
        inode_budget['free_filesystem_inodes']=os.statvfs(a.output).f_favail
        assert inode_budget['free_filesystem_inodes']>=inode_budget['required_free_user_inodes'],'insufficient filesystem inode headroom'
    env=dict(os.environ)
    env.update(PATH=str(toolkit/'bin')+os.pathsep+env.get('PATH',''),
               PYTHONPATH=str(a.accel_root/'util/tracer_nvbit/others')+os.pathsep+env.get('PYTHONPATH',''),
               CUDA_INJECTION64_PATH=str(tool),NVBIT_INSTRUMENTATION_ENABLED='0',ACTIVE_FROM_START='1',
               ALLOW_REG_VAL_TRACING='1',SKIP_TMA_MEM='0',SPINLOCK_HANDLING_MODE='0',ENABLE_WATCHDOG='0',
               TRACE_FILE_COMPRESS='1',USER_DEFINED_FOLDERS='1',TRACES_FOLDER=str(raw),
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_ENABLE_V1_MULTIPROCESSING='0')
    env.pop('DYNAMIC_KERNEL_RANGE',None) # runtime hooks provide the exact decode/layer boundary.
    runner='trace_vllm_decode.py' if a.engine=='vllm' else 'trace_decode.py'
    cmd=[sys.executable,str(package/'scripts'/runner),'--accel-root',str(a.accel_root),'--config',str(a.config),
         '--gpu-uuid',a.gpu_uuid,'--output',str(a.output/'metadata'),'--batch',str(a.batch),'--context',str(a.context),
         '--steps',str(a.steps),'--layers',a.layers]
    binaries={}
    for name,path in {'tracer':tool,'postprocessor':a.accel_root/'util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing','simulator':a.accel_root/'gpu-simulator/bin/release/accel-sim.out'}.items():
        assert path.is_file(),name+' binary missing'
        binaries[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    toolkit_versions={name:subprocess.check_output([str(toolkit/'bin'/name),'--version'],text=True,stderr=subprocess.STDOUT) for name in ['nvcc','cuobjdump']}
    config=json.loads(a.config.read_text())
    dump(a.output/'resolved_run_config.json',config)
    modelconfig=Path(config['model_path'])/'config.json'
    dump(a.output/'checkpoint_model_config.json',json.loads(modelconfig.read_text()))
    source_state={name:{'unstaged':subprocess.check_output(['git','-C',str(path),'diff','--stat'],text=True),'staged':subprocess.check_output(['git','-C',str(path),'diff','--cached','--stat'],text=True)} for name,path in {'framework':a.accel_root,'backend':a.accel_root/'gpu-simulator/gpgpu-sim'}.items()}
    patch_spec=pin['reviewed_local_patch']
    patch_file=package/patch_spec['path']
    source_file=a.accel_root/patch_spec['source_path']
    assert hashlib.sha256(patch_file.read_bytes()).hexdigest()==patch_spec['sha256'],'reviewed patch hash mismatch'
    assert hashlib.sha256(source_file.read_bytes()).hexdigest()==patch_spec['patched_source_sha256'],'reviewed patched source hash mismatch'
    changed=subprocess.check_output(['git','-C',str(a.accel_root),'diff','--name-only'],text=True).splitlines()
    assert changed==[patch_spec['source_path']] and not source_state['framework']['staged'],'unreviewed framework changes'
    assert not source_state['backend']['unstaged'] and not source_state['backend']['staged'],'unreviewed backend changes'
    package_scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (package/'scripts').glob('*.py')}
    summary={'status':'running','framework_sha':actual,'backend_sha':backend,'nvbit_version':'1.8','command':cmd,
             'engine':a.engine,'trace_env':{k:env[k] for k in ['ACTIVE_FROM_START','NVBIT_INSTRUMENTATION_ENABLED','ALLOW_REG_VAL_TRACING','SKIP_TMA_MEM','SPINLOCK_HANDLING_MODE','ENABLE_WATCHDOG','USER_DEFINED_FOLDERS','TRACE_FILE_COMPRESS','TRACES_FOLDER']},
             'runtime_packages':{dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()},'external_binaries':binaries,'CUDA_HOME':str(toolkit),'toolkit_versions':toolkit_versions,'source_changes':source_state,'package_script_sha256':package_scripts,'model_config_sha256':hashlib.sha256(modelconfig.read_bytes()).hexdigest(),
             'reviewed_local_patch':patch_spec,'budget_gib':a.budget_gib,'inode_budget':inode_budget,'minimum_reserved_disk_gib':reserve,'timeout_s':a.timeout_s,'replay':'pending'}
    dump(a.output/'launch_manifest.json',summary)
    start=time.monotonic()
    failure=None
    with (a.output/'capture.log').open('w') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=a.output,start_new_session=True)
        try:
            while process.poll() is None:
                size=0;file_count=0
                for path in raw.rglob('*'):
                    try:
                        if path.is_file():size+=path.stat().st_size;file_count+=1
                    except FileNotFoundError:pass # compressor may rename during this snapshot
                if size>a.budget_gib*1024**3 or file_count>a.budget_files or time.monotonic()-start>a.timeout_s:
                    failure='trace byte/file/time budget exceeded; partial traces are not validated'
                    break
                time.sleep(1)
        finally:
            stop_owned_child(process)
    finished_files=[p for p in raw.rglob('*') if p.is_file()]
    finished_bytes=sum(p.stat().st_size for p in finished_files)
    dump(a.output/'capture_budget_observation.json',{'raw_files':len(finished_files),'raw_bytes':finished_bytes,
        'maximum_raw_files':a.budget_files,'maximum_raw_bytes':a.budget_gib*1024**3,'failure':failure})
    assert finished_bytes<=a.budget_gib*1024**3 and len(finished_files)<=a.budget_files,'completed capture exceeded byte/file budget; preserve partial evidence'
    assert failure is None,failure
    assert process.returncode==0,'trace process failed; preserve capture.log'
    metadata=json.loads((a.output/'metadata/trace_metadata.json').read_text())
    assert metadata['status']=='complete' and metadata['timed_generated_tokens']==a.batch*a.steps
    summary['status']='native_capture_finished'
    dump(a.output/'launch_manifest.json',summary)
    export(a.output,a.accel_root,a.replay_smoke,a.replay_timeout_s)
    # export_manifest hashes the final launch manifest; do not rewrite it afterward.
    if a.archive_path:archive_verified(a.output,a.archive_path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accel-root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--layers',default='full')
    p.add_argument('--engine',choices=['hf','vllm'],default='hf',help='HF semantic bridge or required eager-vLLM optimized-kernel calibration route')
    p.add_argument('--budget-gib',type=float,default=40)
    p.add_argument('--budget-files',type=int,default=2048)
    p.add_argument('--timeout-s',type=int,default=1800)
    p.add_argument('--replay-smoke',action='store_true')
    p.add_argument('--replay-timeout-s',type=int,default=300)
    p.add_argument('--archive-path',type=Path,help='lossless tar outside capture tree, with SHA256 sidecar')
    a=p.parse_args()
    a.accel_root=a.accel_root.resolve();a.config=a.config.resolve();a.output=a.output.resolve()
    try:main(a)
    except BaseException as error:
        if a.output.exists():
            dump(a.output/'failure.json',{'status':'partial_failed','type':type(error).__name__,'message':str(error)})
            if (a.output/'launch_manifest.json').exists():
                manifest=json.loads((a.output/'launch_manifest.json').read_text())
                manifest.update(status='partial_failed',failure_reason=str(error))
                dump(a.output/'launch_manifest.json',manifest)
        raise
