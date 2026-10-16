#!/usr/bin/env python3
"""Check nonempty NVBit traces, postprocess, optionally smoke replay and hash export."""
import argparse
import csv
import hashlib
import json
import lzma
from pathlib import Path
import re
import subprocess
import shutil
import tarfile
from software_decode_experiments import dump


def replay_evidence(log,expected):
    values={key:[int(x) for x in re.findall(key+r'\s*=\s*(\d+)',log)] for key in ['gpu_tot_sim_cycle','gpu_tot_sim_insn']}
    kernels=len(re.findall(r'^kernel_name\s*=',log,re.M))
    assert values['gpu_tot_sim_cycle'] and values['gpu_tot_sim_insn'],'no substantive simulator counters; exit0 is not success'
    assert values['gpu_tot_sim_cycle'][-1]>0 and values['gpu_tot_sim_insn'][-1]>0 and kernels==expected and len(values['gpu_tot_sim_cycle'])==expected and len(values['gpu_tot_sim_insn'])==expected,'missing/extra completed kernel statistics'
    assert 'GPGPU-Sim: *** exit detected ***' in log,'missing completion marker'
    assert not re.search(r'unsupported.*(architecture|SM|opcode)|unsupported binary version|undefined opcode|Assertion .*failed|Segmentation fault|deadlock detected',log,re.I),'unsupported replay'
    return {'cycles':values['gpu_tot_sim_cycle'][-1],'active_thread_instructions':values['gpu_tot_sim_insn'][-1],'kernel_stats_sections':kernels,'scope':'replay completion only; architecture, same-program calibration and functional correctness checked separately'}


def export(root,accel,replay=False,timeout=300):
    stats=[p for p in root.rglob('stats*') if p.is_file()]
    instruction_count=0
    traced_kernels=0
    for path in stats:
        with path.open() as file:
            for row in csv.DictReader(file):
                row={k.strip():v.strip() for k,v in row.items() if k is not None}
                count=int(row['total_reported_insts'])
                instruction_count+=count
                traced_kernels+=count>0
    assert instruction_count>0 and traced_kernels>0,'native completion with zero trace instructions is failure'
    binary_versions=set()
    for path in root.rglob('*.trace*'):
        if path.name.endswith(('.tracez','.traceg')):continue
        opener=lzma.open if path.name.endswith('.xz') else open
        with opener(path,'rt',errors='replace') as file:
            header=''.join(file.readline() for _ in range(40))
        binary_versions.update(int(x) for x in re.findall(r'binary version\s*=\s*(\d+)',header,re.I))
    assert binary_versions,'raw trace binary architecture header missing'
    tool=accel/'util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing'
    assert tool.is_file(),'build pinned postprocessor first'
    raw_lists=[p for p in root.rglob('kernelslist*') if '.g' not in p.name]
    assert raw_lists and any(p.stat().st_size for p in raw_lists),'empty raw kernel list'
    for folder in sorted({p.parent for p in raw_lists}):
        subprocess.run([str(tool),str(folder),'-j','2'],check=True,timeout=timeout)
    lists=list(root.rglob('kernelslist*.g'))
    entries=[]
    for path in lists:
        for line in path.read_text().splitlines():
            if '.tracez' in line or '.traceg' in line:
                candidate=path.parent/line.strip()
                assert candidate.is_file() and candidate.stat().st_size>0,'missing/empty grouped trace '+str(candidate)
                entries.append(candidate)
    assert entries,'postprocessing generated no grouped kernel traces'
    summary={'status':'capture_and_postprocess_verified','reported_warp_instructions':instruction_count,
             'traced_kernels':traced_kernels,'grouped_files':len(entries),'replay':'pending',
             'binary_versions':sorted(binary_versions),'native_hopper_trace':binary_versions=={90},
             'scope':'capture/postprocess integrity only; decode scope requires matching trace_metadata; timing excluded optimized baseline comparison'}
    if replay:
        assert binary_versions=={90},'H100 replay requires actual sm_90 traces; header relabeling is prohibited'
        executable=accel/'gpu-simulator/bin/release/accel-sim.out'
        configs=[accel/'gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100/gpgpusim.config',accel/'gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config']
        assert executable.is_file() and all(p.is_file() for p in configs)
        # First actual trace kernel only: rental integrity smoke, not whole model simulation.
        first=entries[0]
        smoke=first.parent/'rental_replay_smoke.g'
        smoke.write_text(first.name+'\n')
        replay_dir=root/'rental_replay'
        assert not replay_dir.exists(),'use a fresh owned replay directory'
        shutil.copytree(configs[0].parent,replay_dir)
        cmd=[str(executable),'-config',str(replay_dir/'gpgpusim.config'),'-config',str(configs[1]),'-trace',str(smoke)]
        with (root/'rental_replay.log').open('w') as file:
            result=subprocess.run(cmd,stdout=file,stderr=subprocess.STDOUT,timeout=timeout,cwd=replay_dir)
        assert result.returncode==0,'simulator process failed'
        summary['replay']=replay_evidence((root/'rental_replay.log').read_text(),1)
    artifacts=[]
    for path in sorted(p for p in root.rglob('*') if p.is_file() and p.name!='export_manifest.json'):
        h=hashlib.sha256()
        with path.open('rb') as file:
            for chunk in iter(lambda:file.read(1024**2),b''):h.update(chunk)
        artifacts.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    summary['artifacts']=artifacts
    dump(root/'export_manifest.json',summary)
    return summary


def archive_verified(root,destination):
    """Archive already verified outputs outside the capture tree and checksum it."""
    assert (root/'export_manifest.json').is_file()
    destination=destination.resolve()
    assert not destination.is_relative_to(root),'archive destination must be outside trace tree'
    destination.parent.mkdir(parents=True,exist_ok=True)
    size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    assert shutil.disk_usage(destination.parent).free>size*1.1,'insufficient disk for lossless archive'
    assert not destination.exists(),'refuse to overwrite archive'
    with tarfile.open(destination,'w') as archive:
        archive.add(root,arcname=root.name)
    h=hashlib.sha256()
    with destination.open('rb') as file:
        for chunk in iter(lambda:file.read(1024**2),b''):h.update(chunk)
    dump(destination.with_suffix(destination.suffix+'.sha256.json'),{'archive':str(destination),'bytes':destination.stat().st_size,
          'sha256':h.hexdigest(),'scope':'lossless raw+grouped traces/metadata; transferred-copy checksum must be verified before rental termination'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-root',type=Path,required=True)
    p.add_argument('--accel-root',type=Path,required=True)
    p.add_argument('--replay-smoke',action='store_true')
    p.add_argument('--timeout-s',type=int,default=300)
    p.add_argument('--archive-path',type=Path)
    a=p.parse_args()
    summary=export(a.trace_root.resolve(),a.accel_root.resolve(),a.replay_smoke,a.timeout_s)
    print(json.dumps({k:v for k,v in summary.items() if k!='artifacts'}|{'hashed_artifacts':len(summary['artifacts'])},indent=2))
    if a.archive_path:archive_verified(a.trace_root.resolve(),a.archive_path)
