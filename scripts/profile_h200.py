#!/usr/bin/env python3
"""Bounded decode profiles, actual worker metadata and separately scoped counters."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
from h200_session import DEFAULT_ROOT, PACKAGE, PYTHON, UUID, environment, python_command, run, dump
from run_h200_phase01 import case, config_for
from summarize_h200 import summarize

NSYS=Path('/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/none-none/cuda-12.9.1-leytg2766xv4h3xx3vg4rkxawtyj5qvl/bin/nsys')
NCU=NSYS.with_name('ncu')
BASE_METRICS=['gpu__time_duration','dram__bytes_read','dram__bytes_write','dram__throughput',
              'lts__t_sector_hit_rate','lts__t_sectors_op_read','lts__t_sectors_op_write',
              'sm__warps_active','sm__pipe_tensor_cycles_active','smsp__inst_executed_pipe_lsu',
              'smsp__warp_issue_stalled_long_scoreboard_per_warp_active',
              'smsp__warp_issue_stalled_short_scoreboard_per_warp_active',
              'smsp__warp_issue_stalled_barrier_per_warp_active',
              'l1tex__t_sectors_pipe_lsu_mem_local_op_ld','l1tex__t_sectors_pipe_lsu_mem_local_op_st',
              'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld',
              'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st']
METRICS=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum',
         'dram__throughput.avg.pct_of_peak_sustained_elapsed','lts__t_sector_hit_rate.pct',
         'lts__t_sectors_op_read.sum','lts__t_sectors_op_write.sum',
         'sm__warps_active.avg.pct_of_peak_sustained_active',
         'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
         'smsp__inst_executed_pipe_lsu.sum',
         'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
         'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
         'smsp__warp_issue_stalled_barrier_per_warp_active.pct',
         'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
         'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
         'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum']


def command(root,label,config,output,metadata=False):
    cmd=python_command('controlled_vllm_experiments.py','--config',config,'--gpu-uuid',UUID,
         '--mode','timing','--output',output,'--generated-tokens','4','--external-profiler')
    if metadata:cmd+=['--collect-worker-metadata']
    return cmd


def verify_prefix(root,output,reference):
    data=json.loads((output/'run.json').read_text())
    expected=json.loads((reference/'run.json').read_text())['cases'][0]['trials'][0]['output_ids']
    actual=data['cases'][0]['trials'][0]['output_ids']
    passed=actual==[row[:4] for row in expected]
    dump(output/'reference_prefix_check.json',{'passed':passed,'reference':str(reference.relative_to(root)),
          'reference_generated_tokens':len(expected[0]),'profile_generated_tokens':4,
          'scope':'exact prefix IDs of matching uninstrumented engine/shape'})
    assert passed,'profile output differs from matching uninstrumented prefix'


def parse_csv_log(text,first_column):
    lines=text.splitlines()
    start=next((i for i,line in enumerate(lines) if line.startswith(first_column)),None)
    return list(csv.DictReader(io.StringIO('\n'.join(lines[start:])))) if start is not None else []


def profile(root,model,batch,context,extended=False):
    base=root/'configs'/f'{model}_baseline.json'
    assert base.exists()
    reference=case(root,model,base,batch,context,'eager')
    assert reference is not None,'eager reference must pass before profiling'
    identity=f'{model}_profile_b{batch}_c{context}'
    config=config_for(root,base,identity,batch,context,eager=True)
    metadata_output=root/'runs'/(identity+'_metadata')
    if not metadata_output.exists():
        result=run(root,identity+'_metadata',command(root,identity,config,metadata_output,True),300,gpu=True)
        if result['status']!='complete':return
        verify_prefix(root,metadata_output,reference)
    eager_timeline=root/'profiles'/(identity+'_eager')
    output=root/'runs'/(identity+'_nsys_eager')
    if not output.exists():
        cmd=[NSYS,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx',
             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(eager_timeline)]
        result=run(root,identity+'_nsys_eager',cmd+command(root,identity,config,output),300,gpu=True)
        if result['status']!='complete':return
        verify_prefix(root,output,reference)
    stats_label=identity+'_kernel_summary'
    stats_log=root/'logs'/(stats_label+'.log')
    if not stats_log.exists():
        run(root,stats_label,[NSYS,'stats','--report','cuda_gpu_kern_sum','--format','csv',
                             str(eager_timeline)+'.nsys-rep'],120)
    kernels=parse_csv_log(stats_log.read_text(),'Time (%)')
    assert kernels,'timeline contains no kernel summary'
    dump(root/'profiles'/(identity+'_kernel_inventory.json'),{'classification':'measured-real-GPU-instrumented',
         'kernels':kernels,'timing_scope':'three eager pure-decode steps from first measured trial',
         'share_denominator':'sum of GPU kernel durations; not synchronized wall time'})
    layers=json.loads(base.read_text())['layout']['num_hidden_layers']
    selected=sorted(set([0,layers//2,layers-1])) if extended else [layers//2]
    catalog=root/'environment/ncu_decode_metric_catalog.log'
    if not catalog.exists():
        run(root,'ncu_decode_metric_catalog',[NCU,'--query-metrics','--query-metrics-mode','suffix',
                                            '--metrics',','.join(BASE_METRICS)],120)
        shutil.copy2(root/'logs/ncu_decode_metric_catalog.log',catalog)
    available=catalog.read_text()
    requested=[m for m in METRICS if re.search(r'^'+re.escape(m)+r'\s',available,re.M)]
    missing=[m for m in METRICS if m not in requested]
    for layer in selected:
        label=identity+f'_ncu_layer{layer}'
        output=root/'runs'/label
        if output.exists():continue
        cmd=[NCU,'--profile-from-start','off','--clock-control','none','--cache-control','none',
             '--replay-mode','kernel','--nvtx','--nvtx-include',f'layer_{layer}/',
             '--kernel-name-base','demangled','--kernel-name','regex:.*(nvjet|gemv|gemm|cutlass|flash|cache).*',
             '--launch-count','8','--section','LaunchStats','--section','Occupancy',
             '--metrics',','.join(requested),'--csv']
        outcome=run(root,label,cmd+command(root,label,config,output),600,gpu=True)
        records=parse_csv_log((root/'logs'/(label+'.log')).read_text(),'"ID",')
        records=[r for r in records if r.get('Metric Name') and r.get('Kernel Name')]
        dump(root/'profiles'/(label+'.json'),{'status':'complete' if records and outcome['status']=='complete' else 'failed',
             'model':model,'batch':batch,'context':context,'layer':layer,'engine':'vllm_eager',
             'classification':'measured-real-GPU-instrumented','metrics':records,
             'unavailable_metrics':missing,'requested_metrics':requested,
             'cache_control':'none','clock_control':'none','replay_mode':'kernel',
             'scope':'bounded matched eager kernel samples; replay changes residency; not full-step DRAM totals',
             'log':str((root/'logs'/(label+'.log')).relative_to(root))})
        if outcome['status']=='complete':verify_prefix(root,output,reference)
    summarize(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--model',required=True)
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--context',type=int,required=True)
    parser.add_argument('--extended',action='store_true')
    args=parser.parse_args()
    profile(args.root.resolve(),args.model,args.batch,args.context,args.extended)
