#!/usr/bin/env python3
"""Profile the graph baseline separately from the eager layer controls."""
import argparse
import json
from pathlib import Path
import re
from h200_session import DEFAULT_ROOT,UUID,python_command,run,dump
from run_h200_phase01 import config_for
from profile_h200 import NSYS,NCU,METRICS,command,parse_csv_log,verify_prefix
from analyze_h200 import analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--model',required=True)
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--context',type=int,required=True)
    parser.add_argument('--cache-control',choices=['none','all'],default='none')
    args=parser.parse_args();root=args.root.resolve()
    identity=f'{args.model}_profile_graph_b{args.batch}_c{args.context}'
    base=root/'configs'/f'{args.model}_baseline.json'
    reference=root/'runs'/f'{args.model}_graph_b{args.batch}_c{args.context}'
    assert (reference/'run.json').exists()
    config=config_for(root,base,identity,args.batch,args.context)
    timeline=root/'profiles'/identity
    output=root/'runs'/(identity+'_nsys')
    if not output.exists():
        cmd=[NSYS,'profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx',
             '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop',
             '--output='+str(timeline)]
        outcome=run(root,identity+'_nsys',cmd+command(root,identity,config,output),300,gpu=True)
        if outcome['status']!='complete':return
        verify_prefix(root,output,reference)
    stats_label=identity+'_kernel_summary'
    stats_log=root/'logs'/(stats_label+'.log')
    if not stats_log.exists():
        run(root,stats_label,[NSYS,'stats','--report','cuda_gpu_kern_sum','--format','csv',
                             str(timeline)+'.nsys-rep'],120)
    kernels=parse_csv_log(stats_log.read_text(),'Time (%)')
    dump(root/'profiles'/(identity+'_kernel_inventory.json'),{'classification':'measured-real-GPU',
         'external_instrumentation':True,'kernels':kernels,'timing_scope':'three graph pure-decode steps',
         'share_denominator':'sum of GPU kernel durations; not synchronized wall time'})
    assert kernels,'graph capture contained no kernel summary'
    # Select observed optimized linears rather than importing RTX kernel names.
    names=' '.join(row.get('Name','') for row in kernels).lower()
    expression='.*nvjet.*' if 'nvjet' in names else '.*(gemv|gemm|cublas).*'
    available=(root/'environment/ncu_decode_metric_catalog.log').read_text()
    metrics=[m for m in METRICS if re.search(r'^'+re.escape(m)+r'\s',available,re.M)]
    for category,pattern,count in [('linears',expression,4),('attention','.*(flash|FlashAttn|reshape_and_cache).*',3)]:
        label=identity+'_ncu_'+category+('_cold' if args.cache_control=='all' else '')
        output=root/'runs'/label
        if output.exists():continue
        cmd=[NCU,'--profile-from-start','off','--clock-control','none','--cache-control',args.cache_control,
             '--replay-mode','kernel','--graph-profiling','node',
             '--kernel-name-base','demangled','--kernel-name','regex:'+pattern,
             '--launch-count',str(count),'--section','LaunchStats','--section','Occupancy',
             '--metrics',','.join(metrics),'--csv']
        outcome=run(root,label,cmd+command(root,label,config,output),600,gpu=True)
        records=parse_csv_log((root/'logs'/(label+'.log')).read_text(),'"ID",')
        records=[r for r in records if r.get('Metric Name') and r.get('Kernel Name')]
        dump(root/'profiles'/(label+'.json'),{'status':'complete' if records and outcome['status']=='complete' else 'failed',
             'model':args.model,'batch':args.batch,'context':args.context,'layer':None,'engine':'vllm_graph',
             'classification':'measured-real-GPU','external_instrumentation':True,'metrics':records,
             'requested_metrics':metrics,'cache_control':args.cache_control,'clock_control':'none','replay_mode':'kernel',
             'scope':f'first {count} matching optimized graph {category} launches after the prefill barrier; cache-control {args.cache_control}; kernel replay, not full-step traffic',
             'log':str((root/'logs'/(label+'.log')).relative_to(root))})
        if outcome['status']=='complete':verify_prefix(root,output,reference)
    print(json.dumps(analyze(root),indent=2))


if __name__=='__main__':main()
