#!/usr/bin/env python3
"""Attempt bounded whole-engine decode traffic with explicit range replay scope."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from h200_session import DEFAULT_ROOT,PACKAGE,PYTHON,run,dump
from run_h200_phase01 import config_for
from profile_h200 import NCU,command,parse_csv_log
from analyze_h200 import number


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--batch',type=int,required=True)
    parser.add_argument('--context',type=int,required=True)
    parser.add_argument('--attempt',default='application_range')
    args=parser.parse_args();root=DEFAULT_ROOT
    assert re.fullmatch(r'[a-z0-9_]+',args.attempt)
    label=f'{args.model}_range_b{args.batch}_c{args.context}_{args.attempt}'
    output=root/'runs'/label
    assert not output.exists(),'range attempt must be fresh'
    config=config_for(root,root/'configs'/f'{args.model}_baseline.json',label,args.batch,args.context)
    report=root/'profiles'/(label+'.ncu-rep')
    # CUDA graph APIs are unsupported by captured range replay. App-range
    # re-executes the program, so host allocations and graph calls stay native.
    cmd=[NCU,'--replay-mode','app-range','--range-filter',':1:',
         '--launch-count','1','--kill','yes',
         '--clock-control','none','--cache-control','none',
         '--metrics','dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum',
         '--export',report]
    native_command=command(root,label,config,output)
    native_command[1]=PACKAGE/'scripts/h200_app_range_target.py'
    outcome=run(root,label,cmd+native_command,600,gpu=True)
    pass_outputs=sorted(output.parent.glob(label+'_pass*'))
    for pass_output in pass_outputs:
        partial=pass_output/'run.json'
        if not partial.exists():continue
        original=json.loads(partial.read_text())
        if original['status']=='running':
            original.update(status='partial',termination_reason='bounded range profiler intentionally terminated target; ordinary three-trial run incomplete')
            dump(partial,original)
    result={'status':'failed','model':args.model,'batch':args.batch,'context':args.context,
            'classification':'measured-real-GPU','external_instrumentation':True,
            'scope':'three whole engine.step decode iterations after all-request first token; includes LM head/sampling; instrumented range replay',
            'host_restore':'not required by application-range replay; each pass is a fresh process and has its own output directory',
            'clock_control':'none','cache_control':'none','replay_mode':'app-range',
            'range_output_proof':None,'native_pass_proofs':[],'metrics':[],
            'timing_scope':'range replay can change host gaps; duration is not an uninstrumented decode measurement'}
    reference=root/'runs'/f'{args.model}_graph_b{args.batch}_c{args.context}'/'run.json'
    for pass_output in pass_outputs:
        native=pass_output/'profile_interval_output.json'
        if not native.exists():continue
        saved=json.loads(native.read_text());expected=json.loads(reference.read_text())['cases'][0]['trials'][0]['output_ids']
        exact=(saved['output_ids']==[row[:4] for row in expected]
               and saved['decode_iterations']==3 and saved['timed_generated_tokens']==args.batch*3)
        proof={'source':str(native.relative_to(root)),
            'sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'exact_matching_reference_prefix':exact,
            'decode_iterations':saved['decode_iterations'],'timed_generated_tokens':saved['timed_generated_tokens']}
        result['native_pass_proofs'].append(proof)
    if result['native_pass_proofs']:
        result['range_output_proof']={'exact_matching_reference_prefix':all(p['exact_matching_reference_prefix'] for p in result['native_pass_proofs']),
            'completed_native_passes':len(result['native_pass_proofs']),
            'all_passes_have_proof':len(result['native_pass_proofs'])==len(pass_outputs)}
    log=(root/'logs'/(label+'.log')).read_text()
    result['reported_replay_passes']=[int(x) for x in re.findall(r'(\d+)\s+passes',log)] or None
    if report.exists():
        csv_label=label+'_raw_metrics'
        run(root,csv_label,[NCU,'--import',report,'--page','raw','--csv'],120)
        records=parse_csv_log((root/'logs'/(csv_label+'.log')).read_text(),'"ID",')
        result['metrics']=[r for r in records if r.get('Metric Name')]
        values={r['Metric Name']:r['Metric Value'] for r in result['metrics']}
        read=number(values.get('dram__bytes_read.sum'))
        written=number(values.get('dram__bytes_write.sum'))
        ids={r['ID'] for r in result['metrics']}
        if (len(ids)==1 and read is not None and read>0 and written is not None and written>=0
            and result['range_output_proof'] and result['range_output_proof']['exact_matching_reference_prefix']
            and result['range_output_proof']['all_passes_have_proof']
            and '==ERROR==' not in log):
            result['status']='complete'
            result['dram_read_bytes']=read
            result['dram_write_bytes']=written
            result['timed_tokens']=args.batch*3
    result['command_outcome']=outcome
    result['missing_reason']=None if result['status']=='complete' else 'range replay unsupported, output proof unavailable/mismatched, or no substantive DRAM metrics; see raw logs'
    dump(root/'profiles'/(label+'_range_result.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','command_outcome')},indent=2))


if __name__=='__main__':main()
