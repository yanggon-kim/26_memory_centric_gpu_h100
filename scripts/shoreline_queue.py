#!/usr/bin/env python3
"""Serial rental campaign driver with source-family boundaries and closeout reserve."""
import argparse
import datetime
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time


def main(a):
    root=a.session.resolve().parent;package=Path(__file__).resolve().parents[1]
    state_path=root/'results/campaign_queue.json'
    queue_attempt=0
    if state_path.exists():
        for number in range(100):
            previous=root/'results'/f'campaign_queue_attempt{number:02d}.json'
            if not previous.exists():
                previous.write_bytes(state_path.read_bytes());queue_attempt=number+1;break
        else:raise RuntimeError('queue history limit reached')
    content=Path(__file__).read_bytes();sha=hashlib.sha256(content).hexdigest()
    snapshot=root/'environment/source_snapshots'/(sha+'.py');snapshot.write_bytes(content)
    state={'status':'running','steps':[],'queue_source_sha256':sha,'start_at':a.start_at,'queue_attempt':queue_attempt}
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    def execute(name,command):
        session=json.loads(a.session.read_text())
        if time.time()>=datetime.datetime.fromisoformat(session['launch_cutoff']).timestamp():
            state['steps'].append({'id':name,'status':'pending','reason':'launch cutoff preserves closeout buffer'});save();return False
        log_path=root/'logs'/f'queue_{name}_attempt{queue_attempt:02d}.log'
        row={'id':name,'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'log':str(log_path.relative_to(root))}
        state['steps'].append(row);save();print('QUEUE_START',name,flush=True)
        with log_path.open('x') as log:
            code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT,cwd=package)
        row.update(exit_code=code,status='command_complete' if code==0 else 'command_failed',
                   ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
        print('QUEUE_END',name,code,flush=True)
        if code!=0:
            state['status']='needs_diagnosis';save();return False
        return True
    python=str(Path(sys.executable));jobs=str(package/'scripts/shoreline_jobs.py')
    base=[python,jobs,'--session',str(a.session.resolve())]
    def closeout_cpu(stage='stop'):
        # These fixed commands analyze saved files only; no GPU or new allocation.
        commands=[('analysis',[python,str(package/'scripts/shoreline_analyze.py'),'--root',str(root)]),
                  ('figures',[python,str(package/'scripts/plot_shoreline.py'),'--root',str(root)]),
                  ('checkpoint',[python,str(package/'scripts/checkpoint_h200.py'),'--root',str(root),
                    '--message',f'Campaign {stage}: publish analysis and scientific gate dispositions','--push'])]
        state['cpu_closeout']=[]
        for name,command in commands:
            log=root/'logs'/f'queue_cpu_{stage}_{name}_attempt{queue_attempt:02d}.log'
            with log.open('x') as stream:
                code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT,cwd=package)
            state['cpu_closeout'].append({'id':name,'exit_code':code,'log':str(log.relative_to(root))})
            save()
        state.setdefault('cpu_boundaries',[]).append({'stage':stage,'results':state['cpu_closeout']})
        save()
    # J1 is completed/supervised by the caller before this queue begins.
    steps=[('J1_saved_text',base+['--job','J1','--representative-only']),('J2',base+['--job','J2']),('J3',base+['--job','J3']),
           ('freeze_new_treatments',[python,str(package/'scripts/freeze_shoreline.py'),'--session',str(a.session.resolve()),'--activate']),
           ('J2_metadata',base+['--job','J2','--metadata-only']),
           ('J4_native',base+['--job','J4']),('J4_extensions',base+['--job','J4','--extensions']),
           ('J1_current_controls',[python,str(package/'scripts/shoreline_campaign.py'),'--session',str(a.session.resolve()),'--job','J1']),
           ('J11_semantic_pilot',base+['--job','J11','--semantic-pilot-only']),('J5',base+['--job','J5']),
           ('J8',base+['--job','J8']),('J9_analysis',[python,str(package/'scripts/shoreline_analyze.py'),'--root',str(root)]),
           ('J7',base+['--job','J7']),('J10',base+['--job','J10']),('J6',base+['--job','J6']),
           ('J11_closeout',base+['--job','J11']),
           ('final_analysis',[python,str(package/'scripts/shoreline_analyze.py'),'--root',str(root)]),
           ('final_figures',[python,str(package/'scripts/plot_shoreline.py'),'--root',str(root)])]
    if a.start_at:steps=steps[next(i for i,(name,_) in enumerate(steps) if name==a.start_at):]
    for name,command in steps:
        if not execute(name,command):
            if state['status']=='running':state['status']='launch_cutoff_reached';save()
            closeout_cpu();return
        if name in ('J5','J8','J7','J10','J6'):
            # The GPU producer has exited: analyze and publish this job before
            # starting the next one, so interruption cannot lose its analysis.
            closeout_cpu('after_'+name)
            if any(row['exit_code']!=0 for row in state['cpu_closeout']):
                state['status']='analysis_needs_diagnosis';save();return
    state['status']='commands_finished_scientific_gates_separate';save()
    with (root/'logs'/f'queue_final_checkpoint_attempt{queue_attempt:02d}.log').open('x') as log:
        code=subprocess.call([python,str(package/'scripts/checkpoint_h200.py'),'--root',str(root),
            '--message','Campaign command sequence finished: publish final evidence and remaining scientific gates','--push'],
            stdout=log,stderr=subprocess.STDOUT,cwd=package)
    if code!=0:
        state['status']='final_publication_failed';save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--start-at')
    main(p.parse_args())
