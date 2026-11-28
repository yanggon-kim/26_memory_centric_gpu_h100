"""Read-only Nsight SQLite coverage candidate; never scientific qualification.

The observed export's low24-bit thread encoding is accepted only when every
API/NVTX thread maps to an explicit PROCESSES row. Kernel ownership is explicit.
GPU activity is joined through process-scoped CUDA correlation IDs and the
launch API's host-thread NVTX scopes, never GPU-time containment in a host span.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        assert type(a) is int and type(b) is int and b>=a
        total+=b-a if end is None or a>=end else max(0,b-end)
        end=b if end is None else max(end,b)
    return total


def read(path,actor_pid,range_name):
    assert type(actor_pid) is int and actor_pid>0
    path=Path(path).resolve()
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        db.execute('PRAGMA query_only=ON')
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required={'PROCESSES','StringIds','NVTX_EVENTS','CUPTI_ACTIVITY_KIND_KERNEL'}
        assert required<=tables,'required Nsight export tables missing'
        process_rows=list(db.execute('SELECT globalPid,pid FROM PROCESSES'))
        processes={r['globalPid']:r['pid'] for r in process_rows}
        assert len(processes)==len(process_rows),'ambiguous exported process map'
        actors=[g for g,p in processes.items() if p==actor_pid]
        assert len(actors)==1,'actor PID missing or ambiguous in exported process map'
        actor=actors[0]
        def owner(tid):
            assert type(tid) is int
            base=tid & ~0xffffff
            assert base in processes,'unsupported thread encoding: no explicit process mapping'
            return base
        strings={r['id']:r['value'] for r in db.execute('SELECT id,value FROM StringIds')}
        ranges=[]
        for raw in db.execute('SELECT * FROM NVTX_EVENTS'):
            row=dict(raw)
            if owner(row['globalTid'])!=actor or row['end'] is None:continue
            text=row.get('text')
            if text is None:text=strings.get(row.get('textId'))
            if text is None:continue
            row['resolved_text']=text;ranges.append(row)
        selected=[r for r in ranges if r['resolved_text']==range_name]
        assert len(selected)==1,'selected NVTX range missing or ambiguous'
        selected=selected[0]
        assert selected['end']>=selected['start']
        assert selected.get('endGlobalTid') in (None,selected['globalTid']),'cross-thread selected range unsupported'
        apis=defaultdict(list)
        api_tables=[t for t in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER') if t in tables]
        assert api_tables,'no exported CUDA launch API table'
        for table in api_tables:
            for raw in db.execute('SELECT * FROM '+table):
                row=dict(raw);base=owner(row['globalTid'])
                if base!=actor:continue
                assert row['end']>=row['start']
                row['table']=table
                apis[(base,row['correlationId'])].append(row)
        kernels=[];unresolved=[];outside=[];all_actor=0;foreign=set()
        for raw in db.execute('SELECT rowid AS event_rowid,* FROM CUPTI_ACTIVITY_KIND_KERNEL'):
            row=dict(raw);base=row['globalPid']
            assert base in processes,'kernel owner missing from process map'
            if base!=actor:foreign.add(base);continue
            all_actor+=1
            assert row['end']>=row['start']
            row['kernel_name']=strings.get(row.get('demangledName'))
            candidates=apis[(base,row['correlationId'])]
            if len(candidates)!=1:
                unresolved.append({'event_rowid':row['event_rowid'],'correlationId':row['correlationId'],
                    'start':row['start'],'end':row['end'],'candidate_API_count':len(candidates),
                    'reason':'missing_or_ambiguous_process_scoped_launch_API'})
                continue
            api=candidates[0]
            if not selected['start']<=api['start']<=api['end']<=selected['end']:
                outside.append(row['event_rowid']);continue
            scopes=[r for r in ranges if r['resolved_text'].startswith('h200_connector/')
                and r['globalTid']==api['globalTid']
                and r.get('endGlobalTid') in (None,r['globalTid'])
                and r['start']<=api['start']<=api['end']<=r['end']]
            innermost=[]
            if scopes:
                duration=min(r['end']-r['start'] for r in scopes)
                innermost=[r['resolved_text'] for r in scopes if r['end']-r['start']==duration]
            row.update(launch_API={'table':api['table'],'start':api['start'],'end':api['end'],
                'globalTid':api['globalTid'],'name':strings.get(api.get('nameId'))},
                innermost_connector_NVTX=innermost[0] if len(innermost)==1 else None,
                connector_NVTX_ambiguous=len(innermost)>1,
                launch_on_selected_range_thread=api['globalTid']==selected['globalTid'])
            kernels.append(row)
        assert all_actor==len(kernels)+len(unresolved)+len(outside)
        covered=[r for r in kernels if r['innermost_connector_NVTX'] is not None]
        return {'classification':'Read-only structural Nsight coverage summary; no new acquisition or scientific qualification',
            'database':str(path),'actor_pid':actor_pid,'exported_actor_globalPid':actor,
            'selected_range':{k:selected[k] for k in ('start','end','globalTid','resolved_text')},
            'actor_kernel_records':all_actor,'selected_launch_correlated_kernel_records':len(kernels),
            'missing_or_ambiguous_API_kernel_records':unresolved,'outside_selected_launch_window_kernel_rowids':outside,
            'foreign_kernel_globalPids':sorted(foreign),'nonempty_selected_launch_coverage':bool(kernels),
            'selected_kernel_duration_sum_ns':sum(r['end']-r['start'] for r in kernels),
            'selected_kernel_active_union_ns':union_ns([(r['start'],r['end']) for r in kernels]),
            'connector_NVTX_linked_kernel_records':len(covered),'unmatched_connector_kernel_records':len(kernels)-len(covered),
            'selected_kernels':kernels,'physical_HBM_bytes':None,'physical_link_bytes':None,
            'GPU_layer_compute_attributed':False,'scientific_qualified':False,
            'scope':'Each actor kernel remains selected, outside, or unresolved. CUDA activity may execute after its host NVTX range; matching uses launch API correlation and same-thread containment. Nested ranges choose a unique innermost connector label; ties remain unmatched. Labels link observed host scopes, not semantic compute, request ownership, physical traffic, clean time, steady-state goodput or calibrated energy. Source/UUID/monitor/trace provenance and request sidecar checks remain separate mandatory gates.'}
    finally:db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True);p.add_argument('--actor-pid',type=int,required=True)
    p.add_argument('--range-name',required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();result=read(args.database,args.actor_pid,args.range_name)
    if args.output:
        with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('selected_kernels','missing_or_ambiguous_API_kernel_records')}))
