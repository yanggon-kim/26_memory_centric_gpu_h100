"""Read-only CUDA copy-event analysis candidate, separate from GPU programs.

The bytes field is reported literally per exported activity record. It is not
physical HBM/NVLink traffic. copyCount is retained and never used to silently
multiply bytes. Missing copy tables are unavailable, not measured zeros.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

from h200_connector_timeline import read,union_ns


def read_copies(path,actor_pid,range_name):
    path=Path(path).resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest()
    kernel_scope=read(path,actor_pid,range_name)
    actor=kernel_scope['exported_actor_globalPid'];selected=kernel_scope['selected_range']
    result={'classification':'Read-only CUDA copy activity accounting; no new acquisition or scientific qualification',
        'database':str(path),'database_sha256':digest,'actor_pid':actor_pid,'selected_range':selected,
        'copy_table_available':False,'actor_copy_records':None,'selected_launch_correlated_copy_records':None,
        'copy_record_bytes_field_sum':None,'unresolved_launch_copy_records':[],
        'outside_selected_launch_window_copy_rowids':[],'foreign_copy_globalPids':[],
        'selected_copies':[],'by_copy_kind':{},'scientific_qualified':False,
        'physical_HBM_bytes':None,'physical_link_bytes':None,'primary_J_per_token':None,
        'scope':'Literal CUDA activity-record bytes and durations, not physical traffic, calibrated bandwidth, per-layer compute or a clean performance comparison. API correlation is process-scoped; host NVTX association uses launch thread/time. Unresolved and outside-range copies remain explicit. Event bytes are never multiplied by copyCount, whose interpretation remains unqualified.'}
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    try:
        db.execute('PRAGMA query_only=ON')
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' not in tables:return result
        result['copy_table_available']=True
        processes={r['globalPid']:r['pid'] for r in db.execute('SELECT globalPid,pid FROM PROCESSES')}
        def owner(tid):
            assert type(tid) is int and tid & ~0xffffff in processes
            return tid & ~0xffffff
        strings={r['id']:r['value'] for r in db.execute('SELECT id,value FROM StringIds')}
        kinds={r['id']:{'name':r['name'],'label':r['label']} for r in db.execute('SELECT id,name,label FROM ENUM_CUDA_MEMCPY_OPER')} if 'ENUM_CUDA_MEMCPY_OPER' in tables else {}
        apis=defaultdict(list)
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
            if table not in tables:continue
            for raw in db.execute('SELECT * FROM '+table):
                row=dict(raw)
                if owner(row['globalTid'])==actor:
                    row['table']=table;apis[row['correlationId']].append(row)
        ranges=[]
        for raw in db.execute('SELECT * FROM NVTX_EVENTS'):
            row=dict(raw)
            if owner(row['globalTid'])!=actor or row['end'] is None:continue
            text=row.get('text') if row.get('text') is not None else strings.get(row.get('textId'))
            if text is not None and text.startswith('h200_connector/'):
                row['resolved_text']=text;ranges.append(row)
        copies=[];unresolved=[];outside=[];foreign=set();count=0
        for raw in db.execute('SELECT rowid AS event_rowid,* FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
            row=dict(raw);assert row['globalPid'] in processes
            if row['globalPid']!=actor:foreign.add(row['globalPid']);continue
            count+=1
            assert row['end']>=row['start'] and type(row['bytes']) is int and row['bytes']>=0
            candidates=apis[row['correlationId']]
            if len(candidates)!=1:
                unresolved.append({'event_rowid':row['event_rowid'],'bytes_field':row['bytes'],
                    'candidate_API_count':len(candidates),'reason':'missing_or_ambiguous_process_scoped_copy_API'})
                continue
            api=candidates[0]
            if not selected['start']<=api['start']<=api['end']<=selected['end']:
                outside.append(row['event_rowid']);continue
            scopes=[r for r in ranges if r['globalTid']==api['globalTid']
                and r.get('endGlobalTid') in (None,r['globalTid'])
                and r['start']<=api['start']<=api['end']<=r['end']]
            closest=[]
            if scopes:
                duration=min(r['end']-r['start'] for r in scopes)
                closest=[r['resolved_text'] for r in scopes if r['end']-r['start']==duration]
            row.update(copy_kind=kinds.get(row['copyKind'],{'name':'unresolved_copy_kind_'+str(row['copyKind']),'label':None}),
                launch_API={'table':api['table'],'start':api['start'],'end':api['end'],'globalTid':api['globalTid'],
                    'name':strings.get(api.get('nameId'))},
                innermost_connector_NVTX=closest[0] if len(closest)==1 else None,
                connector_NVTX_ambiguous=len(closest)>1)
            copies.append(row)
        assert count==len(copies)+len(unresolved)+len(outside)
        groups=defaultdict(list)
        for row in copies:groups[row['copy_kind']['name']].append(row)
        result.update(actor_copy_records=count,selected_launch_correlated_copy_records=len(copies),
            copy_record_bytes_field_sum=sum(r['bytes'] for r in copies),selected_copies=copies,
            unresolved_launch_copy_records=unresolved,outside_selected_launch_window_copy_rowids=outside,
            foreign_copy_globalPids=sorted(foreign),
            copy_duration_sum_ns=sum(r['end']-r['start'] for r in copies),
            copy_active_union_ns=union_ns([(r['start'],r['end']) for r in copies]),
            copy_count_not_single_record_rowids=[r['event_rowid'] for r in copies if r.get('copyCount') not in (None,1)],
            unmatched_connector_copy_records=sum(r['innermost_connector_NVTX'] is None for r in copies),
            by_copy_kind={kind:{'copy_records':len(rows),'bytes_field_sum':sum(r['bytes'] for r in rows),
                'duration_sum_ns':sum(r['end']-r['start'] for r in rows),
                'active_union_ns':union_ns([(r['start'],r['end']) for r in rows])} for kind,rows in groups.items()})
        return result
    finally:
        db.close()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,'source database changed during read-only analysis'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True);p.add_argument('--actor-pid',type=int,required=True)
    p.add_argument('--range-name',required=True);p.add_argument('--output',type=Path)
    args=p.parse_args();data=read_copies(args.database,args.actor_pid,args.range_name)
    if args.output:
        with args.output.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ('selected_copies','unresolved_launch_copy_records')}))
