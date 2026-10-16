"""Read actual bounded NVTX windows; keep kernel union separate from wall time."""
import sqlite3
from pathlib import Path


def union_ns(intervals):
    total=0;end=None
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def native_overlap(path):
    """Measure simultaneous kernel intervals in the same native NSYS pass."""
    db=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    ranges=db.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'overlap_ratio%_concurrent' AND end IS NOT NULL").fetchall()
    rows=[]
    for start,end,name in ranges:
        kernels=[dict(r) for r in db.execute('SELECT k.start,k.end,s.value AS symbol FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=?',(start,end))]
        memory=[(k['start'],k['end']) for k in kernels if 'transfer' in k['symbol']]
        compute=[(k['start'],k['end']) for k in kernels if 'arithmetic' in k['symbol']]
        assert len(memory)==4 and len(compute)==int(name.split('_')[1][5:]),'native operation identity/count mismatch'
        intersections=[(max(a,c),min(b,d)) for a,b in memory for c,d in compute if max(a,c)<min(b,d)]
        rows.append({'condition':name,'memory_kernel_count':len(memory),'compute_kernel_count':len(compute),
                     'same_pass_overlap_ms':union_ns(intersections)/1e6,
                     'memory_union_ms':union_ns(memory)/1e6,'compute_union_ms':union_ns(compute)/1e6,
                     'scope':'separate instrumented timing, actual same-pass intersections; not ordinary latency'})
    db.close()
    assert {r['condition'] for r in rows}=={f'overlap_ratio{i}_concurrent' for i in (1,2,4,8,16,32)}
    assert all(sum(r['condition']==f'overlap_ratio{i}_concurrent' for r in rows)==10 for i in (1,2,4,8,16,32))
    return {'status':'complete','conditions':rows,'classification':'measured-real-GPU-instrumented'}


def read(path,expected_green_id=None):
    db=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    ranges=db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='shoreline_ticks_4_6' AND end IS NOT NULL").fetchall()
    assert ranges,'selected decode NVTX window missing'
    windows=[]
    for start,end in ranges:
        kernels=[dict(r) for r in db.execute('SELECT k.*,s.value AS symbol FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=?',(start,end))]
        assert kernels,'selected decode window has no CUDA kernels'
        grouped={}
        for k in kernels:
            name=k['symbol'];v=grouped.setdefault(name,{'launches':0,'duration_ns_sum':0})
            v['launches']+=1;v['duration_ns_sum']+=k['end']-k['start']
        contexts=sorted(set(k['greenContextId'] or 0 for k in kernels))
        cross_context=[(max(a['start'],b['start']),min(a['end'],b['end'])) for i,a in enumerate(kernels)
                       for b in kernels[i+1:] if a['greenContextId'] and b['greenContextId'] and
                       a['greenContextId']!=b['greenContextId'] and max(a['start'],b['start'])<min(a['end'],b['end'])] if len(contexts)>1 else []
        windows.append({'start_ns':start,'end_ns':end,'instrumented_wall_ms':(end-start)/1e6,
            'kernel_union_ms':union_ns([(k['start'],k['end']) for k in kernels])/1e6,
            'kernel_sum_ms':sum(k['end']-k['start'] for k in kernels)/1e6,
            'launches':len(kernels),'green_context_ids':contexts,
            'distinct_green_context_kernel_overlap_ms':union_ns(cross_context)/1e6,
            'restriction_passed':None if expected_green_id is None else contexts==[expected_green_id],
            'graph_kernel_count':sum(bool(k.get('graphNodeId')) for k in kernels),
            'kernels':grouped,'scope':'instrumented ticks 4/5/6; kernel union excludes copies; wall remainder is not entirely host idle'})
    db.close()
    return {'status':'complete','windows':windows,'expected_green_context_id':expected_green_id,
            'restriction_passed':None if expected_green_id is None else all(w['restriction_passed'] for w in windows)}
