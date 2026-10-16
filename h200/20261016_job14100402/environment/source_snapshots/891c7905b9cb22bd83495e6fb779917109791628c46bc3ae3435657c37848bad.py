"""Read actual bounded NVTX windows; keep kernel union separate from wall time."""
import sqlite3
from pathlib import Path


def union_ns(intervals):
    total=0;end=None
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


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
        windows.append({'start_ns':start,'end_ns':end,'instrumented_wall_ms':(end-start)/1e6,
            'kernel_union_ms':union_ns([(k['start'],k['end']) for k in kernels])/1e6,
            'kernel_sum_ms':sum(k['end']-k['start'] for k in kernels)/1e6,
            'launches':len(kernels),'green_context_ids':contexts,
            'restriction_passed':None if expected_green_id is None else contexts==[expected_green_id],
            'graph_kernel_count':sum(bool(k.get('graphNodeId')) for k in kernels),
            'kernels':grouped,'scope':'instrumented ticks 4/5/6; kernel union excludes copies; wall remainder is not entirely host idle'})
    db.close()
    return {'status':'complete','windows':windows,'expected_green_context_id':expected_green_id,
            'restriction_passed':None if expected_green_id is None else all(w['restriction_passed'] for w in windows)}
