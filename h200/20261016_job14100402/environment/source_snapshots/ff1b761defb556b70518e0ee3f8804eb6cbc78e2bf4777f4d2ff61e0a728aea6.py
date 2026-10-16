#!/usr/bin/env python3
"""Negative and boundary contracts for new statistical/replay qualification."""
import math
from shoreline_statistics import latency_bounds,sensitivity,hierarchical_interval,paired_ratio


def rejected(function,*args):
    try:function(*args)
    except (AssertionError,ValueError):return
    raise AssertionError('invalid inputs accepted')


def main():
    import tempfile,json
    from pathlib import Path
    from shoreline_evidence import qualify_replays
    import os,threading
    from types import SimpleNamespace
    from h200_session import ProcessMonitor
    monitor=ProcessMonitor.__new__(ProcessMonitor);monitor.process=SimpleNamespace(pid=os.getpid())
    assert monitor.owned(os.getpid()) is True
    observed=[]
    def thread_identity():observed.append(monitor.owned(threading.get_native_id()))
    thread=threading.Thread(target=thread_identity);thread.start();thread.join()
    assert observed==[True], 'owned CUDA thread IDs must resolve through Tgid'
    assert monitor.owned(1) is False
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);base=root/'capture';base.mkdir()
        out=base/'pass000';out.mkdir()
        pub=base/'pass000.json'
        expected={'prompt_sha256':'fixed','trials':[{'output_ids':[[1,2,3]]}]}
        (out/'run.json').write_text(json.dumps({'status':'complete','cases':[{
            'prompt_sha256':'fixed','trials':[{'output_ids':[[1,2,3]]}]*3}]}))
        (out/'selected_interval.json').write_text(json.dumps({'status':'complete','ticks':[4,5,6]}))
        pub.write_text(json.dumps({'status':'complete','output':str(out)}))
        assert qualify_replays(base,expected,root)['passed']
        pub.write_text(json.dumps({'status':'running','output':str(out)}))
        assert not qualify_replays(base,expected,root)['passed']
        pub.write_text(json.dumps({'status':'complete','output':str(out)}))
        (base/'pass001').mkdir()
        assert not qualify_replays(base,expected,root)['passed']
    b=latency_bounds(10,2,8)
    assert b['O_interval']==[0,2]
    for e in b['coupled_endpoints']:
        assert math.isclose(10,2+8-e['X']+e['O'])
        assert 0<=e['X']<=2
    assert not latency_bounds(1,2,8)['feasible']
    rejected(latency_bounds,10,-1,2)
    losing=sensitivity(b,1,.7,.2)
    assert not losing['memory_service_improves'] and losing['M_new']>8
    assert sensitivity(b,1,1.2,0)['memory_service_improves']
    assert sensitivity(b,1,1,0)['same_residual_same_eta_time']==[10,10]
    rejected(hierarchical_interval,[[1]*10]*2)
    rejected(hierarchical_interval,[[1]*9]*3)
    rejected(paired_ratio,[[1]*10]*3,[[0]*10]*3)
    equal=paired_ratio([[1]*10]*3,[[1]*10]*3,draws=100)
    assert equal['ci95']==[1,1] and equal['passes_5_percent_noninferiority']
    losing=paired_ratio([[1.1]*10]*3,[[1]*10]*3,draws=100)
    assert not losing['passes_5_percent_noninferiority']
    print('PASS: acquisition sufficiency, positive rates, coupled bounds, impossible model, losing bandwidth and noninferiority controls')


if __name__=='__main__':main()
