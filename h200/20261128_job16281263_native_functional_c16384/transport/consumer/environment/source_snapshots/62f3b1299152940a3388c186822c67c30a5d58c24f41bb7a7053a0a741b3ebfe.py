#!/usr/bin/env python3
"""Reaccount preserved traffic using matching durable model headers; no recapture."""
import argparse
import hashlib
import json
from pathlib import Path
from shoreline_accounting import useful


def main(a):
    source=a.source_root.resolve();current=a.model_root.resolve();rows=[]
    def load(p):return json.loads(p.read_text())
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((source/'profiles').glob('J2_*_reader_v3.json')):
        d=load(p)
        if d['status']!='complete':continue
        config=load(source/'configs'/(d['model']+'_smoke.json'))
        header=current/'environment/model_headers'/(d['model']+'_v2.json')
        h=load(header)
        assert h['model_id']==config['model_id'] and h['model_revision']==config['model_revision']
        assert h['model_config_sha256']==config['config_sha256']
        q=load(source/'results/qualifications'/f'{d["model"]}_{d["engine"]}_b{d["batch"]}_c{d["context"]}_acq00.json')
        reference=source/q['output']/'run.json';ids=load(reference)['cases'][0]['trials'][0]['output_ids']
        tokens={tick:[v[tick-1] for v in ids] for tick in (4,5,6)}
        inputs=None
        if d['layer'] is not None:
            from shoreline_timeline import union_ns
            candidate={}
            for publication in d['native_replay_publications']:
                native=Path(load(source/publication['publication'])['output'])/'shoreline_worker_metadata.json'
                if not native.exists():continue
                for worker in load(native):
                    for entry in worker.get('layer_inputs',[]):
                        if entry['layer']!=d['layer']:continue
                        ranges=[(t['data_ptr'],t['data_ptr']+t['logical_tensor_bytes']) for t in entry['tensors']
                                if t['name'] in ('hidden_states','residual') and t['contiguous']]
                        if ranges:candidate[entry['tick']]=union_ns(ranges)
            if set(candidate)=={4,5,6}:inputs=candidate
        u=useful(header,d['batch'],d['context'],d['layer'],inputs,tokens)
        rows.append({'id':d['id'],'classification':'historical instrumented physical bytes; analytical demand reprocessed without GPU execution',
            'source':str(p),'source_sha256':sha(p),'header_source':str(header),'header_sha256':sha(header),
            'reference_source':str(reference),'reference_sha256':sha(reference),
            'D_read':d['hbm_read_bytes'],'D_write':d['hbm_write_bytes'],'logical':u,
            'D_read_over_U_tick_sum':d['hbm_read_bytes']/u['U_tick_sum'] if u['U_tick_sum'] else None,
            'D_read_over_U_window_unique':d['hbm_read_bytes']/u['U_window_unique'] if u['U_window_unique'] else None,
            'ordinary_clock_timing_join':None})
    assert not a.output.resolve().is_relative_to(source), 'preserve original session derived rows'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'reader_version':2,'source_session':str(source),'cases':rows,
        'scope':'Source/revision-checked analytical reprocessing; no new traffic measurements or cross-allocation timing join'},indent=2)+'\n')
    print(json.dumps({'reaccounted_captures':len(rows),'output':str(a.output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
