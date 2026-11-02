#!/usr/bin/env python3
"""E6 (CPU only): per-variant wvSplitK counters from e6_counters rocprofv3 runs of e6_shapes.py --light.

Each shape runs wvsplitk_dev then wvsplitk_n, 5 dispatches each (correctness, warmup, 3 timed); the 3 timed
dispatches are averaged. Prints counter means per shape/variant at 304/232/200/152.

  e6_counters.py <job root> <out.json>
"""
import csv,glob,sys,json,collections,statistics as st
root=sys.argv[1]
out={}
for d in sorted(glob.glob(root+'/e6_counters/*')):
    tag=d.rsplit('/',1)[1]
    fs=glob.glob(d+'/**/*counter_collection.csv',recursive=True)
    if not fs: continue
    rows=collections.OrderedDict()
    for r in csv.DictReader(open(fs[0])):
        k=int(r['Dispatch_Id'])
        e=rows.setdefault(k,{'name':r['Kernel_Name'][:40],'grid':int(r['Grid_Size']),'v':0.0,'dur':int(r['End_Timestamp'])-int(r['Start_Timestamp'])})
        e['v']+=float(r['Counter_Value'])
    # GEMM-like dispatches in order; group by shape via sequence of wvSplitK dispatches
    wv=[e for e in rows.values() if e['name'].startswith('void wvSplitK')]
    res={}
    shapes=['qkv','o_proj','gate_up','down','lm_head']
    for i,s in enumerate(shapes):
        for j,var in enumerate(['dev','n']):
            seg=wv[(2*i+j)*5:(2*i+j)*5+5][2:]
            if len(seg)<3: continue
            res[f'{s}:{var}']={'v':st.mean(e['v'] for e in seg),'dur_us':st.mean(e['dur'] for e in seg)/1e3,'grid':seg[0]['grid']}
    out[tag]=res
json.dump(out,open(sys.argv[2],'w'),indent=1)
cnts=['FETCH_SIZE','TCC_EA0_RDREQ','TCC_EA0_RDREQ_LEVEL','GRBM_GUI_ACTIVE']
for s in ['qkv','o_proj','gate_up','down','lm_head']:
  for var in ['dev','n']:
    line=f'{s:8s}{var:4s}'
    for n in (304,232,200,152):
        cells=[]
        for c in cnts:
            x=out.get(f'n{n}_{c}',{}).get(f'{s}:{var}')
            cells.append('-' if x is None else f"{x['v']:.3g}")
        g=next((out[f'n{n}_{c}'][f'{s}:{var}']['grid'] for c in cnts if f'{s}:{var}' in out.get(f'n{n}_{c}',{})),None)
        line+=f' | {n} g{g} '+' '.join(cells)
    print(line)
