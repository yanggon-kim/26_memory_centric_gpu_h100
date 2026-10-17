#!/usr/bin/env python3
"""Export scoped research figures from saved measurements and explicit scenarios."""
import argparse
import json
from pathlib import Path
import statistics


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output=root/'analysis/figures';output.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(output/(name+'.'+suffix),dpi=180)
        plt.close(fig)
    runs=[json.loads(p.read_text()) for p in sorted((root/'runs').glob('J4_native_v4_acq*/run.json'))]
    if len(runs)==3 and all(d['status']=='complete' for d in runs):
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for pattern in ('stream','stride17','gather','paged128'):
            xs=[];ys=[]
            for size in (1,28,32,48,60,96,512):
                cases=[next(c for c in d['cases'] if c['id']==f'{pattern}_{size}mib') for d in runs]
                values=[statistics.median(t['gpu_event_ms']/t['repetitions'] for t in c['trials']) for c in cases]
                xs.append(size*2);ys.append(statistics.median(values))
            axes[0].plot(xs,ys,'o-',label=pattern)
        axes[0].set(xscale='log',yscale='log',xlabel='Input + output footprint (MiB)',ylabel='GPU-event ms / launch',title='Independent native access patterns')
        axes[0].axvline(60,color='k',linestyle=':',label='queried L2 60 MiB');axes[0].legend(fontsize=8)
        for chains in (1,4,32):
            xs=[];ys=[]
            for size in (.03125,48,96):
                cases=[next(c for c in d['cases'] if c['id']==f'pointer_{size:g}mib_chains{chains}') for d in runs]
                values=[statistics.median(t['gpu_event_ms']/t['repetitions']*1e6/c['metadata']['dependent_steps'] for t in c['trials']) for c in cases]
                xs.append(size);ys.append(statistics.median(values))
            axes[1].plot(xs,ys,'o-',label=f'{chains} independent chains')
        axes[1].set(xscale='log',xlabel='Pointer working set (MiB)',ylabel='ns / dependent step per chain',title='Full randomized cycle; concurrent chains')
        axes[1].axvline(60,color='k',linestyle=':');axes[1].legend(fontsize=8)
        fig.suptitle('Raw native measurements: clock/physical-service joins are separate gates',fontsize=11)
        save(fig,'native_service_raw')
    from shoreline_statistics import latency_bounds,sensitivity
    fig,axes=plt.subplots(1,3,figsize=(12,3.8));tables=[]
    for ax,(name,C,M) in zip(axes,[('Memory-heavy',.1,.9),('Balanced',.55,.55),('Compute-heavy',.9,.1)]):
        bounds=latency_bounds(1,C,M)
        values=[]
        for c in (1,.75,.5,.25):
            row=[]
            for r in (.7,1,1.2,1.4):
                s=sensitivity(bounds,c,r,.1)
                row.append(1/max(s['same_residual_same_eta_time']))
                tables.append({'regime':name,'assumed_T':1,'assumed_C':C,'assumed_M':M,**s})
            values.append(row)
        im=ax.imshow(values,vmin=.35,vmax=1.4,cmap='RdYlGn',aspect='auto')
        ax.set(xticks=range(4),xticklabels=(.7,1,1.2,1.4),yticks=range(4),yticklabels=(1,.75,.5,.25),xlabel='Delivered service ratio r',ylabel='Effective compute ratio c',title=name)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{values[i][j]:.2f}',ha='center',va='center')
    fig.suptitle('Illustrative sensitivity only: f=0.10; conservative coupled speedup; no H200 prediction',fontsize=10)
    save(fig,'illustrative_joint_sensitivity')
    (output/'illustrative_joint_inputs.json').write_text(json.dumps({'classification':'illustrative analytical inputs; not fitted H200 measurements','cases':tables},indent=2)+'\n')
    print(json.dumps({'output':str(output),'figures':sorted(p.name for p in output.glob('*.png'))}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
