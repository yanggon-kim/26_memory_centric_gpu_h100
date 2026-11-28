#!/usr/bin/env python3
"""Plot retained acquisition medians without promoting failed timing gates."""
import argparse
from pathlib import Path
import statistics
from h200_priority_campaign import write,sha


def plot(root,records):
    selected=[r for r in records if not r['pilot'] and not r.get('suffix')
              and r['status']=='complete' and len(r.get('step_ms',[]))==10]
    if not selected:return None
    assert len({r['gpu_uuid'] for r in selected})==1
    assert len({r['source_identity'] for r in selected})==1
    assert len({r.get('repair_route','original') for r in selected})==1
    shapes={(r['batch'],r['context']) for r in selected}
    assert len(shapes)==1, 'one plotted curve must contain one workload shape'
    batch,context=next(iter(shapes))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for acquisition in sorted({r['acquisition'] for r in selected}):
        rows=sorted([r for r in selected if r['acquisition']==acquisition and not r['stock']],key=lambda r:r['actual_sm_count'])
        if rows:
            ax.plot([r['actual_sm_count'] for r in rows],[statistics.median(r['step_ms']) for r in rows],
                    'o-',label=f'Green path, acquisition {acquisition}')
        stock=[r for r in selected if r['acquisition']==acquisition and r['stock']]
        for row in stock:
            ax.scatter([row['actual_sm_count']],[statistics.median(row['step_ms'])],marker='x',s=75,
                       label=f'Stock full device, acquisition {acquisition}')
    ax.set(xlabel='Actual granted SMs (decoder placement proved separately)',
           ylabel='Raw median wall ms / decode tick',
           title=('Conditioned ' if selected[0].get('conditioning') else '')+f'Qwen2.5-32B BF16 eager decode · batch {batch} / context {context}\nRetained observations; see qualification gates before inference')
    ax.set_xticks([40,56,80,96,112,132]);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.text(.01,.005,'No confidence bounds or NVIDIA thresholds are inferred from clock-rejected or incomplete acquisitions.',fontsize=8)
    path=root/'analysis/eager_raw_sm_curve.png';fig.savefig(path,dpi=180);plt.close(fig)
    write(root/'results/eager_raw_sm_plot.json',{'classification':'visualization of retained measured observations',
          'plot':str(path),'plot_sha256':sha(path),'batch':batch,'context':context,'points':[{'id':r['id'],'actual_sm_count':r['actual_sm_count'],
              'raw_median_ms_per_tick':statistics.median(r['step_ms']),'qualified':r.get('qualified',False),
              'run':r['run'],'run_sha256':r['run_sha256']} for r in selected],
          'scope':'each acquisition kept separate; ten trial medians only; no error bars, cross-device pooling or threshold inference'})
    return str(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    from h200_review_allocation import case_records
    records,issues,interrupted=case_records(a.root)
    print(plot(a.root,records))
