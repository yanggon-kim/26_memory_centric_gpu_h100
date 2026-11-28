#!/usr/bin/env python3
"""Plot only a completed CPU-reviewed curve and its acquisition-aware intervals."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(root):
    review_path=root/'results/resident_curve_CPU_review.json'
    review=json.loads(review_path.read_text())
    curve_path=Path(review['curve'])
    assert review['status']=='qualified' and review['qualified_timing_curve']
    assert sha(curve_path)==review['curve_sha256']
    curve=json.loads(curve_path.read_text())
    assert curve['complete_qualified_curve'] and all(p['qualified'] for p in curve['points'])
    assert review['protocol']==curve['protocol']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=sorted(curve['points'],key=lambda p:p['actual_sm_count'])
    counts=[p['actual_sm_count'] for p in rows]
    def values(key):
        central=[p[key]['value'] for p in rows]
        lower=[p[key]['value']-p[key]['ci95'][0] for p in rows]
        upper=[p[key]['ci95'][1]-p[key]['value'] for p in rows]
        assert all(x>=0 for x in lower+upper)
        return central,[lower,upper]
    fig,axes=plt.subplots(1,2,figsize=(11,5.1))
    central,errors=values('latency_ms')
    axes[0].errorbar(counts,central,yerr=errors,fmt='o-',capsize=4,label='Restricted eager path')
    stock=curve['unrestricted_stock'];interval=stock['latency_ms']
    axes[0].errorbar([stock['actual_sm_count']],[interval['value']],
        yerr=[[interval['value']-interval['ci95'][0]],[interval['ci95'][1]-interval['value']]],
        fmt='D',color='black',capsize=4,label='Unrestricted eager stock')
    axes[0].set_ylabel('Wall ms / pure decode tick')
    axes[0].legend(fontsize=8)
    central,errors=values('ratio')
    axes[1].errorbar(counts,central,yerr=errors,fmt='o-',capsize=4)
    axes[1].set_ylabel('Latency / paired unrestricted eager stock')
    for margin,color in ((.05,'#43864a'),(.10,'#c67b19'),(.25,'#a85454')):
        axes[1].axhline(1+margin,color=color,linewidth=.8,linestyle='--',label=f'{margin:.0%} margin')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(counts);axis.set_xlabel('Actual granted SMs');axis.grid(alpha=.2)
    conditioned=bool(curve.get('conditioning'))
    fig.suptitle(f"Qwen2.5-32B BF16 · batch {curve['batch']} · context {curve['context']}\n"+
                 ('Conditioned eager decode' if conditioned else 'Resident eager decode')+
                 ' · 3 acquisitions × 10 timed trials',fontsize=12)
    fig.text(.025,.015,'95% hierarchical bootstrap intervals; tested SM levels only. '+
        ('Fixed untimed 1 s buffer conditioning changes cache history and TTFT. ' if conditioned else '')+
        'Optimized graph restriction remains blocked.',fontsize=8,wrap=True)
    fig.tight_layout(rect=(0,.09,1,.88))
    output=root/'analysis/eager_qualified_sm_curve.png'
    fig.savefig(output,dpi=180);plt.close(fig)
    receipt={'classification':'visualization of CPU-qualified measured eager curve',
        'plot':str(output),'plot_sha256':sha(output),'curve':str(curve_path),'curve_sha256':sha(curve_path),
        'review':str(review_path),'review_sha256':sha(review_path),'protocol':curve['protocol'],
        'script_sha256':sha(Path(__file__)),'thresholds':curve['thresholds'],
        'scope':'Same physical UUID/frozen source; 95% intervals from saved acquisition-aware statistics. Conditional eager scope only; no optimized, area, traffic or cross-device inference.'}
    (root/'results/eager_qualified_sm_plot.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();print(plot(args.root.resolve()))
