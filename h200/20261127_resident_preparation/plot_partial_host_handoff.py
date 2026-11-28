#!/usr/bin/env python3
"""Plot validated raw host windows, never isolated transfer/GPU durations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from h200_partial_host_handoff import derive


def main(root):
    root=Path(root).resolve();checked=derive(root,0)
    original_path=root/'results/partial_a0_host_handoff_timeline.json'
    original=json.loads(original_path.read_text())
    assert checked['raw_host_summary_ms']==original['raw_host_summary_ms']
    producer=json.loads(Path(checked['provenance']['producer']['run']).read_text())
    consumer=json.loads(Path(checked['provenance']['consumer']['run']).read_text())
    png=root/'analysis/partial_a0_host_handoff_timeline.png'
    pdf=png.with_suffix('.pdf')
    receipt=root/'results/partial_host_handoff_figure.json'
    assert not any(path.exists() for path in (png,pdf,receipt))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(1,2,figsize=(13,6.6),sharey=True,
        gridspec_kw={'width_ratios':[1.55,1]})
    colors={'producer':'#687b8e','first':'#e58c37','later':'#2677b8','send':'#8b4795'}
    labels=[]
    for p,c in zip(producer['trials'],consumer['trials']):
        if p['warmup']:continue
        row=len(labels);labels.append('Trial '+str(p['trial']))
        anchor=p['final_step_return_unix_ns']
        offset=lambda value:(value-anchor)/1e6
        for axis in axes:
            axis.broken_barh([(offset(p['submit_unix_ns']),-offset(p['submit_unix_ns']))],
                (row-.29,.58),facecolors=colors['producer'])
            for index,span in enumerate(c['host_step_intervals']):
                axis.broken_barh([(offset(span['start_unix_ns']),
                    (span['end_unix_ns']-span['start_unix_ns'])/1e6)],
                    (row-.29,.58),facecolors=colors['first'] if index==0 else colors['later'])
            axis.scatter([offset(p['all_native_sends_observed_unix_ns'])],[row],
                marker='D',s=24,color=colors['send'],zorder=5)
    for axis in axes:
        axis.axvline(0,color='#333333',linestyle='--',linewidth=.8)
        axis.set_xlabel('Host time from producer first-token return (ms)')
        axis.grid(axis='x',alpha=.2);axis.set_axisbelow(True)
        axis.set_ylim(-.7,len(labels)-.3);axis.invert_yaxis()
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(labels)),labels)
    axes[0].set_title('Full two-process harness window',fontsize=11)
    axes[1].set_xlim(-5,200);axes[1].set_title('Handoff detail',fontsize=11)
    fig.suptitle('Raw host timeline: ten requests from one paired acquisition',fontsize=14,y=.98)
    fig.text(.5,.922,'Incomplete cohort; producer clock gate rejected. No qualified slowdown ratio or serving goodput.',
        ha='center',fontsize=10,color='#9b302d')
    legend=[Patch(color=colors['producer'],label='Producer submit to first-token return'),
        Patch(color=colors['first'],label='Consumer first handoff call'),
        Patch(color=colors['later'],label='Following 62 consumer calls'),
        Line2D([0],[0],marker='D',color='none',markerfacecolor=colors['send'],
            markeredgecolor=colors['send'],label='All sends observed by host poll')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,
        frameon=False,fontsize=9)
    fig.text(.5,.02,'Host windows include CPU/IPC effects and overlap. Poll timestamps are observation bounds, not wire time. No GPU kernel durations are plotted.',
        ha='center',fontsize=8.4)
    fig.subplots_adjust(top=.86,bottom=.22,left=.07,right=.985,wspace=.15)
    fig.savefig(png,dpi=160);fig.savefig(pdf);plt.close(fig)
    record={'classification':'CPU figure from archived, source-verified raw host observations',
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reader_sha256':checked['implementation_sha256'],
        'input':str(original_path),'input_sha256':hashlib.sha256(original_path.read_bytes()).hexdigest(),
        'matplotlib':matplotlib.__version__,'timed_paired_requests':10,
        'artifacts':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (png,pdf)},
        'GPU_execution':False,'performance_qualified':False,'qualified_ratio':None,
        'scope':'Original actual call windows shown separately, including inter-call gaps. Host interval width is not isolated GPU or physical communication duration; no additive breakdown. One acquisition and failed producer clocks remain binding.'}
    with receipt.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    print(json.dumps({'png':str(png),'pdf':str(pdf),'GPU_execution':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
