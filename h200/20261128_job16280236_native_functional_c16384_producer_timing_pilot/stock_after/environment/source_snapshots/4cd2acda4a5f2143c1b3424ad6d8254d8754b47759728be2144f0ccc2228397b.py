#!/usr/bin/env python3
"""Standalone figures from qualified H200 run JSON; no historical GPU overlays."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
from h200_session import DEFAULT_ROOT,WORKSPACE,dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    root=parser.parse_args().root.resolve()
    os.environ['MPLCONFIGDIR']=str(WORKSPACE/'cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((root/'results/fixed_bandwidth_case_manifest.json').read_text())
    valid=[case for case in manifest['cases'] if case['status']=='complete']
    models=[name for name in ['bridge','qwen7b','mistral7b','qwen14b','qwen32b'] if any(c['model']==name for c in valid)]
    fig,axes=plt.subplots((len(models)+1)//2,2,figsize=(11,3.6*((len(models)+1)//2)),squeeze=False)
    colors={1:'#245a81',4:'#b65c27',8:'#477849'}
    sources=[]
    for ax,model in zip(axes.flat,models):
        selected=[c for c in valid if c['model']==model]
        for batch in sorted({c['batch'] for c in selected}):
            values=[]
            for case in sorted([c for c in selected if c['batch']==batch],key=lambda c:c['context']):
                source=root/'runs'/case['id']/'run.json';data=json.loads(source.read_text())
                actual=data['cases'][0]
                trials=[t['wall_ms']/(actual['generated_tokens']-1) for t in actual['trials']]
                med=statistics.median(trials)
                values.append((case['context'],med,med-min(trials),max(trials)-med))
                sources.append({'path':str(source.relative_to(root)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
            ax.errorbar([v[0] for v in values],[v[1] for v in values],
                        yerr=[[v[2] for v in values],[v[3] for v in values]],
                        marker='o',capsize=3,color=colors.get(batch),label=f'Batch {batch}')
        ax.set_title(model);ax.set_xscale('log',base=2)
        ax.set_xticks([128,1024,4096],['128','1024','4096'])
        ax.set_xlabel('Initial prompt context (tokens)');ax.set_ylabel('ms per batch decode step')
        ax.grid(alpha=.2);ax.legend(frameon=False)
    for ax in list(axes.flat)[len(models):]:ax.set_visible(False)
    fig.suptitle('H200 controlled BF16 graph decode · measured-real-GPU')
    fig.text(.5,.01,'Three warmups and three trials; error bars show trial min–max. 64 generated tokens, 63 timed steps; prefill excluded.\nOnly qualified cases are plotted. Different models are different workloads.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95))
    for suffix in ['png','pdf']:fig.savefig(root/'results'/('decode_latency.'+suffix),dpi=180)
    plt.close(fig)
    energy=[]
    with (root/'results/measurements.csv').open() as stream:
        for row in csv.DictReader(stream):
            if (row['selected_case_status']=='complete' and row['external_instrumentation']=='False'
                and row['design']=='vllm_graph' and row['joules_per_token']!='NA'):
                source=root/row['source_run_json'];data=json.loads(source.read_text())
                trials=[t['telemetry']['joules_per_token'] for t in data['cases'][0]['trials']]
                med=statistics.median(trials)
                energy.append((row['run_id'],med,med-min(trials),max(trials)-med))
                sources.append({'path':str(source.relative_to(root)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    if energy:
        fig,ax=plt.subplots(figsize=(10,max(3,.5*len(energy)+1.5)))
        labels=[v[0].replace('_graph_',' ').replace('_energy_', ' ') for v in energy]
        ax.barh(labels,[v[1] for v in energy],xerr=[[v[2] for v in energy],[v[3] for v in energy]],color='#245a81',capsize=3)
        ax.set_xlabel('Whole-board joules per generated decode token')
        ax.set_title('H200 BF16 graph decode energy · measured-real-GPU')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.invert_yaxis()
        fig.text(.5,.015,'Three trials; median and min–max. Qualified ≥5 s windows; no idle subtraction.\nDifferent generated lengths and model/batch/context workloads; no component-energy or serving claim.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.08,1,1))
        for suffix in ['png','pdf']:fig.savefig(root/'results'/('board_energy.'+suffix),dpi=180)
        plt.close(fig)
    dump(root/'results/plot_manifest.json',{'classification':'measured-real-GPU',
         'figures':['decode_latency','board_energy'] if energy else ['decode_latency'],
         'sources':sources,'statistic':'median and min/max of three trials',
         'scope':'controlled BF16 graph batch-step latency; no p99 serving or architectural comparison'})


if __name__=='__main__':main()
