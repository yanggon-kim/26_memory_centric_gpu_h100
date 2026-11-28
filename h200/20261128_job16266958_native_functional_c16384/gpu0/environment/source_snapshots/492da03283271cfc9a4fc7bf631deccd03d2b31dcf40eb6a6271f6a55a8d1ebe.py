#!/usr/bin/env python3
"""Read-only diagnostic of saved clocks inside the full declared decode window.

No samples/trials are discarded, and no scientific qualification is changed.
"""
import argparse
from pathlib import Path
import h200_priority_campaign as campaign


def audit(case_path):
    case=campaign.load(case_path)
    run=Path(case['run'])
    process_path=case_path.parent/'processes'/(case['id']+'.json')
    process=campaign.load(process_path)
    monitor=Path(process['monitor']['source'])
    rows=[__import__('json').loads(line) for line in monitor.read_text().splitlines()]
    result={'classification':'CPU clock-window diagnostic; original quality gates unchanged',
            'case':case['id'],'case_sha256':campaign.sha(case_path),'run_sha256':campaign.sha(run),
            'monitor_sha256':campaign.sha(monitor),'source_sha256':campaign.sha(Path(__file__)),
            'gpu_uuid':case['gpu_uuid'],'trials':[],
            'scope':'all saved interior samples inclusively inside wall_start_unix_ns/wall_end_unix_ns; prefill is outside these timestamps; endpoint readouts retained separately'}
    for i,trial in enumerate(campaign.load(run)['cases'][0]['trials']):
        selected=[row for row in rows if trial['wall_start_unix_ns']<=row['unix_ns']<=trial['wall_end_unix_ns']]
        metrics={}
        for name in ('sm_clock_mhz','memory_clock_mhz'):
            values=[r.get(name) for r in selected]
            valid=bool(values) and all(isinstance(v,(int,float)) and v>0 for v in values)
            metrics[name]={'minimum':min(values) if valid else None,'maximum':max(values) if valid else None,
                           'max_min_ratio':max(values)/min(values) if valid else None}
        result['trials'].append({'trial':i,'interior_samples':len(selected),'interior_clocks':metrics,
            'before':{m:trial.get('hardware_before',{}).get(m) for m in metrics},
            'after':{m:trial.get('hardware_after',{}).get(m) for m in metrics}})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-result',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();campaign.write(args.output,audit(args.case_result))
