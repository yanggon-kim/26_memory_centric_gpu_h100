#!/usr/bin/env python3
"""Independent bounded known-payload counter control on one owned spare.

Run in its own coordinator process: H200Session's module-global UUID must not
be shared with the other GPU's reference coordinator. No Slurm submission.
"""
import argparse
from pathlib import Path
import traceback
from h200_priority_campaign import Worker,load,write,sha,limitation

SOURCE_AT_IMPORT=Path(__file__).read_bytes()


def main(args):
    worker=Worker(args.root.resolve(),args.session.resolve())
    assert SOURCE_AT_IMPORT==(worker.script/Path(__file__).name).read_bytes()
    worker.hs.verify_allocation()
    result={'status':'preparing','classification':'known-payload profiler functional control',
        'clean_bandwidth_qualified':False,'decoder_traffic_measured':False}
    try:
        from h200_resident_campaign import build_environment
        build_environment(worker)
        label='P0_native_L2_counter';output=worker.root/'runs'/label
        report=worker.root/'profiles'/(label+'.ncu-rep');report.parent.mkdir(exist_ok=True)
        command=[worker.cuda/'bin/ncu','--replay-mode','kernel','--clock-control','none','--cache-control','none',
            '--profile-from-start','off','--metrics','dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,lts__t_bytes.sum',
            '--export',report,worker.py,worker.script/'shoreline_native.py','--gpu-uuid',worker.uuid,
            '--output',output,'--acquisition','0','--pilot','--profile-case','stream_512mib']
        proc=worker.execute(label,command,timeout=180)
        from h200_ncu_read import resolve_report,parse_csv
        actual=resolve_report(report)
        imported=worker.execute(label+'_import',[worker.cuda/'bin/ncu','--import',actual,'--page','raw',
            '--print-units','base','--csv'],gpu=False,timeout=60)
        values=parse_csv((worker.root/'logs'/(label+'_import.log')).read_text(),include_l2=True)
        native=load(output/'run.json')
        assert proc['status']=='complete' and imported['status']=='complete' and native['status']=='complete'
        assert len(native['cases'])==1 and native['cases'][0]['id']=='stream_512mib'
        assert native['cases'][0]['correctness']=='passed' and len(values)==1
        assert all(v['dram__bytes_read.sum']>0 and v['dram__bytes_write.sum']>=0 and
            v['lts__t_bytes.sum']>0 and v['gpu__time_duration.sum']>0 for v in values.values())
        result.update(status='permission_functional_counter_pilot_passed',gpu_uuid=worker.uuid,
            source_identity=worker.settings['frozen_source_identity'],ranges=values,
            native_run=str(output/'run.json'),native_run_sha256=sha(output/'run.json'),
            counter_artifact=str(actual),counter_artifact_sha256=sha(actual),
            process=str(worker.root/'results/processes'/(label+'.json')),
            scope='One correct known512MiB native control with exact metric units and positive DRAM/L2/time counters; this exercises permissions on this device. Not3independent acquisitions, calibrated service/BW, model traffic, timing normalization or physical link bytes.')
    except BaseException as error:
        result.update(status='blocked_counter_control',error=str(error),traceback=traceback.format_exc())
        limitation(worker.root,'optional_native_counter',str(error),'results/native_counter_disposition.json; logs/P0_native_L2_counter.log',
            'One bounded native512MiB counter/control before model profiling; no clock/cache control or system changes',
            'Retain all native/profiler/reader failure evidence; independent model functional work remains eligible')
    finally:
        write(worker.root/'results/native_counter_disposition.json',result)
        # The parent publishes this completed comparison. Its bounded child
        # cleanup must never interrupt a Git publication inside this process.


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--session',type=Path,required=True)
    main(p.parse_args())
