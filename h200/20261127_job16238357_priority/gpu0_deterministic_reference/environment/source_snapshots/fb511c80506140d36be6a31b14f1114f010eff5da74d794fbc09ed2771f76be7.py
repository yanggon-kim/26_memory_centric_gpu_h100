#!/usr/bin/env python3
"""Select exactly the driver-visible H200 pair inside an owned Slurm allocation.

Fails closed when the driver enumerates more than the two allocated devices;
never guesses UUIDs from Slurm ordinals or NVIDIA minor numbers.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess


def validate(state, inventory, job_id, hostname, uid):
    fields=dict(item.split('=',1) for item in state.split() if '=' in item)
    assert job_id.isdigit() and fields['JobId']==job_id
    assert fields['JobState']=='RUNNING' and fields['NumNodes']=='1'
    assert fields['NodeList'].split('.')[0]==hostname.split('.')[0]
    assert fields['UserId'].endswith(f'({uid})')
    assert 'gres/gpu:h200=2' in fields['AllocTRES'].split(',')
    rows=[[x.strip() for x in row] for row in csv.reader(io.StringIO(inventory)) if row]
    assert len(rows)==2,'Driver visibility must be exactly the allocated pair; refusing ordinal guesses'
    assert all(len(row)==2 and row[0].startswith('GPU-') and row[1]=='NVIDIA H200' for row in rows)
    uuids=[row[0] for row in rows]
    assert len(set(uuids))==2
    return uuids


def main(output):
    job_id=os.environ.get('SLURM_JOB_ID','')
    assert job_id.isdigit(),'requires an owned allocated Slurm step'
    state=subprocess.check_output(['scontrol','show','job','-o',job_id],text=True,timeout=30)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True,timeout=30)
    uuids=validate(state,inventory,job_id,socket.gethostname(),os.getuid())
    record={'slurm_job_id':job_id,'hostname':socket.gethostname(),'gpu_uuids':uuids,
            'inherited_cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'selected_cuda_visible_devices':','.join(uuids),'slurm_state':state,
            'driver_inventory':inventory,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'owned RUNNING allocation with exactly two driver-visible H200s; explicit UUID selection'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'allocated_pair_validated','gpu_uuids':uuids,'manifest':str(output)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
