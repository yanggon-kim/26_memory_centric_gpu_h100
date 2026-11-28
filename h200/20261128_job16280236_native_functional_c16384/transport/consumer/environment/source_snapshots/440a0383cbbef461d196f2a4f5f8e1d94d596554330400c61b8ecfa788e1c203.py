#!/usr/bin/env python3
"""Upload an existing owned archive privately and verify an independent readback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main(a):
    from huggingface_hub import HfApi,hf_hub_url,get_token
    from huggingface_hub.utils import build_hf_headers
    import requests
    from software_decode_experiments import dump
    assert a.file.is_file() and not a.file.is_symlink()
    expected=file_hash(a.file);size=a.file.stat().st_size
    api=HfApi();info=api.repo_info(a.repo,repo_type='dataset')
    assert info.private,'raw research archive must remain in a private dataset'
    result={'status':'uploading','local_file':str(a.file.resolve()),'bytes':size,'sha256':expected,
            'dataset':a.repo,'remote_path':a.remote_path,'private':True}
    dump(a.proof,result)
    commit=api.upload_file(path_or_fileobj=str(a.file),path_in_repo=a.remote_path,
                          repo_id=a.repo,repo_type='dataset',commit_message='Preserve H200 '+a.remote_path.rsplit('/',1)[-1])
    revision=commit.oid
    remote=api.get_paths_info(a.repo,[a.remote_path],repo_type='dataset',revision=revision)[0]
    assert remote.size==size,'remote object size mismatch'
    url=hf_hub_url(a.repo,a.remote_path,repo_type='dataset',revision=revision)
    h=hashlib.sha256();received=0
    with requests.get(url,headers=build_hf_headers(token=get_token()),stream=True,timeout=(30,120)) as response:
        response.raise_for_status()
        for b in response.iter_content(8*1024**2):
            if b:h.update(b);received+=len(b)
    assert received==size and h.hexdigest()==expected,'independent remote readback mismatch'
    result.update(status='verified',revision=revision,remote_readback_bytes=received,
                  remote_readback_sha256=h.hexdigest(),verified_unix_ns=time.time_ns())
    dump(a.proof,result)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file',type=Path,required=True);p.add_argument('--proof',type=Path,required=True)
    p.add_argument('--repo',default='yang-gon/h100-memory-centric-accelsim-traces')
    p.add_argument('--remote-path',required=True)
    a=p.parse_args()
    try:main(a)
    except Exception as error:
        from software_decode_experiments import dump
        # HTTP errors may contain signed download URLs. Keep status/type without
        # leaking query credentials into the compact public evidence log.
        response=getattr(error,'response',None)
        previous=json.loads(a.proof.read_text()) if a.proof.exists() else {}
        previous.update(status='failed',error_type=type(error).__name__,
                        http_status=getattr(response,'status_code',None),
                        reason='private archive access/upload/readback failed; no credentials or signed URLs recorded')
        dump(a.proof,previous)
        print(json.dumps(previous))
        raise SystemExit(1)
