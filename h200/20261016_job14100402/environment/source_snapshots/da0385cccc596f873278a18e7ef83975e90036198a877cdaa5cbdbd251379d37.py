#!/usr/bin/env python3
"""Resolve this pinned repository's legacy tokenizer LFS alias with SHA256 proof."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def resolve(config,probe_tokenizer=True):
    from huggingface_hub import hf_hub_download
    assert config['model_id']=='mistralai/Mistral-7B-Instruct-v0.3'
    revision=config['model_revision']
    url='https://huggingface.co/'+config['model_id']+'/raw/'+revision+'/tokenizer.model'
    with urllib.request.urlopen(url,timeout=30) as response:pointer=response.read(4096)
    text=pointer.decode('ascii')
    assert text.startswith('version https://git-lfs.github.com/spec/v1\n')
    expected=re.search(r'^oid sha256:([0-9a-f]{64})$',text,re.M).group(1)
    size=int(re.search(r'^size (\d+)$',text,re.M).group(1))
    assert size<10*1024**2,'bounded tokenizer artifact only'
    actual=Path(hf_hub_download(config['model_id'],'tokenizer.model.v3',revision=revision))
    digest=hashlib.sha256(actual.read_bytes()).hexdigest()
    assert actual.stat().st_size==size and digest==expected,'versioned tokenizer differs from immutable LFS alias'
    target=Path(config['model_path'])/'tokenizer.model'
    if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==expected
    else:target.symlink_to(actual.resolve())
    proof={'model_id':config['model_id'],'revision':revision,'pointer_source':url,
           'pointer_sha256':hashlib.sha256(pointer).hexdigest(),'resolved_source':'tokenizer.model.v3',
           'resolved_size_bytes':size,'resolved_sha256':digest,'expected_pointer_sha256':expected,
           'local_alias':str(target),
           'reason':'repository metadata reports the 130-byte LFS pointer; resolve returns 587404 bytes; original LlamaTokenizer initialization requires the SentencePiece artifact',
           'scope':'completes the same immutable tokenizer; checkpoint/config and token IDs are not modified'}
    if probe_tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(config['model_path'],local_files_only=True,use_fast=True,trust_remote_code=False)
        probe='Memory bandwidth and on chip storage affect decoding.'
        proof.update(tokenizer_class=type(tokenizer).__name__,is_fast=tokenizer.is_fast,
                     probe_text=probe,probe_ids=tokenizer.encode(probe,add_special_tokens=False))
    return proof


def main():
    from h200_session import DEFAULT_ROOT,dump
    config=json.loads((DEFAULT_ROOT/'configs/mistral7b_smoke.json').read_text())
    proof=resolve(config)
    dump(DEFAULT_ROOT/'configs/mistral7b_tokenizer_resolution.json',proof)
    print(json.dumps({k:proof[k] for k in ('revision','resolved_size_bytes','resolved_sha256','tokenizer_class','is_fast')},indent=2))


if __name__=='__main__':main()
