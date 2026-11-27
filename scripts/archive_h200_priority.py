#!/usr/bin/env python3
"""Archive a closed priority root to owned project storage and verify every byte.

No GPU work. Symlinks are recorded but never followed. Originals are untouched;
archives/receipts are immutable and all transfer hashes are retained.
"""
import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

PACKAGE=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()


def archive(root,destination,max_bytes=256*1024**2):
    root=root.resolve();entries=[];links=[]
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root).as_posix()
        if '__pycache__' in path.parts:continue
        if path.name.startswith(('archive_content_manifest_','archive_transfer_')):continue
        if path.is_symlink():
            links.append({'path':relative,'target':os.readlink(path)});continue
        if path.is_file():entries.append({'path':relative,'bytes':path.stat().st_size,'sha256':digest(path)})
    total=sum(e['bytes'] for e in entries)
    assert entries and total<=max_bytes,'Source budget exceeded; inspect storage before increasing it'
    identity=hashlib.sha256(encoded({'files':entries,'excluded_symlinks':links})).hexdigest()
    basename=f'{root.name}_{identity[:12]}.tar.gz'
    destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    assert shutil.disk_usage(destination).free>max(512*1024**2,2*total),'Insufficient archive storage budget'
    receipt=root/'results'/f'archive_transfer_{identity[:12]}.json'
    target=destination/basename
    if receipt.exists():
        data=json.loads(receipt.read_text())
        assert Path(data['archive'])==target and digest(target)==data['archive_sha256']
        assert data['source_content_identity']==identity
        return data
    manifest=root/'results'/f'archive_content_manifest_{identity[:12]}.json'
    manifest.parent.mkdir(parents=True,exist_ok=True)
    if manifest.exists():
        data=json.loads(manifest.read_text());assert data['files']==entries and data['excluded_symlinks']==links
    else:
        manifest.write_bytes(encoded({'schema_version':1,'source_root':str(root),'source_content_identity':identity,
            'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'files':entries,'excluded_symlinks':links,
            'scope':'Original regular-file bytes only. Archive timestamps/modes normalized. No model cache or symlink targets copied.'}))
    members=entries+[{'path':manifest.relative_to(root).as_posix(),'bytes':manifest.stat().st_size,'sha256':digest(manifest)}]
    with tempfile.TemporaryDirectory(prefix='h200_archive_') as staging:
        local=Path(staging)/basename
        with local.open('wb') as output,gzip.GzipFile(filename='',fileobj=output,mode='wb',mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w') as tar:
            for entry in members:
                path=root/entry['path']
                info=tarfile.TarInfo(root.name+'/'+entry['path']);info.size=entry['bytes'];info.mode=0o644
                with path.open('rb') as stream:tar.addfile(info,stream)
        expected={root.name+'/'+e['path']:e for e in members}
        with tarfile.open(local,'r:gz') as tar:
            actual=tar.getmembers();assert {m.name for m in actual}==set(expected)
            for member in actual:
                assert member.isfile()
                stream=tar.extractfile(member);value=hashlib.sha256();count=0
                for block in iter(lambda:stream.read(1024**2),b''):value.update(block);count+=len(block)
                assert count==expected[member.name]['bytes'] and value.hexdigest()==expected[member.name]['sha256'],'Source changed or archive bytes differ'
        source_hash=digest(local)
        if target.exists():assert digest(target)==source_hash,'Existing archive differs; refusing overwrite'
        else:
            with tempfile.NamedTemporaryFile(prefix='.'+basename+'.',suffix='.partial',dir=destination,delete=False) as out:
                partial=Path(out.name)
                with local.open('rb') as source:shutil.copyfileobj(source,out,1024**2)
                out.flush();os.fsync(out.fileno())
            assert digest(partial)==source_hash,'Transfer checksum mismatch; partial retained'
            try:os.link(partial,target)
            except FileExistsError:assert digest(target)==source_hash,'Concurrent archive differs'
            partial.unlink()
        assert digest(target)==source_hash
    sidecar=target.with_name(target.name+'.sha256')
    check=f'{source_hash}  {target.name}\n'
    if sidecar.exists():assert sidecar.read_text()==check
    else:
        with sidecar.open('x') as stream:stream.write(check)
    data={'schema_version':1,'status':'archive_transfer_verified','verified_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'source_root':str(root),'source_content_identity':identity,'regular_files':len(members),
          'original_bytes':total,'archive':str(target),'archive_bytes':target.stat().st_size,
          'archive_sha256':source_hash,'local_archive_sha256':source_hash,'destination_sha256':digest(target),
          'every_archive_member_verified':True,'source_manifest':str(manifest),'manifest_sha256':digest(manifest),
          'implementation_sha256':digest(Path(__file__)),'scope':'closed evidence-unit backup; allocation may still be active; no new GPU measurement or qualification'}
    with receipt.open('xb') as stream:stream.write(encoded(data))
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--max-source-bytes',type=int,default=256*1024**2)
    args=parser.parse_args();assert args.root.resolve().is_relative_to(PACKAGE/'h200')
    print(json.dumps(archive(args.root,args.destination,args.max_source_bytes)),flush=True)
