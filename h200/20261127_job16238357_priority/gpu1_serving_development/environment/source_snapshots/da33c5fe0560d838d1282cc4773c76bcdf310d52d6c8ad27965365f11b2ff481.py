#!/usr/bin/env python3
"""Create an immutable numerical-program tree while orchestration evolves."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--activate',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    session=json.loads(a.session.read_text())
    files=[x for folder in ('scripts','configs','patches') for x in (root/folder).rglob('*')
           if x.is_file() and '__pycache__' not in x.parts and x.suffix in ('.py','.sh','.cu','.json','.txt','.patch')]
    contents={str(x.relative_to(root)):x.read_bytes() for x in files}
    hashes={name:hashlib.sha256(b).hexdigest() for name,b in contents.items()}
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    destination=Path(session['workspace'])/'frozen'/identity
    for name,content in contents.items():
        target=destination/name
        if target.exists():assert target.read_bytes()==content
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    manifest={'identity':identity,'source_root':str(destination),'files':hashes,
              'scope':'immutable numerical entrypoints and dependencies; acquisition commands use this tree',
              'orchestration':'live repository launcher and exact commands remain separately recorded'}
    target=a.session.parent/'environment'/('source_freeze_'+identity[:12]+'.json')
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    if a.activate:
        session['frozen_source_root']=str(destination)
        session['frozen_source_identity']=identity
        a.session.write_text(json.dumps(session,indent=2)+'\n')
    print(json.dumps({'identity':identity,'source_root':str(destination),'activated':a.activate}))


if __name__=='__main__':main()
