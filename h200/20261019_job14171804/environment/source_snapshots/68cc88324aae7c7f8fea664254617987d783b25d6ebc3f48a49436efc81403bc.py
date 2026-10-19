"""Fail-closed replay publications and immutable ordinary-output comparisons."""
import hashlib
import json
from pathlib import Path


def qualify_replays(base,expected,root):
    base,root=Path(base),Path(root)
    publications=sorted(base.glob('pass[0-9][0-9][0-9].json')) if base.exists() else []
    proofs=[]
    directories={p.name for p in base.glob('pass[0-9][0-9][0-9]') if p.is_dir()} if base.exists() else set()
    published={p.stem for p in publications}
    for publication in publications:
        p=json.loads(publication.read_text());out=Path(p['output'])
        native,interval=out/'run.json',out/'selected_interval.json'
        valid=p.get('status')=='complete' and native.exists() and interval.exists()
        if valid:
            d=json.loads(native.read_text());i=json.loads(interval.read_text());cases=d.get('cases',[])
            valid=(d.get('status')=='complete' and i.get('status')=='complete' and i.get('ticks')==[4,5,6]
                   and len(cases)==1 and len(cases[0].get('trials',[]))>=3
                   and cases[0].get('prompt_sha256')==expected['prompt_sha256']
                   and all(t['output_ids']==expected['trials'][0]['output_ids'] for t in cases[0]['trials']))
        proofs.append({'publication':str(publication.relative_to(root)),
                       'sha256':hashlib.sha256(publication.read_bytes()).hexdigest(),'passed':bool(valid)})
    return {'passed':bool(proofs) and directories==published and all(p['passed'] for p in proofs),
            'passes':proofs,'unpublished_directories':sorted(directories-published),
            'missing_output_directories':sorted(published-directories)}
