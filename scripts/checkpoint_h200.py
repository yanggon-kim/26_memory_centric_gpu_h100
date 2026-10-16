#!/usr/bin/env python3
"""Publish compact H200 evidence; keep raw databases and archives outside Git."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess

PACKAGE = Path(__file__).resolve().parents[1]
TEXT = {'.py', '.sh', '.json', '.jsonl', '.csv', '.md', '.log', '.txt', '.cmake', '.cu', '.h', '.cpp', '.patch', '.sha256', '.sass', '.ptx', '.config'}
SECRETS = [re.compile(rb'hf_[A-Za-z0-9]{25,}'),
           re.compile(rb'gh[pousr]_[A-Za-z0-9]{25,}'),
           re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),
           re.compile(rb'(?i)Authorization[\"\s:]+Bearer\s+[A-Za-z0-9._-]{15,}')]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=PACKAGE, text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, action='append', required=True)
    p.add_argument('--message', required=True)
    p.add_argument('--push', action='store_true')
    a = p.parse_args()
    lock = (PACKAGE/'.git/h200_checkpoint.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert git('branch', '--show-current') == 'codex/h200-phase01'
    cached_raw = {}
    for root in a.root:
        inventory=root/'results/checkpoint_raw_inventory.json'
        if inventory.exists():
            cached_raw.update({x['path']:x for x in json.loads(inventory.read_text()).get('raw_artifacts',[])})
        manifest=root/'results/shoreline_case_manifest.json'
        if manifest.exists():
            data=json.loads(manifest.read_text());attempts=[]
            for source in sorted((root/'results/processes').glob('*.json')):
                row=json.loads(source.read_text())
                attempts.append({'id':row['label'],'status':row.get('status','running'),
                    'source':str(source.relative_to(root)),
                    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'gpu_execution':bool(row.get('monitor')),'exit_code':row.get('exit_code'),
                    'numerical_source_freeze':row.get('source_freeze_identity')})
            data['attempts']=attempts
            manifest.write_text(json.dumps(data,indent=2)+'\n')
    candidates = list((PACKAGE/'scripts').glob('*.py')) + [PACKAGE/'h200/README.md']
    for root in a.root:
        root = root.resolve()
        assert root.is_relative_to(PACKAGE/'h200'), 'H200 root required'
        candidates += [x for x in root.rglob('*') if x.is_file() and not x.is_symlink()]
    selected, omitted = [], []
    for path in sorted(set(candidates)):
        if '__pycache__' in path.parts or '.control' in path.parts or path.name.endswith('.tmp'):
            continue
        if path.suffix not in TEXT | {'.png', '.pdf'} or path.stat().st_size > 8*1024**2:
            name=str(path.relative_to(PACKAGE));stat=path.stat();prior=cached_raw.get(name,{})
            content_hash=prior.get('sha256') if prior.get('bytes')==stat.st_size and prior.get('mtime_ns')==stat.st_mtime_ns else None
            omitted.append({'path':name, 'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                            'sha256':content_hash or hashlib.sha256(path.read_bytes()).hexdigest(),
                            'storage':'persistent workspace; excluded from public Git'})
            continue
        content = path.read_bytes()
        if path.suffix in TEXT and any(s.search(content) for s in SECRETS):
            raise RuntimeError('Secret pattern found; publication refused: '+str(path.relative_to(PACKAGE)))
        selected.append(str(path.relative_to(PACKAGE)))
    for root in a.root:
        inventory = root/'results/checkpoint_raw_inventory.json'
        inventory.parent.mkdir(parents=True, exist_ok=True)
        inventory.write_text(json.dumps({'raw_artifacts':omitted},indent=2)+'\n')
        selected.append(str(inventory.resolve().relative_to(PACKAGE)))
    # Explicit paths permit compact JSON in historically ignored results directories.
    for start in range(0, len(selected), 100):
        git('add', '-f', '--', *selected[start:start+100])
    staged = git('diff', '--cached', '--name-only').splitlines()
    for name in staged:
        content = subprocess.check_output(['git', 'show', ':'+name], cwd=PACKAGE)
        if any(s.search(content) for s in SECRETS):
            raise RuntimeError('Secret pattern in staged file; publication refused: '+name)
    if staged:
        print(git('commit', '-q', '-m', a.message), flush=True)
    commit = git('rev-parse', 'HEAD')
    if a.push:
        subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/codex/h200-phase01'], cwd=PACKAGE, check=True)
        remote = git('ls-remote', 'origin', 'refs/heads/codex/h200-phase01').split()[0]
        assert remote == commit, 'remote verification failed'
    print(json.dumps({'commit': commit, 'remote_verified': a.push,
                      'selected_files': len(selected), 'omitted_raw_file_count': len(omitted)}), flush=True)
    for root in a.root:
        pause = root/'.control/pause_after_checkpoint'
        if pause.exists():
            pause.unlink()
            print('Requested safe case boundary reached; checkpoint published before scheduler stop.',flush=True)
            raise SystemExit(75)


if __name__ == '__main__':
    main()
