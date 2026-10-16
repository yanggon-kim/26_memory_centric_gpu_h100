#!/usr/bin/env python3
"""Publish compact H200 evidence; keep raw databases and archives outside Git."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PACKAGE = Path(__file__).resolve().parents[1]
TEXT = {'.py', '.sh', '.json', '.jsonl', '.csv', '.md', '.log', '.txt', '.cmake', '.cu', '.h', '.cpp'}
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
    assert git('branch', '--show-current') == 'codex/h200-phase01'
    candidates = list((PACKAGE/'scripts').glob('*.py'))
    for root in a.root:
        root = root.resolve()
        assert root.is_relative_to(PACKAGE/'h200'), 'H200 root required'
        candidates += [x for x in root.rglob('*') if x.is_file() and not x.is_symlink()]
    selected, omitted = [], []
    for path in sorted(set(candidates)):
        if '__pycache__' in path.parts or path.name.endswith('.tmp'):
            continue
        if path.suffix not in TEXT | {'.png', '.pdf'} or path.stat().st_size > 8*1024**2:
            omitted.append(str(path.relative_to(PACKAGE)))
            continue
        content = path.read_bytes()
        if path.suffix in TEXT and any(s.search(content) for s in SECRETS):
            raise RuntimeError('Secret pattern found; publication refused: '+str(path.relative_to(PACKAGE)))
        selected.append(str(path.relative_to(PACKAGE)))
    # Explicit paths permit compact JSON in historically ignored results directories.
    for start in range(0, len(selected), 100):
        git('add', '-f', '--', *selected[start:start+100])
    staged = git('diff', '--cached', '--name-only').splitlines()
    for name in staged:
        content = subprocess.check_output(['git', 'show', ':'+name], cwd=PACKAGE)
        if any(s.search(content) for s in SECRETS):
            raise RuntimeError('Secret pattern in staged file; publication refused: '+name)
    if staged:
        print(git('commit', '-m', a.message), flush=True)
    commit = git('rev-parse', 'HEAD')
    if a.push:
        subprocess.run(['git', 'push', 'origin', 'HEAD:refs/heads/codex/h200-phase01'], cwd=PACKAGE, check=True)
        remote = git('ls-remote', 'origin', 'refs/heads/codex/h200-phase01').split()[0]
        assert remote == commit, 'remote verification failed'
    print(json.dumps({'commit': commit, 'remote_verified': a.push,
                      'selected_files': len(selected), 'omitted_raw_files': omitted}), flush=True)


if __name__ == '__main__':
    main()
