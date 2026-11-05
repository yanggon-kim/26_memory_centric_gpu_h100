#!/usr/bin/env python3
"""G6 static metadata of vLLM's ROCm skinny-GEMM kernels (wvSplitK, wvSplitKrc, LLMM1) for this GPU's architecture.

Runs inside the vLLM container (CPU only). Finds the extension library that holds the kernels, unbundles the code
object for --arch with clang-offload-bundler (or roc-obj when present), and records for every matching kernel the
AMDGPU metadata notes (VGPR/AGPR/SGPR counts, LDS = group_segment_fixed_size, private segment, wavefront size, max
flat workgroup size) plus an instruction-class census of the disassembly (global/buffer loads and stores, LDS ops,
MFMA/dot, waitcnt). Static metadata explains resource limits; it is not a measured stall breakdown.

  isa_meta.py --arch gfx942 --out <dir>
"""
import argparse
import collections
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

PAT = re.compile(r'(wvSplitK|LLMM1|wvSplitKrc)')


def tool(name):
    for d in ('/opt/rocm/llvm/bin', '/opt/rocm/bin', '/usr/bin'):
        p = Path(d) / name
        if p.exists():
            return str(p)
    return shutil.which(name)


def run(cmd, timeout=600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    import vllm
    root = Path(vllm.__file__).parent
    libs = sorted(glob.glob(str(root / '_rocm_C*.so')) + glob.glob(str(root / '_C*.so')))
    res = {'scope': __doc__.strip(), 'arch': a.arch, 'libraries': libs, 'tools': {}, 'kernels': {}, 'errors': []}
    bundler, readelf, objdump = tool('clang-offload-bundler'), tool('llvm-readelf'), tool('llvm-objdump')
    res['tools'] = {'clang-offload-bundler': bundler, 'llvm-readelf': readelf, 'llvm-objdump': objdump}
    for lib in libs:
        co = a.out / (Path(lib).name + f'.{a.arch}.co')
        target = f'hipv4-amdgcn-amd-amdhsa--{a.arch}'
        ok = False
        for tgt in (target, target + ':sramecc+:xnack-', target + ':xnack-', target + ':sramecc+'):
            rc, out, err = run([bundler, '--type=o', f'--targets={tgt}', f'--input={lib}', f'--output={co}', '--unbundle'])
            if rc == 0 and co.exists() and co.stat().st_size > 0:
                ok = True
                res.setdefault('unbundled', {})[lib] = tgt
                break
        if not ok:
            rc, out, err = run([bundler, '--list', '--type=o', f'--input={lib}'])
            res['errors'].append({'lib': lib, 'step': 'unbundle', 'available_targets': out[-2000:], 'stderr': err[-500:]})
            continue
        rc, notes, err = run([readelf, '--notes', str(co)])
        (a.out / (co.name + '.notes.txt')).write_text(notes + err)
        rc, dis, err = run([objdump, '-d', '--no-show-raw-insn', str(co)], timeout=1800)
        cur = None
        census = collections.defaultdict(collections.Counter)
        for line in dis.splitlines():
            m = re.match(r'^[0-9a-f]+ <(.+)>:', line)
            if m:
                cur = m.group(1) if PAT.search(m.group(1)) else None
                continue
            if cur:
                ins = line.strip().split(' ')[0] if line.strip() else ''
                for cls, rx in (('global_load', r'^global_load'), ('buffer_load', r'^buffer_load'), ('global_store', r'^global_store'),
                                ('buffer_store', r'^buffer_store'), ('ds_read', r'^ds_(read|load)'), ('ds_write', r'^ds_(write|store)'),
                                ('mfma', r'^v_mfma'), ('dot', r'^v_dot'), ('waitcnt', r'^s_waitcnt'), ('total', r'.')):
                    if re.match(rx, ins):
                        census[cur][cls] += 1
        # kernel metadata: parse the YAML-ish msgpack dump of readelf --notes
        for block in re.split(r'\n\s*- \.agpr_count|\n\s*- \.args', notes):
            name = re.search(r'\.name:\s+(\S+)', block)
            if not name or not PAT.search(name.group(1)):
                continue
            meta = {}
            for key in ('.vgpr_count', '.agpr_count', '.sgpr_count', '.group_segment_fixed_size', '.private_segment_fixed_size',
                        '.wavefront_size', '.max_flat_workgroup_size', '.vgpr_spill_count', '.sgpr_spill_count'):
                m = re.search(re.escape(key) + r':\s+(\d+)', block)
                if m:
                    meta[key.lstrip('.')] = int(m.group(1))
            res['kernels'][name.group(1)] = {'library': Path(lib).name, 'metadata': meta}
        for k, c in census.items():
            res['kernels'].setdefault(k, {'library': Path(lib).name})['census'] = dict(c)
    res['kernel_count'] = len(res['kernels'])
    (a.out / 'isa_meta.json').write_text(json.dumps(res, indent=1) + '\n')
    print(f"{res['kernel_count']} matching kernels; errors {len(res['errors'])}")


if __name__ == '__main__':
    q = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    q.add_argument('--arch', required=True)
    q.add_argument('--out', type=Path, required=True)
    main(q.parse_args())
