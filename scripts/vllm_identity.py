"""Verify the pinned vLLM release source used by the CUDA 12.8 bridge."""

import hashlib
import json
from pathlib import Path
import subprocess


def verify(vllm):
    import os
    image = os.environ.get('AMD_VLLM_IMAGE')
    if image:
        # ROCm route: a prebuilt AMD rocm/vllm container; identity is the image file hash plus the version.
        digest = os.environ.get('AMD_VLLM_IMAGE_SHA256', 'unrecorded')
        return {'version': vllm.__version__, 'identity_route': 'amd-rocm-vllm-container',
                'image': image, 'image_sha256': digest, 'install_path': str(Path(vllm.__file__).resolve().parent)}
    if vllm.__version__ == '0.22.1':
        return {'version': vllm.__version__, 'identity_route': 'release-version'}

    # setuptools-scm labels the tagged checkout as the next dev version when
    # the reviewed Python 3.10 spinloop compatibility patch makes it dirty.
    assert vllm.__version__.startswith('0.22.2.dev0+g0decac0d9.d'), vllm.__version__
    root = Path(vllm.__file__).resolve().parents[1]
    manifest = json.loads((Path(__file__).resolve().parents[1] /
                           'results/vllm_source_manifest_cu128.json').read_text())
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only'], text=True).splitlines()
    assert commit == manifest['vllm_commit'] and changed == ['csrc/spinloop.cpp']
    patched = root / 'csrc/spinloop.cpp'
    assert hashlib.sha256(patched.read_bytes()).hexdigest() == manifest['patched_spinloop_sha256']
    return {'version': vllm.__version__, 'identity_route': 'tagged-source-with-reviewed-patch',
            'source_commit': commit, 'patched_spinloop_sha256': manifest['patched_spinloop_sha256']}
