#!/usr/bin/env python3
"""One separately labeled deterministic HF diagnostic; no gate relaxation.

CLI is the unchanged hf_full_reference.py interface. Its dynamic generation and
controlled StaticCache/SDPA functions run unchanged. Deterministic algorithms
and explicit cuBLAS workspace are experiment-local process settings. The failed
default reference remains failed even if this diagnostic passes.
"""
import hashlib
import os
from pathlib import Path

HF_SHA = 'bdd6f485477e82ab904151e21a0299a4893866664aeb73a082b75a9ef37096aa'


def tag(record, settings):
    record['deterministic_reference_diagnostic'] = dict(settings)
    record['promotes_original_default_reference'] = False
    return record


def main():
    path = Path(__file__).with_name('hf_full_reference.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HF_SHA
    from h200_session import verify_allocation
    assert verify_allocation() is not None
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    assert not torch.cuda.is_initialized(), 'deterministic workspace must precede CUDA initialization'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    settings = {'protocol': 'HF_deterministic_diagnostic_v1',
        'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'unchanged_HF_script_sha256': HF_SHA,
        'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'precision': 'BF16 model weights/KV; no precision or attention-mask change',
        'scope': 'Stricter algorithm control on identical inputs, exact outputs and unchanged StaticCache functions; no accepted ties, tolerance, default-reference promotion or timing claim.'}
    assert settings['torch_deterministic_algorithms'] and not settings['warn_only']
    import hf_full_reference as reference
    original_dump = reference.dump
    reference.dump = lambda path, record: original_dump(path, tag(record, settings))
    reference.main()


if __name__ == '__main__':
    main()
