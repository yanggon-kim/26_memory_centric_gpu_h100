"""cubw E6 hook (loaded by Python inside the vLLM container because PYTHONPATH includes scripts/).

Does nothing unless CUBW_CU_COUNT_OVERRIDE is set. When set, vLLM's ROCm skinny-GEMM dispatch sizes
wvSplitK with that CU count instead of the device CU count (vllm.utils.platform_utils.num_compute_units),
so a decoder restricted by HSA_CU_MASK can launch the kernel sized for the CUs it actually has. The
override is applied in every process that imports vLLM (engine-core subprocesses inherit the variable) and
announced on stderr.
"""
import os

_N = os.environ.get('CUBW_CU_COUNT_OVERRIDE')
if _N:
    import importlib.abc
    import sys

    _TARGET = 'vllm.utils.platform_utils'

    class _Patch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                import importlib.util
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            loader = spec.loader
            orig = loader.exec_module

            def exec_module(module):
                orig(module)
                n = int(_N)
                module.num_compute_units = lambda device_id=0: n
                sys.stderr.write(f'cubw sitecustomize: num_compute_units -> {n} (pid {os.getpid()})\n')
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Patch())
