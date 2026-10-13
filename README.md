# H100 LLM decode research package

A portable package for evidence-driven H100 Phase 0–2 research and Accel-Sim 2.0 trace collection. A selected BF16 H100 bridge, real sm_90 micro/partial-layer/two-step whole-model HF capture with one-kernel replay smokes and analytical reuse study have now been run; the optimized baseline, quantitative bottleneck, semantic calibration and architecture benefit remain unestablished. For a new rental, start with [the continuation handoff](docs/RENTAL_RESUME.md). Read [the primary specification](docs/primary_specification.md), Sections 0 and 14 first, then [the execution playbook](docs/phase02_playbook.md) and [current opportunity report](docs/opportunity_report.md).

The historical [RTX 5080 findings](docs/rtx5080_findings.md) motivate falsifiable tests: small-batch weight delivery can be busy, larger batches consume compute, ideal pinning differs from naive LRU expansion, and disagreement between energy sensors can invalidate J/token. None transfers automatically to H100.

The next rental must also follow [fixed-bandwidth collection requirements](docs/fixed_bandwidth_collection.md). The [interactive explanation](docs/fixed-bandwidth-serving.html) separates traffic reduction from bandwidth utilization and whole-model gain.

## Before renting

Reserve an explicitly named H100 GPU and enough CPU, RAM, disk and transfer time. Start with the 1.5B bridge. The 32B model is conditional on actual HBM/workspace fit; Conditional Qwen3-30B-A3B follows dense baselines if memory/runtime budgets permit; active parameters do not determine weight fit. Model downloads resolve immutable HF revisions and respect declared size budgets. Do not start with all models or a full-model large trace.

The runtime previously tested on RTX 5080 is Linux x86_64, Python 3.10, torch 2.11.0/CUDA 13, vLLM 0.22.1, transformers 4.57.6. The lock is reproducibility input, not proof of H100 compatibility. Supply an existing matching CUDA 13 toolkit/compatible driver; setup does not install or change either. `nvtx==0.2.16` is a separately pinned dependency of the upstream hook ([PyPI metadata](https://pypi.org/project/nvtx/)); its H100 hook execution has been exercised on the HF eager route.

```bash
CUDA_HOME=/existing/cuda13 PYTHON_BIN=python3.10 scripts/setup.sh --install
export CUDA_HOME=/existing/cuda13
export PATH="$CUDA_HOME/bin:$PATH"
.venv/bin/python scripts/inventory.py --gpu-uuid GPU-ALLOCATED-UUID --check-cuda --output results/environment.json
.venv/bin/python scripts/materialize_config.py --model bridge --stage smoke --download --output results/bridge_smoke.json
.venv/bin/python scripts/controlled_vllm_experiments.py --config results/bridge_smoke.json --gpu-uuid GPU-ALLOCATED-UUID --mode timing --output results/bridge_smoke_run
```

Alternatively provide `--snapshot /cache/.../snapshots/IMMUTABLE_SHA` to materialize without downloading. Every output directory must be fresh. The controlled runner uses synchronous in-process vLLM and proves the all-request prefill boundary. StaticCache/compiled HF is a distinct software control; its strict eager/compiled correctness gate can fail and must not be bypassed.

Build external pinned tools with `CUDA_HOME=/existing/toolkit scripts/setup_accelsim.sh /owned/external/tools` in a child shell; dependencies and version/source gates are explicit. Open a fresh shell for native capture.

Use [trace collection](docs/trace_collection.md) for the actual pinned upstream hook API, runnable HF and eager-vLLM routes (native HF pilot verified; vLLM pending), the ACTIVE_FROM_START workaround, nonempty-capture checks, postprocessing, replay and export. Accel-Sim and NVBit are external source dependencies, not vendored binaries. See [the source audit and calibration gates](docs/simulator_audit.md).

CPU-only checks:

```bash
python3 scripts/validate_package.py
python3 scripts/materialize_config.py --help
python3 scripts/run_trace.py --help
python3 scripts/test_cpu_contracts.py
```

New `results/` and `traces/` outputs are ignored by default; selected historical evidence is explicitly tracked. Model weights, environments, raw traces, archives and profiler binaries are not included. See the continuation handoff for private archive recovery and missing local-only evidence. `docs/rtx5080_evidence/` is historical scoped evidence. Runtime configs with machine-local cache paths belong in ignored outputs, not portable model templates. Parent coordination owns repository creation/publishing; this package itself performs no Git operations.
