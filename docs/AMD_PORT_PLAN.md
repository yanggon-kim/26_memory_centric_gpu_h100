# AMD Instinct port plan (branch `amd/mi300x-phase0`)

Status: plan only. No AMD measurement exists yet. H100/H200/RTX evidence on other branches is not AMD evidence, and nothing here relabels it.

## Goal

Pursue the same research question as `docs/primary_specification.md`: under fixed die-area and fixed HBM bandwidth, could moving area from compute into on-chip memory and data movement lower joules per decoded token? Run it on the AMD HPC Fund cluster instead of a rented H100. The research contract in `AGENTS.md` still applies unchanged: BF16, one declared GPU, no TP, seed 20261013, warmup ≥3 and trials ≥3, exact greedy output gates, the prefill/decode separation, energy qualification, and measured/analytical/simulated labels.

## Why AMD is still a meaningful testbed

MI300X/MI325X have a different memory hierarchy: per-XCD L2 (4 MB × 8) plus a 256 MB memory-side Infinity Cache (MALL) in front of HBM3/HBM3E. That last-level cache is already much larger than H100's 50 MiB L2. It is therefore a real-hardware control for part of the hypothesis: does a large on-chip cache at fixed HBM bandwidth actually lower decode traffic and time? Every such claim must still be measured with counters (TCC and MALL/data-fabric), not assumed from spec sheets.

## Cluster facts (observed 2026-10-18)

| Item | Value |
|---|---|
| Partitions | `mi3001x` (1× MI300X, VM, billing 0.0125/node-h), `mi3008x` (8× MI300X, 0.1), `mi3258x` (8× MI325X, 0.12), `mi2104x` (4× MI210, 0.04) |
| Allocation | `alloc_blaisetine_10012026_09302027`, 450 node-hours |
| ROCm | 7.2.0 default; `pytorch/2.10.0` module (rocm 7.1, needs `python3.12`) |
| Profilers | `rocprofv3`, `rocprof-compute` (Omniperf), `rocprof-sys` present; counter permission on compute nodes untested |
| Containers | `apptainer`/`podman` present, no root; HF Hub reachable from the login node |
| Storage | `$WORK=/work1/proj_blaisetine/yanggon` (1.9 TB); compute nodes cannot see the login node's `/tmp` |
| Known issue | On `devel` MI210 VMs, torch 2.10 BLAS fails ("HIP error: file not found"); it works on `mi2104x` |

## Tool mapping

| H100 package component | AMD replacement | Notes |
|---|---|---|
| `nvidia-smi`, pynvml (inventory, power, clocks) | `amd-smi` (+ `rocm-smi`) | Energy-counter and power-sample access must be re-qualified; keep raw values |
| CUDA 13 / cu128 lock, Python 3.10 venv | ROCm 7.x; AMD `rocm/vllm` container via apptainer, or module torch + venv | New lock file; record image digest |
| vLLM CUDA source build (6 failures) | Prebuilt ROCm vLLM container | Record vLLM, torch, AITER/CK/Triton attention backend versions |
| FlashAttention-3 | ROCm attention backends (AITER/CK, Triton) | Record which backend actually runs; tiles/splits from kernel names or source |
| Nsight Systems | `rocprofv3 --kernel-trace --hip-trace`, `rocprof-sys` | Timeline + kernel inventory |
| Nsight Compute counters | `rocprofv3 --pmc`, `rocprof-compute` | HBM bytes (e.g. `FETCH_SIZE`/`WRITE_SIZE`, TCC_EA*), TCC hit/miss, MALL where exposed, VALU/MFMA busy, LDS bank conflicts, occupancy; query `rocprofv3-avail` and record exact names |
| NVBit instruction/address traces | rocprofv3 thread trace (ATT) for sampled kernels; tensor-level logical traces (existing `reuse_analysis.py`) | No NVBit equivalent for full address streams; address-level reuse will be sampled or logical, labelled as such |
| Accel-Sim / GPGPU-Sim (sm_90) | gem5 GPU model (check gfx942/MI300 support), or analytical model only | Accel-Sim cannot replay AMD traces. Feasibility check before any simulator claim |
| `roofline_ceilings` | Re-measure BF16 MFMA GEMM and useful-copy ceilings on the allocated GPU | Same scripts after device-API port |

## Phased execution

1. **Phase 0 (inventory):** on one MI300X, record SKU/serial, CU count, L2/MALL, HBM, clocks, power, ROCm/driver, and profiler permissions (`rocprofv3-avail`, a trial `--pmc` run). Write `results/amd_<date>_<host>/environment.json`. Port `scripts/inventory.py` and `scripts/gpu_target.py` to detect ROCm.
2. **Runtime:** pull the AMD `rocm/vllm` image to `$WORK`, then run the bridge model (Qwen2.5-1.5B, pinned revision) at B1/C128 with HF eager and vLLM. Check exact-token gates and the HF vs. vLLM comparison.
3. **Phase 1 baselines:** bridge at B1/4/8 × C128/1024/4096, then 7B/14B/32B. 192 GB HBM fits the whole dense portfolio on one GPU. Collect timing, energy (qualified or NA), timelines and bounded counter sets per kernel class.
4. **Phase 2:** logical reuse/working-set curves with measured L2/MALL provenance; fixed-bandwidth artifacts from `docs/fixed_bandwidth_collection.md` with AMD counter names. Evaluate the MALL as an existing large-LLC control.
5. **Simulator feasibility:** document gem5/alternatives; no calibration claims without same-program comparison.

## Cost policy

The study needs one GPU. `mi3001x` gives one MI300X at about 1/8 the billing of a full 8-GPU `mi3008x` node. Use `mi3001x` by default once it is verified to work (it is a VM partition, like `devel`, where torch BLAS failed). Use 8-GPU nodes only when they are the only route, and expose a single GPU to the job.
