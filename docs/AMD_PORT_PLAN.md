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

## Profiling workflow

1. `rocprofv3 --kernel-trace --stats` (plus `--marker-trace` for prefill/decode ranges) on `scripts/amd/decode_profile_bench.py` finds the kernels that dominate decode time. `--sys-trace --output-format pftrace` gives a Perfetto timeline (the Nsight Systems role).
2. `rocprof-compute profile` (formerly Omniperf; the Nsight Compute role) on the same short workload, filtered to those kernels. It reports L1/L2 hit rates and traffic, HBM bandwidth and memory stalls, LDS bank conflicts, occupancy and VGPR/SGPR use, MFMA utilization and the per-level roofline. It replays the program once per counter pass, so the workload must be short and deterministic. Each replay appends its output IDs to a log for the identity check.
3. Raw `rocprofv3 --pmc` covers specific counters (for example HBM read/write bytes) on bounded ranges where the full analysis is unnecessary.

Profiled runs are never baseline timings. Tool setup: `rocprof-compute` 3.3.0 (ROCm 7.1.0, matching the torch module) runs from `$WORK/venvs/rocprof_compute_py312` with `PYTHONPATH=` cleared; the profiled program runs from `$WORK/venvs/hf_rocm_py312`.

## Phased execution

1. **Phase 0 (inventory):** on one MI300X, record SKU/serial, CU count, L2/MALL, HBM, clocks, power, ROCm/driver, and profiler permissions (`rocprofv3-avail`, a trial `--pmc` run). Write `results/amd_<date>_<host>/environment.json`. Port `scripts/inventory.py` and `scripts/gpu_target.py` to detect ROCm.
2. **Runtime:** pull the AMD `rocm/vllm` image to `$WORK`, then run the bridge model (Qwen2.5-1.5B, pinned revision) at B1/C128 with HF eager and vLLM. Check exact-token gates and the HF vs. vLLM comparison.
3. **Phase 1 baselines:** bridge at B1/4/8 × C128/1024/4096, then 7B/14B/32B. 192 GB HBM fits the whole dense portfolio on one GPU. Collect timing, energy (qualified or NA), timelines and bounded counter sets per kernel class.
4. **Phase 2:** logical reuse/working-set curves with measured L2/MALL provenance; fixed-bandwidth artifacts from `docs/fixed_bandwidth_collection.md` with AMD counter names. Evaluate the MALL as an existing large-LLC control.
5. **Simulator feasibility:** document gem5/alternatives; no calibration claims without same-program comparison.

## Multi-GPU extension (TP4, TP8)

**Scope authorization.** On 2026-10-18 the project owner explicitly authorized adding 4- and 8-GPU tensor-parallel runs. This is the scope authorization `AGENTS.md` requires. It adds cases. It does not replace or relabel the single-GPU (TP1) portfolio, which stays primary. BF16, no quantization, and the other contract rules are unchanged.

**Why it matters for the hypothesis.** Tensor parallelism divides each layer's weights (and the KV heads, where the head count allows) across N GPUs. Each GPU keeps its own HBM bandwidth and its own 256 MB Infinity Cache, so the per-GPU working set shrinks roughly as W/N at fixed per-GPU cache and bandwidth. A TP1 → TP4 → TP8 sweep is a real-hardware probe of residency: does a working set that fits better in on-chip cache lower per-GPU HBM bytes per token and step time? Collective communication (all-reduce over xGMI) is the new cost that can erase any gain, and it must be measured, not assumed.

| Size | Hardware | Notes |
|---|---|---|
| TP4 | 4 of the 8 MI300X on a `mi3008x` node | Record which GPUs and their xGMI connectivity (`amd-smi topology`) |
| TP8 | all 8 MI300X on `mi3008x` (and optionally `mi3258x` MI325X) | MI325X results stay labelled separately |

**Route.** Use vLLM in the ROCm container with `tensor_parallel_size` 4 or 8 and RCCL. The HF eager runner stays TP1 only. `gpu_target.select`/`verify_cuda` enforce exactly one visible GPU, so multi-GPU runs need a separate selector. It must record every participating UUID and fail on any other compute process on those GPUs.

**Order and gates.**
1. Run only after the same model and case pass TP1 vLLM smoke and baseline.
2. Start with bridge B1/C128 at TP4 then TP8, then the cases where per-GPU weights approach the cache size (bridge, then 7B/14B), then 32B. Head counts must divide by N, otherwise record the case as unsupported.
3. Correctness: output IDs identical across warmups and trials (required). Compare against TP1 and HF; changed reduction order can legitimately alter BF16 greedy tokens. A mismatch is recorded as a finding, never bypassed, and timing comparisons across TP sizes are reported only where the tokens match or the divergence is documented.

**Measurements.**
- Per-step decode timing and tokens/s, with prefill kept separate.
- Per-GPU kernel time split into compute kernels vs RCCL collectives, from `rocprofv3` traces on every rank.
- Per-GPU HBM read/write bytes and TCC/MALL hit rates from `rocprofv3 --pmc` and `rocprof-compute`, giving bytes per token per GPU and summed.
- Energy as the sum of all participating GPUs' counters, under the same qualification rules. Idle non-participating GPUs are recorded but excluded; any idle subtraction is reported separately.
- Interpretation separates three effects: (a) less traffic per GPU from residency, (b) more aggregate bandwidth from more GPUs, and (c) communication overhead. Only (a) bears on the on-chip-memory hypothesis; (b) is not a fixed-bandwidth result.

## Cost policy

The study needs one GPU. `mi3001x` gives one MI300X at about 1/8 the billing of a full 8-GPU `mi3008x` node. Use `mi3001x` by default once it is verified to work (it is a VM partition, like `devel`, where torch BLAS failed). Use `mi3008x` for the TP4/TP8 extension, and expose only the declared GPUs to each run. On a full-node allocation, schedule the TP4/TP8 cases in the same job as the TP1 work so the other GPUs are not idle while billed.
