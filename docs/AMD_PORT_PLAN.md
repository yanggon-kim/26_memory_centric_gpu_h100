# AMD Instinct port plan (branch `amd/mi300x-phase0`)

**Active follow-up, 2026-10-26:** read [the new E0–E8 execution handoff](amd/compute_bandwidth_followup_20261026.md) first. AMD measurements now exist; the "plan only" statement and tool-feasibility questions below describe the original starting state. The new handoff governs follow-up ordering and comparison gates, preserves original results, and requires reconciling current MI325X work before any new launch. It does not claim that its new experiments have run.

Status: plan only. No AMD measurement exists yet. H100/H200/RTX evidence on other branches is not AMD evidence, and nothing here relabels it.

## Goal and governing documents

The research question is the shoreline-aware one from `docs/shoreline_aware_research_plan_2.0.md` (copied from `codex/h200-phase01`). It supersedes the architecture scope of `docs/primary_specification.md`:

> Can some GPU compute resources be reallocated to on-chip SRAM, to external-memory PHY/controller capacity and delivered HBM bandwidth, or to both, to improve LLM decode latency/throughput and energy per token under explicit silicon, shoreline, package, thermal and cost constraints?

Candidates (plan 2.0 §7) are compared, not preselected: stock, compute-only trim, compute → SRAM, compute → HBM/MC/PHY (rectangular or stepped die), joint HBM + L2 banks + NoC, a minimal memory-serving engine, and external alternatives (chiplets, bridges).

Precedence on this branch:
1. This plan, for AMD hardware, allocation and tool routes.
2. `docs/shoreline_hopper_measurements.md` for the M1–M8 and A–F measurement methods, statistics and acceptance. Its Hopper/NCU/green-context specifics are translated in the AMD mapping below.
3. `docs/h200_shoreline_jobs.md` for the job order J0–J11 (used as the template).
4. `AGENTS.md` and the older supplements for the correctness, energy, isolation and labelling gates. The stronger rule always wins.

Statistics follow the common contract for any new quantitative comparison: three independent process acquisitions per condition, ≥3 full warmups and 10 accepted timed trials per acquisition, 64 generated tokens (63 pure-decode ticks after the verified one-token barrier), seed 20261013, and a pooled ≤5% clock envelope over valid samples, declared before looking at performance. Three-trial smoke runs are pilots only. Physical architecture changes, geometry/PPA and simulation are later stages; the real GPU supplies motivation and service-limit evidence.

## Why AMD is a meaningful testbed for this question

- **A real "more HBM, larger LLC" design point.** MI300X uses 8 HBM3 stacks (~5.3 TB/s nominal; H100 SXM has 5 stacks) and a 256 MB memory-side Infinity Cache (MALL) on its I/O dies, in front of per-XCD 4 MB L2s. Its compute dies (XCDs) are stacked on the I/O dies, a chiplet/3D answer to the shoreline limit (plan 2.0 candidate B7). Specs are only context; every traffic or cache effect must be measured with counters.
- **A same-compute bandwidth pair.** MI325X has the same gfx942 compute (304 CUs) and the same Infinity Cache, with HBM3E at ~6.0 TB/s nominal and 256 GB capacity. MI300X vs MI325X is a real-hardware bandwidth-ratio (`r`) data point at near-fixed compute (`c`). It is confounded by power limit, clocks and HBM type, so it is labelled a cross-SKU comparison, never a controlled stack addition.
- **Resource restriction without green contexts.** ROCm supports compute-unit masks (`HSA_CU_MASK` per process, `hipExtStreamCreateWithCUMask` per stream). This is the likely route for the M6/M7 SM-allocation sweep and may cover library and graph kernels that CUDA green contexts could not (the H200 graph-restriction attempt failed). Support, granularity and actual placement must be proven before use.

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

## AMD mapping of the required measurements (M1–M8, A–F)

| ID | Contract requirement | AMD route | Open feasibility question |
|---|---|---|---|
| M1 | Strongest graph-optimized and eager baselines; timelines, launch counts, busy-union shares | vLLM (ROCm container) with HIP graphs vs `enforce_eager`; HF eager/compiled; `rocprofv3 --kernel-trace --marker-trace`, `--sys-trace` Perfetto timelines | Container runs on host driver amdgpu 6.16.13 |
| M2 | Logical bytes/FLOPs vs physical HBM read/write for first/middle/last layers and full decode, ticks 4/5/6 | `rocprofv3 --pmc` on HBM-side counters (TCC→EA read/write requests and sizes) and MALL counters if exposed; `rocprof-compute` memory chart | Exact gfx942 counter names/units via `rocprofv3-avail`; whether MALL hit counters exist; full-range collection |
| M3 | Per-kernel arithmetic, issue, stalls, occupancy, registers/LDS, launch geometry, actual instructions | `rocprof-compute` (VALU/MFMA busy, wave occupancy, VGPR/SGPR/AGPR, LDS bank conflicts, instruction mix); code objects disassembled with `llvm-objdump`/`roc-obj` | Mapping attention backend (AITER/CK/Triton) tiles and splits |
| M4, A–D | Independent stream/strided/gather/pointer-chase, scalar/MFMA/reduction, concurrent compute+memory microbenchmarks; footprints below/near/above L2 **and** the 256 MB MALL | New HIP microbenchmarks (`hipcc --offload-arch=gfx942`) with event timing and `--pmc` bytes | None beyond implementation; footprints must straddle both 4 MB L2 and 256 MB MALL |
| M5 | Normal vs competing-read cache-history pressure; retention policies | Pressure buffer above L2 + MALL capacity (≥512 MiB) between layers | Write-back attribution across MALL |
| M6/M7 | Software SM-allocation sweep 1.0/.75/.5/.3, repeated with service microbenchmarks | CU masking balanced across the 8 XCDs; verify actual CU use with counters | Mask granularity, graph/library coverage, effect on per-XCD L2 |
| M8 | Long-window energy with counter vs sampled-power agreement; prefill and larger-batch counterexamples | `amd_nvml.py`: amdsmi energy accumulator and socket power | Counter resolution and update cadence on MI300X; socket-power scope |
| E/F | Layer events, instrumentation-perturbation and PM pass-layout controls; `T = C + M − X + O` bounds; c/r/f sensitivity | HIP events per layer; rocprofv3 pass layout; same analysis math | Same as contract |

## Phased execution (adapted J0–J11)

1. **J0 – inventory and capability ledger:** SKU, UUID, CU count, L2/MALL, HBM, clocks, power, ROCm/driver; profiler counter permission (a trial `--pmc` run); CU-mask support; container start. Write `results/amd_<date>_<host>/environment.json` and a `shoreline_case_manifest.json` listing every M/A–F family as implemented / needs adapter / unavailable.
2. **Correctness pilots:** bridge B1/C128 on HF eager (StaticCache gate) and vLLM graph/eager; exact-token checks and HF vs vLLM comparison.
3. **J1–J3 on the focused contrasts:** bridge B1/C1024 and B8/C4096; a 7B model (Qwen2.5-7B or Mistral-7B) B1/C4096 and B8/C4096 (a compute-active counterexample); 14B, then 32B B1/C4096 (192 GB fits all on one GPU). Then timing, timelines, M2 bytes and M3 kernel evidence.
4. **J4:** independent service microbenchmarks (A–D).
5. **J5:** CU-mask sweep (M6) with repeated service controls (M7).
6. **J7, J8, J10:** retention/pressure, layer timing/overlap bounds, qualified energy and prefill/large-batch negative regimes.
7. **J9:** c/r/f joint sensitivity using measured inputs, including the MI300X vs MI325X bandwidth pair and the multi-GPU cases below.
8. **Simulator/trace feasibility:** no NVBit/Accel-Sim on AMD. Check gem5's gfx942 support, and use sampled thread traces (rocprofv3 ATT) plus logical traces. No calibration claim without a same-program comparison.

The original full dense portfolio (bridge/7B/14B/32B × B1/4/8 × C128/1024/4096) remains available as extra controls, run after the focused contrasts.

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
- Interpretation separates three effects: (a) less traffic per GPU from residency, (b) more aggregate bandwidth from more GPUs, and (c) communication overhead. Under plan 2.0, (a) bears on the compute → SRAM candidate and (b) on the compute → bandwidth candidate; neither is a fixed-bandwidth result on its own.

**Compute-matched variant (if CU masking passes J5).** TP-N with each GPU restricted to about 1/N of its CUs keeps total active CUs near one GPU's while multiplying HBM stacks, bandwidth and Infinity Cache by N. This is a real-hardware probe of "same compute, more delivered bandwidth" (`c` ≈ 1, `r` > 1). It is confounded by inter-GPU communication, per-GPU clocks and power, and by remaining per-GPU fixed costs. It must be reported with those confounds and never presented as an added-stack implementation.

The common contract (`shoreline_hopper_measurements.md`) forbids tensor parallelism. On this branch, the 2026-10-18 owner authorization above overrides that for these labelled cases only.

## Cost policy

The study needs one GPU. `mi3001x` gives one MI300X at about 1/8 the billing of a full 8-GPU `mi3008x` node. Use `mi3001x` by default once it is verified to work (it is a VM partition, like `devel`, where torch BLAS failed). Use `mi3008x` for the TP4/TP8 extension, and expose only the declared GPUs to each run. On a full-node allocation, schedule the TP4/TP8 cases in the same job as the TP1 work so the other GPUs are not idle while billed.

## Added 2026-10-18: bandwidth effect at fixed compute

Requested by the owner during execution. These experiments add r ≠ 1 points at constant compute (c):

| Experiment | How bandwidth changes | How compute is held | Confounds recorded |
|---|---|---|---|
| Same-GPU competitor (contract J6) | An owned process on CUs 232–303 streams HBM at 0/25/50/75/100% duty (about 0–2.25 TB/s taken); its windowed bandwidth is logged | Decoder fixed on CUs 0–231 in every treatment | Shared L2/MALL/fabric contention, not only HBM; power/clock interference, isolated by an arithmetic-only competitor at 100% |
| TP at fixed total CUs | TP1 → TP2 → TP4 (→ TP8) multiplies HBM stacks, bandwidth and Infinity Cache | Total active CUs ≈ 304 (one GPU) and, separately, ≈ 152–160 | Inter-GPU all-reduce cost; per-GPU fixed overheads; slightly different totals after rounding to multiples of 8 |
| MI325X vs MI300X, compute-matched | MI325X HBM3E (+7–10% measured stream bandwidth) | MI325X restricted to 264 CUs to match MI300X's measured BF16 MFMA rate; full CUs in the same job for pairing | Different power cap and clock behaviour; HBM type; cross-node |

Results are labelled measured; any architecture projection drawn from them stays analytical.
