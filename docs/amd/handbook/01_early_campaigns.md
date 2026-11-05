# 01 — Early campaigns: MI300X 468885, MI325X 468949 / 469084, MI325X full 470176 + 481072

This chapter covers the first AMD measurements on branch `amd/mi300x-phase0`: one long MI300X campaign on
2026-10-18, two short MI325X jobs the same day, and the unattended MI325X repeat of the whole campaign
(2026-10-25 and 2026-10-28). Everything later (the "cubw" compute-vs-bandwidth follow-up, E0–E8) builds on
these results. Acronyms are defined on first use here; the shared glossary is in
[README.md](README.md).

## 0. Background you need first

### 0.1 The research question

The governing plan is `docs/shoreline_aware_research_plan_2.0.md` ("plan 2.0"). Its question, quoted in
`docs/AMD_PORT_PLAN.md`:

> Can some GPU compute resources be reallocated to on-chip SRAM, to external-memory PHY/controller capacity
> and delivered HBM bandwidth, or to both, to improve LLM decode latency/throughput and energy per token
> under explicit silicon, shoreline, package, thermal and cost constraints?

"Shoreline" is the die edge where memory interfaces (PHY = physical layer) sit. HBM = high-bandwidth memory.
Real hardware cannot be re-floorplanned, so these campaigns measure *motivation* evidence: how much decode
(generating one token per step) depends on compute, on bandwidth and on on-chip cache.

### 0.2 The measurement contract (M1–M8, A–F, J0–J11)

The methods come from `docs/shoreline_hopper_measurements.md` (written for NVIDIA Hopper; "the common
contract"). The job order comes from `docs/h200_shoreline_jobs.md`. In short:

| ID | What the contract asks for (paraphrased from `docs/shoreline_hopper_measurements.md` §3–4) |
|---|---|
| M1 | Strongest graph-mode and eager baselines; timelines, launch counts, three acquisitions |
| M2 | Logical bytes/FLOPs vs physical HBM reads/writes per decode tick |
| M3 | Per-kernel arithmetic, occupancy, stalls, launch geometry |
| M4 | Independent service microbenchmarks: stream, strided, gather, pointer chase, arithmetic, mixed |
| M5 | Normal vs competing-read cache-history pressure (does cache retention across tokens matter?) |
| M6 | Software compute-allocation sweep at nominal 1.0 / .75 / .5 / .3 of the SMs (CUs on AMD) |
| M7 | Repeat M4 service controls under exactly the M6 allocations |
| M8 | Long-window qualified energy; prefill and larger batch as counterexamples |
| A–D | Latency chains, compute service, memory service, concurrent compute+memory (on AMD: folded into M4) |
| E–F | Layer timing / instrumentation controls; model bounds `T = C + M − X + O` |

The latency model (contract §4): T = measured step time, C = FLOPs / compute rate, M = bytes / bandwidth,
X = overlap, O = residual. The feasible O interval is `[max(0, T−C−M), T−max(C,M)]`. A nonempty interval is
consistency, not validation. Sensitivity uses c (compute scale), r (delivered memory-service scale) and
f (fraction of memory traffic eliminated): `C_new = C/c`, `M_new = M(1−f)/r`.

Job order J0–J11 (from `docs/h200_shoreline_jobs.md`): J0 inventory, J1 baselines (M1), J2 traffic (M2),
J3 kernels (M3), J4 service (M4, A–D), J5 allocation sweeps (M6/M7), J6 same-GPU memory competitor,
J7 retention (M5), J8 layer timing / bounds (E/F), J9 c/r/f sensitivity, J10 energy (M8), J11 traces.

### 0.3 How the AMD port mapped the contract

`docs/AMD_PORT_PLAN.md` ("AMD mapping of the required measurements") translated each item:

| Contract | AMD route used in these campaigns |
|---|---|
| NVIDIA green contexts / SM allocation (M6/M7) | `HSA_CU_MASK=<physical index>:0-(k-1)` per process; k a multiple of 8 so each of the 8 XCDs (accelerator complex dies, 38 CUs each) loses CUs equally |
| Nsight Systems (M1 timelines) | `rocprofv3 --kernel-trace --marker-trace` |
| Nsight Compute (M2/M3 counters) | `rocprofv3 --pmc` (per-dispatch counters) and `rocprof-compute` 3.3.0 (Omniperf) |
| nvidia-smi / NVML energy (M8) | `amd-smi` energy accumulator and socket power through a pynvml-style shim (`amd_nvml`) |
| CUDA microbenchmarks (M4, A–D) | new HIP program `scripts/amd/svc_bench.hip`, built as `$WORK/bin/svc_bench_gfx942` (`--offload-arch=gfx942`) |
| vLLM CUDA build | prebuilt AMD container `rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0`, run with Apptainer |
| Tensor parallelism | forbidden by the contract; **explicitly authorized by the project owner on 2026-10-18** for TP2/TP4/TP8 as labelled extra cases (`docs/AMD_PORT_PLAN.md`, "Multi-GPU extension") |

Two AMD facts made the port interesting (`docs/AMD_PORT_PLAN.md`): MI300X has 8 HBM3 stacks plus a 256 MB
memory-side Infinity Cache (also called MALL, memory attached last-level cache) in front of per-XCD 4 MB L2
caches; and MI325X has the same gfx942 compute (304 CUs) with faster HBM3E, so MI300X vs MI325X is a
near-same-compute bandwidth pair (confounded by power cap, clocks and HBM type).

### 0.4 Statistics and labels used everywhere in this chapter

- Fixed by the contract (`docs/AMD_PORT_PLAN.md`): three independent process **acquisitions** per
  condition; ≥3 warmups and 10 timed trials per acquisition; 64 generated tokens, so 63 pure-decode ticks;
  seed 20261013; BF16; no quantization.
- **ms/step** = HIP-event elapsed time / decode iterations (`scripts/amd/summarize_timing.py`).
- **95% CI**: hierarchical bootstrap, 10,000 draws, seed 20261018, resampling acquisitions then trials
  within each acquisition (`scripts/amd/summarize_timing.py` docstring).
- **Clock gate** (declared before looking at data): a condition is clock-qualified when pooled GFX and
  memory clock samples inside all trial windows stay within max/min ≤ 1.05. It almost always failed on
  these power-capped GPUs, so results are **observational** (clock-confounded). It was reported, never used
  to drop data.
- Labels (`docs/amd/shoreline_opportunity_report.md`): **measured** = real GPU; **analytical** = computed
  from measured inputs; **profiled** = under a profiler (timings perturbed, never baselines).
- Models: "bridge" = Qwen2.5-1.5B-Instruct; qwen7b = Qwen2.5-7B-Instruct; mistral7b =
  Mistral-7B-Instruct-v0.3; qwen14b / qwen32b = Qwen2.5-14B/32B-Instruct (revisions pinned in
  `results_published/amd_20261018_mi300x/configs/*_base.json`). Case names read
  `<model>_<graph|eager>_b<batch>_c<context>`.

### 0.5 Shared machinery: the runner, the container and the "lanes"

All decode timing goes through `scripts/controlled_vllm_experiments.py` (repo root `scripts/`), run inside
the vLLM container by `scripts/amd/vllm_in_container.sh`. That wrapper records a clean environment and gives
each process its own Triton/vLLM/Inductor compile cache (a shared cache caused launch failures, see 1.0).
`--mode timing` does the 3 warmups + 10 trials; `--mode energy` runs a long generation; `--external-profiler`
marks a profiled run.

A **lane** is a bash script that pins one experiment family to one GPU and loops over acquisitions and cases.
Lanes skip acquisitions whose `run.json` already says `complete`, and rename failed directories to
`*_failed_<time>` (failures are never deleted). The lane scripts in `scripts/amd/`:

| Lane script | Family | What it does |
|---|---|---|
| `lane_m1_baselines.sh <uuid> <root>` | M1 | every config in `configs/index.json` (except C128 smoke), acq 0–2, graph and eager interleaved |
| `lane_cases.sh <uuid> <root> <subdir> <cfg>…` | M1 extras | same, for named configs (GPU-6 repro, representative text) |
| `lane_profile.sh` | M1 timelines / M2 | rocprofv3 trace and counters on vLLM cases (first version) |
| `lane_counters.sh <uuid> <root> <case>…` | M2/M3 | 4 counter groups per case, one group per rocprofv3 run, up to 4 attempts each |
| `lane_rocprof_compute.sh <idx> <root>` | M3 | curated rocprof-compute sets with a 600 s hard limit each |
| `service_sweep_lane.sh <uuid> <idx> <root>` → `service_sweep.py` | M4/M7 | 34 microbenchmark conditions × CU levels 304/232/152/96 × 3 acq + 1 counter acq |
| `lane_pressure.sh <uuid> <root> "<model> <b> <c>"…` → `pressure_bench.py` | M5 | normal vs pressure, HF compiled StaticCache path |
| `lane_cumask.sh <uuid> <idx> <root> <case>…` | M6 | CU levels 304/232/152/96 × acq 0–2; FMA proof run under each mask first |
| `lane_energy.sh` / `lane_energy_cu.sh` | M8 | long-window energy at full CUs / at 304, 152, 96 CUs |
| `lane_prefill.sh` | M8 counterexample | prefill wall time vs CU level (2 generated tokens per run) |
| `lane_tp.sh <tp> <uuids> <idxs> <root> <case>…` → `tp_vllm_experiments.py` | TP extension | TP-N full and compute-matched, 3 acq, up to 3 attempts |
| `lane_competitor.sh <uuid> <idx> <root> <case>…` | J6 | decoder on CUs 0–231, owned competitor on 232–303 |

Summaries are rebuilt on the login node (CPU only) by `scripts/amd/refresh_and_publish.sh`, which calls
`summarize_timing.py`, `summarize_service.py`, `summarize_pressure.py`, `summarize_energy.py`,
`analyze_counters.py`, `summarize_m6.py`, `summarize_tp.py`, `summarize_competitor.py`, `latency_bounds.py`,
`classify_kernels.py`, `consolidate_shoreline.py` and finally `publish_results.py` (copies compact files to
`results_published/` and writes a SHA-256 manifest of raw files).

---

## #1 MI300X first campaign — job 468885

- **At a glance:** partition `mi3008x`; 8× AMD Instinct MI300X (gfx942, 304 CUs, 192 GB HBM3, 750 W cap);
  node k002-003; 2026-10-18 12:11:55 – 18:57 CDT; 6 h 45 min of a 12 h limit; **cancelled by the operator
  when all planned work was done** (not a failure). No frozen release: scripts were written and fixed during
  the allocation (commits `585d77cb` port plan … `eb7efdfa` final report, all 2026-10-18; see
  `git log`). Status: complete (sub-experiment exceptions listed below).
  Sources: `docs/amd/experiment_log.md` (top), `docs/amd/shoreline_opportunity_report.md`.
- **Question:** the first AMD pass over the whole contract. Is batch-1 decode limited by bandwidth,
  arithmetic, cache, or something else on a GPU with a large Infinity Cache? How far can compute be cut
  before decode slows? Does delivered bandwidth at fixed compute matter?
- **Design (whole campaign):** one interactive 8-GPU allocation; each experiment family ran as a lane on its
  own GPU, all lanes concurrently. Software: ROCm 7.2.0 host, amdgpu 6.16.13; HF route on the `pytorch/2.10.0`
  module (torch 2.10.0+rocm7.1, Python 3.12, transformers 4.57.6); vLLM route = the container above (vLLM
  0.23.1.dev1+g9ddef7117, torch 2.11.0+rocm7.14.1). Focused cases: bridge B1/C1024 and B8/C4096; qwen7b and
  mistral7b B1 and B8 at C4096; qwen14b and qwen32b B1/C4096; graph and eager each (16 production M1
  conditions plus a smoke).
- **How it was run — GPU roles** (`docs/amd/experiment_log.md`, "GPUs used"):

| GPU (physical index) | ID | Role |
|---|---|---|
| 0 | `GPU-e5d1f2010c8b902a` | M1 baselines (later TP8, all GPUs) |
| 1 | `GPU-ff01d97deccdb374` | HF smoke, profiling, counters (later TP4) |
| 2 | `GPU-a72ad0fc3c4efc83` | M4/M7 service sweep, then Qwen7B batch × CU sweep (later TP4) |
| 3 | `GPU-870c8c5c21427604` | M6 CU-mask sweep, prefill-vs-CU (later TP4) |
| 4 | `GPU-3bb38eeb949c256a` | M8 energy, energy-vs-CU (later TP2, TP152) |
| 5 | `GPU-05c7e838addcf74b` | profiling, counters (later TP4) |
| 6 | `GPU-77e3d22110fc21c1` | rocprof-compute; M1 cross-GPU repro |
| 7 | `GPU-3c9417ac76059843` | M5 pressure, then M6 for Mistral/14B/32B (later TP2) |

  Lanes shared host CPU, PCIe root complexes and node power; no foreign users were on the node
  (`docs/amd/experiment_log.md`, "Concurrency disclosure").

- **Timeline (CDT, from `docs/amd/experiment_log.md`):** 12:12 J0; 12:39 vLLM graph smoke (bridge B1/C128,
  2.18 ms/step, exact outputs); 12:40 M1 lane; 12:43 HF StaticCache gate passed; 12:47 service sweep;
  12:48 profiling, CU-mask and energy lanes; 12:50–13:05 M5; 12:53–13:30 counter-crash investigation; 13:41 M1
  complete; 15:19 all 44 counter groups complete; 14:45 TP2 pilot; 15:22 TP4; 17:05 owner asks for more
  fixed-compute bandwidth evidence (J6, TP at ~160 CUs, MI325X matched job); 17:08–17:36 TP8; 17:49–18:31
  J6; 18:23–18:55 gap fills; 18:56 monitor stopped, allocation cancelled.

### 1.0 Campaign-wide tool findings (read before trusting any sub-result)

From `docs/amd/experiment_log.md`, "Tool findings" and "Additional tool and runtime findings":

| Finding | Consequence |
|---|---|
| `rocprofv3 --pmc` works on compute nodes | hardware traffic counters usable (they were blocked on H100) |
| A 1 GiB copy gives FETCH_SIZE 1,048,588 KB, WRITE_SIZE 1,048,576 KB | FETCH/WRITE_SIZE trusted as L2→memory bytes |
| No Infinity Cache (MALL) or data-fabric counters exposed | counted bytes are an **upper bound** on HBM bytes |
| FETCH_SIZE + WRITE_SIZE in one pass exceeds the counter budget and hangs rocprofv3 | one derived counter per pass; watchdog (`rocprof_watchdog.sh`) |
| Per-dispatch counters crash/hang on real LLM runs right after runtime Triton kernels load | exclude `^(_|triton)` kernels; up to 4 attempts per group; failed attempts kept |
| `HSA_CU_MASK` uses the **physical** device index; comma lists are ignored; ranges work | masks are written `idx:0-(k-1)` |
| Contiguous CU indices interleave across the 8 XCDs; an XCD left with an empty mask ignores it | contiguous k (multiple of 8) trims every XCD equally |
| Shared Triton cache across concurrent containers → "hip_utils.so could not be loaded" | per-process caches from ~13:35 |
| Energy lane first pass failed silently (missing temperature call in the shim; exit code 0) | lanes judge success by `run.json` status |
| Sporadic `unspecified launch failure` in ~1 in 20 vLLM runs, no ECC errors | bounded retries; failed runs kept |
| GFX clock rises when CUs are masked (~1650 → ~1950 MHz for 7B B1); memory clock ramps 900→1300 MHz | clock gate fails for most conditions |

### 1.1 J0 — environment, identity, counters, CU-mask checks

- **Design / how:** inventory of SKU, IDs, CU count, caches, clocks, power, ROCm/driver; a trial `--pmc`
  run; CU-mask checks with the FMA microbenchmark; container start. Outputs in
  `results_published/amd_20261018_mi300x/j0/` (`environment.json`, `amd_smi_static_gpu.txt`,
  `amd_smi_topology.txt`, `rocprofv3_avail.txt`); raw trial runs (`pmc_FETCH_SIZE`, `pmc_WRITE_SIZE`,
  `pmc_TCC_HIT_TCC_MISS`, `pmc_TCC_EA0_RDREQ_DRAM_…`) under `results/amd_20261018_mi300x/j0/`.
- **Results:**
  - GPU identity: amdsmi `hip_uuid` differs between amdsmi versions and torch's ROCm UUID is not the amdsmi
    UUID. The package settled on `GPU-<ASIC serial>` and verifies the HIP device by PCI bus.
  - Topology: all 8 GPUs connected by xGMI with weight 15 and 1 hop to every other GPU (`amd_smi_topology.txt`).
  - Counters permitted and validated (table in 1.0). No MALL counters.
  - CU masks: per-stream exact bitmask of 38 contiguous CUs gave 15.9 TFLOP/s FMA; an every-8th list (all on
    one XCD) gave the full rate, proving the XCD interleave.
  - A case manifest (`analysis/shoreline_case_manifest.json`) lists every family's acquisitions, attempts
    and disposition.
- **What it means:** AMD gives what Hopper did not: counters and a way to restrict compute that also covers
  graph and library kernels. The caveat is that HSA_CU_MASK is a software proxy (masked CUs keep their area,
  leakage and clock domain).

### 1.2 M1 — baselines (plus GPU-6 repro, text check, HF gate)

- **Question:** what is the strongest correct baseline and where does step time go?
- **Design:** 16 production conditions (8 cases × graph/eager), 3 acquisitions × 10 trials, 64 tokens,
  vLLM 0.23 in the container (graph = HIP graphs; eager = `enforce_eager`). Engine settings (from
  `configs/bridge_graph_b1_c1024.json`): `max_model_len` 8192, `gpu_memory_utilization` 0.7,
  `max_num_seqs` = batch. Extras: (a) the 8 graph cases repeated on GPU 6 for cross-GPU reproducibility;
  (b) "representative text" prompts at bridge B1/B4 C128 (`m1_text`); (c) the HF StaticCache correctness gate
  and compiled timing on the HF route (`results/amd_20261018_mi300x/hf/bridge_hf_compiled_b1_c128`).
- **How:** `scripts/amd/lane_m1_baselines.sh GPU-e5d1f201… <root>` on GPU 0; `lane_cases.sh … m1_gpu6`
  on GPU 6; `lane_cases.sh … m1_text` for text. A profiled eager trace
  (`m1_profile/trace_bridge_eager_b1_c1024/analysis.json`) came from `lane_profile.sh`.
- **Analysis:** `summarize_timing.py` (bootstrap CI, clock join with `node_monitor.jsonl`).
- **Results** (`results_published/amd_20261018_mi300x/m1/summary.txt`, graph mode):

| Case | ms/step | 95% CI | tok/s | Eager ms/step |
|---|---:|---|---:|---:|
| bridge B1 C1024 | 2.187 | 2.182–2.194 | 457 | 8.281 |
| bridge B8 C4096 | 3.155 | 3.145–3.169 | 2536 | 14.210 |
| qwen7b B1 C4096 | 4.682 | 4.667–4.704 | 214 | 7.839 |
| qwen7b B8 C4096 | 6.654 | 6.601–6.696 | 1202 | 13.537 |
| mistral7b B1 C4096 | 5.111 | 5.072–5.176 | 196 | 9.493 |
| mistral7b B8 C4096 | 7.601 | 7.566–7.643 | 1053 | 16.559 |
| qwen14b B1 C4096 | 8.638 | 8.625–8.651 | 116 | 13.299 |
| qwen32b B1 C4096 | 17.532 | 17.460–17.617 | 57 | 19.004 |

  - GPU 6 repeat: within 0.1–0.8% of GPU 0 (e.g. qwen32b 17.669 vs 17.532; `m1_gpu6/summary.txt`).
  - Text check: bridge B1/C128 2.180 ms, B4/C128 2.591 ms (both clock-qualified).
  - Eager profiled trace (bridge B1/C1024): 337 kernel launches per step; kernel busy ≈1.9 ms in a ≈12.5–13.7 ms
    profiled step (report: GPU busy ≈15% of step wall). Eager is host-bound and treated as a separate program,
    not a baseline.
  - Clock gate: 2 of 17 conditions pass (bridge graph B1/C1024 and bridge eager B1/C1024); a post-hoc
    interior-80% diagnostic gives the same 2.
- **What it means:** graph mode is the baseline for everything later. Eager vLLM at batch 1 measures the host,
  not the GPU. MI300X cannot meet the ≤5% clock envelope without clock locking, which the contract forbids, so
  every later comparison is observational.
- **Caveats:** the M1 runs started before `prefill_wall_ms` was recorded (added 13:32), so M1 has no prefill
  times. A smoke condition (bridge graph B1/C128, n=1) is in the summary and marked partial.

### 1.3 M2 — traffic counters (logical vs physical bytes)

- **Question:** does decode read each weight and KV byte about once per step, or is there extra physical
  traffic that more cache could remove?
- **Design:** 12 counter cases (eager for all models; graph for bridge B1, bridge B8, qwen7b B1, mistral7b B1)
  × 4 counter groups: `FETCH_SIZE`; `WRITE_SIZE`; `TCC_HIT TCC_MISS` (TCC = the L2); and
  `SQ_INSTS_VALU_MFMA_MOPS_BF16 SQ_WAVES GRBM_GUI_ACTIVE` (MFMA = matrix fused multiply-add). 16 generated
  tokens per profiled run; decode ticks 4, 5, 6 analysed. Logical demand U is computed from safetensor sizes
  and KV size.
- **How:** `lane_counters.sh` on GPUs 1 and 5 (lane logs `m2_counters/lane_ff01d97d.log`, `lane_05c7e838.log`,
  also `lane_e5d1f201.log` on GPU 0 during gap fills). Each group: `rocprofv3 --kernel-trace --marker-trace
  --pmc <group> --kernel-exclude-regex '^(_|triton)'`, own process group, killed on crash-handler hang or
  900 s; up to 4 attempts; GPU lock wait. Analysis: `analyze_counters.py`.
- **Results** (`m2_counters/summary.txt`, tick 5; report §3):

| Case | Read GB | Logical U GB | D/U | Written MB | L2 hit fraction |
|---|---:|---:|---:|---:|---:|
| bridge B1 C1024 (eager / graph) | 3.133 / 3.131 | 3.117 | 1.005 | 7 | 0.13 |
| bridge B8 C4096 (eager / graph) | 4.174 / 4.175 | 4.028 | 1.036 | 161–164 | 0.21 |
| qwen7b B1 C4096 | 14.41 | 14.38 | 1.002 | 31 | 0.06 |
| qwen7b B8 C4096 eager | 16.47 | 16.02 | 1.028 | 550 | 0.09 |
| mistral7b B1 C4096 | 14.80 | 14.77 | 1.002 | 11–12 | 0.05 |
| mistral7b B8 C4096 eager | 18.78 | 18.53 | 1.014 | 171 | 0.17 |
| qwen14b B1 C4096 eager | 28.85 | 28.79 | 1.002 | 95 | 0.05 |
| qwen32b B1 C4096 eager | 65.14 | 65.05 | 1.001 | 130 | 0.04 |

  Issued MFMA work / useful FLOPs (MFMA count × 512, the rocprof-compute convention, not independently
  verified) is 3.6–4.9 at B1 (GEMVs padded to 16-row MFMA tiles) and 1.9–2.2 at B8.
- **What it means:** at batch 1, L2→memory reads match the logical demand within 0.2–0.5% (3–4% at B8): every
  weight and KV byte streams once per step and the L2 keeps nothing across steps. More on-chip capacity can
  only help if it holds data across steps, which M5 tests.
- **Caveats:** bytes are measured before the Infinity Cache, so they are an upper bound on HBM bytes. D/U ≈ 1
  is aggregate, not a per-address proof. Runtime Triton kernels (fused norm/activation/rotary) were excluded
  and so are uncounted. Failed and interrupted attempts are kept in the raw root.

### 1.4 M3 — rocprof-compute (and the eager timeline)

- **Question:** per-kernel arithmetic, occupancy and memory behaviour.
- **Design:** rocprof-compute 3.3.0 from `$WORK/venvs/rocprof_compute_py312` (pandas pinned 2.2.3). Targets:
  three service benchmarks (`svc_read4g`, `svc_mixed16`, `svc_gather1g`) and the HF decode workload
  `decode_profile_bench.py` (bridge, 8 tokens, eager attention) at B1/C1024 and B8/C4096 with five sets each
  (`launch_stats`, `mem_thruput`, `compute_thruput_util`, `compute_thruput_flops`, roofline), filtered to
  Tensile GEMV/GEMM kernels (`-k Cijk`).
- **How:** `lane_rocprof_compute.sh 6 <root>` on GPU 6, each set with a 600 s hard limit, `rocpd` output
  format, and a clean ROCm 7.2 `amd-smi` shim (`$WORK/bin/rpc_shim`) because the 7.1 `amd-smi` could not open
  `/dev/dri`.
- **Results:** partial. The full default profile crashed real LLM kernels (illegal address / launch failure)
  even with `-k Cijk`; CSV output failed (`Agent_Id` str/int merge). Published analyses:
  `m3_rocprof_compute/hf_bridge_b1_c1024_compute_thruput_flops_rocpd.analysis.txt`,
  `…_compute_thruput_util_rocpd.analysis.txt`, `hf_bridge_b8_c4096_mem_thruput_rocpd.analysis.txt`. Per
  `m3_rocprof_compute/lane_sets.log`, rocpd attempts of `svc_mixed16`, `svc_gather1g`, B1 `mem_thruput`, B1
  roofline, B8 `compute_thruput_util/flops` were killed at the limit, and B8 roofline ended rc=1.
- **What it means:** M3 contributed kernel inventories (for example, Tensile `Cijk_…MT16x16x512_MI16x16x1…`
  GEMV tiles dominate the HF B1 decode) but no complete per-kernel stall picture. The per-kernel
  classification (1.11) was built from rocprofv3 counters instead.

### 1.5 M4 (A–D) — service microbenchmarks at full CUs

- **Question:** what can this GPU deliver for each access pattern? These are the independent rates
  (β for bandwidth, P for compute) used in the bounds.
- **Design:** `service_sweep.py` defines 34 conditions: read at 2 MiB–16 GiB footprints (straddling the
  4 MB L2 and 256 MB MALL), write and copy 4 GiB, strided (stride 2/4/8/16), gather (64 MiB, 1 GiB), single
  pointer chase at 1 MiB–4 GiB, multi-chain chase (8–32,768 chains), FMA, MFMA, and mixed compute+memory
  (intensity 0–256). Each: 3 acquisitions × (10 trials, 3 warmups) + 1 counter acquisition (FETCH_SIZE and
  WRITE_SIZE in separate rocprofv3 passes). Repeated at 304/232/152/96 CUs (M7, 1.8).
- **How:** `service_sweep_lane.sh GPU-a72ad0fc… 2 <root>` on GPU 2; binary `$WORK/bin/svc_bench_gfx942`
  (source `scripts/amd/svc_bench.hip`), default grid 8 × CUs blocks of 256 threads. Summary
  `summarize_service.py`.
- **Results** (`m4_service/summary.txt`, 304 CUs, 0 failures):

| Control | MI300X |
|---|---:|
| Read 8 MiB / 32 MiB (L2-resident) | 31,968 / 24,242 GB/s |
| Read 128 / 256 MiB (Infinity Cache) | 5,639 / 5,655 GB/s |
| Read 512 MiB / 1 GiB / 4 GiB / 16 GiB (HBM) | 4,547 / 4,594 / 4,549 / 4,478 GB/s |
| Write / copy 4 GiB | 4,853 / 3,816 GB/s |
| Gather 64 MiB / 1 GiB | 878 / 631 GB/s |
| Dependent-load latency at 1 MiB / 128 MiB / 4 GiB | 103 / 245 / 268 ns |
| BF16 MFMA (register-resident) | 1,071 TFLOP/s |
| FP32 FMA | 94.9 TFLOP/s |

  The report notes ~32k independent chains (~1.2 MB in flight) are needed to approach HBM bandwidth.
- **What it means:** the Infinity Cache serves streaming reads only ~1.25× faster than HBM (5.65 vs
  4.55 TB/s). The reference rates for later analysis are β = 4.55 TB/s and P = 1071 TFLOP/s.

### 1.6 M5 — retention / cache-history pressure

- **Question:** does any data survive in L2 or the Infinity Cache from one token to the next, and does that
  help? If evicting it costs nothing, extra SRAM capacity would not help today's decode.
- **Design:** `pressure_bench.py` on the HF compiled StaticCache path (not vLLM). In one process, trials
  alternate "normal" and "pressure". Pressure reads a 1 GiB buffer (> L2 + 256 MB MALL) after step k
  synchronizes and before step k+1's start event, so it is outside every timed interval. 64 tokens, 3 warmups,
  10 trials, 3 acquisitions. Cases: bridge B1 at C1024/2048/4096, bridge B8/C4096, qwen7b B1/C4096. Outputs
  must be identical.
- **How:** `lane_pressure.sh GPU-3c9417ac… <root> 'bridge 1 1024' …` on GPU 7 (host module env,
  `scripts/amd/env.sh`), 12:50–13:05, 15 acquisitions. Summary `summarize_pressure.py`.
- **Results** (`m5_pressure/summary.txt`):

| Case | normal ms/step | pressure / normal [CI] | outputs identical |
|---|---:|---|---|
| bridge B1 C1024 | 4.992 | 1.0140 [1.0111, 1.0168] | yes |
| bridge B1 C2048 | 7.114 | 1.0085 [1.0064, 1.0108] | yes |
| bridge B1 C4096 | 10.834 | 1.0051 [1.0040, 1.0069] | yes |
| bridge B8 C4096 | 13.095 | 1.0029 [1.0021, 1.0041] | yes |
| qwen7b B1 C4096 | 14.369 | 1.0023 [1.0020, 1.0027] | yes |

- **What it means:** evicting all cache contents between tokens costs 0.2–1.4%, even where the whole KV
  (~115 MB at bridge B1/C4096) fits in the Infinity Cache. Cross-token retention is worth little for this
  decode. This is the main evidence against "compute → SRAM capacity" on this hardware.
- **Caveats:** the HF path is slower than vLLM graph (4.99 vs 2.19 ms for bridge B1/C1024), so the absolute
  share could differ for vLLM. The pressure read is an intervention, not a guaranteed flush.

### 1.7 M6 — CU-mask sweep of decode

- **Question:** how far can compute be cut before decode slows (the "compute-only trim" candidate)?
- **Design:** CU levels 304 / 232 / 152 / 96 (nominal 1.0 / .75 / .5 / .3, multiples of 8 so all XCDs are
  trimmed equally); masks `HSA_CU_MASK=<idx>:0-(k-1)`. Before each level, an FMA run under the same mask is
  saved as proof (`proof_gpu<idx>_cu<k>_a<acq>.json`). Cases: bridge graph B1/C1024, bridge eager B1/C1024,
  bridge graph B8/C4096, qwen7b graph B1 and B8 C4096 (GPU 3); mistral7b B1/B8, qwen14b B1, qwen32b B1
  (GPU 7, extension from 13:06); qwen7b graph C1024 at B1/B4/B16/B32/B64 (batch × CU sweep, GPU 2). 56 cells,
  3 acquisitions each. Each level has its own 304-CU reference run on the same GPU.
  Pre-registered check: a 5% noninferiority envelope on the upper 95% CI bound (contract J5).
- **How:** `lane_cumask.sh <uuid> <idx> <root> <cases>` on GPUs 3, 7, 2; gap fills 18:23–18:55.
  Summary `summarize_timing.py` and `summarize_m6.py` (`analysis/m6_prefill_summary.txt`).
- **Results** (ratio to 304 CUs on the same GPU, `analysis/m6_prefill_summary.txt`; median GFX MHz rises as
  CUs are cut):

| Case | 232 | 152 | 96 | GFX MHz 304 → 96 |
|---|---:|---:|---:|---|
| bridge graph B1 C1024 | 1.015 | 1.128 | 1.209 | 1958 → 2088 |
| bridge graph B8 C4096 | 1.098 | 1.208 | 1.246 | 1874 → 2078 |
| qwen7b B1 C4096 | 1.056 | 1.132 | 1.239 | 1628 → 1917 |
| mistral7b B1 C4096 | 1.048 | 1.109 | 1.166 | 1759 → 2032 |
| qwen14b B1 C4096 | 1.046 | 1.097 | 1.105 | 1622 → 1855 |
| qwen32b B1 C4096 | 1.034 | 1.084 | 1.092 | 1348 → 1679 |
| qwen7b B8 C4096 | 1.192 | 1.425 | 1.637 | 1680 → 2095 |
| qwen7b B4 / B16 / B32 / B64 C1024 | 1.135 / 1.216 / 1.086 / 1.177 | 1.306 / 1.484 / 1.255 / 1.378 | 1.402 / 1.699 / 1.418 / 1.514 | — |
| bridge eager B1 C1024 | 1.006 | 0.978 | 0.974 | host-bound |

  Clock gate: 19/56 cells pass (`m6_cumask/summary.txt`). Noninferiority (≤5%) at 232 CUs holds only for the
  largest models at batch 1 (32B; 14B and Mistral borderline).
- **What it means:** at batch 1, cutting to 32% of CUs slows decode only 9–24%, least for the largest models;
  at batch ≥4 the penalty is 40–70%. A naive "c = FLOP capacity" model predicts <1% (see 1.10), so CUs matter
  for more than arithmetic: they generate the memory requests (M7).
- **Later revisions:** the cubw follow-up measured finer levels and found R(n) = T(n)/T(304) is **non-monotone**
  on MI300X/MI325X, traced to vLLM's wvSplitK kernel and to shader-engine imbalance of CU masks (see the
  cubw chapters and `docs/amd/mi300x_se_review_20261105.md`). The 4-level curve here is a coarse sampling of
  that shape and is clock-confounded.

### 1.8 M7 — service under the M6 masks

- **Question:** does trimming CUs reduce independently attainable bandwidth, and for which patterns?
- **Design / how:** the same `service_sweep.py` run, all 34 conditions at 232/152/96 CUs on GPU 2 (part of
  1.5).
- **Results** (ratio to 304 CUs, `m4_service/summary.txt`):

| Control | 232 | 152 | 96 |
|---|---:|---:|---:|
| BF16 MFMA | 0.739 | 0.529 | 0.356 |
| FP32 FMA | 0.813 | 0.644 | 0.471 |
| HBM read 4 GiB | 0.949 | 0.880 | 0.874 |
| Copy 4 GiB | 0.997 | 0.953 | 0.895 |
| L2-resident read 32 MiB | 0.869 | 0.630 | 0.459 |
| Random gather 1 GiB | 0.963 | 0.717 | 0.495 |
| Mixed, intensity 64 | 0.865 | 0.644 | 0.448 |
| Dependent latency (1 chain) | ≈1.0 (0.97–1.0) | | |

- **What it means:** HBM streaming keeps 87% of its rate at 96 CUs, but cache-resident reads and gathers fall
  roughly with CU count. CUs are request generators; removing them removes in-flight requests. This coupling
  became the "bandwidth per CU" (B_CU) idea in later reports (box in
  `docs/amd/shoreline_opportunity_report.md`, added 2026-11-03).

### 1.9 M8 — energy, energy under masks, and prefill

- **Question:** is J/token measurable and qualified; does compute trimming save energy; where does a
  decode-oriented design lose (prefill, larger batch)?
- **Design:**
  - Energy: 8 graph cases from `configs_energy/` with long generations (`energy_generated_tokens`, e.g. 2048
    for bridge B1, 4096 for qwen7b B1), 3 acquisitions × 3 trials, sensor interval 0.02 s, minimum window 5 s.
    Qualification = amdsmi energy counter vs integrated sampled socket power.
  - Energy under masks: qwen7b and qwen32b B1/C4096 at 304/152/96 CUs.
  - Prefill vs CUs: qwen7b B1 and B8, bridge B8, qwen32b B1 (all C4096) at 304/232/152/96; each run generates
    2 tokens and records `prefill_wall_ms` (first `engine.step`).
- **How:** `lane_energy.sh` and `lane_energy_cu.sh` on GPU 4 (`--mode energy`); `lane_prefill.sh` on GPU 3.
  Summaries `summarize_energy.py`, `summarize_m6.py`.
- **Results:**

| Case (graph) | J/token [CI] | mean W | acq |
|---|---|---:|---:|
| bridge B1 C1024 | 1.246 [1.240, 1.256] | 563 | 3 |
| bridge B8 C4096 | 0.249 | 607 | 3 |
| qwen7b B1 C4096 | 3.594 | 749 | 3 |
| qwen7b B8 C4096 | 0.659 | 751 | 3 |
| mistral7b B1 C4096 | 3.908 | 740 | 3 |
| mistral7b B8 C4096 | 0.782 | 748 | 2 |
| qwen14b B1 C4096 | 6.742 | 752 | 3 |
| qwen32b B1 C4096 | 14.001 | 751 | 3 |

  (`m8_energy/summary.txt`; counter vs power agreement within 0.2–1.4%, windows 7–77 s per the report.)
  Energy under masks (report §6): 7B 3.55 → 3.94 → 4.23 J/token and 32B 13.5 → 14.5 → 14.6 at
  304 → 152 → 96 CUs, board power ~722–751 W throughout. Prefill ratio vs 304 CUs at 96 CUs: 2.08 (bridge B8),
  2.20 (qwen7b B1), 2.17 (qwen7b B8), 2.25 (qwen32b B1) (`analysis/m6_prefill_summary.txt`).
- **What it means:** decode runs at the 750 W cap from 7B upward. Masking CUs does not lower power (clocks rise
  instead) and raises J/token 10–19%. A software mask cannot show silicon savings. Prefill is the clear loser
  of compute trimming (2.1–2.3× at 96 CUs).
- **Caveats:** energy is socket scope and internal consistency only, not absolute calibration. The Mistral B8
  third acquisition was cut by the allocation end. The first energy pass failed silently (1.0) and was rerun.
  Some prefill cells have 2 acquisitions (n shown in `m6_prefill_summary.txt`). The report table lists
  mistral7b B1 as 3.916 J/token; the current summary file says 3.908 (small refresh difference).

### 1.10 E/F latency bounds and J9 c/r/f sensitivity

- **Question:** how much of a step is explained by bandwidth (M) and arithmetic (C)? What is left (O)?
- **Design (analytical):** `scripts/amd/latency_bounds.py`. T = M1 graph ms/step; F = useful FLOPs per step
  (2·B·linear parameters + attention term); D = counter bytes (FETCH + WRITE, eager program, ticks 4–6);
  P = 1071 TFLOP/s; β = 4.55 TB/s. C = F/P, M = D/β, O_feasible = [max(0, T−C−M), T−max(C,M)]. c/r/f grid:
  c ∈ {1, .75, .5, .25}, r ∈ {.7, 1, 1.2, 1.4}, f ∈ {0, .05, .1, .2}, under same-O/X assumptions and a
  free-overlap envelope.
- **How:** run on the login node by `refresh_and_publish.sh`; output `analysis/shoreline_latency_bounds.json`
  and `.txt`.
- **Results** (`analysis/shoreline_latency_bounds.txt`):

| Case | T ms | C ms | M ms | M/T | O feasible ms |
|---|---:|---:|---:|---:|---|
| bridge B1 C1024 | 2.187 | 0.003 | 0.690 | 0.32 | 1.494–1.497 |
| bridge B8 C4096 | 3.155 | 0.028 | 0.954 | 0.30 | 2.173–2.201 |
| qwen7b B1 C4096 | 4.682 | 0.015 | 3.175 | 0.68 | 1.493–1.508 |
| qwen7b B8 C4096 | 6.653 | 0.118 | 3.742 | 0.56 | 2.794–2.911 |
| mistral7b B1 C4096 | 5.111 | 0.015 | 3.256 | 0.64 | 1.840–1.855 |
| mistral7b B8 C4096 | 7.601 | 0.122 | 4.167 | 0.55 | 3.311–3.434 |
| qwen14b B1 C4096 | 8.638 | 0.030 | 6.363 | 0.74 | 2.245–2.275 |
| qwen32b B1 C4096 | 17.532 | 0.065 | 14.349 | 0.82 | 3.118–3.183 |

- **What it means:** decode is not bandwidth-saturated for small and medium models: streaming the bytes at
  4.55 TB/s takes 30–82% of the step; arithmetic is under 2%; 1.5–3.4 ms per step is residual (latency,
  small kernels, launches). **M6 falsifies reading c as FLOP capacity:** C/c predicts <1% slowdown at
  c = 0.32, but 9–24% was measured. Any architecture model must tie "compute removal" to request-generation
  and on-chip delivery capacity.

### 1.11 Per-kernel classification of the residual O

- **Question:** what is O made of?
- **Design (analytical over profiled inputs):** `scripts/amd/classify_kernels.py` (added 2026-10-20, commit
  `5fc029aa`). For each M2 case, the FETCH, WRITE and MFMA counter runs are aligned per decode step by kernel
  position, with names checked. Per kernel: bytes, issued FLOPs, duration, ideal = max(bytes/β, FLOPs/P).
  Labels declared before inspection: memory-bound (≥60% of β), compute-bound (≥60% of P), small/overhead
  (<10 µs otherwise), latency-bound (the rest), uncounted (excluded Triton kernels). 50% and 70% thresholds are
  in the JSON. Idle/launch = clean M1 T minus profiled busy union.
- **Results** (graph mode, ms per step, `analysis/kernel_classification.txt`):

| Case | T | Memory kernels at peak | Their excess | Latency-bound | Small (<10 µs) | Uncounted | Idle/launch |
|---|---:|---:|---:|---:|---:|---:|---:|
| bridge B1 C1024 | 2.187 | 0.600 | 0.164 | 0.025 | 0.660 | 0.457 | 0.282 |
| bridge B8 C4096 | 3.155 | 0.171 | 0.071 | 1.369 | 0.608 | 0.468 | 0.468 |
| qwen7b B1 C4096 | 4.682 | 3.047 | 0.280 | 0.015 | 0.514 | 0.452 | 0.375 |
| mistral7b B1 C4096 | 5.111 | 3.132 | 0.347 | 0 | 0.555 | 0.536 | 0.540 |

  Memory-bound GEMVs achieve 3.6 TB/s (1.5B), 4.2 TB/s (7B), 4.6 TB/s (32B eager) of counted bytes.
- **What it means:** the weight-streaming GEMVs are near the stream rate; their excess is only 6–7% of T. The
  residual is many small per-layer kernels (about 56 attention calls of ~6 µs, KV writes, fused Triton
  norm/activation/rotary) plus launch gaps: 37–64% of T at batch 1. For small models, fewer/fused kernels
  would help more than bandwidth or SRAM.
- **Caveats:** durations come from profiled runs (counter collection serializes dispatches). Later work
  identified the dominant GEMV as vLLM's wvSplitK skinny-GEMM kernel (cubw E6).

### 1.12 TP2 / TP4 / TP8 at full and fixed total compute

- **Question (owner-authorized 2026-10-18):** at fixed total compute, does spreading decode across more GPUs
  (more HBM stacks, bandwidth and Infinity Cache) speed it up? This gives r > 1 at c ≈ 1.
- **Design:** vLLM `tensor_parallel_size` N with RCCL over xGMI. Variants:
  - **full**: every GPU at 304 CUs;
  - **matched**: each GPU masked to `per = ⌈(304/N)/8⌉·8` CUs: TP2 152, TP4 80, TP8 40 per GPU (total 304,
    320, 320) — verified in the lane logs, e.g. `HSA_CU_MASK=0:0-39;…;7:0-39` for TP8;
  - **matched152** (added 17:05): total ≈160 (TP2 80 per GPU, TP4 40 per GPU), compared with TP1 at 152 CUs
    from M6.
  Cases: bridge B1/C1024, qwen7b B1 and B8, mistral7b, qwen14b, qwen32b (C4096). TP8 only for mistral7b,
  qwen14b, qwen32b (other head counts not divisible by 8). 3 acquisitions, 64 tokens. Correctness: output IDs
  compared with TP1.
- **How:** `lane_tp.sh <tp> <uuids> <idxs> <root> <cases>` → `scripts/amd/tp_vllm_experiments.py`
  (records every participating GPU ID; metric = synchronized wall ms/step). GPU sets (lane logs in
  `results_published/amd_20261018_mi300x/tp/`): TP2 on 4,7 (from 14:50); TP4 on 1,2,3,5 (from 15:22); TP8 on
  0–7 (17:08–17:36); matched152 TP2 on 4,5 and 6,7, TP4 on 4,5,6,7 (from ~17:37). After a TP4 qwen14b hang at
  16:00 (Triton cache race), runs used persistent per-case compile caches, rank-failure detection and up to
  3 attempts. Summary `summarize_tp.py` → `results/amd_20261018_mi300x/tp/summary.txt`.
- **Results** (speedup over TP1 = TP1 HIP-event ms / TP-N wall ms):

| Case | TP2 full | TP4 full | TP8 full | TP2 matched | TP4 matched | TP8 matched | TP2 m152 | TP4 m152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bridge B1 C1024 | 1.01 | 1.12 | n/a | 0.92 | 0.89 | n/a | — | — |
| qwen7b B1 | 1.34 | 1.71 | n/a | 1.16 | 1.33 | n/a | 1.12 | 1.05 |
| qwen7b B8 | 1.38 | 1.87 | n/a | 1.11 | 1.21 | n/a | — | — |
| mistral7b B1 | 1.31 | 1.64 | 1.90 | 1.19 | 1.29 | 1.07 | — | — |
| qwen14b B1 | 1.44 | 1.93 | 2.28 | 1.28 | 1.45 | 1.23 | 1.21 | 1.11 |
| qwen32b B1 | 1.63 | 2.43 | 3.16 | 1.48 | 1.86 | 1.74 | 1.39 | 1.33 |

  Batch-1 output IDs match TP1 in every case; at batch 8 they differ (reduction order), recorded as a finding.
- **What it means:** at fixed total CUs, more delivered bandwidth helps in proportion to the bandwidth share
  of the step (32B > 14B > 7B ≈ Mistral > 1.5B, which loses to communication). Gains saturate and reverse
  (TP8 matched < TP4 matched). With few CUs per GPU the extra bandwidth is worth less (TP4 × 40 < TP2 × 80):
  bandwidth needs enough request issuers per HBM stack.
- **Caveats:** TP ratios mix HIP-event (TP1) and wall-clock (TP-N) scope. The eight GPUs also bring 8× L2/MALL
  capacity, power budget and all-reduce cost — not an isolation of bandwidth. No TP energy (windows < 5 s).
  Two TP2 full cells have 2 acquisitions (bridge and qwen32b, per `tp/summary.txt`; the report footnote
  mentions only qwen32b, whose first attempt hit a launch failure). `docs/amd/interim_synthesis_20261028.md`
  says "38 per GPU" for TP8 matched; the lane logs show 40.

### 1.13 J6 — same-GPU memory competitor

- **Question (added 17:05 at the owner's request):** at constant decoder compute, how much does decode slow
  when part of the memory service is taken away? (r < 1 at c fixed.)
- **Design:** decoder masked to CUs 0–231 in every treatment; an owned competitor process on the disjoint CUs
  232–303 of the same GPU. Treatments: solo; read25/50/75/100 (streaming read of a 1 GiB buffer at the
  requested duty cycle; the host sleeps between launches); fma100 (register FMA, a power/clock control). The
  competitor logs achieved GB/s or TFLOP/s in 0.25 s windows, joined to the decoder's trial windows. Treatment
  order rotates per acquisition (`(k + 2·acq) mod 6`) so drift does not align with one treatment.
  3 acquisitions. Cases: bridge B1/C1024, qwen7b B1 and B8 C4096, qwen32b B1/C4096.
- **How:** `lane_competitor.sh <uuid> <idx> <root> <cases>` with binary `$WORK/bin/svc_bench_gfx942_v2`
  (`--seconds 900 --duty d`, launched with `setsid` and `HSA_CU_MASK=<idx>:232-303`); the decoder gets
  `HSA_CU_MASK=<idx>:0-231` and `RESEARCH_ALLOWED_PIDS=<competitor pid>` so the foreign-process check allows
  the competitor. Four lanes (`j6_competitor/lane_0..3.log`). First pass at 17:37 failed (the runner needs a
  fresh output directory that the lane had pre-created); fixed, rerun 17:49–18:31. Summary
  `summarize_competitor.py`.
- **Results** (ratio to solo; competitor GB/s during decoder trials in parentheses;
  `j6_competitor/summary.json`):

| Case | solo ms | read25 | read50 | read75 | read100 | fma100 |
|---|---:|---:|---:|---:|---:|---:|
| bridge B1 C1024 | 2.296 | 1.042 (482) | 1.065 (946) | 1.101 (1355) | 1.147 (1934) | 1.009 |
| qwen7b B1 C4096 | 4.921 | 1.051 (375) | 1.105 (723) | 1.158 (1077) | 1.236 (1488) | 1.044 |
| qwen7b B8 C4096 | 7.858 | 1.048 (413) | 1.115 (788) | 1.166 (1161) | 1.259 (1623) | 1.032 |
| qwen32b B1 C4096 | 18.224 | 1.059 (285) | 1.114 (558) | 1.172 (815) | 1.247 (1094) | 1.069 |

- **What it means:** decode slows nearly linearly with bandwidth taken, at constant decoder CUs (up to
  15–26%). Power/clock interference alone (FMA) explains 1–7%. The competitor gets less bandwidth next to
  heavier decoders, so the two genuinely contend. This is shared-service sensitivity (L2, MALL and fabric are
  shared too), not a change of memory interfaces.
- **Caveats / later revisions:** qwen32b read25 has 2 complete acquisitions (one failed; the report says 3).
  The interim synthesis notes competitor cells raise the decoder's clock, biasing slowdowns downward. Whether
  the masked decoder "leaks" onto the competitor's CUs was not tested here; cubw built leak tests v1–v4 for
  that (E3), and the arithmetic competitor ran at only ~0.6 of its solo rate beside the decoder in the
  follow-up pilot.

### Campaign #1 — what it means overall

The report's bottom line (`docs/amd/shoreline_opportunity_report.md`): decode streams every byte once per
step; extra cache barely helps (M5); decode is far from bandwidth-saturated for small models (E/F); compute
trimming costs batch-1 decode little but prefill and larger batches a lot (M6, M8); and delivered bandwidth at
fixed compute moves decode time (TP matched, J6, MI325X). This favoured **compute → delivered bandwidth (plus
request-generation capacity)** over **compute → SRAM** for dense small-batch decode, with the caveat that
20–70% of step time is neither bandwidth nor arithmetic.

### Campaign #1 — failures and caveats (summary)

- Clock gate failed for 15/17 M1 conditions and 37/56 M6 cells; all comparisons are observational.
- Counter collection crashed or hung intermittently; Triton kernels excluded; every failed attempt kept.
- rocprof-compute mostly failed on LLM kernels.
- ~1 in 20 vLLM runs had `unspecified launch failure`; retried, kept under `*_failed_*`.
- 17:07: two stale TP8 waiters started duplicate lanes while GPU 0 was busy; both killed within a minute; one
  clean TP8 lane ran.
- HSA_CU_MASK is a software proxy; masked CUs are not removed silicon.

### Campaign #1 — files

- Methodology: `docs/AMD_PORT_PLAN.md`, `docs/shoreline_hopper_measurements.md`, `docs/h200_shoreline_jobs.md`.
- Results: `docs/amd/shoreline_opportunity_report.md`; execution log `docs/amd/experiment_log.md` (top
  sections); artifact index `docs/amd/artifacts.md`.
- Compact evidence (138 files): `results_published/amd_20261018_mi300x/` — `j0/`, `m1/`, `m1_gpu6/`,
  `m1_text/`, `m1_profile/`, `m2_counters/`, `m3_rocprof_compute/`, `m4_service/`, `m5_pressure/`,
  `m6_cumask/`, `m8_energy/`, `m8_energy_cu/`, `m8_prefill/`, `tp/` (lane logs), `j6_competitor/`,
  `analysis/`, `configs*/`, `cohort.json`, `raw_manifest.sha256` (6.67 GB indexed).
- Raw: `/home1/yanggon/00_first_dir/26_memory_centric_gpu_h100/results/amd_20261018_mi300x/` (includes
  `tp/summary.txt`, lane logs such as `m6_lane7_rerun.log`, `tp8_waiter.sh`, node `monitor/`).
- Commits: `585d77cb` … `eb7efdfa` (2026-10-18), `00d8a58b` (469084 results), `5fc029aa` (kernel classifier).

---

## #2 MI325X comparison — job 468949

- **At a glance:** partition `mi3258x`; 1 of 8 MI325X used (GPU 0, `GPU-aa71f114af3bed6d`); node not
  recorded in the published files (469084 later ran on k002-001 with the same GPU ID); 2026-10-18
  15:21:32–16:03:02 CDT (2490 s, Omnistat report in `$WORK/mi325x_compare_468949.out`); limit 3 h;
  script added in commit `a31513e8`; status complete.
- **Question:** MI325X has the same gfx942 compute (304 CUs) and faster HBM3E. Does decode speed up in step
  with bandwidth? Intended as the same-compute bandwidth pair (c ≈ 1, r > 1).
- **Design:** self-contained batch job on GPU 0 only: M4 service sweep at full CUs (all 34 conditions, 3 acq)
  and the 8 graph M1 cases copied from the MI300X configs (3 acq × 10 trials), with a GPU monitor.
- **How:** `sbatch scripts/amd/mi325x_compare.sbatch` (runs `inventory.py`, `gpu_monitor.py`,
  `service_sweep.py --cus 304`, `lane_m1_baselines.sh`, then the summarizers). Ran while 468885 was still
  running on another node.
- **Results** (`results_published/amd_20261018_mi325x_job468949/`):

| Quantity | MI325X | MI300X (468885) | ratio |
|---|---:|---:|---:|
| Read 4 GiB / 16 GiB | 5,002 / 4,795 GB/s | 4,549 / 4,478 | 1.10 / 1.07 |
| Write 4 GiB / copy 4 GiB | 5,314 / 4,084 GB/s | 4,853 / 3,816 | 1.10 / 1.07 |
| Read 256 MiB (MALL) / 32 MiB (L2) | 5,654 / 24,357 GB/s | 5,655 / 24,242 | 1.00 |
| Chase latency 1 MiB / 4 GiB | 104 / 267 ns | 103 / 268 | 1.0 |
| BF16 MFMA / FP32 FMA | 1,230 / 107.8 TFLOP/s | 1,071 / 94.9 | 1.15 / 1.14 |

| Graph decode ms/step | MI325X | MI300X | speedup |
|---|---:|---:|---:|
| bridge B1 C1024 | 2.107 | 2.187 | 1.04 |
| bridge B8 C4096 | 3.040 | 3.155 | 1.04 |
| qwen7b B1 C4096 | 4.308 | 4.682 | 1.09 |
| qwen7b B8 C4096 | 6.242 | 6.654 | 1.07 |
| mistral7b B1 C4096 | 4.746 | 5.111 | 1.08 |
| mistral7b B8 C4096 | 7.088 | 7.601 | 1.07 |
| qwen14b B1 C4096 | 7.779 | 8.638 | 1.11 |
| qwen32b B1 C4096 | 15.299 | 17.532 | 1.15 |

  Clock gate: 1/8 pass (bridge B1).
- **What it means:** decode is 1.04–1.15× faster, more for larger models, but MI325X also has +14–15% compute
  (higher power limit), so this is **not** a same-compute pair. That motivated #3.
- **Caveats:** cross-node, cross-SKU, cross-date; clocks power-managed.
- **Files:** `results_published/amd_20261018_mi325x_job468949/` (17 files: `m1/`, `m4_service/`, `configs/`,
  `environment.json`, `selected_gpu.txt`, `cohort.json`, `raw_manifest.sha256`); raw
  `results/amd_20261018_mi325x_job468949/`; report §8c of `docs/amd/shoreline_opportunity_report.md`.

## #3 MI325X compute-matched — job 469084

- **At a glance:** partition `mi3258x`; GPU 0 (`GPU-aa71f114af3bed6d`); node k002-001; 2026-10-18
  19:47:20–20:44:20 CDT (Omnistat, `$WORK/mi325x_matched_469084.out`; the log says 19:46–20:41); limit 3 h;
  script added in commit `f1fa6966`; status complete, no failed runs; published 2026-10-19 (`00d8a58b`).
- **Question:** if MI325X is restricted to the CU count that matches MI300X's measured MFMA rate, is it still
  faster? That would be r > 1 at c ≈ 1.
- **Design:** 264 CUs = 304 × 1071/1230 rounded to a multiple of 8 (`mi325x_matched.sbatch` header). Full
  (304) and matched (264) conditions interleaved in the same job on the same GPU (paired), 3 acquisitions.
  M4 service at 304 and 264. Cases: qwen7b B1 and B8, mistral7b B1, qwen14b B1, qwen32b B1 (graph, C4096).
- **How:** `sbatch scripts/amd/mi325x_matched.sbatch`; masks `HSA_CU_MASK=0:0-263`; runner as in M1.
- **Results** (`results_published/amd_20261018_mi325x_matched_job469084/`):

| Service | 304 CUs | 264 CUs | ratio |
|---|---:|---:|---:|
| BF16 MFMA | 1,231 TFLOP/s | 997 | 0.81 |
| FP32 FMA | 108.2 | 98.0 | 0.91 |
| Read 4 GiB | 4,992 GB/s | 4,427 | 0.89 |
| Read 16 GiB | 4,632 | 4,257 | 0.92 |
| Copy 4 GiB | 4,083 | 4,082 | 1.00 |

| Decode ms/step | MI325X 304 | MI325X 264 | MI300X 304 (468885) |
|---|---:|---:|---:|
| qwen7b B1 | 4.314 | 4.945 | 4.682 |
| qwen7b B8 | 6.207 | 8.151 | 6.654 |
| mistral7b B1 | 4.748 | 5.273 | 5.111 |
| qwen14b B1 | 7.783 | 8.729 | 8.638 |
| qwen32b B1 | 15.319 | 17.446 | 17.532 |

- **What it means:** the match failed to isolate bandwidth. Masking to 264 CUs gave 997 (not 1071) TFLOP/s and
  also cut streaming bandwidth below MI300X's full-CU rate (r ≈ 0.97, c ≈ 0.93 per the report). MI325X@264 is
  as slow as or slower than MI300X@304. It independently confirms M7: removing CUs removes request-generation
  capacity and so deliverable bandwidth.
- **Caveats:** no condition passes the clock gate (all clockOK=False in `m1/summary.txt`).
- **Files:** `results_published/amd_20261018_mi325x_matched_job469084/` (12 files); raw
  `results/amd_20261018_mi325x_matched_job469084/`; report §8c.

---

## MI325X full campaign: design shared by #5 and #6

The MI325X full campaign is an **unattended repeat of the whole MI300X campaign** on one 8× MI325X node, with
the same scripts, configs and lane roles. It is a separate cohort: never pooled with MI300X
(`cohort.json` rule: "cohorts are never pooled").

- **Driver:** `scripts/amd/mi325x_full.sbatch` (commit `e79a7aae`, 2026-10-19; `#SBATCH -p mi3258x -N 1
  -t 12:00:00`). Raw root `$WORK/results_amd/amd_<date>_mi325x_full_job<id>`, symlinked into `results/`.
  Progress in `<root>/RUN_STATUS.log`. A dry run (`DRY_RUN=1 bash scripts/amd/mi325x_full.sbatch`) prints
  every lane command with placeholder GPUs and checks inputs without GPU use.
- **Phase 0 — preflight** (`scripts/amd/preflight_node.py`): writes `gpus.tsv` and `preflight.json`; checks
  8 GPUs, torch, the container (vLLM version string), models, a counter run (FETCH on a 64 MiB copy:
  65,548 KB vs 65,536 expected) and a CU mask (FMA 102.9 TFLOP/s full vs 64.5 at 152 CUs in 470176). Then 51
  input files are checked.
- **Phase 1 — eight single-GPU lanes in parallel** (each family run twice, "p1" and "p2", so the second pass
  fills sporadic failures):

| GPU | Lanes |
|---|---|
| 0 | `lane_m1_baselines.sh` (M1) |
| 1 | `lane_counters.sh` bridge eager B1, bridge graph B1, bridge eager B8, qwen7b eager B1 and B8 (M2) |
| 2 | `service_sweep_lane.sh` (M4/M7), then `lane_cumask.sh` qwen7b C1024 B1/B4/B16/B32/B64 |
| 3 | `lane_cumask.sh` bridge graph/eager B1, bridge B8, qwen7b B1 and B8 (M6), then `lane_prefill.sh` |
| 4 | `lane_energy.sh` 8 cases (M8), then `lane_energy_cu.sh` qwen7b and qwen32b |
| 5 | `lane_counters.sh` bridge graph B8, mistral7b eager/graph B1, mistral7b eager B8, qwen14b and qwen32b eager B1, qwen7b graph B1 |
| 6 | `lane_rocprof_compute.sh` (M3), then `lane_cases.sh … m1_gpu6` (repro) |
| 7 | `lane_pressure.sh` (M5), `lane_cumask.sh` mistral7b B1/B8, qwen14b, qwen32b, `lane_cases.sh … m1_text` |

- **Phase 2:** TP4 on GPUs 0–3 and TP2 on 4,5 (full and matched); J6 competitor on GPU 6 (bridge B1, qwen7b B1)
  and GPU 7 (qwen32b B1, qwen7b B8).
- **Phase 3:** TP8 on all 8 GPUs (mistral7b, qwen14b, qwen32b).
- **Phase 4:** TP at ~160 total CUs: TP4 on 0–3 (40 CUs each) and TP2 on 4,5 and 6,7 (80 each)
  (`TP_VARIANTS=matched152 TP_MATCH_TOTAL=152`).
- Before each phase the remaining Slurm time is checked; a phase that cannot finish is recorded as skipped.

GPU IDs (`preflight.json`): 0 `GPU-aa71f114af3bed6d` (the same GPU used by 468949 and 469084), 1
`GPU-9d95c7d965754a23`, 2 `GPU-e2ec0ce821299f33`, 3 `GPU-183763f7d77caea7`, 4 `GPU-9d10d9528c677241`,
5 `GPU-24d82f87c792a716`, 6 `GPU-ce14bc05c9da4a79`, 7 `GPU-4b2c461fdef6ba6a`.

Attempts 1 and 2 of job 470176 (2026-10-20 and 2026-10-23) ended in NODE_FAIL; their partial phase-1 roots
(`$WORK/results_amd/amd_20261020_mi325x_full_job470176`, `…amd_20261023_…`) are kept as failures and are
superseded. They are described in [05_minor_checks.md](05_minor_checks.md).

## #5 MI325X full campaign, phase 1 — job 470176 (attempt 3)

- **At a glance:** partition `mi3258x`; 8× MI325X; node k002-001; 2026-10-25 17:57:51 – 2026-10-26 05:57 CDT
  (TIMEOUT at the 12 h limit); all phase-1 lanes ended rc=0 by 21:20:13; driver commit `e79a7aae` (the exact
  commit checked out at submission is not recorded). Status: phase 1 complete; phases 2–4 **never started**.
- **Question:** does the MI300X picture (M1–M8, E/F) hold on a GPU with ~10% more bandwidth and ~13–15% more
  compute? Same cases, same statistics.
- **Design / how:** phase 0 and phase 1 as in the shared design above. Timeline from
  `RUN_STATUS.log`: preflight 17:57:51–17:58:01; M5 done 18:06; service sweep done 18:49; M1 p1 done 19:06;
  M6 GPU 3 done 19:26; prefill done 20:43; last counter lane done 21:20.
- **Bug:** the phase-1 `wait` also waited for the background node monitor, which never exits, so the job idled
  ~8.6 h until TIMEOUT (no "phase1 end" line in `RUN_STATUS.log`). Fixed in commit `d0131769`
  (2026-10-26): `wait_lanes` skips the monitor, and `PHASES` selects phases.
- **Completion** (`docs/amd/mi325x_full_campaign_results_20261028.md` §2):

| Family | Cells / groups | Acquisitions | Failed attempts kept |
|---|---|---|---|
| M1 baselines | 16 | 3/3 each | 3 |
| M1 GPU-6 repro / text | 8 / 2 | 3/3 | 0 |
| M6 CU masks incl. batch sweep | 56 | 3/3 | 6 |
| M8 prefill vs CU | 4 × 4 | 3/3 | 1 |
| M8 energy | 8 | **2/3** (every `a0` failed twice: `AMDSMI_STATUS_FILE_ERROR`) | 3 dirs + retries |
| M8 energy vs CU | 6 | 1–2 of 3 | 1+ |
| M5 pressure | 5 | 3, but bridge B1/C1024 and qwen7b at 2 | – |
| M4 service | 140 rows | 3 each | – |
| M2 counters | 12 × 4 = 48 groups | 48/48 done | 113 failed/interrupted attempts |
| M3 rocprof-compute | 13 sets | **5/13** completed | – |

- **Results** (MI325X; `results_published/amd_20261025_mi325x_full_job470176/`):

| M1 graph ms/step | MI325X | MI300X | MI325X/MI300X |
|---|---:|---:|---:|
| bridge B1 C1024 | 2.208 | 2.187 | 1.009 |
| qwen7b B1 C4096 | 4.351 | 4.682 | 0.929 |
| qwen7b B8 C4096 | 6.270 | 6.654 | 0.942 |
| mistral7b B1 C4096 | 4.815 | 5.111 | 0.942 |
| qwen14b B1 C4096 | 7.816 | 8.638 | 0.905 |
| qwen32b B1 C4096 | 15.757 | 17.532 | 0.899 |

| M6 ratio to 304 CUs, graph | MI325X 232 / 152 / 96 | MI300X 232 / 152 / 96 |
|---|---|---|
| qwen32b B1 | 1.031 / 1.115 / 1.124 | 1.034 / 1.084 / 1.092 |
| qwen14b B1 | 1.091 / 1.152 / 1.172 | 1.046 / 1.097 / 1.105 |
| qwen7b B1 | 1.090 / 1.204 / 1.353 | 1.056 / 1.132 / 1.239 |
| qwen7b B8 | 1.280 / 1.602 / 1.896 | 1.192 / 1.425 / 1.637 |
| bridge B1 | 1.045 / 1.167 / 1.260 | 1.015 / 1.128 / 1.209 |

  - Batch sweep at 96 CUs (qwen7b C1024): 1.314 (B1), 1.562 (B4), 1.989 (B16), 1.596 (B32), 1.716 (B64).
  - Prefill at 96 CUs: qwen32b B1 2.70 (MI300X 2.25); qwen7b B8 2.53 (2.17).
  - M4 at 304 CUs: read 4 GiB 5,021 GB/s, MFMA 1,215 TFLOP/s, FMA 106.5. At 96 CUs MI325X keeps 80.8% of
    read bandwidth and 31.3% of MFMA (MI300X 87.4% and 35.6%).
  - M8 J/token (2 acq): qwen32b B1 15.43 at 967 W (MI300X 14.00 at 751 W); qwen7b B1 3.96 at 893 W;
    qwen7b B8 0.708 at 853 W. Energy vs CU (qwen32b B1): 15.45 (304, 1 acq), 16.89 (152, 2), 16.80 (96, 1).
  - M2: read bytes per tick within 0.14–3.6% of logical demand (qwen32b eager 65.14 vs 65.05 GB).
  - M5: pressure/normal 1.002–1.011, outputs identical (`m5_pressure/summary.txt`).
  - E/F (`analysis/shoreline_latency_bounds.txt`): M/T 0.28 (bridge) to 0.83 (qwen32b); O feasible
    1.46–3.40 ms.
  - Clock gate: M1 1/16 (bridge eager B1), M6 6/56, repro 0/8, text 0/2; GFX ~1405–2095 MHz.
- **What it means:** the MI300X picture holds. MI325X is ~6–10% faster on the medium and large models (the
  decode-time ratio tracks the inverse streaming-bandwidth ratio, not the MFMA ratio), uses ~10% more energy
  per token, and is **more** CU-sensitive at 152 and 96 CUs. The interim synthesis reads this as: with more
  bandwidth, compute becomes a larger share of each step. Later cubw work expressed this as higher bandwidth
  per CU (B_CU).
- **Caveats:** observational (clock gate); cross-node, cross-SKU, cross-date comparison with MI300X; energy has
  2/3 acquisitions; rocprof-compute 5/13 sets (`mem_thruput`, roofline and three service sets failed with
  illegal memory access / profiling execution failed).
- **Files:** `results_published/amd_20261025_mi325x_full_job470176/` (108 files, incl. `RUN_STATUS.log`,
  `preflight.json`, `gpus.tsv`, `analysis/`, `raw_manifest.sha256` indexing 3.18 GB); raw
  `$WORK/results_amd/amd_20261025_mi325x_full_job470176`; Slurm output `$WORK/mi325x_full_470176.out`;
  report `docs/amd/mi325x_full_campaign_results_20261028.md`; reconciliation
  `docs/amd/cubw_e0_reconciliation_20261026.md`.

## #6 MI325X full campaign, phases 2–4 — job 481072

- **At a glance:** partition `mi3258x`; 8× MI325X; node k002-001; 2026-10-28 07:20:33–09:17:06 CDT; submitted
  2026-10-26 with `PHASES="2 3 4"` after a dry run; driver includes the `d0131769` fix; status complete
  (180/180 `run.json` complete: TP 108, J6 72; 0 failed attempts). Allocation used before submission:
  286.38 / 450 (`docs/amd/experiment_log.md`).
- **Question:** repeat the fixed-compute bandwidth experiments (TP at full / matched / ~160 CUs and the J6
  competitor) on MI325X.
- **Design / how:** phase 0 preflight (FMA 99.5 TFLOP/s full in this job), phase 1 skipped ("not selected"),
  then phases 2–4 as in the shared design (`RUN_STATUS.log`: phase 2 07:20–08:34; TP8 08:34–09:02; phase 4
  09:02–09:17). The TP summary needs the TP1 baselines from #5, so it was produced through a view directory
  `$WORK/results_amd/view_mi325x_full_470176_481072` that links 470176 `m1`/`m6_cumask` with 481072 `tp`;
  summarizers unchanged.
- **Results** (`results_published/amd_20261028_mi325x_full_job481072/tp/summary.txt`,
  `j6_competitor/summary.json`):

| TP speedup vs TP1 | TP2 full | TP4 full | TP8 full | TP2 matched | TP4 matched | TP8 matched | TP2 m152 | TP4 m152 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bridge B1 C1024 | 1.01 | 1.07 | n/a | 0.88 | 0.81 | n/a | — | — |
| qwen7b B1 | 1.31 | 1.66 | n/a | 1.09 | 1.17 | n/a | 1.04 | 0.93 |
| qwen7b B8 | 1.35 | 1.77 | n/a | 1.03 | 1.09 | n/a | — | — |
| mistral7b B1 | 1.25 | 1.55 | 1.73 | 1.13 | 1.16 | 0.93 | — | — |
| qwen14b B1 | 1.41 | 1.82 | 2.08 | 1.18 | 1.21 | 1.01 | 1.10 | 0.95 |
| qwen32b B1 | 1.62 | 2.34 | 2.90 | 1.36 | 1.53 | 1.40 | 1.24 | 1.11 |

  Output IDs matched TP1 in every row except TP4 qwen7b B8 (full and matched).

| J6 ratio to solo (MI325X) | solo ms | read25 | read50 | read75 | read100 (GB/s taken) | fma100 |
|---|---:|---:|---:|---:|---:|---:|
| bridge B1 C1024 | 2.225 | 1.051 | 1.085 | 1.090 | 1.159 (1650) | 1.001 |
| qwen7b B1 C4096 | 4.729 | 1.031 | 1.060 | 1.090 | 1.157 (1492) | 0.998 |
| qwen7b B8 C4096 | 7.912 | 1.045 | 1.051 | 1.074 | 1.111 (1623) | 1.004 |
| qwen32b B1 C4096 | 16.572 | 1.029 | 1.073 | 1.101 | 1.146 (1233) | 1.015 |

- **What it means:** the same direction as MI300X but weaker. Matched TP still helps the large models (qwen32b
  TP4 matched 1.53× vs MI300X 1.86×) and saturates earlier (TP8 matched < TP4 matched; TP4 × 40 CUs is slower
  than TP1 at 152 for 7B and 14B). MI325X absorbs a memory competitor better (qwen32b read100 1.146 vs 1.247 on
  MI300X), and the FMA control costs almost nothing (≤1.5%). With the higher bandwidth per CU, taking bandwidth
  away matters less and taking CUs away matters more.
- **Caveats:** TP1 baselines come from #5, run on the same node 2.5 days earlier; TP ratios mix HIP-event and
  wall scope; TP also scales cache capacity and power; observational clocks.
- **Files:** `results_published/amd_20261028_mi325x_full_job481072/` (66 files: `tp/`, `j6_competitor/`,
  `RUN_STATUS.log`, `preflight.json`, `gpus.tsv`, configs, `raw_manifest.sha256` indexing 0.11 GB); raw
  `$WORK/results_amd/amd_20261028_mi325x_full_job481072`; Slurm output `$WORK/mi325x_full_481072.out`;
  report `docs/amd/mi325x_full_campaign_results_20261028.md`; cross-run reading
  `docs/amd/interim_synthesis_20261028.md`; publication commit `65af30ec`.

---

## How these early campaigns fed the later work

- The **compute-vs-bandwidth follow-up** (`docs/amd/compute_bandwidth_followup_20261026.md`, E0–E8) was
  written after #1–#5 to test their observational findings under randomized, gated designs. Its E0
  reconciliation (`docs/amd/cubw_e0_reconciliation_20261026.md`) took stock of 470176 and 481072 first.
- **M6 → E2:** the 4-level CU curve became a fine-grained R(n) study; it turned out non-monotone (wvSplitK
  kernel sizing and shader-engine balance). Treat the 232/152/96 ratios here as coarse samples.
- **M7 → E4 and B_CU:** the request-generation coupling was re-measured (E4) and summarized as HBM bandwidth
  per CU.
- **J6 → E3:** the competitor design was kept, with leak tests (v1–v4) added to prove the decoder stays off the
  competitor's CUs.
- **Clock confounding** was never removed: no clock or power changes were allowed in any campaign.
