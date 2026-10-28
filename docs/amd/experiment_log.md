# AMD MI300X experiment log (allocation 468885, node k002-003, 2026-10-18)

All times CDT. Result root: `results/amd_20261018_mi300x/` (large raw files stay in `$WORK`/results, see `docs/amd/artifacts.md`).

## Allocation and environment (J0)

- SLURM job 468885, partition `mi3008x`, node `k002-003`, 8× AMD Instinct MI300X (gfx942, 304 CUs, 192 GB HBM3, 750 W cap), started 12:11:55, limit 12 h.
- Host: ROCm 7.2.0 default, amdgpu 6.16.13; HF route uses the `pytorch/2.10.0` module (torch 2.10.0+rocm7.1, Python 3.12, transformers 4.57.6, numpy 2.4.3 vs lock 2.2.6).
- vLLM route: AMD container `rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0` (SHA256 017554e0…632e), vLLM 0.23.1.dev1+g9ddef7117, torch 2.11.0+rocm7.14.1, transformers 5.16.1, run via apptainer.
- GPU identity: amdsmi `hip_uuid` differs between amdsmi versions (host 26.1 returns the ASIC serial; the container returns torch's HIP UUID), and torch's ROCm `uuid` is not the amdsmi UUID. The package now uses `GPU-<ASIC serial>` and verifies the HIP device by PCI bus.
- GPUs used: 0 `GPU-e5d1f2010c8b902a` (M1), 1 `GPU-ff01d97deccdb374` (HF smoke, profiling, counters), 2 `GPU-a72ad0fc3c4efc83` (M4/M7 service), 3 `GPU-870c8c5c21427604` (M6), 4 `GPU-3bb38eeb949c256a` (M8 energy), 5 `GPU-05c7e838addcf74b` (profiling, counters), 6 `GPU-77e3d22110fc21c1` (rocprof-compute), 7 `GPU-3c9417ac76059843` (M5 pressure, then M6 for larger models).
- Concurrency disclosure: lanes ran simultaneously on different GPUs of the same node (shared host CPU, PCIe root complexes and node power). No foreign users were on the node. ROCr opens a small context (~27 KB VRAM, no GFX time) on every GPU for any HIP process, so the foreign-use check rejects only ≥64 MiB VRAM or GFX activity and records the rest.

## Tool findings

| Finding | Evidence | Consequence |
|---|---|---|
| Counter access is permitted | `rocprofv3 --pmc` works on compute nodes | Hardware traffic gates are reachable on AMD (they were blocked on H100) |
| Counters validated on a known-traffic copy | 1 GiB copy: FETCH_SIZE 1,048,588 KB, WRITE_SIZE 1,048,576 KB; TCC_EA0_RDREQ_DRAM×128 B and WRREQ_DRAM×64 B ≈ 1.07 GB each | FETCH/WRITE_SIZE are trusted for L2→memory bytes |
| No Infinity Cache (MALL) or data-fabric counters exposed | `rocprofv3-avail` list | Measured bytes are L2→memory-side traffic, an upper bound on HBM bytes; MALL effects come from timing controls |
| FETCH_SIZE + WRITE_SIZE in one pass exceeds the counter budget | "Request exceeds the capabilities of the hardware", then rocprofv3 hangs in its signal handler | One derived counter per pass; hard-kill timeouts and a watchdog |
| Per-dispatch counters crash/hang on real LLM workloads | vLLM eager: "AQL packet is malformed" / illegal address right after runtime Triton loads; HF: crash at lazily loaded `attn_fwd`; simple HIP microbenchmarks always fine | Exclude `^(_\|triton)` kernels from collection, up to 4 attempts per group, all attempts kept; rocprof-compute filtered with `-k Cijk` and eager attention |
| `HSA_CU_MASK` device index is the physical (pre-HIP-masking) index | Mask `0:` had no effect on GPU 2; `2:` worked | Lanes use the physical index |
| `HSA_CU_MASK` ignores comma lists; ranges work | every-8th list gave full FMA rate | Only contiguous ranges used |
| CU indices interleave across XCDs | per-stream exact bitmask: 38 contiguous CUs → 15.9 TFLOP/s; every-8th (all on one XCD) → full rate, i.e. an XCD left with an empty mask ignores it | Contiguous k (multiple of 8) masks trim every XCD equally; masks leaving an XCD empty are invalid |
| ROCm 7.1 `amd-smi` refuses `/dev/dri` access; mixed 7.1/7.2 amdsmi libraries break the CLI | rocprof-compute "No GPU detected" | rocprof-compute uses a clean-environment ROCm 7.2 `amd-smi` shim |
| Shared Triton cache across concurrent container processes | "hip_utils.so could not be loaded", unspecified launch failure (qwen14b cu232 a0) | From ~13:35, per-process Triton/vLLM/inductor caches; earlier completed runs passed exact-output validation |
| MI300X clocks are power-managed | GFX clock rises when CUs are masked (≈1650 → ≈1950 MHz for 7B B1); memory clock ramps 900→1300 MHz at trial start | Contract clock envelope (≤5%) often fails; reported as declared, with a separately labelled interior-sample diagnostic |

## Timeline

- 12:12 job start; J0 inventory, identity, counter, CU-mask, container checks.
- 12:39 vLLM graph smoke (bridge B1/C128): exact outputs, 2.18 ms/step.
- 12:40 M1 lane (GPU 0); 12:43 HF StaticCache gate passed (GPU 1).
- 12:46 rocprofv3 trace of vLLM eager works, with per-step and per-layer markers.
- 12:47 service sweep (GPU 2); 12:48 profiling lanes (GPU 1, 5), CU-mask lane (GPU 3), energy lane (GPU 4).
- 12:50 M5 pressure (GPU 7), complete at 13:05 (15 acquisitions).
- 12:53–13:30 counter-collection crash investigation (see tool findings); lanes restarted with exclusion and retries.
- 13:06 M6 extension lane on GPU 7 (Mistral, 14B, 32B).

## Additional tool and runtime findings (13:35–15:45)

| Finding | Evidence | Consequence |
|---|---|---|
| Energy lane first pass failed silently | `amd_nvml` lacked `nvmlDeviceGetTemperature`; vLLM printed the traceback but exited 0 | Shim fixed (hotspot temperature); lane restarted 13:3x; lanes judge success by `run.json` status |
| Watchdog kills left lock-holding orphans | next attempts failed instantly with `BlockingIOError` on the GPU lock | Counter lane rewritten: per-attempt process group, in-lane hang detection, lock wait |
| rocprof-compute 3.3 CSV path broken | `Agent_Id` str/int merge error; analysis fails under pandas 3.0.6 | `--format-rocprof-output rocpd` and pandas 2.2.3 in the tool venv |
| rocprof-compute full profile crashes LLM kernels | illegal address / launch failure in the first pass even with `-k Cijk`; service benchmarks profile fine (but also failed intermittently later) | Only curated single-pass sets on the HF workload; several sets/roofline still failed and are recorded |
| Sporadic `unspecified launch failure` | about 1 in 20 vLLM runs across GPUs, with or without CU masks or profilers (e.g. TP2 32B full a0, 7B B4 cu232); zero ECC errors | Bounded retries; all failed runs kept under `*_failed_*`; possibly related to every HIP process opening queues on all 8 GPUs while 8 lanes run |
| Prefill timing added | `controlled_vllm_experiments.py` now records `prefill_wall_ms` (first `engine.step`) | Present in M6/M8-prefill/TP runs after 13:32, absent from earlier M1 runs |

## Timeline (continued)

- 13:41 M1 complete (17 conditions × 3 acquisitions, 10 trials each).
- 13:44 counter lanes with Triton exclusion and retries; 15:19 all counter lanes complete (44 groups).
- 13:56 M6 lane GPU 3 complete; 14:34 GPU 7 extension complete; 15:21 batch-size × CU sweep (GPU 2) complete.
- 14:25 M8 energy first pass complete (fixed sensor); 14:47 representative text complete.
- 14:45 TP2 pilot OK; TP2 lane (GPUs 4,7) from 14:50; TP4 lane (GPUs 1,2,3,5) from 15:22.
- 15:20 MI325X comparison job 468949 started on `mi3258x` (separate node, self-contained).
- 15:22 prefill-vs-CU lane complete.

## Close-out (15:45–18:57)

- 16:00 TP4 qwen14b hung 30 min after one rank's Inductor/Triton cache race ("Cubin file saved by TritonBundler not found"); TP lanes switched to persistent per-case compile caches with rank-failure detection and retries.
- 17:05 Owner requested more bandwidth-at-fixed-compute measurements: added the same-GPU competitor (J6), TP at about 160 total CUs, and a compute-matched MI325X job (469084, queued for 2026-10-19 04:29).
- 17:07 Two stale TP8 waiters started duplicate lanes while GPU 0 was busy; both were killed within a minute and one clean TP8 lane ran 17:08–17:36.
- 17:37 J6 first pass failed (the runner requires a fresh output directory, which the lane had pre-created); fixed and rerun 17:49–18:31.
- 18:23–18:55 gap fills: all M6 cells at 3 acquisitions, all counter groups complete, energy at 3 acquisitions (Mistral B8 at 2).
- 18:56 node monitor stopped; allocation 468885 cancelled at 6 h 45 min elapsed with all planned work done.
- Clock gate: 2 of 17 M1 conditions pass the contract's ≤5% envelope, and the post-hoc interior-sample diagnostic gives the same 2 of 17; MI300X power management varies clocks by more than 5%.
- 2026-10-18 19:46–20:41: MI325X compute-matched job 469084 completed on k002-001 (no failed runs). 264 CUs gave 997 TFLOP/s BF16 (target 1071), and HBM stream fell 9–11% versus the same GPU at 304 CUs, so the pairing did not isolate bandwidth. Published 2026-10-19.

## 2026-10-26 — MI325X job 470176 attempt 3 and resume job 481072

- Attempt 3 (Oct 25 17:57 – Oct 26 05:57, k002-001) completed all of phase 1; every lane ended rc=0 by 21:20
  (`$WORK/results_amd/amd_20261025_mi325x_full_job470176`, which supersedes the partial attempts of Oct 20 and Oct 23).
- Phases 2–4 never started: the phase-1 `wait` also waited on the background node monitor, which never exits,
  so the job idled ~8.6 h until TIMEOUT. Fixed in `scripts/amd/mi325x_full.sbatch` (`wait_lanes` skips the
  monitor), and added `PHASES` to select phases.
- Resubmitted phases 2–4 only as job 481072 (`PHASES="2 3 4"`); dry run checked. Allocation used: 286.38 / 450.

## 2026-10-26 — compute/bandwidth follow-up (cubw_v1): E0 and submissions

- Owner authorized running the whole follow-up (docs/amd/compute_bandwidth_followup_20261026.md) without further
  intervention: E0, MI300X pilot and production on mi3008x, and a separate MI325X cohort job after 481072.
- E0 delivered: `scripts/amd/cubw/` (commits b428ec06, 2da9ae35) and
  [cubw_e0_reconciliation_20261026.md](cubw_e0_reconciliation_20261026.md). CPU fixtures 17/17 pass;
  svc_bench_v3 smoke-tested on an MI210 devel node; dry runs enumerate 216 (pilot) and 763 (MI325X auto) steps.
- Submitted from frozen release `$WORK/cubw_release/2da9ae35…`:
  - 481167 `cubw-mi300x-pilot` (mi3008x, 1:45:00) — plan `$WORK/cubw_plans/mi300x_pilot_20261026_144349_2da9ae35.json`.
  - 481168 `cubw-mi325x-auto` (mi3258x, 3:45:00, `afterany:481072`, user hold until the MI300X pilot validates
    the tools) — plan `$WORK/cubw_plans/mi325x_auto_20261026_144357_2da9ae35.json`.
- MI300X production is submitted only after the pilot is reviewed (gates, boundary level, cost).

## 2026-10-28 — MI300X pilot 481167, v1.1 repair, resubmissions; MI325X 481072 complete

- 481167 ran 08:34–09:06 on k002-002 (after ~42 h in the queue). Review:
  [cubw_pilot_review_20261028.md](cubw_pilot_review_20261028.md). Placement passed on all 8 GPUs; the v1
  lifecycle and outputs gates failed on tool/reader defects; leak and clock gates failed on measurements;
  graph-mode rocprofv3 counters crashed/hung. Nothing was released under v1 rules.
- v1.1 (commit 79d8b597): per-GPU residual check, tag-based orphan sweep, outputs gate excludes profiled runs,
  native co-runner leak references (`leak_ok_v11`), graph-mode counters dropped. 23/23 CPU fixture checks;
  MI210 devel smoke (job 483497) of co-runner and sweep.
- 481168 (v1 release, held) cancelled unrun. Submitted from release 79d8b597:
  483498 `cubw-mi300x-auto` (mi3008x, 2:45:00, minimal re-pilot → gates → production, boundary 200) and
  483499 `cubw-mi325x-auto` (mi3258x, 3:15:00, `afterany:481072`). Dry runs: 691 and 759 steps.
- 481072 (MI325X phases 2–4) ran 07:20–09:17 on k002-001: phase 2 (J6 competitor, TP2, TP4), phase 3 (TP8),
  phase 4 (152-CU-matched TP). 180/180 run.json `complete` (TP 108, J6 72). With 470176 phase 1, the old
  MI325X campaign now has all four phases on disk; not yet published.
