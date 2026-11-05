# Goal follow-up G1–G8: execution plan (goal_v1, 5 November 2026)

This plan was committed and pushed **before any data from the jobs it describes**. It adapts
[goal_based_followup_20261105.md](goal_based_followup_20261105.md) (pinned to branch head `5b988063`; handoff commit
`e9c76bc5`) to the AMD hardware actually available.

## 0. Owner instructions and authorization (2026-11-05)

- Execution ordered: "implement the experiments for each machine and submit the slurm jobs so that we can execute and
  finish all the jobs."
- Budget: no cap. Cohorts: all three (MI300X, MI325X, MI350X). G2: **mixed serving** is in scope.
- **Controlled clocks: not authorized, so not attempted.** In the owner's words, "if we don't have any authorization,
  we can't do it. mark that." Every G1 timing stays observational unless it passes the unchanged pooled clock gate.
- Raw archive: there is still no off-cluster destination. Raw data stays in `$WORK/results_amd/`. Compact,
  trial-level evidence goes into git.
- No clock, power, driver, firmware or cluster setting is changed. Cohorts are never pooled. Failed acquisitions are
  preserved. At most 2 attempts per step, and only for execution failures.

## 1. Inventory and reconciliation

| cohort | partition (GPUs) | arch | CUs | SEs/XCD | HBM peak | B_CU peak | charge |
|---|---|---|---:|---:|---:|---:|---:|
| MI300X | `mi3008x` (8), `mi3001x` (1) | gfx942 | 304 | 4 | 5.3 TB/s | 17.4 GB/s | 1.0 / 0.125 |
| MI325X | `mi3258x` (8; one node, k002-001) | gfx942 | 304 | 4 | 6.0 TB/s | 19.7 GB/s | 1.2 |
| MI350X | `mi3508x` (8), `mi3501x` (1) | gfx950 | 256 | 4 | 8.0 TB/s | 31.2 GB/s | 1.4 / 0.175 |

- **Running jobs before this plan:** none.
- **Allocation:** 324.8 of 450 units used, shared with a teammate.
- **Cost-rate correction:** `scripts/amd/cubw/plan.py` estimated costs with 8.0, 12.8 and 11.2 units per node-hour.
  Measured billing is node-hours × 1.0, 1.2 and 1.4, so earlier estimates in the docs were about 8× too high. goal_v1
  uses the real multipliers.
- **Known tool limits** (handbook §8 and reviews):
  - Graph-mode `rocprofv3 --pmc` failed; eager counters work, except for some excluded kernels.
  - MI325X counters were refused (CP firmware 177 < 186).
  - MI350X `FETCH_SIZE` read 0.50× on a known copy.
  - No MALL or data-fabric counters were found on MI300X.
  - The v1.6.1 dispatch-tag fix had not yet run on a GPU (it runs here).
- **Kernel route of the CU-count override.** This was read from the container's
  `vllm/model_executor/layers/utils.py`. `CUBW_CU_COUNT_OVERRIDE` changes:
  - `wvSplitK` when the token count n ≤ 5 (gfx9);
  - `wvSplitKrc` when 10 ≤ n ≤ 128 (gfx950 only).

  Prefill and batch-8 decode use hipBLASLt (`torch.nn.functional.linear`). For those, the "reduced grid" is **no
  intervention** unless the G2 kernel-route trace shows a changed skinny-kernel grid.

## 2. Source pins and new code

| item | status |
|---|---|
| `scripts/amd/cubw/cubw.py` | reused; new step kind `serve` (runs `serve_mix.py`) |
| `cubw/analyze.py` | reused unchanged (acquisitions, clocks, placement, leak v2/v4, `ratio_ci` bootstrap) |
| `cubw/plan.py` | reused helpers (Plan, `placement_core`, `dispatch_suite`, `e4_suite`, `e6_cmd`, `derived_configs`) |
| `cubw/e6_shapes.py`, `svc_bench_v3`–`v6` binaries (hashes in `$WORK/bin/svc_bench.sha256`) | reused unchanged |
| `scripts/sitecustomize.py` (override route) | reused unchanged |
| **new** `scripts/amd/goal/serve_mix.py` | G2/G3 mixed-serving runner |
| **new** `goal/counter_audit.py` | G4 counter inventory plus known-traffic audit |
| **new** `goal/isa_meta.py` | G6 static metadata for wvSplitK, wvSplitKrc and LLMM1 |
| **new** `goal/plan_goal.py`, `goal/goal.sbatch`, `goal/submit_goal.sh` | plans and job wrapper |
| **new** `goal/analyze_goal.py` | goal gates, G1–G7 tables, `goal_status.json` |
| **new** `goal/g8_model.py` + **frozen** `docs/amd/goal/g8_frozen_model.json` | G8 model, predictions, judge |
| **new** `goal/test_goal.py` | plan invariants; `cubw/test_analyze.py` also still passes |

- **Release:** jobs run from a frozen `git archive` of the commit that contains this plan (`$WORK/cubw_release/<sha>`).
- **Container:** the same vLLM container as every earlier run (`rocm/vllm … vllm_0.23.0`, recorded per run).
- **Models:** the same pinned BF16 revisions, TP1, prompts and seed policy (seed 20261013 in configs; plan seed
  20261105).

## 3. Design per cohort

Shorthand:
- n = the decoder's CU mask (contiguous, interleaved over XCDs);
- grid = the `wvSplitK` CU count, via the override. Stock means the device count.
- Every reduced tuned grid is shader-engine balanced (a multiple of 32).

| | MI300X | MI325X | MI350X |
|---|---|---|---|
| promising point n_p (grid = n_p) | 128 | 160 | 192 |
| boundary point n_b (grid = n_b) | 96 | 128 | 128 |
| G1 tuned levels (grid = n) | 256, 224, 192, 160, 128, 96 | same | 224, 192, 160, 128, 96 |
| tuned full device (mask full, grid 288) | yes | yes | — (256 is balanced) |
| G5 competitor pool (fixed) | CUs 256–303 | 256–303 | 224–255 |
| G5 decoder masks (grid = n) | 256, 192, 128 | same | 224, 192, 128 |

**Why these points (from the evidence before this plan):**
- MI300X at 128 CUs had R 0.993 (499334).
- MI350X at 192 had R 1.009; at 160, 1.067; at 128, 1.168 (497082).
- MI325X has no balanced-grid data. The G8 model predicts 1.000 at 160 and 1.047 at 128.

**G2 conditions:**
- A: full mask, stock kernel;
- B: full mask, grid n_p (software launch tuning alone);
- C: mask n_p, grid n_p;
- D: mask n_p, stock kernel;
- Cb: mask n_b, grid n_b.

### Stage 1 (`pilot`; characterisation, no timing comparison promoted)

1. **Placement:** every mask used, on all 8 GPUs (exact distinct CUs, per-XCD counts within 1).
2. **G5 isolation:** leak test v4, with the **tuned** decoder, on every GPU at the two pool-edge levels.
   - The rule is the v1.6 rule unchanged.
   - The required decoder positive on 0..n+7 is the stock decoder, as in 499332.
   - gfx942 also runs leak v2 with the tuned decoder on 2 GPUs. This is the pre-registered fallback, used only if
     leak v4 is insensitive anywhere.
3. **G1/G6 placement of the tuned grid:** the dispatch probe at full and at every tuned level, on 2 GPUs
   (grids = device, n, balanced(n)).
4. **G4:** `counter_audit.py`, covering:
   - counter inventory, definition files and firmware;
   - ≤ 16 memory-side counters × 8 known-traffic loads (L2-resident, MALL-resident and streaming reads; writes;
     copies) × 2 repetitions.
5. **G6:**
   - static ISA/resource metadata;
   - native service curves (E4 suite: ILP, grid, stride, gather, pointer chase, reuse) at 4 masks;
   - the CU × grid kernel-shape sweep: Qwen32B GEMM shapes at 8 masks × 9 (MI350X: 7) wvSplitK grid sizes,
     3 acquisitions.
6. **G2 kernel routes:** kernel-trace timelines of the 7 isolated cases at A and C.
7. **G7 (not MI325X):** eager Qwen32B decode with `FETCH_SIZE` and `WRITE_SIZE`, one counter per pass, at A and C.
   Kernels matching `^(_|triton)` are excluded from collection, as before. That scope limit is recorded.

### Gate (`scripts/amd/goal/analyze_goal.py gate`, frozen here)

- `release_g1 = placement_ok`.
- `release_g5 = placement_ok ∧ lifecycle_ok ∧ isolation`, where isolation is:
  - `leak_ok_v4` (tuned decoder, all GPUs and levels) if v4 is sensitive on every GPU and level;
  - otherwise, on gfx942 only, `leak_ok_v2` with the tuned decoder;
  - otherwise false.

### Stage 2 (`production`; 8 randomized complete blocks, one per GPU)

- **Every block, G1:** Qwen32B graph B1/C4096.
  - Stock full device is run **first and last** in the block (drift check).
  - Randomized in between: the tuned full device, the full mask with grid n_p, every tuned level, and stock at n_p
    and n_b.
  - That is 10–12 cells × 8 acquisitions per cell.
- **Blocks 0–3:**
  - G2 isolated cases × A, B, C, D, Cb:
    - Qwen32B B1/C16384 (long KV);
    - Qwen7B B8/C4096 (larger batch; hipBLASLt);
    - Qwen7B B4/C1024 (larger batch, still on the skinny route);
    - Qwen14B B1/C4096 (second size);
    - Mistral-7B B1/C4096 (second family);
    - Qwen32B prefill (2 tokens; prefill wall).
  - G5 surface: 3 decoder masks × {absent, inactive, mem25, mem50, mem100, arith100}.
- **Blocks 4–7:**
  - G2 **mixed serving** (`serve_mix.py`): Qwen32B, 32 requests per trace.
    - Prompt lengths cycle through 256/1024/4096/512/2048; output lengths through 64/256/128.
    - Closed loop at concurrency 2 and 8, so chunked prefill and decode batches of 1–8 mix.
    - 3 warmups of the first 2C requests, then 10 timed traces, × A, B, C, D, Cb.
  - G3 decode energy: 10 trials × 1023 ticks, × A, B, C, Cb.
  - G3 serving energy: C = 8, at A and C.
- **Statistics:**
  - 3 warmups and 10 timed trials per acquisition.
  - Block-paired bootstrap: 10,000 draws, seed 20261018, trials nested (`analyze.py ratio_ci`).

### Single-GPU smokes (pilot-labelled, judge nothing)

- **Partitions:** `mi3001x` (gfx942) and `mi3501x` (gfx950). MI325X has no single-GPU partition, so the gfx942 smoke
  covers its code path.
- **Content:** one acquisition of every step kind above, including serving, serving energy, decode energy, a G5 cell,
  the counter audit (3 counters) and the ISA step.
- **Sequence:** the 8-GPU jobs are submitted **held** and released only after the smoke review. A port bug found by a
  smoke is fixed first, and the held job is replaced from the fixed release.

## 4. Pre-registered acceptance rules and terminal dispositions

- **G1:**
  - **Qualified noninferiority:** upper 95% CI of R ≤ 1.05 **and** clock_ok (pooled in-window max/min ≤ 1.05 over the
    cell and its stock reference) **and** placement_ok **and** output digest identical to stock. Otherwise the result
    is observational.
  - **Clock sensitivity** (reported, never a substitute): within-cell elasticity of log T on log gfx clock, and the
    clock-adjusted ratios.
  - **Drift:** the stock last/first ratio is reported.
- **G2:**
  - Per case, ratios B/A, C/A, D/A and Cb/A with CIs.
  - Isolated cases require identical outputs; a mismatch is reported as a numerics change.
  - Serving reports throughput, TTFT p50/p99, ITL p50/p99 and E2E p99 ratios vs A. Request-level output agreement
    with A is recorded, not asserted: closed-loop batching changes numerics.
  - Kernel route per case: does the override reach a kernel?
  - No workload-weighted overall score is defined, so none is reported. Regressions are reported alongside
    successes.
- **G3:**
  - J/token (board energy accumulator), total J, window and average power, for the same scope.
  - The (P'/P)·(T'/T) decomposition.
  - Primary J/token is NA whenever the runner's sensor checks fail (window ≥ 5 s, ≥ 5 accumulator updates,
    counter-vs-sampled-power agreement ≤ 10%). There is no idle subtraction.
  - A CU mask does not remove silicon: lower power is not leakage savings.
- **G4:** a counter is a **physical-HBM read candidate** only if all three hold:
  - streaming reads give a ratio in [0.9, 1.15] in some unit;
  - the MALL-resident read (128 MiB × 16) gives ≤ 0.25 in that unit;
  - the L2-resident read gives ≤ 0.25.

  A counter with streaming ≈ 1 and MALL-resident ≈ 1 is **pre-MALL** (request side). If no counter qualifies, the
  physical-HBM status is **unavailable**. In that case, D/T and pre-MALL counters are never relabelled as HBM.
- **G5:**
  - Released by `release_g5`. Decoder ratios are vs absent (same block, same mask and grid), with achieved competitor
    GB/s and coverage, plus mem100/arith100.
  - A local `−Δlog T/Δlog B_decoder` is estimated only if G4 validates a decoder-side physical counter. Otherwise
    latency is reported against treatment and achieved competitor activity.
- **G6:**
  - Best wvSplitK grid per mask and shape (3 acquisitions).
  - Balanced vs unbalanced grids, dispatch placement and serialization, service curves and static resources.
  - The 42 GB/s per CU figure is a hypothesis, not a limit.
- **G7:**
  - Logical demand per token from the model config: weights + KV read at 4096 + KV write.
  - Eager pre-MALL counters per decode tick at A and C.
  - The HBM amplification ratio is **unavailable** unless G4 finds a physical counter.
- **G8:**
  - Frozen **now** in `docs/amd/goal/g8_frozen_model.json` (committed with this plan). The model is
    R(g) = 1 + k·max(0, B(g) − B0), with B = peak/g. It was fitted on MI350X 497082 only, giving k = 0.00767 per
    GB/s and B0 = 40.79 GB/s per CU.
  - **Held-out points:** MI325X g ∈ {288 on the full mask, 256, 224, 192, 160, 128, 96}; MI300X 160 and 96;
    MI350X 96.
  - **Pass** iff every |obs − pred| ≤ 0.05 and the mean absolute error ≤ 0.025.
  - **Unavailable** below 3 paired blocks.
  - No refit on held-out points.

## 5. Budget, timing and failure limits

| job | partition | steps | est. longest lane | node-h | units |
|---|---|---:|---:|---:|---:|
| smoke gfx942 | `mi3001x` | 54 | 2.0 h | ~2.4 | ~0.3 |
| smoke gfx950 | `mi3501x` | 54 | 2.0 h | ~2.4 | ~0.4 |
| MI300X | `mi3008x` | 778 | 5.8 h | ~6.2 | ~6.2 |
| MI325X | `mi3258x` | 774 | 5.8 h | ~6.2 | ~7.4 |
| MI350X | `mi3508x` | 710 | 5.7 h | ~6.1 | ~8.5 |

- **Time limits:** 9 h per 8-GPU job; 3 h per smoke. There is a 40-minute reserve for analysis and hashing.
- **Deadline:** steps that cannot finish before it are recorded as `not_run (deadline)`, not silently dropped.
- **Total:** about 23 units, against no cap.
- **Failure limits:** 2 attempts per step for execution failures only. Hung profilers get no retry. A cohort's
  goals end with the dispositions above, not with repeated attempts.

## 6. Delivery

- **Per job:** `results_published/<cohort>_<date>_goal_followup_job<id>/`, containing:
  - the launch plan, gates, `goal_tables.json` (trial-level acquisitions included), `goal_summary.md` and
    `goal_status.json`;
  - the counter audit, ISA metadata and serving per-request tables;
  - SHA-256 of the raw root.
- **Cross-cohort:** `docs/amd/goal_followup_results_<date>.md`, the G8 judgment, the experiment log, the cohort index
  and a handbook chapter.
- **Raw data:** stays in `$WORK/results_amd/` (no archive destination exists; reported as a delivery gap).

AMD acknowledgement for any presentation: "This work was supported in part by Advanced Micro Devices, Inc. under the
AMD University Program's AI & HPC Cluster".
