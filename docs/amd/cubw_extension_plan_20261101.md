# cubw v1.3 extension — plan, pre-registration and CPU-side results (1 November 2026)

This note follows [compute_bandwidth_followup_results_20261101.md](compute_bandwidth_followup_results_20261101.md)
§7, which lists the work that was not yet done. It was committed **before** the extension jobs ran.

| item | GPU needed | status |
|---|---|---|
| n=200 vs 232 anomaly (finer levels) | yes | job submitted (§4) |
| E6 kernel identification | no | done (§1) |
| E6 kernel-shape controls and correctly-sized kernel | yes | job submitted (§4) |
| Counters (graph-mode `--pmc` unavailable) | yes | job submitted: eager decode and shape controls (§4) |
| MI300X E3 (needs a new leak test) | yes | test built, smoke-tested, rule pre-registered (§2), job submitted |
| E7 model fit and held-out check | no | done (§3); blind predictions registered |
| Raw archive | no | checksum readback done (§5); off-cluster copy needs a destination from the owner |
| Fixed clocks, full E3 grid | — | unavailable (clock/power changes not authorized) |

## 1. E6, step 1 — dominant decode kernels (CPU, from existing traces)

`scripts/amd/cubw/e6_kernels.py` was run on the graph-mode kernel traces of jobs 487341 (MI300X) and 489594
(MI325X), Qwen32B B1/C4096, at n=304 and n=200. The tables are in
`results_published/amd_20261101_cubw_extension_prep/e6_kernels_*.json`.

| kernel | share of decode kernel time | per tick | n200/n304 time, MI300X / MI325X |
|---|---:|---:|---|
| `wvSplitK_hf_sml_<bf16,64,2,16,8,2,1>` (vLLM ROCm skinny GEMM) | 83.5% / 84.0% | 257 (64 layers × 4 GEMMs + lm_head) | 1.034 / 1.059 |
| `paged_attention_ll4mi_QKV_mfma16` | 4.4% | 64 | 1.073 / 1.112 |
| reshape/cache, RMS norm, SiLU, attention reduce (Triton and vLLM) | ~1.7–2.1% each | 64 | 0.95–1.03 |

Key fact: `wvSplitK` workgroups are 64 threads with 64 KiB of LDS, which is all of a CU's LDS. So at most one
workgroup can be resident per CU, and memory-level parallelism scales directly with the number of CUs.

vLLM sizes this kernel with `num_compute_units()`, which returns the device CU count (304) even under
`HSA_CU_MASK`. The grid is 19456 = 64 × 304 at every mask level. Whether a kernel sized for the mask changes
the penalty is the E6 question the extension measures (`e6_decode`, `CUBW_CU_COUNT_OVERRIDE`).

## 2. Leak test v2 (per-CU probe), pre-registered

**Why a new test.** The v1.1 test failed on MI300X (decoder probe ratio 0.77–0.82 of the gentlest native
co-runner). The decoder slowed an FMA probe about as much as a native streaming co-runner did, so the aggregate
probe cannot separate CU leakage from chip-wide contention.

**The probe.** `svc_bench_v4 --mode cuprobe`:
- Each FMA workgroup reserves all 64 KiB of LDS, so one probe workgroup is resident per CU — the same footprint
  as `wvSplitK`.
- Each workgroup's completion is counted per physical CU (HW_ID/XCC_ID).
- A foreign workgroup resident on a probe CU therefore lowers *that CU's* rate. Chip-wide effects (power,
  clocks, memory) lower all CUs together.

**MI210 smoke** (devel job 492951), probe on CUs 52–103:

| run | CUs below 0.85 | min q | lowest q values |
|---|---:|---:|---|
| native read co-runner on 0–51 | 0 | 0.95 | – |
| native FMA co-runner on 0–51 | 0 | 0.95 | – |
| positive-control probe on 0–59 (shares 8 CUs) | **8** | 0.46 | 0.46 ×4, 0.64–0.66 ×4 |

The test is sensitive to shared CUs and specific against contention.

**Rule** (in the launch plan's `gates.leak_ok_v2`, analysed by `analyze.leak_v2_eval`):
- r_c = per-CU rate beside X / per-CU rate alone, on the same GPU and probe mask.
- q_c = r_c / median_c(r_c).
- Every run must cover all 304−n probe CUs.
- The positive control must show ≥ 4 CUs with q_c < 0.85; otherwise the test is insensitive and fails.
- **Pass iff** every decoder run has min_c q_c ≥ (lowest min_c q_c of the native read/FMA co-runner runs) − 0.05.

`release_e3_minimal_v2` = placement, lifecycle, outputs and leak_ok_v2. On MI300X it gates E3-minimal inside
the same job. On MI325X the suite runs as a cross-check only; its E3-minimal is already released.

## 3. E7 — service-response model and blind predictions

`scripts/amd/cubw/e7_model.py` fits R(n) = (1 − s) + s·P_x(n) on Qwen32B production E2 at n = 232/152/96. P_x is
the native E4 penalty of one control x. The tables are in `results_published/amd_20261101_cubw_e7_model/`.

**r1, unconstrained** (procedure as first written):
- It selected latency rows or rows with s = −1.3 (MI300X) and s = 2.9 (MI325X). These have no meaning as a
  fraction of decode time.
- Held-out results: 1/5 on MI300X and 2/5 on MI325X.
- **Rejected.** Three training points do not support choosing among 22 controls.

**r2, constrained** (revised *after* seeing r1): throughput controls only, 0 ≤ s ≤ 1. Its held-out checks on
existing data are therefore not blind.

| cohort | selected control | s | held-out within 0.02 + CI half-width |
|---|---|---:|---|
| MI300X | 4 GiB streaming read | 0.50 | 4/5 (miss: Qwen14B n=152, +0.035) |
| MI325X | 4 GiB streaming read | 0.92 | 4/5 (miss: Qwen14B n=96, −0.035) |

Even this constrained fit has no term that can produce n=200 faster than n=232. A monotone model can only absorb
that inside its tolerance.

**Blind predictions** (Qwen32B R(n), committed before any measurement at these levels):

| n | MI300X | MI325X |
|---:|---:|---:|
| 272 | 1.013 | 1.025 |
| 248 | 1.023 | 1.043 |
| 216 | 1.026 | 1.047 |
| 184 | 1.039 | 1.082 |

The model is "validated" on a cohort only if every held-out and blind point lies within 0.02 + the CI
half-width.

## 4. Extension jobs (release de2baecb)

| job | cohort | limit | plan |
|---|---|---|---|
| 492956 | MI300X (mi3008x) | 2:45:00 | `$WORK/cubw_plans/mi300x_auto_20261101_155830_de2baecb.json` (285 steps) |
| 492957 | MI325X (mi3258x) | 2:30:00 | `$WORK/cubw_plans/mi325x_auto_20261101_155837_de2baecb.json` (235 steps); node k002-001 down since 11:48 (cluster side) |

**Pilot stage:**
- placement at 304/272/248/232/216/200/184/152 on all 8 GPUs;
- leak v2 suites at n=232 and n=152 (two GPUs on MI300X, one on MI325X).

**Gate**, then **production stage:**
- E2-fine: Qwen32B, 8 blocks.
- `e6_decode`: wvSplitK sized for the active CUs, 7 levels × 5 blocks, same GPUs as E2-fine.
- Kernel traces at all 8 levels, plus the sized kernel at 232/200.
- E6 shape controls: 8 levels × 3 acquisitions.
- Pre-MALL counters on shape controls: FETCH_SIZE, TCC_EA0_RDREQ, TCC_EA0_RDREQ_LEVEL and GRBM_GUI_ACTIVE at
  304/232/200/152.
- Eager decode counters: FETCH_SIZE and RDREQ_LEVEL at 304/200.
- MI300X only: E3-minimal (5 blocks) if release_e3_minimal_v2.

**Reader:** the same release, re-run afterwards with the current analyzer. `e6_decode` has its own output check,
since a different split-K configuration may legitimately change the output digest; it is excluded from
outputs_ok.

## 5. Raw data integrity

`sha256sum -c` readback on 2026-11-01 for every cubw root (481167, 483498, 483499, 487341, 489594): all data files
match.

The only mismatches are logs appended after hashing:
- every root's `RUN_STATUS.log`, which gets the final "COMPLETE" line after the hash;
- two `runner.log` files in 481167, written by the orphaned rocprofv3 processes that v1.1 fixed.

The off-cluster durable copy is **not done**: it needs a destination chosen by the owner. `$WORK` is not backed
up. The sizes are 14 GB (481167), plus 16 MB, 373 MB, 401 MB and 438 MB.
