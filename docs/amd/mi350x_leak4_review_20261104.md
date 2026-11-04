# MI350X leak test v4 + E3-minimal 499332 — review (4 November 2026)

**Job:** 499332 on `mi3508x` (k007-003, 8× MI350X), cubw v1.6, release f737d1b1. Smoke:
[mi350x_leak4_smoke_review_20261104.md](mi350x_leak4_smoke_review_20261104.md).
- Ran 38 min (pilot 15 min, production 22 min). All 153 steps that report a return code had rc = 0. The other
  112 journal records are co-run entries, which carry no return code.
- Judged against the pre-registration [followups_v16_plan_20261104.md](followups_v16_plan_20261104.md) §A.
- Tables: `results_published/amd_20261104_mi350x_cubw_v1_6_leak4_job499332/` (`gates.json` holds every leak test).
- **Cohort: MI350X only.** Every timing is observational and clock-confounded: pooled gfx ratio 1.37, gfx
  1583–2172 MHz, memory clock held at 2000 MHz.

## 1. Gates

| gate | value |
|---|---|
| placement_ok | true: all 8 GPUs, 5 levels and the competitor mask 192–255, exact per-XCD counts |
| lifecycle_ok | true |
| outputs_ok | true: 105/105 runs complete, 1 digest |
| **leak_ok_v4** | **true: 16 of 16 decoder tests pass** (8 GPUs × n = 192, 128) |
| **release_e3_minimal_v4** | **true**, so E3-minimal ran on 8 blocks |
| clock_ok | false |

## 2. Leak test v4: the masked decoder does not leak onto the competitor CUs

Per GPU and level (16 tests), from `gates.json` → `leak_v4`:

| | n = 192 (probe CUs 192–255) | n = 128 (probe CUs 128–255) |
|---|---|---|
| alone: coverage / largest late fraction | 64/64 on every GPU / ≤ 0.0042 | 128/128 / ≤ 0.0058 |
| natives (read, FMA on 0..n−1): flagged CUs | 0 in all 16 runs | 0 in all 16 runs |
| **required positive: full-LDS probe on 0..n+7** | 25–33 flagged (top-8 excess 0.68–0.77) | 25–36 (0.60–0.67) |
| **required positive: decoder on 0..n+7** | 28 flagged (0.42–0.46) | 28–33 (0.49–0.54) |
| recorded: FMA / read on 0..n+7 | 24–41 / 25–28 | 24–29 / 25–43 |
| **decoder on 0..n−1: flagged CUs (largest excess)** | **0** on all 8 GPUs (≤ 0.0020) | **0** on all 8 GPUs (≤ 0.0017) |

- Every positive cleared the ≥ 4 requirement on every GPU, so the test is sensitive everywhere.
- Every decoder run flags 0, equal to the natives' maximum of 0, so every test passes.
- The largest decoder excess, 0.002, is 50× below the 0.10 flag threshold.

**Not pre-registered: positives flag more CUs here than in the smoke.** In the smoke (one GPU on a
`mi3501x` slice) each positive flagged exactly the 8 shared CUs. Here each flags those 8 plus about 17 more
(GPU L0, n = 192, from the raw per-CU counts):
- **The 8 shared CUs:** the probe gets no workgroup there in about 60% of launches (seen ≈ 40k of 99k). The
  displaced workgroup runs as a second one on another CU (8 CUs show multi), as in the smoke.
- **About 17 unshared probe CUs:** a workgroup lands there in every launch, but starts > 5 µs late in about 67%
  of them. Late starts spread to CUs the co-runner cannot reach.
- A likely cause is in-order dispatch: a workgroup waiting for a busy shared CU holds back the ones queued
  behind it. Why this shows on the 8-GPU node and not on the slice is not known.

This only makes the positives stronger. It changes no flag, threshold or verdict, and the decoder runs show no
such spread (0 flagged).

## 3. E3-minimal: memory competition slows decode much more than arithmetic

Qwen32B decode on CUs 0..n−1, with an owned competitor on CUs n..255. 8 blocks, one per GPU. Values are decode
time relative to no competitor at the same n:

| n | arithmetic 100% | memory 50% | memory 100% |
|---|---:|---:|---:|
| 192 | 1.019 [1.015, 1.023] (29.4 TFLOP/s) | 1.116 [1.115, 1.118] (1183 GB/s) | 1.273 [1.269, 1.278] (2536 GB/s) |
| 128 | 1.022 [1.018, 1.027] (29.6 TFLOP/s) | 1.107 [1.104, 1.109] (1269 GB/s) | 1.253 [1.248, 1.258] (2683 GB/s) |

- Decoder alone: 14.75 ms at 192 and 14.57 ms at 128; E2 at 256 in the same job is 12.28 ms.
- Inactive control at 192: 1.0001 [0.9992, 1.0010] (1 block).

**Pre-registered expectation (direction only): met.**

| | MI300X 492956 | MI325X 489594 | **MI350X 499332** |
|---|---:|---:|---:|
| memory 100% / arithmetic 100% | 1.18 (232), 1.21 (152) | 1.14, 1.16 | **1.25 (192), 1.23 (128)** |
| decoder alone on fewer CUs / decoder beside memory 100% | 0.846 | 0.924 | **0.775** (128 alone vs 192 + mem100) |

- On MI350X, a co-runner holding 64 CUs of arithmetic costs the decoder only about 2%. A co-runner streaming
  2.5 TB/s costs it 27%.
- **The B_CU reading:** the decoder's cost tracks the HBM bandwidth taken from it, not the CUs taken from it.
  128 CUs with uncontested HBM decode 22% faster than 192 CUs beside a memory-bound neighbour.
- Like every E3 result, this is observational and clock-confounded (`clock_ok` false). The leak test shows
  that the contention is not through stolen CUs. It goes through the shared memory system (HBM, MALL, fabric).
- **Note on the decoder kernel:** E3 used the stock `wvSplitK` (grid = 256), so the decoder's own absent time
  carries the ~1.20 dispatch cost at 192 and 128 found in 497082. The ratios above are relative to that same
  stock decoder, so they are not affected.

## 4. Status of the two follow-ups

- **A (this job): done.** E3 is no longer blocked on MI350X. All three cohorts now have E3-minimal, and all show
  the same ordering (memory ≫ arithmetic), largest on MI350X.
- **B (MI300X shader-engine study, 499334):** queued; reviewed separately when it ends.

If any of this is presented or published, the AMD acknowledgement is: "This work was supported in part by
Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
