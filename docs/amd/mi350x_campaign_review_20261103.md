# MI350X campaign 496074 — review (3 November 2026)

**Job:** 496074 `cubw-mi350x-auto` on `mi3508x` (k007-004, 8× MI350X, gfx950, 256 CUs).
- Release e7116c1f, cubw v1.4, plan `mi350x_auto_20261102_233252_e7116c1f.json`.
- Ran 04:47–05:56 CST: 1 h 09 min of the 3:45 limit, about 12.9 units (1.154 node-h × 11.2).

**Plan and rationale:** [mi350x_campaign_plan_20261102.md](mi350x_campaign_plan_20261102.md).
**Smoke runs:** [mi350x_smoke_review_20261102.md](mi350x_smoke_review_20261102.md).
**Compact tables:** `results_published/amd_20261103_mi350x_cubw_v1_4_auto_job496074/`.
**Raw root:** `$WORK/results_amd/amd_20261103_mi350x_cubw_v1_auto_job496074` (1885 files hashed).

MI350X is a separate cohort. Cross-cohort rows below compare matched per-XCD CU fractions only and pool nothing.
Every timing comparison is **observational, clock-confounded**: the pooled gfx clock ratio was 1.53 (the gate
is ≤ 1.05). The memory clock held at 2000 MHz throughout.

## 1. Execution and gates

- **Steps:**
  - 727 planned; 686 ran, all with rc = 0. 421 had a check, and every check passed.
  - The other 41 are E3-minimal steps. They were not run, by gate (below).
- **Outputs:** 94/94 decode runs complete, 1 distinct output digest.
- **Placement:** exact on all 8 GPUs at every level, evenly split per XCD. Levels:
  - 256/232/208/192/184/168/152/128/80;
  - competitor 192–255: 64 CUs, 8 per XCD;
  - comma-list and wrong-index negative controls.
- **Gates:**

  | gate | value |
  |---|---|
  | placement_ok, lifecycle_ok, outputs_ok | true |
  | leak_ok_v11 | true |
  | leak_ok_v2 | **false**: positive control insensitive, 0 CUs flagged (every q ≈ 1.0) |
  | clock_ok | false |
  | release_e2, release_e5e8 | true |
  | release_e3_minimal_v2 | **false**, so E3-minimal did not run |

  The leak-v2 failure is the same insensitivity the smoke run found, as pre-registered. The threshold was not
  relaxed.

## 2. E2 — decode latency R(n) = T(n)/T(full), Qwen32B, B1/C4096 graph mode

**MI350X** (8 blocks per level; 95% bootstrap CI):

| n | 232 | 208 | 192 | 184 | 168 | 152 | 128 | 80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R | 1.251 | 1.177 | 1.180 | 1.197 | 1.191 | 1.180 | 1.159 | 1.858 |
| CI | 1.19–1.32 | 1.15–1.20 | 1.14–1.23 | 1.15–1.25 | 1.15–1.23 | 1.14–1.22 | 1.12–1.20 | 1.81–1.90 |

T(256) = 12.70 ms per step.
- Qwen14B (3 blocks): 192 1.160, 168 1.214, 128 1.139, 80 1.819.
- The pilot gave the same shape: 192 1.208, 168 1.186, 128 1.172, 80 1.878.

The curve is a **step**. Any restriction below 256 costs about 16–25%. The cost then stays flat down to 128 and
jumps at 80. There is no gradual bandwidth slope.

**Matched per-XCD fractions across cohorts** (Qwen32B, production E2; each cohort's own full device = 1):

| fraction per XCD (MI3x0X / MI350X) | MI300X 487341 | MI325X 489594 / 492957 | MI350X 496074 |
|---|---:|---:|---:|
| 34/38 ≈ 29/32 | — | 272: — / 1.116 | 232: **1.251** |
| 31/38 ≈ 26/32 | — | 248: — / 1.080 | 208: **1.177** |
| 29/38 ≈ 24/32 | 232: 1.024 | 232: 1.074 / 1.052 | 192: **1.180** |
| 27/38 ≈ 23/32 | — | 216: — / 1.026 | 184: **1.197** |
| 25/38 ≈ 21/32 | 200: 1.012 | 200: 1.056 / 1.021 | 168: **1.191** |
| 23/38 ≈ 19/32 | — | 184: — / 1.008 | 152: **1.180** |
| 19/38 = 16/32 | 152: 1.070 | 152: 1.187 / 1.141 | 128: **1.159** |
| 12/38 ≈ 10/32 | 96: 1.083 | 96: 1.186 | 80: **1.858** |

At every matched fraction, restricting CUs costs more on MI350X than on either earlier cohort. The ranking
follows HBM bandwidth per CU: MI300X 17, MI325X 20, MI350X 31 GB/s. This is the direction prediction 1 of the
plan expected from the memory-level-parallelism (MLP) explanation: more bandwidth per CU means each CU must keep
more reads in flight. It is observational, and three cohorts are not a fit.

## 3. E6 — where the step comes from

**Kernel traces** (`e6_kernels_all_levels.json`; 8 fine levels plus 2 sized-kernel runs; one trace each):
- `wvSplitK_hf_sml_<bf16,64,2,16,8,2,1>`: 256 workgroups × 160 KiB LDS, one per CU.
  - 80.8% of decode kernel time at n = 256.
  - Kernel-time ratio to 256: 232 1.288, 208 1.242, 192 1.212, 184 1.215, 168 1.210, 152 1.198, 128 1.209.

  This flat plateau has the same shape as R(n).
- Everything else is small:
  - paged attention: 4.1% of time, ratio about 1.2;
  - the other kernels: ≤ 2.6% of time each, ratios 0.8–1.0.

**A `wvSplitK` sized for the active CUs** (`e6_decode`: 5 blocks per level; outputs bit-identical):

| n | 232 | 208 | 192 | 184 | 168 | 152 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R, default kernel (E2) | 1.251 | 1.177 | 1.180 | 1.197 | 1.191 | 1.180 | 1.159 |
| R, kernel sized for n | 1.159 | 1.349 | **0.989** (0.965–1.009) | 1.412 | 1.330 | 1.547 | 1.135 |

- **At 192 CUs (75%), the sized kernel removes the decode penalty.** The kernel trace agrees: the sized
  `wvSplitK` is 1.078× instead of 1.212×. The kernel-shape controls agree for all five GEMM shapes: `wvsplitk_n`
  at 192 is 1.002–1.020×.
- At 232 and 128 the sized kernel helps a little. At 208, 184, 168 and 152 it is much worse. A per-shape
  breakdown is in `summary.md` → E6 kernel-shape controls.
- So the MI350X penalty is not a fixed hardware property of having fewer CUs. It depends on how the skinny GEMM's
  work is cut into workgroups. Some cuts (24 or 16 CUs per XCD) work well, and others do not.
- On MI325X, the sized kernel was slower at every level ([review](cubw_mi325x_extension_review_20261102.md)).
  MI350X is the first cohort where a sized kernel reaches full-device speed at a reduced CU count.

**Counters** (`e6_counters_by_variant.json`, `e6_inflight.json`; light single runs; FETCH_SIZE is unvalidated
in absolute terms on gfx950):
- FETCH_SIZE and TCC_EA0_RDREQ are identical across 256/192/168/128 for each shape. Restricting CUs moves the
  same bytes.
- The table shows mean HBM reads in flight (TCC_EA0_RDREQ_LEVEL / GRBM_GUI_ACTIVE), then busy cycles. Each is a
  ratio to the default kernel at 256:

  | shape | default 192 | default 128 | sized 192 | sized 168 |
  |---|---|---|---|---|
  | qkv | 0.60 / 1.20 | 0.50 / 1.20 | 0.73 / 1.02 | 0.55 / 1.23 |
  | o_proj | 0.59 / 1.22 | 0.50 / 1.21 | 0.69 / 1.06 | 0.62 / 1.07 |
  | gate_up | 0.60 / 1.29 | 0.53 / 1.32 | 0.83 / 1.08 | 0.45 / 1.71 |
  | down | 0.59 / 1.30 | 0.50 / 1.23 | 0.77 / 1.00 | 0.66 / 1.07 |
  | lm_head | 0.67 / 1.23 | 0.60 / 1.22 | 0.89 / 1.01 | 0.45 / 1.73 |

  Where the kernel keeps more reads in flight, it runs close to full-device time. Where in-flight reads fall to
  about half, it takes 20–70% longer.
- This is the same mechanism found on MI325X: same bytes, fewer reads in flight. Here it also shows the other
  way round, since the sized-192 kernel raises in-flight reads and recovers the time. Single runs; observational.
- Eager-mode counters ran on gfx950 without the MI325X crash (n = 256 and 168). FETCH_SIZE per tick is equal at
  both levels (63.6 M). RDREQ_LEVEL per tick is 0.69× at 168.

## 4. E4 — service curves (3 acquisitions; ratio to 256)

| control | 192 | 168 | 128 | 80 |
|---|---:|---:|---:|---:|
| seqread4g (GB/s) | 0.835 | 0.876 | 0.895 | 0.557 |
| ilp1 / ilp8 | 0.673 / 1.019 | 0.671 / 0.899 | 0.678 / 0.756 | 0.341 / 0.390 |
| grid8192 | 1.000 | 0.977 | 0.897 | 0.574 |
| gather1g | 0.911 | 0.819 | 0.704 | 0.390 |
| reuse8m / reuse32m (cache-resident) | 0.654 / 0.671 | 0.577 / 0.649 | 0.545 / 0.592 | 0.278 / 0.301 |
| fma / mfma (TFLOP/s) | 0.854 / 0.668 | 0.765 / 0.603 | 0.638 / 0.502 | 0.334 / 0.250 |
| chase c4096 (ns/load) | 1.021 | 1.007 | 1.009 | 0.988 |

- With enough parallelism (ilp8, grid8192), the native stream holds 90–100% of full-device bandwidth down to
  128–192 CUs.
- One load in flight per thread (ilp1) loses a third of the bandwidth at every reduced level. This matches the
  in-flight explanation above.
- Single-chain pointer chase (c1) is *faster* at reduced CU counts (about 0.72×). This is consistent with higher
  gfx clocks when fewer CUs are busy. It is a reminder that every timing here is clock-confounded.

## 5. E8 — guardrails (3 acquisitions each; mean)

| workload | 256 | 168 (boundary) | 80 |
|---|---:|---:|---:|
| Qwen32B B1 decode, energy run (ms per step) | 12.61 | 14.84 | — |
| Qwen32B J/token (W) | 11.46 (909) | 12.47 (841) | — |
| Qwen32B prefill, C4096 (ms) | 327 | 395 | 742 |
| Qwen32B C16384 decode (ms per step) | 13.82 | 15.82 | — |
| Qwen7B B8 decode (ms per step) | 4.67 | 5.56 | 8.69 |
| Mistral-7B B1 decode (ms per step) | 4.16 | 4.73 | — |

- At the boundary level, every guardrail workload is 14–21% slower.
- Energy per token is about 9% higher: lower power does not offset the longer runtime.
- On MI350X, the default software stack gives no free CU reduction at 168.

## 6. E3 and E7

- **E3 (fixed partition):** the pre-registered gate did not release it, because leak v2 is insensitive on gfx950.
  The pilot's single acquisition at n = 192 is labelled pilot, releases nothing and is no isolation proof:
  - a memory competitor (2326 GB/s) on the other 64 CUs slowed decode 1.32×;
  - an arithmetic competitor (28 TFLOP/s) slowed it 0.99×.

  The direction matches MI325X.
- **E7 (service-response model):** not run on MI350X.
  - There was no pre-registered train/held/blind split for this cohort, and all levels are already observed. A
    "blind" check is no longer possible.
  - E7 also failed its blind check on MI325X.
  - The step-shaped R(n) is set by kernel geometry, not by a smooth service curve. A post-hoc fit would add
    nothing that can be tested.

## 7. What this means for the project

1. **The shoreline question needs the kernel in the loop.** Measured with vLLM's default kernel, MI350X decode
   needs all 256 CUs: any reduction costs 16–25%. With the GEMM cut to fit, 192 CUs (75%) give full-device
   decode speed. So the real CU requirement for memory-bound decode is set by achievable reads in flight per CU,
   not by the CU count itself.
2. **More bandwidth per CU raises the cost of a naive CU cut.** The cohorts rank MI300X < MI325X < MI350X at
   every matched fraction. The MLP explanation from MI325X now has a third, consistent point and a positive
   control: the sized-192 kernel raised in-flight reads and recovered the time.
3. **Limits:**
   - Every timing is clock-confounded, so no point is a qualified non-inferiority result.
   - The counters are single light runs.
   - E3 isolation on gfx950 is unproven.

## 8. Open items (not submitted)

- **A gfx950 leak probe.** E3 needs one, for example duration-based or with smaller LDS. It must be
  pre-registered and smoke-tested on `mi3501x` before any E3 run.
- **Why the sized kernel works at 24 and 16 CUs per XCD and fails at 26/23/21/19.** A CPU-side reading of the
  vLLM `wvSplitK` sizing for those n, then one short trace job, would settle it.
- **MI300X extension 492956** is still pending. Its node is down or drained. After it runs, a combined
  three-cohort results report follows.

If any of this is presented or published, the AMD acknowledgement is: "This work was supported in part by
Advanced Micro Devices, Inc. under the AMD University Program's AI & HPC Cluster".
