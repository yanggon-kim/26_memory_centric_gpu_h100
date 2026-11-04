# Follow-ups v1.6 — pre-registration (4 November 2026)

This document was written and committed **before any data from the jobs it describes**. It records the rules and
predictions those jobs are judged against.
- Owner's instruction (2026-11-04): "Do Two follow-ups". These are the two items offered at the end of
  [mi350x_followup_review_20261103.md](mi350x_followup_review_20261103.md):
  - A: a leak test that works on gfx950, so E3 can run on MI350X;
  - B: whether shader-engine imbalance (H_SE) also shapes MI300X.
- Cohorts stay separate: A is MI350X only, B is MI300X only.
- Code: cubw v1.6:
  - `svc_bench_v6.hip` (`gapprobe`);
  - `plan.py --leak4` / `--se-study`;
  - reader `analyze.py` (`leak_v4_eval`, `leak_ok_v4`, `release_e3_minimal_v4`).
- Binaries: `svc_bench_gfx950_v6`, `svc_bench_gfx942_v6` and `svc_bench_gfx942_v5` (v5 source unchanged, newly
  built for gfx942). Hashes are in `$WORK/bin/svc_bench.sha256`.
- No clock, power or cluster setting is changed. Every timing stays observational and clock-confounded unless the
  pooled clock gate passes.

## A. Leak test v4 (MI350X): watch when the probe can start, not how fast it runs

**Why v2 and v3 were blind** (jobs 496074, 497081, 497082):
- A `wvSplitK`-like intruder holds all 160 KiB of a CU's LDS.
- When the probe runs continuously, it wins the CU and the intruder starves. So the probe's completions (v2) and
  residency (v3) are unchanged.

**v4 probe (`svc_bench_v6 --mode gapprobe`):**
- **Launch shape:**
  - each launch has one workgroup per probe CU (grid = 256 − n);
  - each workgroup has 16 waves and the full 160 KiB of LDS, so it cannot share a CU with any LDS-holding
    workgroup;
  - about 600 FMA iterations, which is roughly 20 µs.
- **The gap:** the host sleeps 200 µs between launches, so a foreign workgroup can take a probe CU while the
  probe is idle. When the next launch comes, the probe's workgroup for that CU must wait for the foreign one to
  finish, or it is placed elsewhere.
- **What each workgroup records:** its physical CU, start time and end time.
- **Per window and CU:**
  - launches in which the CU ran a probe workgroup;
  - launches in which its first workgroup started more than 5 µs after the launch's first start ("late");
  - launches in which it ran two or more workgroups.
- **Why 5 µs:** in 497082, start skew within a launch of a balanced full-LDS grid was ≤ 0.81 µs in every trial,
  so 5 µs is more than 6× that.

**Pre-registered rule (`leak_ok_v4`; the same text is in the plan's `gates`):**
- **Late fraction:** f_c = (late launches + launches with no probe workgroup on c) / launches, over the run's
  steady windows. For decoder runs, only the windows inside decode trials count.
- **Excess:** e_c = f_c(beside X) − f_c(alone), on the same GPU and probe mask.
- **Flag:** CU c is flagged if e_c ≥ 0.10.
- **Coverage:** the alone run must see all 256 − n probe CUs.
- **Natives:** read and FMA co-runners on 0..n−1.
- **Required positive controls, on 0..n+7** (8 shared probe CUs, one per XCD), each of which must flag ≥ 4 CUs:
  - the v4 full-LDS probe (`cuprobe`);
  - **the Qwen32B decoder itself** with mask 0..n+7.

  If either misses, the test is insensitive and `leak_ok_v4` is false.
- FMA and read co-runners on 0..n+7 are recorded but not required. They hold no LDS and can share a CU with the
  probe, so they are not expected to delay its start.
- **Pass:** every decoder run at n (mask 0..n−1) flags no more CUs than the most flagged by any native run on the
  same GPU and level.
- **Scope:**
  - levels n = 192 and 128 (probe masks 192–255 and 128–255);
  - all 8 GPUs, so 16 decoder tests;
  - `leak_ok_v4` needs every test to pass.
- **Release:** `release_e3_minimal_v4` = placement_ok, lifecycle_ok, outputs_ok and leak_ok_v4. Only then do the
  E3-minimal cells run:
  - Qwen32B, n ∈ {192, 128} × {absent, mem50, mem100, arith100};
  - 8 blocks, one per GPU, each also holding E2 at 256;
  - plus one inactive control.
- **The threshold will not be relaxed after data.**

**E3 expectation (direction only, observational):** memory-100% slows the decoder more than arithmetic-100% at
both levels, as on MI300X (1.18× and 1.21×) and MI325X (1.14× and 1.16×).

## B. Shader-engine balance on MI300X (H_SE on gfx942)

**Known before these jobs:**
- **SE layout** (placement keys, 492956): MI300X has 4 shader engines (SEs) per XCD. Contiguous masks give these
  active CUs per SE:

  | n | 304 | 288 | 272 | 256 | 232 | 224 | 200 | 192 | 152 | 128 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | CUs per SE | **9,10,10,9** | 9,9,9,9 | 9,9,8,8 | 8,8,8,8 | 8,7,7,7 | 7,7,7,7 | 7,6,6,6 | 6,6,6,6 | 5,5,5,4 | 4,4,4,4 |

  The levels 288, 256, 224, 192 and 128 are balanced. 304, 272, 232, 200 and 152 are not; the 288 and 304 rows
  are inferred from the same mask order.
- **`wvSplitK` on gfx942:** one 64 KiB (full-LDS) workgroup per CU, with grid = the device CU count (304) under
  any mask.
- **What MI300X has shown so far:**
  - The stock kernel has a non-monotone R(n): 272 1.068, 232 1.033, 184 1.010, 152 1.085.
  - The kernel sized to n is slower at every level (e6_decode, 492956).

**H_SE on MI300X:** the gfx942 dispatcher, like gfx950's, does not follow unequal active-CU counts per SE. Because
the full device itself is unbalanced, the **stock full-device decode is itself affected**.

**Predictions, judged on the 8-GPU job:**
- **P1m (dispatch probe on 2 GPUs; 5 trials; 16 waves; 64 KiB LDS; grid = n under mask n):**
  - (a) At the balanced masks 288, 256, 224, 192 and 128: max one workgroup per CU and 0 serialized, in every
    trial on both GPUs.
  - (b) At the full device 304, and at ≥ 3 of 272, 232, 200 and 152: max ≥ 2 workgroups per CU and serialized > 0,
    on both GPUs.
  - (c) Grid = balanced(n) under mask n gives max one workgroup per CU at every level.
- **P2m (kernel shapes, 3 acquisitions):** at each unbalanced level, including 304, the size balanced(n) is faster
  than the size n for ≥ 4 of 5 Qwen32B GEMM shapes. At 304, size n is the device size.
- **P3m (Qwen32B decode, 8 blocks, paired within GPU):**
  - (a) R_balanced(n) < R_e2(n) at each of 272, 232, 200 and 152. R is relative to the stock kernel at 304 in the
    same block.
  - (b) At the full device: T(304, kernel sized for 288) / T(304, stock) < 1, with upper 95% CI < 1. That is,
    using 288 of 304 CUs' worth of workgroups is faster than the stock launch.
  - Recorded but not judged: R_e6_decode(n) at the balanced masks, and R_balanced(272 → 256) vs R_e6_decode(256).

**Verdict:**
- **Holds on MI300X** if P1m and P3m pass.
- **Refuted on MI300X** if P1m fails.
- If P1m passes and P3m fails, the dispatch artefact exists on gfx942 but does not drive MI300X decode.

## C. Jobs

| job | partition | plan | steps | estimate |
|---|---|---|---:|---|
| A smoke | `mi3501x` (1 GPU) | `--kind pilot --leak4 --smoke`: placement at 256/200/192/136/128 and the competitor mask; the leak v4 suite at 192 and 128 | 22 | 0.63 node-h, ~0.9 units |
| A main | `mi3508x` (8 GPUs) | `--kind auto --leak4`: placement and the leak v4 suite on every GPU → gate → E3-minimal and E2 at 256 on 8 blocks | 249 | 1.12 node-h, ~12.6 units |
| B smoke | `mi3001x` (1 GPU) | `--kind pilot --se-study --smoke`: placement at the 10 levels; the dispatch suite; E2, `e6_balanced` at 304 and 272; `e6_decode` at 256; shapes at 304 and 272 | 52 | 0.69 node-h, ~0.7 units |
| B main | `mi3008x` (8 GPUs) | `--kind auto --se-study`: placement; dispatch on 2 GPUs; E2 pilot → gate → on 8 blocks: E2 at 10 levels, `e6_balanced` at 5, `e6_decode` at 5; kernel traces; shapes ×3 | 371 | 1.74 node-h, ~13.9 units |

**Sequence:**
1. Each 8-GPU job is submitted **held** behind its smoke and released only after the smoke is reviewed.
2. A port bug found by a smoke is fixed first, and the held job is replaced by a resubmission from the fixed
   release.
3. **Rule for A:** if the smoke shows v4 insensitive (either required positive flags < 4 at either level), the
   8-GPU job A is **cancelled, not released**, and E3 on MI350X stays blocked. No threshold is changed.
4. At most 2 attempts per step.
