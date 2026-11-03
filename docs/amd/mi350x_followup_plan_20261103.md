# MI350X follow-up — pre-registration (3 November 2026)

This document was written and committed **before any data from the jobs it describes**. It records the rules and
predictions those jobs are judged against.
- Owner's instruction (2026-11-03): "yes, submit them", covering the two open items of
  [mi350x_campaign_review_20261103.md](mi350x_campaign_review_20261103.md) §8.
- Cohort: MI350X only (gfx950, 256 CUs). Nothing here is pooled with MI300X or MI325X.
- Code: cubw v1.5. `svc_bench_v5.hip`, `plan.py --followup`, reader v1.5 `analyze.py`, `e6_shapes.py --cu-counts`.

## A. Leak test v3: why v2 failed on gfx950, and the redesign

**What the 496074 raw data show (leak v2, `e1_leak_v2/*/competitor.jsonl`):**
- The v4 probe (one full-LDS workgroup per CU, completion counts) completed between 506,264 and 506,282
  workgroups on every one of its 64 CUs when run alone. That near-identical count is the signature of a fixed
  per-launch assignment of workgroups to CUs: 2048 workgroups per launch, 32 per CU.
- Beside the positive control, which shares 8 CUs, the probe kept its counts on the shared CUs (about 519,000).
  The *co-runner* was starved there instead: 11,767–14,266 workgroups on the shared CUs, against about 496,000
  elsewhere.
- So on gfx950, a completion count cannot show a foreign workgroup on a probe CU, whoever wins the CU.

**v3 probe (`svc_bench_v5 --mode cuprobe_t`):**
- No LDS, 256 threads.
- Each thread runs a dependent chase through an L2-resident 256 KiB ring, plus 4 FMA chains, for 500 iterations.
- Thread 0 adds the workgroup's residency time (`s_memrealtime`) and one completion to its physical CU.
- Each window reports per-CU completions and mean residency. The per-CU speed is
  **s_c = completions / summed residency**.
- A foreign workgroup co-resident on a probe CU competes for SIMD issue and the vector-memory path, so it
  lengthens residency on that CU. Without LDS, a full-LDS workgroup like `wvSplitK` can co-reside with the probe.

**Pre-registered rule (`leak_ok_v3`, also in the plan's `gates` text):**
- It is the v1.3 rule (`leak_v2_eval`) applied to s_c instead of completion rate:
  - q_c = (s_c beside X / s_c alone) / median over c of the same ratio;
  - every run must cover all 256 − n probe CUs;
  - natives are co-runners on 0..n−1;
  - positive controls are co-runners sharing 8 probe CUs (0..n+7).
- **Sensitivity:** each of three positive controls must flag ≥ 4 CUs with q_c < 0.85:
  - FMA;
  - streaming read;
  - the v4 full-LDS probe, which has `wvSplitK`-like residency.

  If any of them misses, the test is insensitive and `leak_ok_v3` is false.
- **Pass** iff, for every decoder run, min q_c ≥ (lowest min q_c of the native runs on the same GPU and mask)
  − 0.05.
- **Scope:** levels n = 192 and 128, on two GPUs.
- **Release:** `release_e3_minimal_v3` = placement_ok, lifecycle_ok, outputs_ok and leak_ok_v3. Only then do
  the E3-minimal cells run: Qwen32B, n ∈ {192, 128} × {absent, mem50, mem100, arith100}, 5 blocks, plus one
  inactive control.
- **The threshold will not be relaxed after data.**

## B. Shader-engine balance of `wvSplitK` (hypothesis H_SE)

**What is known before these jobs:**
- **vLLM's `wvSplitK`** (CPU reading of `/app/vllm/csrc/rocm/skinny_gemms.cu` in the container):
  - it launches `grid = CuCount` workgroups;
  - each workgroup has 16 waves and holds all 160 KiB of LDS, so one fits per CU;
  - each workgroup strides through output rows in rounds of `CuCount × waves × YTILE`;
  - `mindiv` trims active waves without adding rounds.

  The host-side rounds and waves do **not** explain the MI350X pattern. For example, o_proj gets an identical
  configuration at 192 and 184 (YTILE 2, 14 waves, 1 round), but runs 1.02× vs 1.55×.
- **Placement** (496074, `e1_placement` CU keys): contiguous masks interleave over XCDs and then over each
  XCD's 4 shader engines (SEs). Active CUs per SE per XCD:

  | n | 256 | 232 | 208 | 192 | 184 | 168 | 152 | 128 |
  |---|---|---|---|---|---|---|---|---|
  | CUs per SE | 8,8,8,8 | 8,7,7,7 | 7,7,6,6 | 6,6,6,6 | 6,6,6,5 | 6,5,5,5 | 5,5,5,4 | 4,4,4,4 |

  The levels where a kernel sized for n reached full speed (256, 192, 128) are exactly the balanced levels. On
  MI300X even the full device is unbalanced (9,10,10,9).

**H_SE:** with one 160 KiB workgroup per CU, the workgroup dispatcher spreads a launch's workgroups over the SEs
in a way that does not follow the active-CU count per SE. Where SEs differ, some CUs get a second workgroup that
runs only after the first finishes, and the kernel takes up to about two workgroup times.

**Predictions, with pass criteria judged on the jobs below:**
- **P1 (dispatch probe, `e6_dispatch`; 5 trials, 160 KiB LDS, 16 waves, grid = n under mask n):**
  - At the balanced levels 256, 192 and 128: max one workgroup per CU, and serialized workgroups = 0 in every
    trial.
  - At ≥ 3 of the 4 strongly unbalanced levels (208, 184, 168, 152): max workgroups per CU ≥ 2 and serialized
    workgroups > 0.
  - Grid = balanced(n) under mask n: max one workgroup per CU at every level.
  - The no-LDS control (`_lds0`) is recorded and not judged.
- **P2 (kernel shapes, `wvsplitk_c<balanced(n)>` under mask n, 3 acquisitions in the 8-GPU job):** at each
  unbalanced level, the balanced size is faster than the size n for ≥ 4 of 5 GEMM shapes.
- **P3 (decode, `e6_balanced`):** Qwen32B with `CUBW_CU_COUNT_OVERRIDE = balanced(n)` under mask n, paired by
  block with E2 at n and at 256, 8 blocks:
  - R_balanced(n) < R_e2(n) at every unbalanced level;
  - R_balanced(208) (size 192) is within ±0.03 of R_e6_decode(192) from the same job.

  Like every timing here, P3 is clock-confounded and observational.

H_SE is **supported** if P1 and P3 pass, and **refuted** if P1 fails. If P1 passes and P3 fails, the cause of
the decode effect is still open.

## C. Jobs

| job | partition | what | steps | estimate |
|---|---|---|---:|---|
| smoke | `mi3501x` (1 GPU, 0.175×) | placement at the 8 fine levels, the whole leak v3 suite at 192 and 128, the dispatch suite, E2 pilot at 256 and 208, `e6_balanced` at 208, `e6_decode` at 192, shapes at 256, 208 and 192 | 58 | 0.94 node-h, ≤ 1.3 units |
| follow-up | `mi3508x` (8 GPUs, 1.4×) | `--kind auto --followup`: pilot (placement on 8 GPUs, leak v3 on 2 GPUs, dispatch, E3 lifecycle, E2 pilot) → gate → production (E2-fine and `e6_balanced` on 8 blocks, `e6_decode` at 192 and 128, E3-minimal if released, kernel traces with the balanced size, shapes ×3) | 337 | ≤ 2.0 node-h, ≤ 23 units at estimate; earlier jobs used about a third of their estimate |

**Sequence:**
1. The 8-GPU job is submitted **held** behind the smoke.
2. It is released only after the smoke is reviewed.
3. A port bug found by the smoke is fixed first, and the held job is replaced by a resubmission from the fixed
   release. This follows the 496073 → 496074 pattern.
4. At most 2 attempts per step.

No clock, power or cluster setting is changed.
