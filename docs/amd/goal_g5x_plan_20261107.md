# G5 extension on gfx942 (MI300X, MI325X): leak test v5 and the G5 surface — pre-registration

Owner order (2026-11-07): "Do this G5 on MI300X/MI325X by submitting the slurm job."

Context:

- `docs/amd/goal_followup_results_20261107.md`: G5 was not released on MI300X (502425) or MI325X (502426) because the
  isolation gate (leak test v4 with the tuned decoder) failed on both.
- This plan was written and committed before any data from the jobs below.

No clock, power, driver or firmware change. Frozen G8 predictions and the goal_v1 results are not modified.

## 1. Why v4 failed, and what v5 adds

Leak v4 runs a start-delay probe (svc_bench_v6 `gapprobe`) on CUs n..303 while the decoder runs on 0..n-1. In
502425/502426 the probe was sensitive on every GPU. The positives (8 overlapping CUs) gave top-8 excess 0.42–0.64.
At n = 128 the tuned decoder flagged 0 CUs on every GPU. At n = 256 it flagged 32 of 48 probe CUs on most GPUs, while
the native controls (mem100, arith100 on 0..255) flagged 0. That is why v4 failed.

The following was seen before this plan, in a reread of the 502425 data with the v5 reader on the old root, and is
disclosed here:

- At n = 256 the decoder's excess is **uniform**: the top-8 mean equals the maximum, 0.251, on all 32 flagged CUs.
- On one GPU (L1), a native control also produced 0.246 on the same probe.

A decoder workgroup that took a probe CU would show the positive pattern: a few CUs with large excess. The uniform
pattern fits a probe launch delayed as a whole. One candidate cause is contention in the command processor and
workgroup dispatcher: vLLM graph decode issues **39,249 kernels/s on MI300X and 43,725 kernels/s on MI325X** (731
kernels per decode step, goal_v1 G2 route traces), against a handful of long kernels/s for the v4 natives.

Leak v5 keeps the v4 suite unchanged and adds two **launch-storm negative controls** on the decoder mask 0..n-1. Both
use `svc_bench_v8`:

| Control | Kernel | Grid | Launch pattern |
|---|---|---|---|
| `storm_lds` | full-LDS 1024-thread workgroups, 100 FMA iterations | one workgroup per CU of the mask (like wvSplitK) | 64 launches enqueued per synchronize |
| `storm_small` | 256-thread no-LDS workgroups | 4 per CU | same |

Every storm workgroup stays on the storm's own CU mask, so it cannot occupy a probe CU. Whatever probe delay a storm
causes is a non-leak effect at a known launch rate.

## 2. Jobs

| Job | Partition | Content | Estimate |
|---|---|---|---|
| smoke | `mi3001x` (1 GPU) | placement; leak v5 at n = 256; G5 absent/mem100/arith100 at 256; pilot-labelled, judges nothing | ~0.8 node-h, ~0.1 units |
| MI300X main | `mi3008x` (8 GPUs) | placement (every mask used); leak v5 at n = 256 and 128 on every GPU; then G5, 8 blocks | ~1.7 node-h (plan estimate), ~1.7–3.5 units |
| MI325X main | `mi3258x` (8 GPUs) | same | ~1.7 node-h, ~2.1–4.2 units |

Release rules:

- The mains are submitted held. They are released after the smoke has been reviewed for one thing: the storm controls
  run, and their kernel rate is recorded.
- If the smoke shows a storm rate below the decoder rate, the storm iteration count may be lowered once, before the
  mains are released. This is a pilot decision, made before any judged data.

G5 cells, unchanged from goal_v1:

- Qwen2.5-32B graph B1/C4096, tuned decoder (grid = n), n ∈ {256, 192, 128}.
- Treatments: absent, inactive, mem25, mem50, mem100, arith100. The competitor runs on CUs 256–303.
- 3 warmups and 10 trials per acquisition; one acquisition per block; 8 blocks (one per GPU).
- Randomized order per block (seed 20261105).

G5 runs whenever placement passes, as the owner ordered. The isolation verdict sets the **disposition** of the G5 rows,
not whether they run.

## 3. Leak test v5 rule (frozen)

The rule is evaluated per GPU and level n (probe on n..303). It is implemented in `scripts/amd/goal/analyze_g5x.py`.

**Quantities:**
- e_c: the v4 excess late fraction of probe CU c beside a run, relative to the probe alone.
- T8(run): the mean of the 8 largest e_c.
- M(run): the maximum e_c.
- Natives: mem100, arith100, storm_lds and storm_small, all on 0..n-1.
- Positives: cuprobe on 0..n+7, and the stock decoder on 0..n+7.

**Conditions:**
1. **Available:** at least one storm is dispatch-matched. Its measured kernel launch rate must be at least the
   decoder rate above (39,249/s MI300X, 43,725/s MI325X).
2. **Sensitive:** every positive has T8 ≥ max_native T8 + 0.10 and at least 4 flagged CUs (e_c ≥ 0.10).
3. **OK:** available, and sensitive, and both of:
   - T8(decoder) ≤ max_native T8 + (min_positive T8 − max_native T8) / 2;
   - M(decoder) ≤ max_native M + 0.10.

**Verdict:**
- `leak_ok_v5` = OK at every GPU and level.
- G5 disposition:
  - `isolated (leak v5)` if placement_ok, lifecycle_ok and leak_ok_v5;
  - otherwise `observational: isolation not established`. In that case mem100/arith100 is reported as the contrast
    robust to shared-CU effects.
- The v4 verdict is computed and reported unchanged.

**Falsifiers:**
- The storms are dispatch-matched but the decoder's T8 exceeds the midpoint. Isolation then stays unestablished, and
  the dispatch-contention explanation is weakened.
- A storm reproduces the decoder pattern. The v4 failure is then attributed to launch contention, not CU leakage.

## 4. Analysis and delivery

The in-job report is `analyze_g5x.py report`: `g5x_tables.json` and `g5x_summary.md`. The G5 statistics are identical
to goal_v1 (`analyze_goal.g5_surface`: block-paired bootstrap, 10,000 draws).

Published folders: `results_published/amd_<date>_<cohort>_goal_g5x_g5x_job<id>/`. A results note will be added to
`docs/amd/goal_followup_results_20261107.md` as a dated addendum, and to the experiment log and handbook.

## 5. Amendment after smoke 507394 (2026-11-07, before the mains were released)

Smoke result (`mi3001x`, 20/20 steps, 19.5 min; pilot-labelled, judges nothing):

- **storm_small:** ran at 137,056 kernels/s, 3.5× the MI300X decoder rate, so it is dispatch-matched.
- **storm_lds:** exited at start (`illegal memory access`). Its per-CU counter buffer was not allocated in `storm` mode
  (`svc_bench_v8.hip`: the `cuprobe` allocation branch did not include `storm`).
- **Fix:** add `storm` to that branch. The binary was rebuilt (sha256 7da07a48…). The smoke binary is kept as
  `svc_bench_gfx942_v8.smoke507394`.
- **Unchanged:** the kernels, iteration counts, rule and plans.
- **Smoke leak v5 at n = 256:**
  - decoder T8 0.001, against the T8 threshold of 0.329 and the minimum positive T8 of 0.645;
  - leak_ok_v5 true;
  - leak v4 also passed on this GPU.
- **Smoke G5 at n = 256, one block, pilot:** mem100 1.197, arith100 1.070.

The mains (507396, 507397) use the same release and plan. They read the rebuilt binary from `$WORK/bin`.
