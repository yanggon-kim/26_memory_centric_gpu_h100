# G5 on gfx942 with leak test v6 (shader-engine-balanced probe) — pre-registration

Owner order (2026-11-09), after the G5 extension results: prepare and submit a job with an isolation probe that behaves
consistently on gfx942 at 256 CUs. This plan was written and committed before any data from the jobs below.

No clock, power, driver or firmware change. Frozen G8 predictions, goal_v1 and goal_g5x results are not modified.
New results are a separate run: they are never pooled with 507396/507397.

## 1. Why v4 and v5 failed at n = 256 (diagnosis from existing data, disclosed)

Re-reading the probe-alone runs of 507396 (`e1_leak_v4/alone_gapprobe_256-303_L*`) by physical CU key
(xcc<<8 | se<<5 | sh<<4 | cu):

- Probe indices 256–303 give 6 CUs per XCD, split **1, 2, 2, 1** over the four shader engines, on every XCD and every GPU
  read (L1, L2).
- The probe launches 48 workgroups, 6 per XCD. Alone, the single CU of SE0 and of SE3 runs **two** probe workgroups in
  50 % of launches; each CU of SE1 and SE2 runs **none** in 25 % of launches. The same holds beside mem100.
- This is the pattern of round-robin dispatch over shader engines from a starting engine that moves between launches
  (6 workgroups over 4 engines → 2, 2, 1, 1 in a rotating order). The same mechanism as handbook C7 (two full-LDS
  workgroups on one CU when engines have unequal active CUs).
- The v4/v5 excess is computed as (late launches + launches without a probe workgroup) relative to the probe alone. If a
  co-runner shifts the rotation phase, the missing/late pattern moves to other CUs and every probe CU changes by a
  multiple of 0.25. That is the uniform 0.25 step seen with the decoder and, on some GPUs, with every kind of native.

So the v4/v5 probe at 256–303 measures dispatch-phase shifts as well as occupancy. A probe whose workgroups always land
one per CU, whatever the starting engine, removes that term.

## 2. Leak test v6 design

**CU map** (`scripts/amd/goal/cumap.py`, one `shell` step per GPU, first after placement):

- Mask indices are interleaved over the 8 XCDs (index i on XCD i mod 8). A mask that leaves an XCD empty is unsafe
  (work spills; cubw pilot review 2026-10-28), so indices are mapped 8 at a time: a placement run (svc_bench_v3) on
  `j..j+7` must expose exactly 8 CUs, one per XCD. All mapped CUs must be distinct (`map_ok`).
- Per level n (probe indices n..303), group the indices by (XCD, shader engine): sizes c_g, k = min c_g.
- List s0 = the first k indices of each group; list s1 = the last k. Both have exactly k CUs on every engine of every
  XCD, so 4k workgroups per XCD land one per CU in every launch. Together they cover every probe index when 2k ≥ max c_g
  (otherwise the level is infeasible: no masks, its steps fail to start, the level is `unavailable`).
- Each list is re-proven by a placement run on the comma-list mask itself (comma lists are accepted on this runtime,
  pilot review 2026-10-28): it must expose exactly the predicted CUs, or the list gets no mask.
- Positive overlap set of list s: from each XCD the index of that list in its lowest shader engine (8 CUs).

**Runs per GPU, level n ∈ {256, 192, 128} and list s ∈ {s0, s1}** (30 s each, randomized per level and GPU, seed 20261105):

| Run | Probe (svc_bench_v6 gapprobe, one full-LDS workgroup per CU of the list) beside |
|---|---|
| alone | nothing |
| natives | mem100, arith100, storm_lds, storm_small on 0..n-1 (as in v5) |
| positive | full-LDS cuprobe (svc_bench_v4) on 0..n-1 plus the 8 overlap CUs of list s |
| decoder | tuned Qwen2.5-32B graph B1/C4096 decoder (grid = n) on 0..n-1 |

The runner (`cubw.py`) resolves `@probe_n<n>_s<j>` / `@pos_n<n>_s<j>` masks from `<root>/cumap/<lane>.json` at run
time; gapprobe and storm block counts equal the CUs in the resolved mask.

Unchanged from goal_g5x: placement at every contiguous mask; G5 production (Qwen2.5-32B graph B1/C4096, tuned decoder
grid = n, n ∈ {256, 192, 128} × {absent, inactive, mem25, mem50, mem100, arith100}, competitor on 256–303, 8 blocks,
one per GPU, 3 warmups and 10 trials, randomized per block with a new seed label `g5y`). G5 runs when placement passes;
leak v6 sets only the disposition.

## 3. Leak test v6 rule (frozen; `scripts/amd/goal/analyze_g5y.py`)

Per GPU, level n and list s:

- **Valid:** the probe alone covers exactly the CUs of its list, and on every CU (launches without a probe workgroup +
  launches with two or more) / launches ≤ 0.02.
- e_c, T8 (mean of the 8 largest e_c), M (max e_c) and flags (e_c ≥ 0.10) as in v5.
- **Sensitive:** positive T8 ≥ max native T8 + 0.10 and positive flags ≥ 4.
- **OK:** valid and sensitive and T8(decoder) ≤ max native T8 + (positive T8 − max native T8) / 2 and
  M(decoder) ≤ max native M + 0.10.

`leak_ok_v6` = OK for every GPU × level × list; a missing run is not OK. G5 disposition: `isolated (leak v6)` if
placement_ok, lifecycle_ok and leak_ok_v6; otherwise `observational: isolation not established`, with mem100 / arith100
as the contrast robust to shared-CU effects. Per-level results are also reported.

**Falsifiers.**

- The balance diagnosis is wrong if alone runs are not valid on balanced lists (missing or multi > 0.02).
- If valid and sensitive lists still show decoder T8 above the midpoint, the decoder does reach probe CUs (or disturbs
  them through another path), and isolation is not established.

## 4. Jobs and release rules

| Job | Partition | Content | Plan estimate |
|---|---|---|---|
| smoke | `mi3001x` (1 GPU) | placement, CU map, leak v6 at 256/192/128, G5 absent/mem100/arith100 at 256; pilot-labelled, judges nothing | ~1.4 node-h est, ~0.17 units |
| MI300X main | `mi3008x` (8 GPUs) | placement, CU map and leak v6 on every GPU; G5, 8 blocks | ~2.1 node-h est, ~2.1 units |
| MI325X main | `mi3258x` (8 GPUs) | same | ~2.1 node-h est, ~2.6 units |

(The goal_g5x mains finished in 1.06 h against a 1.7 h estimate.)

The mains are submitted held. They are released after the smoke has been reviewed for three things only:

1. `map_ok` and both lists proven at each level;
2. the probe alone is valid on both lists at each level;
3. every step type runs (no start failures).

If (1) or (2) fails, the mains stay held and the result is reported. Any change needed before release is written here
as a dated amendment first. The rule in §3 is not changed after any judged data.

## 5. Delivery

Published folders `results_published/amd_<date>_<cohort>_goal_g5y_g5y_job<id>/` (and the smoke folder), with
`g5y_tables.json`, `g5y_summary.md`, `cumap/` files, gates, launch plan and raw SHA-256 list. A dated addendum to
`docs/amd/goal_followup_results_20261107.md`, the experiment log and handbook chapter 06.

AMD acknowledgement for any publication: "This work was supported in part by Advanced Micro Devices, Inc. under the
AMD University Program's AI & HPC Cluster".
