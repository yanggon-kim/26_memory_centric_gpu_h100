# Chapter 06 — Goal follow-up G1–G8 (#29–#33), overlap calibration OC0–OC7 (#34–#39), G5 on gfx942 (#40–#42)

Two assignments ran at the same time on 2026-11-05 to 2026-11-07:

- G1–G8 (`docs/amd/goal_based_followup_20261105.md`): goal-driven follow-up on all three cohorts.
- OC0–OC7 (`docs/amd/overlap_calibration_followup_20261105.md`): compute/memory overlap calibration on single-GPU
  partitions.

They used separate plans, releases and run roots. The OC work never touched the queued G1–G8 releases or the frozen G8
predictions.

Results documents (read these for the full numbers):
- `docs/amd/goal_followup_results_20261107.md` — G1–G8 across the three cohorts, final G8 verdict.
- `docs/amd/overlap_calibration_results_20261105.md` — OC0–OC4, OC6 freeze and held-out verdict.
- `docs/amd/overlap_calibration_results_20261107.md` — transfer check, OC5 mechanism table, OC7 delivery.

---

## #29–#30 G1–G8 smokes (502423 MI300X, 502424 MI350X)

**At a glance.** One-GPU smokes of release fab0ea00 on `mi3001x` and `mi3501x`, run before the 8-GPU mains were
released.

**Question.** Does every G1–G8 step type run, and does the analyzer read its output?

**Results.** All step types ran. The analyzer needed fixes, which went into commit 2d81405a after the mains were
released. The mains were released unchanged, as the OC notice required.

**Files.** Review `docs/amd/goal_smoke_review_20261105.md`; `results_published/amd_20261105_{mi300x_goal_v1_smoke_job502423,mi350x_goal_v1_smoke_job502424}/`.

## #31–#33 G1–G8 mains (502425 MI300X, 502426 MI325X, 502427 MI350X)

**At a glance.**

| Cohort | Partition | Wall | Steps rc 0 | Folder |
|---|---|---|---|---|
| MI300X | `mi3008x` | 3 h 53 min | 593 / 594 | `amd_20261107_mi300x_goal_v1_auto_job502425` |
| MI325X | `mi3258x` | 3 h 12 min | 590 / 590 | `amd_20261106_mi325x_goal_v1_auto_job502426` |
| MI350X | `mi3508x` | 2 h 37 min | 710 / 710 | `amd_20261105_mi350x_goal_v1_auto_job502427` |

**Design.** The pre-registered plan is `docs/amd/goal_followup_execution_plan_20261105.md`. Its parts:

- G1: restriction curve at sized and stock grids.
- G2: isolated phases plus mixed serving (A full/stock, B full/sized, C reduced/sized, Cb boundary, D reduced/stock).
- G3: matched energy.
- G4: counter audit.
- G5: tuned decoder × memory/arithmetic competitors, gated by isolation leak v4.
- G6: kernel-shape and native service sweep.
- G7: traffic counters vs logical demand.
- G8: a frozen model judged on held-out points.

**How it was run.** The in-job goal analysis crashed on all three cohorts: the release snapshot had a `natives` reader
bug. Each report was regenerated after the job with the fixed `analyze_goal.py report` on a devel CPU node, so raw roots
stay untouched. The one MI300X failure is a G7 eager FETCH_SIZE step at the full mask. It timed out after 25 min and,
under the plan, was not retried.

**Results.**

- **G1 (sized grid).** R ≤ 1.005 down to 128 CUs on MI300X and MI325X, and 1.009 / 1.034 at 96. MI350X holds to 192 CUs
  (0.996), then 1.055 (160), 1.155 (128), 1.392 (96). The stock kernel on a reduced mask costs 2–19 %.
- **G2.** Serving at the reduced, sized mask loses 5–11 % of throughput at concurrency 2 and 20–28 % at concurrency 8.
  Prefill is compute-bound.
- **G3.** Board power does not change with the mask, so J/token follows step time.
- **G4.** No physical-HBM counter is validated on any cohort.
- **G5 (MI350X only).** A memory competitor costs the decoder 14–26 %; an arithmetic competitor costs ≤ 3 %. Isolation
  failed on gfx942, so G5 was not released there (later run as #40–#42, observational).
- **G6.** Sequential read per active CU saturates at 41 / 37 / 45 GB/s/CU at 96 CUs.
- **G7 (MI300X).** FETCH_SIZE per token = 1.0016 × logical compulsory read (request side).
- **G8.** REJECT (MAE 0.035; MI325X 96 error 0.133, MI300X 96 0.102, MI350X 96 0.066).

**What it means.** The single "≈ 42 GB/s per CU" threshold (C11) does not transfer:

- The gfx942 cohorts stay within 1–3.4 % of full speed at 55–63 GB/s/CU of advertised bandwidth (96 CUs).
- MI350X loses more than predicted.

What limits sized decode is a cohort-specific per-CU delivery ceiling, not the advertised bandwidth per CU.

**Failures and caveats.** Every timing is observational and clock-confounded (clock control is not authorized). On
MI325X there are no G7 steps (eager pmc is unusable on its firmware). G5 was unavailable on gfx942 in these jobs; see #40–#42.

**Files.** One folder per cohort above: `goal_tables.json`, `goal_summary.md`, `goal_status.json`, `model_inputs.json`,
`counter_audit.json`, `isa_meta_gfx9xx.json`. The final judge is `docs/amd/goal/g8_judge_final.json`.

---

## #34–#35 OC pilots (502623 MI300X, 502625 MI350X)

**At a glance.** One GPU each, on `mi3001x` and `mi3501x`; 45/45 steps each. Release d6dced90.

**Results.**

- Counters validated: SQ_INSTS_MFMA, SQ_INSTS_VMEM_RD, FETCH_SIZE (gfx942) and TCC_EA0_RDREQ(_LEVEL).
- The SQ_ACTIVE_INST_* counters are instruction counts, so candidate A is unavailable.
- ATT per-CU timelines (candidate B, `scripts/amd/oc/oc_att.py`) pass checks ii and iv. Check iii (split) is
  degenerate.

**Files.** Review `docs/amd/oc_pilot_review_20261105.md`; `results_published/amd_20261105_*_oc_v0_pilot_job*/`.

## #36–#37 OC development (502805, 502806) and freeze

**Results.**

- Issued MFMA = 4·m·⌈k/1024⌉ per call, exactly.
- The pre-registered M1 = O + C + M − η·min(C, M) fits η > 1, so η is not identifiable.
- ATT X/A_C = 0.94–1.00 on every shape.
- M1 and the amended M6 = sqrt(O² + (κM)²) were frozen before held-out.

**Files.** Review `docs/amd/oc_dev_review_20261105.md`; `docs/amd/oc/oc6_frozen_{mi300x,mi350x}.json`.

## #38–#39 OC held-out (502898, 502899) and synthesis

**Results.**

| Cohort | M1 within 10 % | M6 within 10 % | Accepted |
|---|---|---|---|
| MI300X | 40/60 | 53/60 | neither model |
| MI350X | 18/60 | 39/60 | neither model |

- The issued-MFMA form is exact on 48 new points per cohort.
- Transfer to Qwen2.5-7B graph decode: per-step projection error within 10 % for both models. Shape-level errors reach
  +31 %.
- OC5 mechanism table:
  - dispatch serialization: supported;
  - arithmetic shortage: not limiting down to 96–192 CUs;
  - per-CU delivery ceiling: supported, conditional;
  - memory-service sensitivity: supported on MI350X;
  - η as a constant: rejected;
  - LDS limit: not supported at probe occupancy.

**Files.** Per cohort, in `results_published/amd_20261105_*_oc_v0_heldout_job*/`: `validation.json`,
`transfer_qwen7b.json`, `observability.json`, `overlap_bounds.json`, `service_references.json`,
`calibration_frozen.json`, `model_inputs.json`, `oc_status.json`.

---

## #40–#42 G5 extension on gfx942 (507394 smoke, 507396 MI300X, 507397 MI325X)

**At a glance.**

| Job | Partition | Wall | Steps | Folder |
|---|---|---|---|---|
| smoke 507394 | `mi3001x` | 19.5 min | 20 / 20 | `amd_20261107_mi300x_goal_g5x_g5x_smoke_job507394` |
| MI300X 507396 | `mi3008x` | 1 h 04 min | 360 / 360 | `amd_20261108_mi300x_goal_g5x_g5x_job507396` |
| MI325X 507397 | `mi3258x` | 1 h 04 min | 360 / 360 | `amd_20261109_mi325x_goal_g5x_g5x_job507397` |

**Question.** Owner order: run G5 on MI300X/MI325X. Can isolation be established there with a probe that controls for
the decoder's kernel launch rate, and what does a competitor cost the tuned decoder?

**Design.** Pre-registration `docs/amd/goal_g5x_plan_20261107.md`. Leak test v5 = the leak v4 suite plus two
launch-storm negative controls on the decoder mask (`svc_bench_v8` `storm`, `storm_small`), at n = 256 and 128 on all
8 GPUs. Then G5: 3 levels × 6 treatments × 8 blocks. G5 runs on placement; leak v5 sets only the disposition.

**How it was run.** The smoke found a crash in `storm` mode (per-CU buffer not allocated). It was fixed and rebuilt,
and the fix was recorded as a §5 amendment before the mains were released. The mains ran unchanged, without retries.

**Results.**

- Leak v5: storms run at 139–176 k kernels/s (decoder 39–44 k/s), positives are sensitive, OK 8 / 8 at n = 128 but 6 / 8
  at n = 256 on each cohort → **leak_ok_v5 false**, G5 **observational: isolation not established**.
- The n = 256 probe step (≈ 0.25, uniform) also appears with storms and with low-rate natives on other GPUs, so it is not
  specific to launch rate.
- G5: mem100 1.19–1.26 (MI300X), 1.16–1.21 (MI325X), rising as the decoder has fewer CUs; arith100 1.03–1.06; inactive
  1.00. mem100 / arith100 = 1.10–1.20 at every level, CIs far from 1.

**What it means.** The MI350X ordering (memory competition hurts, arithmetic competition barely does) holds on gfx942 as
an observational result. The extra arithmetic cost on gfx942 (3–6 % vs ≤ 3 % on MI350X) cannot be assigned to the
memory system or to shared CUs while isolation is not established.

**Failures and caveats.** Clock-confounded like every timing. Isolation not established under the frozen rule.

**Files.** Results: addendum in `docs/amd/goal_followup_results_20261107.md`. Per folder: `g5x_tables.json`,
`g5x_summary.md`, `gates.json`, `launch_plan.json`, `cubw_tables.json`, `raw_SHA256SUMS`, `raw_root.txt`.
