# Chapter 06 — Goal follow-up G1–G8 (#29–#33) and overlap calibration OC0–OC7 (#34–#39)

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
  failed on gfx942, so G5 was not released there.
- **G6.** Sequential read per active CU saturates at 41 / 37 / 45 GB/s/CU at 96 CUs.
- **G7 (MI300X).** FETCH_SIZE per token = 1.0016 × logical compulsory read (request side).
- **G8.** REJECT (MAE 0.035; MI325X 96 error 0.133, MI300X 96 0.102, MI350X 96 0.066).

**What it means.** The single "≈ 42 GB/s per CU" threshold (C11) does not transfer:

- The gfx942 cohorts stay within 1–3.4 % of full speed at 55–63 GB/s/CU of advertised bandwidth (96 CUs).
- MI350X loses more than predicted.

What limits sized decode is a cohort-specific per-CU delivery ceiling, not the advertised bandwidth per CU.

**Failures and caveats.** Every timing is observational and clock-confounded (clock control is not authorized). On
MI325X there are no G7 steps (eager pmc is unusable on its firmware). G5 is unavailable on gfx942.

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
