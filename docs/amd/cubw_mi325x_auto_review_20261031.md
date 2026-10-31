# cubw MI325X auto job 483499 review (31 October 2026)

## Run

- Job 483499 `cubw-mi325x-auto`: mi3258x, k002-001, 2026-10-31 03:57–04:13 CDT (17 min).
- Release 79d8b597 (v1.1).
- Plan: `$WORK/cubw_plans/mi325x_auto_20261028_091903_79d8b597.json`.
- Raw root: `$WORK/results_amd/amd_20261031_mi325x_cubw_v1_auto_job483499` (1311 files, `SHA256SUMS`).
- Compact tables: `results_published/amd_20261031_mi325x_cubw_v1_auto_job483499/`.

MI325X is a separate cohort from MI300X. Its pilot rows are single acquisitions and none of them is
clock-qualified.

## 1. Stages and gates

The v1.1 reader defect diagnosed in [cubw_auto_review_20261030.md](cubw_auto_review_20261030.md) recurred
here:

- The auto gate counted 190 unattempted production steps as incomplete runs.
- As a result, every release was false and only the ungated E4 ×3 production ran (339/339 rc=0).
- The pilot itself was clean:
  - 21/21 decoder runs complete, all with one digest.
  - Placement, lifecycle, timeline, E8 and E4 pilot steps all rc=0.
  - The only exception is the intentional unsafe empty-XCD mask probe `0-6` (rc=3), a revalidation item rather
    than a placement gate.

| gate | v1.1 as run | v1.2 re-gate (`gates_regated_v12.json`) |
|---|---|---|
| placement_ok / lifecycle_ok | pass / pass | pass / pass |
| outputs_ok | fail (reader defect) | pass (21/21, 1 digest) |
| leak_ok (v1) | fail | fail (probe 0.75–0.84 of alone) |
| leak_ok_v11 | pass | **pass**, relative 0.87 / 0.91 / 0.92 / 0.97 (≥ 0.85) |
| clock_ok | fail | fail: pooled gfx 1494–2076 MHz (×1.39), mem 1285–1500 MHz |
| release_e2 / e5e8 / e3_minimal | false | **true / true / true** |
| release_e3_full | false | false (clock gate) |

On MI325X the decoder's slowdown of the probe stays within the native co-runner envelope:

| probe | decoder ratio | native co-runner ratios |
|---|---:|---:|
| arith100 | 0.79–0.84 | 0.81–0.87 |

E3-minimal is therefore released for this cohort, as predeclared. MI300X failed the same gate (0.77–0.82), so the
two cohorts differ on it.

## 2. Pilot observations (one acquisition each, clock-confounded)

**E2, Qwen32B R(n).** Two GPUs, values at 232 / 200 / 152 / 96 CUs:

| GPU | 232 | 200 | 152 | 96 |
|---|---:|---:|---:|---:|
| GPU 1 | 1.069 | 1.014 | 1.153 | 1.108 |
| GPU 2 | 1.012 | 1.038 | 1.100 | 1.135 |

- The full-CU baseline is 16.23 ms/step.
- Qwen14B: 1.178 / 1.053 / 1.172 / 1.210.
- Single-run spread is large, so only the production blocks can settle the boundary. 200 is kept.

**E3, Qwen32B (ratio to absent at the same n):**

| competitor | n=232 | n=152 |
|---|---:|---:|
| mem100 | 1.169 (1170 GB/s) | 1.194 (1299 GB/s) |
| arith100 | 1.002 | 1.045 |
| inactive | 1.010 | – |

**E4 production (3 acquisitions, ratio to 304), at 232 / 200 / 152 / 96 CUs:**

| measurement | 232 | 200 | 152 | 96 |
|---|---:|---:|---:|---:|
| FMA | 0.76 | 0.71 | 0.56 | 0.40 |
| MFMA | 0.66 | 0.58 | 0.42 | 0.30 |
| ILP-1 read | 0.83 | 0.89 | 0.70 | 0.55 |
| 8 MiB reuse | 0.71 | 0.64 | 0.50 | 0.36 |
| 9728-block streaming | 1.01 | 1.02 | 1.00 | 0.95 |

Dependent-chase latency is flat. The pattern matches MI300X, with steeper losses for the compute-bound rows.

## 3. Follow-up

Submitted **489594** `cubw-mi325x-production` (mi3258x, 2:30:00):

- Release 99bcd1ad (v1.2 scripts).
- Plan: `$WORK/cubw_plans/mi325x_production_20261031_041507_99bcd1ad.json`.
- Gates: `CUBW_GATES` = the v1.2 re-gate.
- Content:
  - placement;
  - E2 (Qwen32B ×5 and Qwen14B ×3 blocks);
  - E3-minimal (Qwen32B, n=232/152, absent/mem50/mem100/arith100, plus the inactive control);
  - E5 ktrace and E8 at boundary 200.
- E4 is omitted (done here). Full-grid E3 is journalled not_run.
