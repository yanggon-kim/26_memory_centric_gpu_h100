# cubw MI300X auto job 483498 review and v1.2 repair (30 October 2026)

Job 483498 `cubw-mi300x-auto` (mi3008x, k002-003, 2026-10-30 06:26–06:39 CDT, 13 min), release 79d8b597 (v1.1),
plan `$WORK/cubw_plans/mi300x_auto_20261028_091901_79d8b597.json`. Raw root:
`$WORK/results_amd/amd_20261030_mi300x_cubw_v1_auto_job483498` (968 files, `SHA256SUMS`); compact tables in
`results_published/amd_20261030_mi300x_cubw_v1_auto_job483498/`. Follows
[cubw_pilot_review_20261028.md](cubw_pilot_review_20261028.md). Pilot rows are one acquisition each and are never
promoted to production comparisons; nothing here is clock-qualified.

## 1. What ran

| stage | content | result |
|---|---|---|
| pilot (06:26–06:38) | placement ×8 GPUs; leak suites on L0/L1 (+ native co-runner references); lifecycle at n=232 (absent/inactive/mem100/arith100); Qwen32B E2 on 2 GPUs | 18/18 decoder runs complete, one output digest; 63 native/competitor steps rc=0 |
| gate (06:38) | `analyze.py gate` (v1.1) | **every release false** (see §2) |
| production (06:38–06:39) | 271 gated steps journalled not_run (E2 40, E3 193, E5/E8 38); ungated E4 ×3 ran | E4: 339/339 native steps rc=0 |

## 2. Gates

| gate | v1.1 (as run, `gates.json`) | v1.2 re-gate (`gates_regated_v12.json`) | diagnosis |
|---|---|---|---|
| placement_ok | pass | pass | 8 GPUs × {304, 232, 200, 152, 96, 232–303}, balanced per XCD |
| lifecycle_ok | pass | pass | v1.1 per-GPU residual check and tag sweep worked |
| outputs_ok | **fail** (164 runs, 18 complete) | pass (18/18, 1 digest) | **Reader defect.** The auto gate runs after the pilot stage, when all 146 production decoder steps are still unattempted; v1.1 counted them as incomplete runs. |
| leak_ok (v1) | fail | fail | probe during decode / alone = 0.61 |
| leak_ok_v11 | **fail** | **fail** | measurement, predeclared threshold kept (below) |
| clock_ok | fail | fail | pooled gfx 1300–1964 MHz (×1.51), mem 1049–1300 MHz |
| release_e2 / e5e8 | false | **true** | |
| release_e3_minimal / full | false | false | blocked by leak_ok_v11 |

leak_ok_v11 (decoder probe ratio ÷ highest native co-runner ratio, same probe mask, GPU and lane; must be ≥ 0.85):

| cell | decoder ratio | native arith100 | native mem100 | relative |
|---|---:|---:|---:|---:|
| n232 arith100 L0 | 0.614 | 0.749 | 0.592 | 0.820 |
| n232 arith100 L1 | 0.609 | 0.762 | 0.590 | 0.799 |
| n152 leakprobe L0 | 0.622 | 0.811 | 0.650 | 0.766 |
| n152 leakprobe L1 | 0.667 | 0.870 | 0.701 | 0.766 |

The decoder slows the probe more than a native FMA co-runner, and about as much as a native streaming-read
co-runner (decoder/mem100 = 0.95–1.04). That pattern fits a memory-bound decoder contending for shared memory
service. It is not proof against leakage, though. The gate compared against the gentlest co-runner, as
predeclared, and it fails; **E3 stays unreleased on MI300X**. The threshold is not changed after the fact.

## 3. Pilot observations (one acquisition each, clock-confounded)

- E2 Qwen32B R(n), 2 GPUs: 232 → 1.022; 200 → 1.009; 152 → 1.064; 96 → 1.078. This is consistent with pilot 481167
  (1.03 / 1.02 / 1.09 / 1.10). The boundary stays 200.
- E3 lifecycle cells, n=232 (ratio to absent): mem100 1.231 (competitor 1043 GB/s), arith100 1.066 (10.7 TFLOP/s),
  inactive 0.994. mem100 / arith100 = 1.155.
- E4 production (3 acquisitions, ratio to 304): FMA 0.82/0.76/0.65/0.47 and MFMA 0.76/0.67/0.54/0.36 at
  232/200/152/96. ILP-1 read 0.84/0.88/0.72/0.57. 8 MiB reuse 0.76/0.69/0.57/0.38. 9728-block streaming
  0.97–1.01 at every level. Dependent-chase latency is flat to within ±7%.

## 4. v1.2 repair (commit c7cf8aa9) and follow-up

- `analyze.py`: outputs_ok ignores steps that were never attempted (`not_run`). Attempted runs that failed still
  fail the gate. Two fixtures were added; 25/25 checks pass.
- `plan.py --e4-done-in <root>`: production without E4, because its three acquisitions are already in 483498.
- Submitted **487341** `cubw-mi300x-production` (mi3008x, 1:45:00):
  - Plan: `$WORK/cubw_plans/mi300x_production_20261030_064338_c7cf8aa9.json`.
  - Gates: `CUBW_GATES` = the v1.2 re-gate.
  - Content: placement, E2 (Qwen32B ×5, Qwen14B ×3 blocks), E5 ktrace, E8; boundary 200.
  - E3 cells are journalled not_run.
- MI325X 483499 (still pending on release 79d8b597) is left in the queue to keep its priority age:
  - Its pilot and E4 data are unaffected by the defect.
  - After it ends, its pilot is re-gated with v1.2 and an MI325X production job is submitted the same way.
